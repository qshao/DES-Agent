from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass, field

from rdkit import Chem

from ..analogue_expansion import generate_analogues_tagged
from ..candidate_generation_ligand import generate_ligand_candidates
from ..chemistry_filter import canonicalize_smiles
from ._metal_helpers import _apply_ligand_reality_gate
from ..concurrency import run_concurrent
from ..llm.schemas import CandidateBrainstorm, CandidateReview
from ..multi_cycle import _family_ucb_scores
from ..predictors.stability_constants import predict_log_k
from ..schemas import CandidateProposal
from ..trajectory import CycleSnapshot, SearchTrajectory, TopEntry, shortlist_delta


def _safe_canon(smi: str) -> str:
    try:
        return canonicalize_smiles(smi)
    except ValueError:
        return smi


def _normalize_competitor_metals(competitor_metal: str | list[str]) -> list[str]:
    """Normalize a bare metal string or list of metals into a de-duplicated,
    order-preserving list. Raises ValueError if the result would be empty."""
    raw = [competitor_metal] if isinstance(competitor_metal, str) else list(competitor_metal)
    seen: set[str] = set()
    out: list[str] = []
    for m in raw:
        if m not in seen:
            seen.add(m)
            out.append(m)
    if not out:
        raise ValueError("competitor_metal must contain at least one metal ion")
    return out


@dataclass(frozen=True)
class SelectivityResult:
    """One ligand screened in a metal selectivity run.

    ``log_k_target`` and ``log_k_competitor`` hold raw ML predictions when
    ``stability_rule_weight == 0`` (the default). When the blend is enabled
    (``stability_rule_weight > 0``), they store the ML + rule-based blended
    value rather than the raw ML output.

    ``log_k_competitor``/``delta_log_k`` always hold the WORST-CASE off-target's
    value (the one keeping the ligand from being clean) — for a single off-target
    this is that off-target's value, unchanged from prior behavior.
    """

    ligand_smiles: str
    log_k_target: float
    log_k_competitor: float
    delta_log_k: float
    composite_score: float
    source: str
    source_id: str
    rationale: str
    log_k_competitors: dict[str, float] = field(default_factory=dict)
    worst_competitor_metal: str = ""


@dataclass
class SelectivityScreenOutcome:
    target_metal: str
    competitor_metals: list[str]
    results: list[SelectivityResult]
    n_screened: int
    n_cycles: int
    llm_brainstorm: list[CandidateBrainstorm] = field(default_factory=list)
    llm_candidate_reviews: list[CandidateReview] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    claim_verdicts: list[object] = field(default_factory=list)
    trajectory: object = None   # SearchTrajectory | None
    dft_results: dict = field(default_factory=dict)   # dict[str, DFTResult]


def _compute_composite(log_k_target: float, log_k_competitor: float,
                        w_affinity: float, w_selectivity: float) -> tuple[float, float]:
    delta = log_k_target - log_k_competitor
    score = w_affinity * log_k_target + w_selectivity * delta
    return delta, score


def _deduplicate_proposals(
    proposals: list[CandidateProposal], seen: set[str]
) -> list[CandidateProposal]:
    out: list[CandidateProposal] = []
    for p in proposals:
        canon = canonicalize_smiles(p.smiles)
        if canon is None or canon in seen:
            continue
        seen.add(canon)
        out.append(CandidateProposal(
            smiles=canon,
            rationale=p.rationale,
            family=p.family,
            source=p.source,
            source_id=p.source_id,
        ))
    return out


def _llm_proposals_from_brainstorm(
    brainstorms: list[CandidateBrainstorm],
) -> list[CandidateProposal]:
    out: list[CandidateProposal] = []
    for b in brainstorms:
        mol = Chem.MolFromSmiles(b.smiles)
        if mol is None:
            print(f"[selectivity] invalid SMILES from LLM (skipped): {b.smiles!r}", file=sys.stderr)
            continue
        out.append(CandidateProposal(
            smiles=b.smiles,
            rationale=b.rationale,
            family=b.family,
            source="llm",
            source_id="brainstorm",
        ))
    return out


def _score_proposal_pair(
    target_metal: str,
    competitor_metals: list[str],
    proposal: CandidateProposal,
    model_path,
    w_affinity: float,
    w_selectivity: float,
    stability_rule_weight: float = 0.0,
) -> tuple[SelectivityResult | None, list[str]]:
    warnings: list[str] = []
    try:
        pred_target = predict_log_k(
            target_metal, proposal.smiles, model_path=model_path, allow_fallback=True
        )
        pred_competitors = {
            metal: predict_log_k(metal, proposal.smiles, model_path=model_path, allow_fallback=True)
            for metal in competitor_metals
        }
    except Exception as exc:
        warnings.append(f"Prediction failed for {proposal.smiles}: {exc}")
        return None, warnings

    val_target = pred_target.value
    val_competitors = {metal: pred.value for metal, pred in pred_competitors.items()}
    # Blend in the rule-based (Irving-Williams + HSAB + chelate) log K so the
    # metal *difference* reflects real coordination chemistry rather than the
    # heuristic model's near-uniform output.
    if stability_rule_weight > 0.0:
        try:
            from ..chemistry.stability_rules import rule_based_log_k

            rt = rule_based_log_k(target_metal, proposal.smiles)
            rc_by_metal = {
                metal: rule_based_log_k(metal, proposal.smiles) for metal in val_competitors
            }
            w = stability_rule_weight
            val_target = (1.0 - w) * val_target + w * rt
            for metal, rc in rc_by_metal.items():
                val_competitors[metal] = (1.0 - w) * val_competitors[metal] + w * rc
        except Exception as exc:
            warnings.append(f"stability-rule blend failed for {proposal.smiles}: {exc}")

    worst_metal = max(val_competitors, key=val_competitors.get)
    worst_val = val_competitors[worst_metal]

    delta_log_k, composite_score = _compute_composite(
        val_target, worst_val, w_affinity, w_selectivity
    )
    return SelectivityResult(
        ligand_smiles=proposal.smiles,
        log_k_target=val_target,
        log_k_competitor=worst_val,
        delta_log_k=delta_log_k,
        composite_score=composite_score,
        source=proposal.source,
        source_id=proposal.source_id,
        rationale=proposal.rationale,
        log_k_competitors=val_competitors,
        worst_competitor_metal=worst_metal,
    ), warnings


def _top_k_stable(
    prev: list[SelectivityResult], curr: list[SelectivityResult], k: int = 5
) -> bool:
    prev_smiles = {r.ligand_smiles for r in prev[:k]}
    curr_smiles = {r.ligand_smiles for r in curr[:k]}
    return prev_smiles == curr_smiles


def _build_selectivity_context(
    target_metal: str,
    competitor_metal: str | list[str],
    prev_results: list[SelectivityResult],
    cycle: int,
    w_affinity: float,
    w_selectivity: float,
    des_compatible_hints: list[str] | None = None,
    des_incompatible_hints: list[str] | None = None,
    family_hit_scores: dict[str, list[float]] | None = None,
    saturated_families: set[str] | None = None,
) -> str:
    metals = _normalize_competitor_metals(competitor_metal)
    competitor_line = (
        f"Competitor metal: {metals[0]}" if len(metals) == 1
        else f"Off-target metals: {', '.join(metals)}"
    )
    lines = [
        f"Target metal: {target_metal}",
        competitor_line,
        f"Selectivity weight: {w_selectivity} | Affinity weight: {w_affinity}",
        f"Cycle: {cycle}",
    ]
    if prev_results:
        lines.append("Top ligands from previous cycle (highest composite score first):")
        for r in prev_results[:5]:
            lines.append(
                f"  - {r.ligand_smiles}: log_K({target_metal})={r.log_k_target:.2f}, "
                f"log_K({r.worst_competitor_metal})={r.log_k_competitor:.2f}, "
                f"ΔlogK={r.delta_log_k:.2f}, score={r.composite_score:.2f}"
            )
        failed = [r for r in prev_results if r.delta_log_k <= 0][:3]
        if failed:
            lines.append(f"Non-selective ligands (ΔlogK ≤ 0, avoid similar structures):")
            for r in failed:
                lines.append(f"  - {r.ligand_smiles}: ΔlogK={r.delta_log_k:.2f}")
    if des_compatible_hints:
        lines.append("Ligands that formed DES in previous pass (prefer similar scaffolds):")
        for smiles in des_compatible_hints:
            lines.append(f"  - {smiles}")
    if des_incompatible_hints:
        lines.append("Ligands that did NOT form DES (avoid similar scaffolds):")
        for smiles in des_incompatible_hints:
            lines.append(f"  - {smiles}")
    if family_hit_scores:
        items = sorted(family_hit_scores.items(), key=lambda x: -sum(x[1]) / len(x[1]))
        lines.append("Productive selectivity families (avg composite score):")
        for fam, scores in items[:3]:
            avg = sum(scores) / len(scores)
            lines.append(f"  - {fam}: {len(scores)} selective hits, avg score={avg:.2f}")
    if saturated_families:
        lines.append("Exhausted families (diminishing returns, avoid repeating):")
        for fam in sorted(saturated_families)[:4]:
            lines.append(f"  - {fam}")
    return "\n".join(lines)



_SELECTIVITY_SUCCESS_THRESHOLD = 0.0  # delta_log_k > 0 = selective


def run_metal_selectivity_screen(
    target_metal: str,
    competitor_metal: str | list[str],
    n: int = 20,
    model_path=None,
    llm_provider=None,
    constraints: dict | None = None,
    n_cycles: int = 1,
    w_affinity: float = 0.5,
    w_selectivity: float = 0.5,
    des_compatible_hints: list[str] | None = None,
    des_incompatible_hints: list[str] | None = None,
    stability_rule_weight: float = 0.0,
    binding_pH: float = 7.0,
    dft_validate: bool = False,
    dft_top_n: int = 3,
) -> SelectivityScreenOutcome:
    from ..chemistry.claim_grounding import ground_coordination as _ground_coord
    from ..chemistry_filter import murcko_scaffold_smiles as _scaffold

    competitor_metals = _normalize_competitor_metals(competitor_metal)

    seen_smiles: set[str] = set()
    all_reviews: list[CandidateReview] = []
    all_brainstorm: list[CandidateBrainstorm] = []
    all_warnings: list[str] = []
    all_sel_verdicts: list[object] = []
    all_coord_verdicts: list[object] = []
    cumulative_results: list[SelectivityResult] = []
    prev_cycle_results: list[SelectivityResult] = []
    sel_snapshots: list[CycleSnapshot] = []
    sel_prev_labels: list[str] = []
    sel_converged = False
    cycles_run = 0

    # Cross-cycle tracking for Features 1, 3, 4
    family_hit_scores: dict[str, list[float]] = {}  # family → [composite_score] for selective hits
    family_fail_ledger: Counter[str] = Counter()
    scaffold_counts: dict[str, dict] = {}
    transform_hit_counts: Counter[str] = Counter()
    transform_fail_counts: Counter[str] = Counter()
    analogue_transform_map: dict[str, str] = {}

    for cycle in range(1, n_cycles + 1):
        cycles_run = cycle
        failing_scaffolds: set[str] = set()
        saturated_families: set[str] = set()
        if cycle > 1:
            failing_scaffolds = {
                sc for sc, counts in scaffold_counts.items()
                if counts["fail"] >= 3 and counts["hit"] == 0
            }
            hit_counts = Counter({fam: len(v) for fam, v in family_hit_scores.items()})
            ucb = _family_ucb_scores(hit_counts, family_fail_ledger)
            saturated_families = {
                fam for fam, score in ucb.items()
                if score < 0.5 and (hit_counts.get(fam, 0) + family_fail_ledger.get(fam, 0)) >= 5
            }

        proposals: list[CandidateProposal] = []

        if cycle == 1 or llm_provider is None:
            heuristic_n = max(n // 2, 5) if (llm_provider is not None and cycle == 1) else n
            heuristic = generate_ligand_candidates(target_metal, heuristic_n, constraints)
            proposals.extend(_deduplicate_proposals(heuristic, seen_smiles))

        # Feature 2: generate structural analogues of top selective hits and near-misses
        if cycle > 1 and prev_cycle_results:
            transform_weights: dict[str, float] = {}
            for tx in set(transform_hit_counts) | set(transform_fail_counts):
                h = transform_hit_counts.get(tx, 0)
                f = transform_fail_counts.get(tx, 0)
                transform_weights[tx] = (h + 1) / (h + f + 2)

            top_hits = [r for r in prev_cycle_results if r.delta_log_k > _SELECTIVITY_SUCCESS_THRESHOLD][:2]
            _near_miss_floor = _SELECTIVITY_SUCCESS_THRESHOLD - 0.5
            near_misses = sorted(
                [r for r in prev_cycle_results
                 if _near_miss_floor <= r.delta_log_k <= _SELECTIVITY_SUCCESS_THRESHOLD],
                key=lambda r: r.delta_log_k, reverse=True,
            )[:2]
            analogue_proposals: list[CandidateProposal] = []
            for r in top_hits:
                for analogue_smi, tx_name in generate_analogues_tagged(r.ligand_smiles, max_n=4, transform_weights=transform_weights):
                    analogue_transform_map.setdefault(analogue_smi, tx_name)
                    analogue_proposals.append(CandidateProposal(
                        smiles=analogue_smi,
                        rationale=f"analogue of {r.ligand_smiles} (ΔlogK={r.delta_log_k:.2f}, via {tx_name})",
                        family="analogue",
                        source="analogue",
                        source_id=r.ligand_smiles,
                    ))
            for r in near_misses:
                for analogue_smi, tx_name in generate_analogues_tagged(r.ligand_smiles, max_n=3, transform_weights=transform_weights):
                    analogue_transform_map.setdefault(analogue_smi, tx_name)
                    analogue_proposals.append(CandidateProposal(
                        smiles=analogue_smi,
                        rationale=f"near-miss analogue of {r.ligand_smiles} (ΔlogK={r.delta_log_k:.2f}, via {tx_name})",
                        family="analogue",
                        source="near_miss_analogue",
                        source_id=r.ligand_smiles,
                    ))
            proposals.extend(_deduplicate_proposals(analogue_proposals, seen_smiles))

        if llm_provider is not None:
            context = _build_selectivity_context(
                target_metal, competitor_metals, prev_cycle_results, cycle, w_affinity, w_selectivity,
                des_compatible_hints=des_compatible_hints,
                des_incompatible_hints=des_incompatible_hints,
                family_hit_scores=family_hit_scores if cycle > 1 else None,
                saturated_families=saturated_families if cycle > 1 else None,
            )
            try:
                brainstorms = llm_provider.brainstorm_ligands_selectivity(
                    target_metal, competitor_metals, constraints, context
                )
                all_brainstorm.extend(brainstorms)
                llm_proposals = _llm_proposals_from_brainstorm(brainstorms)
                proposals.extend(_deduplicate_proposals(llm_proposals, seen_smiles))
                for b in brainstorms:
                    if b.rationale:
                        try:
                            v = _ground_coord(b.smiles, b.rationale, pH=binding_pH)
                            all_coord_verdicts.append(v)
                            if v.status == "contradicted":
                                all_warnings.append(
                                    f"[GROUNDING] Coordination contradicted for {b.smiles}: {v.detail}"
                                )
                        except Exception:
                            pass
                proposals = _apply_ligand_reality_gate(target_metal, brainstorms, proposals, all_warnings)
            except Exception as exc:
                all_warnings.append(f"LLM brainstorm failed (cycle {cycle}): {exc}")

        # Feature 4: drop proposals whose scaffold consistently fails selectivity
        if failing_scaffolds:
            proposals = [
                p for p in proposals
                if not (sc := _scaffold(p.smiles)) or sc not in failing_scaffolds
            ]

        proposals = proposals[:n]
        if not proposals:
            break

        cycle_results: list[SelectivityResult] = []
        for proposal in proposals:
            try:
                result, warnings = _score_proposal_pair(
                    target_metal, competitor_metals, proposal, model_path, w_affinity, w_selectivity,
                    stability_rule_weight=stability_rule_weight,
                )
            except Exception as exc:
                all_warnings.append(f"Scoring failed for {proposal.smiles}: {exc}")
                continue
            all_warnings.extend(warnings)
            if result is not None:
                cycle_results.append(result)

        # Ground selectivity claims (rule-based ΔlogK vs. LLM-implied target > competitor)
        from ..chemistry.claim_grounding import ground_selectivity as _ground_sel
        _sel_verdicts: list[object] = []
        for r in cycle_results:
            if r.source != "llm":
                continue
            try:
                v = _ground_sel(target_metal, r.worst_competitor_metal, r.ligand_smiles, "target_selective")
                _sel_verdicts.append(v)
                if v.status == "contradicted":
                    all_warnings.append(
                        f"[GROUNDING] Selectivity contradicted for {r.ligand_smiles}: {v.detail}"
                    )
            except Exception:
                pass
        all_sel_verdicts.extend(_sel_verdicts)

        if llm_provider is not None:
            context = _build_selectivity_context(
                target_metal, competitor_metals, prev_cycle_results, cycle, w_affinity, w_selectivity,
                des_compatible_hints=des_compatible_hints,
                des_incompatible_hints=des_incompatible_hints,
            )
            review_results = run_concurrent(cycle_results, lambda r: llm_provider.review_ligand(target_metal, r.ligand_smiles, context))
            for r, res in zip(cycle_results, review_results):
                if res.error is not None:
                    all_warnings.append(f"LLM review failed for {r.ligand_smiles}: {res.error}")
                    continue
                all_reviews.append(res.value)

        # Update cross-cycle trackers
        ligand_family_map = {
            _safe_canon(b.smiles): b.family for b in all_brainstorm
        }
        for r in cycle_results:
            family = ligand_family_map.get(r.ligand_smiles, "")
            sc = _scaffold(r.ligand_smiles)
            tx_name = analogue_transform_map.get(r.ligand_smiles)
            if r.delta_log_k > _SELECTIVITY_SUCCESS_THRESHOLD:
                if family and family not in ("", "unknown", "analogue", "heuristic"):
                    family_hit_scores.setdefault(family, []).append(r.composite_score)
                if sc:
                    scaffold_counts.setdefault(sc, {"hit": 0, "fail": 0})["hit"] += 1
                if tx_name:
                    transform_hit_counts[tx_name] += 1
            else:
                if family and family not in ("", "unknown", "analogue", "heuristic"):
                    family_fail_ledger[family] += 1
                if sc:
                    scaffold_counts.setdefault(sc, {"hit": 0, "fail": 0})["fail"] += 1
                if tx_name:
                    transform_fail_counts[tx_name] += 1

        by_smiles = {r.ligand_smiles: r for r in cumulative_results}
        for r in cycle_results:
            existing = by_smiles.get(r.ligand_smiles)
            if existing is None or r.composite_score > existing.composite_score:
                by_smiles[r.ligand_smiles] = r
        cumulative_results = sorted(
            by_smiles.values(), key=lambda r: r.composite_score, reverse=True
        )

        top_score = f"{cumulative_results[0].composite_score:.2f}" if cumulative_results else "n/a"
        print(
            f"[cycle {cycle}/{n_cycles}] screened={len(proposals)} top_score={top_score}",
            file=sys.stderr,
            flush=True,
        )

        this_converged = cycle > 1 and _top_k_stable(prev_cycle_results, cumulative_results)
        try:
            top5 = cumulative_results[:5]
            entries = [
                TopEntry(
                    label=r.ligand_smiles, metric_name="composite_score",
                    metric_value=r.composite_score,
                    secondary=f"ΔlogK={r.delta_log_k:.2f}, logK({target_metal})={r.log_k_target:.2f}",
                )
                for r in top5
            ]
            curr_labels = [r.ligand_smiles for r in top5]
            snap_new, snap_drop = shortlist_delta(sel_prev_labels, curr_labels)
            sel_snapshots.append(CycleSnapshot(
                cycle=cycle,
                n_screened=len(proposals),
                n_hits=sum(1 for r in cumulative_results if r.delta_log_k > 0),
                top_entries=entries,
                new_entrants=snap_new if cycle > 1 else [],
                dropouts=snap_drop if cycle > 1 else [],
                converged=this_converged,
                convergence_reason="top-5 ligand set stable vs previous cycle" if this_converged else "",
            ))
            sel_prev_labels = curr_labels
        except Exception as exc:
            all_warnings.append(f"[TRAJECTORY] snapshot capture failed (cycle {cycle}): {exc}")

        if this_converged:
            sel_converged = True
            print(
                f"[cycle {cycle}/{n_cycles}] top-5 stable — converged early",
                file=sys.stderr,
                flush=True,
            )
            break

        prev_cycle_results = list(cumulative_results)

    # --- Optional DFT validation stage (post-loop) ---
    dft_results_map: dict = {}
    if dft_validate and cumulative_results:
        from ..chemistry.dft_cache import cached_compute_dft_properties as _dft
        from ..chemistry.dft_selectivity import dft_selectivity_adjustment as _dft_adj
        from ..llm.base import nominate_for_dft_fallback as _dft_fallback

        top_k_pool = cumulative_results[: min(dft_top_n * 2, len(cumulative_results))]
        if llm_provider is not None and hasattr(llm_provider, "nominate_for_dft"):
            try:
                nominated_smiles = llm_provider.nominate_for_dft(
                    top_k_pool, target_metal, competitor_metals, dft_top_n
                )
            except Exception as exc:
                all_warnings.append(
                    f"[DFT] LLM nomination failed, using top-{dft_top_n} by score: {exc}"
                )
                nominated_smiles = _dft_fallback(top_k_pool, dft_top_n)
        else:
            nominated_smiles = _dft_fallback(top_k_pool, dft_top_n)

        for smi in nominated_smiles:
            res = _dft(smi, pH=binding_pH)
            dft_results_map[smi] = res
            if not res.success:
                all_warnings.append(f"[DFT] Warning: skipping {smi[:40]!r} — {res.error}")

        successful = [r for r in dft_results_map.values() if r.success]
        if nominated_smiles and not successful:
            all_warnings.append(
                "[DFT] Warning: all DFT computations failed — rule-based ranking used"
            )
        elif successful:
            import dataclasses as _dc
            updated = []
            for r in cumulative_results:
                dft_res = dft_results_map.get(r.ligand_smiles)
                if dft_res and dft_res.success:
                    adj = _dft_adj(dft_res, target_metal, r.worst_competitor_metal)
                    r = _dc.replace(r, composite_score=r.composite_score + adj)
                updated.append(r)
            cumulative_results = sorted(updated, key=lambda r: r.composite_score, reverse=True)
    # --- End DFT stage ---

    sel_trajectory = SearchTrajectory(
        workflow="metal-selectivity",
        headline=f"{target_metal} over {', '.join(competitor_metals)} selectivity",
        metric_label="composite score",
        snapshots=sel_snapshots,
        total_cycles=cycles_run,
        converged=sel_converged,
        convergence_reason=(sel_snapshots[-1].convergence_reason if sel_snapshots else ""),
        final_summary=(sel_snapshots[-1].top_entries if sel_snapshots else []),
    )
    return SelectivityScreenOutcome(
        target_metal=target_metal,
        competitor_metals=competitor_metals,
        results=cumulative_results,
        n_screened=len(seen_smiles),
        n_cycles=n_cycles,
        llm_brainstorm=all_brainstorm,
        llm_candidate_reviews=all_reviews,
        warnings=all_warnings,
        claim_verdicts=all_sel_verdicts + all_coord_verdicts,
        trajectory=sel_trajectory,
        dft_results=dft_results_map,
    )
