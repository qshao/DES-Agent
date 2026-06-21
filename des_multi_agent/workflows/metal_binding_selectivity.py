from __future__ import annotations

import sys
from dataclasses import dataclass, field

from rdkit import Chem

from ..candidate_generation_ligand import generate_ligand_candidates
from ..chemistry_filter import canonicalize_smiles
from ..llm.schemas import CandidateBrainstorm, CandidateReview
from ..predictors.stability_constants import predict_log_k
from ..schemas import CandidateProposal
from ..trajectory import CycleSnapshot, SearchTrajectory, TopEntry, shortlist_delta


@dataclass(frozen=True)
class SelectivityResult:
    """One ligand screened in a metal selectivity run.

    ``log_k_target`` and ``log_k_competitor`` hold raw ML predictions when
    ``stability_rule_weight == 0`` (the default). When the blend is enabled
    (``stability_rule_weight > 0``), they store the ML + rule-based blended
    value rather than the raw ML output.
    """

    ligand_smiles: str
    log_k_target: float
    log_k_competitor: float
    delta_log_k: float
    composite_score: float
    source: str
    source_id: str
    rationale: str


@dataclass
class SelectivityScreenOutcome:
    target_metal: str
    competitor_metal: str
    results: list[SelectivityResult]
    n_screened: int
    n_cycles: int
    llm_brainstorm: list[CandidateBrainstorm] = field(default_factory=list)
    llm_candidate_reviews: list[CandidateReview] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    claim_verdicts: list[object] = field(default_factory=list)
    trajectory: object = None   # SearchTrajectory | None


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
    competitor_metal: str,
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
        pred_competitor = predict_log_k(
            competitor_metal, proposal.smiles, model_path=model_path, allow_fallback=True
        )
    except Exception as exc:
        warnings.append(f"Prediction failed for {proposal.smiles}: {exc}")
        return None, warnings

    val_target = pred_target.value
    val_competitor = pred_competitor.value
    # Blend in the rule-based (Irving-Williams + HSAB + chelate) log K so the
    # metal *difference* reflects real coordination chemistry rather than the
    # heuristic model's near-uniform output.
    if stability_rule_weight > 0.0:
        try:
            from ..chemistry.stability_rules import rule_based_log_k

            rt = rule_based_log_k(target_metal, proposal.smiles)
            rc = rule_based_log_k(competitor_metal, proposal.smiles)
            w = stability_rule_weight
            val_target = (1.0 - w) * val_target + w * rt
            val_competitor = (1.0 - w) * val_competitor + w * rc
        except Exception as exc:
            warnings.append(f"stability-rule blend failed for {proposal.smiles}: {exc}")

    delta_log_k, composite_score = _compute_composite(
        val_target, val_competitor, w_affinity, w_selectivity
    )
    return SelectivityResult(
        ligand_smiles=proposal.smiles,
        log_k_target=val_target,
        log_k_competitor=val_competitor,
        delta_log_k=delta_log_k,
        composite_score=composite_score,
        source=proposal.source,
        source_id=proposal.source_id,
        rationale=proposal.rationale,
    ), warnings


def _top_k_stable(
    prev: list[SelectivityResult], curr: list[SelectivityResult], k: int = 5
) -> bool:
    prev_smiles = {r.ligand_smiles for r in prev[:k]}
    curr_smiles = {r.ligand_smiles for r in curr[:k]}
    return prev_smiles == curr_smiles


def _build_selectivity_context(
    target_metal: str,
    competitor_metal: str,
    prev_results: list[SelectivityResult],
    cycle: int,
    w_affinity: float,
    w_selectivity: float,
    des_compatible_hints: list[str] | None = None,
    des_incompatible_hints: list[str] | None = None,
) -> str:
    lines = [
        f"Target metal: {target_metal}",
        f"Competitor metal: {competitor_metal}",
        f"Selectivity weight: {w_selectivity} | Affinity weight: {w_affinity}",
        f"Cycle: {cycle}",
    ]
    if prev_results:
        lines.append("Top ligands from previous cycle (highest composite score first):")
        for r in prev_results[:5]:
            lines.append(
                f"  - {r.ligand_smiles}: log_K({target_metal})={r.log_k_target:.2f}, "
                f"log_K({competitor_metal})={r.log_k_competitor:.2f}, "
                f"ΔlogK={r.delta_log_k:.2f}, score={r.composite_score:.2f}"
            )
    if des_compatible_hints:
        lines.append("Ligands that formed DES in previous pass (prefer similar scaffolds):")
        for smiles in des_compatible_hints:
            lines.append(f"  - {smiles}")
    if des_incompatible_hints:
        lines.append("Ligands that did NOT form DES (avoid similar scaffolds):")
        for smiles in des_incompatible_hints:
            lines.append(f"  - {smiles}")
    return "\n".join(lines)


def run_metal_selectivity_screen(
    target_metal: str,
    competitor_metal: str,
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
) -> SelectivityScreenOutcome:
    from ..chemistry.claim_grounding import ground_coordination as _ground_coord
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

    for cycle in range(1, n_cycles + 1):
        proposals: list[CandidateProposal] = []

        if cycle == 1 or llm_provider is None:
            heuristic_n = max(n // 2, 5) if (llm_provider is not None and cycle == 1) else n
            heuristic = generate_ligand_candidates(target_metal, heuristic_n, constraints)
            proposals.extend(_deduplicate_proposals(heuristic, seen_smiles))

        if llm_provider is not None:
            context = _build_selectivity_context(
                target_metal, competitor_metal, prev_cycle_results, cycle, w_affinity, w_selectivity,
                des_compatible_hints=des_compatible_hints,
                des_incompatible_hints=des_incompatible_hints,
            )
            try:
                brainstorms = llm_provider.brainstorm_ligands_selectivity(
                    target_metal, competitor_metal, constraints, context
                )
                all_brainstorm.extend(brainstorms)
                llm_proposals = _llm_proposals_from_brainstorm(brainstorms)
                proposals.extend(_deduplicate_proposals(llm_proposals, seen_smiles))
                # Ground coordination claims from LLM rationale (species-aware).
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
            except Exception as exc:
                all_warnings.append(f"LLM brainstorm failed (cycle {cycle}): {exc}")

        proposals = proposals[:n]
        if not proposals:
            break

        cycle_results: list[SelectivityResult] = []
        for proposal in proposals:
            result, warnings = _score_proposal_pair(
                target_metal, competitor_metal, proposal, model_path, w_affinity, w_selectivity,
                stability_rule_weight=stability_rule_weight,
            )
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
                v = _ground_sel(target_metal, competitor_metal, r.ligand_smiles, "target_selective")
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
                target_metal, competitor_metal, prev_cycle_results, cycle, w_affinity, w_selectivity,
                des_compatible_hints=des_compatible_hints,
                des_incompatible_hints=des_incompatible_hints,
            )
            for r in cycle_results:
                try:
                    review = llm_provider.review_ligand(target_metal, r.ligand_smiles, context)
                    all_reviews.append(review)
                except Exception as exc:
                    all_warnings.append(f"LLM review failed for {r.ligand_smiles}: {exc}")

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

    sel_trajectory = SearchTrajectory(
        workflow="metal-selectivity",
        headline=f"{target_metal} over {competitor_metal} selectivity",
        metric_label="composite score",
        snapshots=sel_snapshots,
        total_cycles=len(sel_snapshots),
        converged=sel_converged,
        convergence_reason=(sel_snapshots[-1].convergence_reason if sel_snapshots else ""),
        final_summary=(sel_snapshots[-1].top_entries if sel_snapshots else []),
    )
    return SelectivityScreenOutcome(
        target_metal=target_metal,
        competitor_metal=competitor_metal,
        results=cumulative_results,
        n_screened=len(seen_smiles),
        n_cycles=n_cycles,
        llm_brainstorm=all_brainstorm,
        llm_candidate_reviews=all_reviews,
        warnings=all_warnings,
        claim_verdicts=all_sel_verdicts + all_coord_verdicts,
        trajectory=sel_trajectory,
    )
