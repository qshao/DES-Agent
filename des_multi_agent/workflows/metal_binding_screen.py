from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field

from rdkit import Chem

from ..candidate_generation_ligand import generate_ligand_candidates
from ..chemistry_filter import canonicalize_smiles
from ..llm.schemas import CandidateBrainstorm, CandidateReview
from ..predictors.stability_constants import StabilityConstantPrediction, predict_log_k
from ..schemas import CandidateProposal


@dataclass(frozen=True)
class LigandScreenResult:
    metal_ion: str
    ligand_smiles: str
    prediction: StabilityConstantPrediction
    log_k: float
    source: str
    source_id: str
    rationale: str


@dataclass
class MetalBindingScreenOutcome:
    metal_ion: str
    results: list[LigandScreenResult]
    n_screened: int
    n_cycles: int
    llm_candidate_reviews: list[CandidateReview] = field(default_factory=list)
    llm_brainstorm: list[CandidateBrainstorm] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    claim_verdicts: list[object] = field(default_factory=list)


def _score_proposals(
    metal_ion: str,
    proposals: list[CandidateProposal],
    model_path=None,
    allow_fallback: bool = True,
) -> tuple[list[LigandScreenResult], list[str]]:
    results: list[LigandScreenResult] = []
    warnings: list[str] = []
    for proposal in proposals:
        try:
            pred = predict_log_k(
                metal_ion, proposal.smiles, model_path=model_path, allow_fallback=allow_fallback
            )
            results.append(
                LigandScreenResult(
                    metal_ion=metal_ion,
                    ligand_smiles=proposal.smiles,
                    prediction=pred,
                    log_k=pred.value,
                    source=proposal.source,
                    source_id=proposal.source_id,
                    rationale=proposal.rationale,
                )
            )
        except Exception as exc:
            warnings.append(f"Prediction failed for {proposal.smiles}: {exc}")
    results.sort(key=lambda r: r.log_k, reverse=True)
    return results, warnings


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
            print(f"[metal-binding] invalid SMILES from LLM (skipped): {b.smiles!r}", file=sys.stderr)
            continue
        out.append(CandidateProposal(
            smiles=b.smiles,
            rationale=b.rationale,
            family=b.family,
            source="llm",
            source_id="brainstorm",
        ))
    return out


def _top_k_stable(prev: list[LigandScreenResult], curr: list[LigandScreenResult], k: int = 5) -> bool:
    prev_smiles = {r.ligand_smiles for r in prev[:k]}
    curr_smiles = {r.ligand_smiles for r in curr[:k]}
    return prev_smiles == curr_smiles


_LOG_K_SUCCESS_THRESHOLD = 4.0  # minimum log_k considered "good binding"


def _family_ucb_scores(
    hit_scores: dict[str, list[float]],
    fail_counts: "Counter[str]",
    *,
    C: float = 1.4,
) -> dict[str, float]:
    """UCB1 score per family for metal-binding exploration."""
    all_families = set(hit_scores) | set(fail_counts)
    N_total = sum(len(v) for v in hit_scores.values()) + sum(fail_counts.values()) + 1
    scores: dict[str, float] = {}
    for fam in all_families:
        h = len(hit_scores.get(fam, []))
        f = fail_counts.get(fam, 0)
        n = h + f
        if n == 0:
            scores[fam] = float("inf")
            continue
        hit_rate = h / n
        exploration = C * math.sqrt(math.log(N_total) / n)
        scores[fam] = hit_rate + exploration
    return scores


def run_metal_binding_screen(
    metal_ion: str,
    n: int = 20,
    model_path=None,
    llm_provider=None,
    constraints: dict | None = None,
    n_cycles: int = 1,
    binding_pH: float = 7.0,
) -> MetalBindingScreenOutcome:
    from ..chemistry.claim_grounding import ground_coordination as _ground_coord
    from ..chemistry_filter import murcko_scaffold_smiles as _scaffold
    from ..analogue_expansion import generate_analogues_tagged
    from collections import Counter

    seen_smiles: set[str] = set()
    all_reviews: list[CandidateReview] = []
    all_brainstorm: list[CandidateBrainstorm] = []
    all_warnings: list[str] = []
    all_coord_verdicts: list[object] = []
    cumulative_results: list[LigandScreenResult] = []
    prev_cycle_results: list[LigandScreenResult] = []

    # Cross-cycle tracking for Features 1, 3, 4
    family_hit_scores: dict[str, list[float]] = {}  # family → [log_k] for successes
    family_fail_ledger: Counter[str] = Counter()
    scaffold_counts: dict[str, dict] = {}           # scaffold_smi → {"hit": int, "fail": int}
    # Enhancement 4: per-transform hit/fail counts for adaptive selection
    transform_hit_counts: Counter[str] = Counter()
    transform_fail_counts: Counter[str] = Counter()
    # smiles → transform_name for analogues generated this run
    analogue_transform_map: dict[str, str] = {}

    for cycle in range(1, n_cycles + 1):
        # Compute saturation + failing scaffolds for LLM context and proposal filter
        failing_scaffolds: set[str] = set()
        saturated_families: set[str] = set()
        if cycle > 1:
            failing_scaffolds = {
                sc for sc, counts in scaffold_counts.items()
                if counts["fail"] >= 3 and counts["hit"] == 0
            }
            ucb = _family_ucb_scores(family_hit_scores, family_fail_ledger)
            saturated_families = {
                fam for fam, score in ucb.items()
                if score < 0.5 and (len(family_hit_scores.get(fam, [])) + family_fail_ledger.get(fam, 0)) >= 5
            }

        proposals: list[CandidateProposal] = []

        if cycle == 1 or llm_provider is None:
            heuristic = generate_ligand_candidates(metal_ion, n, constraints)
            proposals.extend(_deduplicate_proposals(heuristic, seen_smiles))

        # Feature 2: generate structural analogues of top hits and near-misses in cycle 2+
        if cycle > 1 and prev_cycle_results:
            # Enhancement 4: bias transforms toward historically productive ones
            transform_weights: dict[str, float] = {}
            all_tx = set(transform_hit_counts) | set(transform_fail_counts)
            for tx in all_tx:
                h = transform_hit_counts.get(tx, 0)
                f = transform_fail_counts.get(tx, 0)
                transform_weights[tx] = (h + 1) / (h + f + 2)  # Laplace smoothed hit rate

            top_hits = [r for r in prev_cycle_results if r.log_k >= _LOG_K_SUCCESS_THRESHOLD][:2]
            non_hits = [r for r in prev_cycle_results if r.log_k < _LOG_K_SUCCESS_THRESHOLD]
            _near_miss_floor = _LOG_K_SUCCESS_THRESHOLD - 0.5
            near_misses = sorted(
                [r for r in non_hits if r.log_k >= _near_miss_floor],
                key=lambda r: r.log_k, reverse=True,
            )[:2]
            analogue_proposals: list[CandidateProposal] = []
            for r in top_hits:
                for analogue_smi, tx_name in generate_analogues_tagged(r.ligand_smiles, max_n=4, transform_weights=transform_weights):
                    analogue_transform_map[analogue_smi] = tx_name
                    analogue_proposals.append(CandidateProposal(
                        smiles=analogue_smi,
                        rationale=f"analogue of {r.ligand_smiles} (log_K={r.log_k:.2f}, via {tx_name})",
                        family="analogue",
                        source="analogue",
                        source_id=r.ligand_smiles,
                    ))
            for r in near_misses:
                for analogue_smi, tx_name in generate_analogues_tagged(r.ligand_smiles, max_n=3, transform_weights=transform_weights):
                    analogue_transform_map[analogue_smi] = tx_name
                    analogue_proposals.append(CandidateProposal(
                        smiles=analogue_smi,
                        rationale=f"near-miss analogue of {r.ligand_smiles} (log_K={r.log_k:.2f}, via {tx_name})",
                        family="analogue",
                        source="near_miss_analogue",
                        source_id=r.ligand_smiles,
                    ))
            proposals.extend(_deduplicate_proposals(analogue_proposals, seen_smiles))

        if llm_provider is not None:
            context = _build_context(
                metal_ion, prev_cycle_results, cycle,
                family_hit_scores=family_hit_scores if cycle > 1 else None,
                saturated_families=saturated_families if cycle > 1 else None,
            )
            try:
                brainstorms = llm_provider.brainstorm_ligands(metal_ion, constraints, context)
                all_brainstorm.extend(brainstorms)
                llm_proposals = _llm_proposals_from_brainstorm(brainstorms)
                proposals.extend(_deduplicate_proposals(llm_proposals, seen_smiles))
                # Ground coordination claims from LLM rationale
                _coord_verdicts: list[object] = []
                for b in brainstorms:
                    if b.rationale:
                        try:
                            v = _ground_coord(b.smiles, b.rationale, pH=binding_pH)
                            _coord_verdicts.append(v)
                            if v.status == "contradicted":
                                all_warnings.append(
                                    f"[GROUNDING] Coordination contradicted for {b.smiles}: {v.detail}"
                                )
                        except Exception:
                            pass
                all_coord_verdicts.extend(_coord_verdicts)
            except Exception as exc:
                all_warnings.append(f"LLM brainstorm failed (cycle {cycle}): {exc}")

        # Feature 4: drop proposals whose scaffold consistently fails
        if failing_scaffolds:
            proposals = [
                p for p in proposals
                if not (sc := _scaffold(p.smiles)) or sc not in failing_scaffolds
            ]

        proposals = proposals[:n]
        if not proposals:
            break

        cycle_results, warnings = _score_proposals(metal_ion, proposals, model_path)
        all_warnings.extend(warnings)

        if llm_provider is not None:
            context = _build_context(metal_ion, prev_cycle_results, cycle)
            for r in cycle_results:
                try:
                    review = llm_provider.review_ligand(metal_ion, r.ligand_smiles, context)
                    all_reviews.append(review)
                except Exception as exc:
                    all_warnings.append(f"LLM review failed for {r.ligand_smiles}: {exc}")

        # Update cross-cycle trackers
        def _safe_canon(smi: str) -> str:
            try:
                return canonicalize_smiles(smi)
            except ValueError:
                return smi

        ligand_family_map = {
            _safe_canon(b.smiles): b.family for b in all_brainstorm
        }
        for r in cycle_results:
            family = ligand_family_map.get(r.ligand_smiles, "")
            sc = _scaffold(r.ligand_smiles)
            tx_name = analogue_transform_map.get(r.ligand_smiles)
            if r.log_k >= _LOG_K_SUCCESS_THRESHOLD:
                if family and family not in ("", "unknown", "analogue"):
                    family_hit_scores.setdefault(family, []).append(r.log_k)
                if sc:
                    scaffold_counts.setdefault(sc, {"hit": 0, "fail": 0})["hit"] += 1
                if tx_name:
                    transform_hit_counts[tx_name] += 1
            else:
                if family and family not in ("", "unknown", "analogue"):
                    family_fail_ledger[family] += 1
                if sc:
                    scaffold_counts.setdefault(sc, {"hit": 0, "fail": 0})["fail"] += 1
                if tx_name:
                    transform_fail_counts[tx_name] += 1

        # Merge cycle results into cumulative, keeping best log_k per SMILES
        by_smiles = {r.ligand_smiles: r for r in cumulative_results}
        for r in cycle_results:
            existing = by_smiles.get(r.ligand_smiles)
            if existing is None or r.log_k > existing.log_k:
                by_smiles[r.ligand_smiles] = r
        cumulative_results = sorted(by_smiles.values(), key=lambda r: r.log_k, reverse=True)

        top_log_k = f"{cumulative_results[0].log_k:.2f}" if cumulative_results else "n/a"
        print(
            f"[cycle {cycle}/{n_cycles}] screened={len(proposals)} top log_K={top_log_k}",
            file=sys.stderr,
            flush=True,
        )

        if cycle > 1 and _top_k_stable(prev_cycle_results, cumulative_results):
            print(f"[cycle {cycle}/{n_cycles}] top-5 stable — converged early", file=sys.stderr, flush=True)
            break

        prev_cycle_results = list(cumulative_results)

    return MetalBindingScreenOutcome(
        metal_ion=metal_ion,
        results=cumulative_results,
        n_screened=len(seen_smiles),
        n_cycles=n_cycles,
        llm_candidate_reviews=all_reviews,
        llm_brainstorm=all_brainstorm,
        warnings=all_warnings,
        claim_verdicts=all_coord_verdicts,
    )


def _build_context(
    metal_ion: str,
    prev_results: list[LigandScreenResult],
    cycle: int,
    family_hit_scores: dict[str, list[float]] | None = None,
    saturated_families: set[str] | None = None,
) -> str:
    lines = [f"Metal ion: {metal_ion}", f"Cycle: {cycle}"]
    if prev_results:
        lines.append("Top ligands from previous cycle (highest log K first):")
        for r in prev_results[:5]:
            lines.append(f"  - {r.ligand_smiles}: log_K={r.log_k:.2f}, rationale={r.rationale}")
        # Negative feedback: show poor performers
        failed = [r for r in prev_results if r.log_k < _LOG_K_SUCCESS_THRESHOLD][:3]
        if failed:
            lines.append("Poor-binding ligands from previous cycle (log_K < 4, avoid similar):")
            for r in failed:
                lines.append(f"  - {r.ligand_smiles}: log_K={r.log_k:.2f}")
    if family_hit_scores:
        items = sorted(family_hit_scores.items(), key=lambda x: -sum(x[1]) / len(x[1]))
        lines.append("Productive binding families (avg log_K):")
        for fam, scores in items[:3]:
            avg = sum(scores) / len(scores)
            lines.append(f"  - {fam}: {len(scores)} hits, avg log_K={avg:.2f}")
    if saturated_families:
        lines.append("Exhausted families (diminishing returns, avoid repeating):")
        for fam in sorted(saturated_families)[:4]:
            lines.append(f"  - {fam}")
    return "\n".join(lines)
