from __future__ import annotations

import sys
from dataclasses import dataclass, field

from rdkit import Chem

from ..candidate_generation_ligand import generate_ligand_candidates
from ..chemistry_filter import canonicalize_smiles
from ..llm.schemas import CandidateBrainstorm, CandidateReview
from ..predictors.stability_constants import predict_log_k
from ..schemas import CandidateProposal


@dataclass(frozen=True)
class SelectivityResult:
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
    delta_log_k, composite_score = _compute_composite(
        pred_target.value, pred_competitor.value, w_affinity, w_selectivity
    )
    return SelectivityResult(
        ligand_smiles=proposal.smiles,
        log_k_target=pred_target.value,
        log_k_competitor=pred_competitor.value,
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
) -> SelectivityScreenOutcome:
    seen_smiles: set[str] = set()
    all_reviews: list[CandidateReview] = []
    all_brainstorm: list[CandidateBrainstorm] = []
    all_warnings: list[str] = []
    cumulative_results: list[SelectivityResult] = []
    prev_cycle_results: list[SelectivityResult] = []

    for cycle in range(1, n_cycles + 1):
        proposals: list[CandidateProposal] = []

        if cycle == 1 or llm_provider is None:
            heuristic_n = max(n // 2, 5) if (llm_provider is not None and cycle == 1) else n
            heuristic = generate_ligand_candidates(target_metal, heuristic_n, constraints)
            proposals.extend(_deduplicate_proposals(heuristic, seen_smiles))

        if llm_provider is not None:
            context = _build_selectivity_context(
                target_metal, competitor_metal, prev_cycle_results, cycle, w_affinity, w_selectivity
            )
            try:
                brainstorms = llm_provider.brainstorm_ligands_selectivity(
                    target_metal, competitor_metal, constraints, context
                )
                all_brainstorm.extend(brainstorms)
                llm_proposals = _llm_proposals_from_brainstorm(brainstorms)
                proposals.extend(_deduplicate_proposals(llm_proposals, seen_smiles))
            except Exception as exc:
                all_warnings.append(f"LLM brainstorm failed (cycle {cycle}): {exc}")

        proposals = proposals[:n]
        if not proposals:
            break

        cycle_results: list[SelectivityResult] = []
        for proposal in proposals:
            result, warnings = _score_proposal_pair(
                target_metal, competitor_metal, proposal, model_path, w_affinity, w_selectivity
            )
            all_warnings.extend(warnings)
            if result is not None:
                cycle_results.append(result)

        if llm_provider is not None:
            context = _build_selectivity_context(
                target_metal, competitor_metal, prev_cycle_results, cycle, w_affinity, w_selectivity
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

        if cycle > 1 and _top_k_stable(prev_cycle_results, cumulative_results):
            print(
                f"[cycle {cycle}/{n_cycles}] top-5 stable — converged early",
                file=sys.stderr,
                flush=True,
            )
            break

        prev_cycle_results = list(cumulative_results)

    return SelectivityScreenOutcome(
        target_metal=target_metal,
        competitor_metal=competitor_metal,
        results=cumulative_results,
        n_screened=len(seen_smiles),
        n_cycles=n_cycles,
        llm_brainstorm=all_brainstorm,
        llm_candidate_reviews=all_reviews,
        warnings=all_warnings,
    )
