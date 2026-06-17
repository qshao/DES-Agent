from __future__ import annotations

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
    seen_smiles: set[str] = set()
    all_reviews: list[CandidateReview] = []
    all_brainstorm: list[CandidateBrainstorm] = []
    all_warnings: list[str] = []
    all_coord_verdicts: list[object] = []
    cumulative_results: list[LigandScreenResult] = []
    prev_cycle_results: list[LigandScreenResult] = []

    for cycle in range(1, n_cycles + 1):
        proposals: list[CandidateProposal] = []

        if cycle == 1 or llm_provider is None:
            heuristic = generate_ligand_candidates(metal_ion, n, constraints)
            proposals.extend(_deduplicate_proposals(heuristic, seen_smiles))

        if llm_provider is not None:
            context = _build_context(metal_ion, prev_cycle_results, cycle)
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

        proposals = proposals[:n]
        if not proposals:
            break

        cycle_results, warnings = _score_proposals(metal_ion, proposals, model_path)
        all_warnings.extend(warnings)

        if llm_provider is not None:
            for r in cycle_results:
                context = _build_context(metal_ion, prev_cycle_results, cycle)
                try:
                    review = llm_provider.review_ligand(metal_ion, r.ligand_smiles, context)
                    all_reviews.append(review)
                except Exception as exc:
                    all_warnings.append(f"LLM review failed for {r.ligand_smiles}: {exc}")

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
) -> str:
    lines = [f"Metal ion: {metal_ion}", f"Cycle: {cycle}"]
    if prev_results:
        lines.append("Top ligands from previous cycle (highest log K first):")
        for r in prev_results[:5]:
            lines.append(f"  - {r.ligand_smiles}: log_K={r.log_k:.2f}, rationale={r.rationale}")
    return "\n".join(lines)
