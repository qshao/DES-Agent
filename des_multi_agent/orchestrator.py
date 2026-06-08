from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from .candidate_generation import generate_candidates
from .chemistry_filter import canonicalize_smiles, filter_candidates
from .config import DEFAULT_ABSOLUTE_TM_MAX_K, DEFAULT_RELATIVE_DROP_MIN
from .discovery import load_discovery_library, literature_lookup, merge_discovery_candidates, similarity_search
from .exporting import export_des_run_bundle
from .reporting import format_report
from .evaluation import DesResult, classify_des
from .llm.factory import build_llm_provider
from .llm.schemas import CandidateBrainstorm, CandidateReview, CritiqueNote, ExplanationNote
from .paths import resolve_existing_path
from .prediction import predict_curve
from .run_memory import apply_run_memory_preferences, build_run_memory, load_run_memory_history, write_run_memory
from .predictors.designsolvents import ViscosityPrediction, predict_viscosity
from .property_resolution import resolve_melting_point
from .ranking import rank_results
from .schemas import CandidateProposal, DesThresholds
from .uncertainty import (
    AnnotatedResult,
    MinimumTmUncertainty,
    UncertaintyPolicy,
    apply_uncertainty_policy,
    estimate_min_tm_uncertainty,
    rank_annotated_results,
)


@dataclass(frozen=True)
class SearchOutcome:
    results: list[DesResult]
    annotated_results: list[AnnotatedResult]
    candidate_proposals: list[CandidateProposal]
    candidate_reviews: list[CandidateReview]
    brainstorm_candidates: list[CandidateBrainstorm]
    explanation_notes: list[ExplanationNote]
    critique_notes: list[CritiqueNote]
    llm_warnings: list[str]
    memory_notes: list[str] = field(default_factory=list)
    viscosity_predictions: list[ViscosityPrediction] = field(default_factory=list)


def _merge_candidates(*candidate_groups):
    merged = []
    seen: set[str] = set()
    for group in candidate_groups:
        for candidate in group:
            smiles = candidate.smiles.strip()
            if not smiles:
                continue
            try:
                canonical = canonicalize_smiles(smiles)
            except ValueError:
                continue
            if canonical in seen:
                continue
            seen.add(canonical)
            merged.append(candidate)
    return merged


def _search_context(component_a: str, n: int, checkpoint_path: str, config_path: str) -> str:
    return (
        f"Component A: {component_a}\n"
        f"Requested deterministic candidates: {n}\n"
        f"Checkpoint: {checkpoint_path}\n"
        f"Config: {config_path}"
    )


def _fallback_uncertainty(
    component_a: str,
    component_b: str,
    checkpoint_path: str,
    config_path: str,
    reason: str,
) -> MinimumTmUncertainty:
    return MinimumTmUncertainty(
        component_a=component_a,
        component_b=component_b,
        repeated_values=(),
        mean_tm_k=float("inf"),
        std_tm_k=float("inf"),
        min_tm_k=float("inf"),
        max_tm_k=float("inf"),
        trust_score=0.0,
        uncertainty_flag="high",
        explanation=reason,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
    )


def _build_discovery_candidates(component_a: str, n: int, discovery_path: str | None, llm_warnings: list[str]) -> list[CandidateProposal]:
    if not discovery_path:
        return []
    try:
        library = load_discovery_library(discovery_path)
    except Exception as exc:
        llm_warnings.append(f"Discovery loading failed for {discovery_path}: {exc}")
        return []
    literature = literature_lookup(component_a, library)
    similar = similarity_search(component_a, library, limit=n)
    if not literature and not similar:
        llm_warnings.append(f"No discovery candidates found at {discovery_path}; using heuristic generator only.")
    return merge_discovery_candidates(literature, similar)


def _promote_brainstorm_candidates(candidates: list[CandidateBrainstorm]) -> list[CandidateProposal]:
    return [
        CandidateProposal(
            smiles=candidate.smiles,
            rationale=candidate.rationale,
            family=candidate.family,
            source="llm",
            source_id="brainstorm",
        )
        for candidate in candidates
    ]


def _review_top_candidates(
    provider,
    component_a: str,
    candidate_proposals: list[CandidateProposal],
    context: str,
    top_n: int,
    llm_warnings: list[str],
) -> tuple[list[CandidateReview], dict[str, CandidateReview]]:
    review_notes: list[CandidateReview] = []
    review_by_smiles: dict[str, CandidateReview] = {}
    for proposal in candidate_proposals[: max(0, top_n)]:
        try:
            review = provider.review_candidate(component_a, proposal.smiles, context)
        except Exception as exc:
            llm_warnings.append(f"LLM candidate review failed for {proposal.smiles}: {exc}")
            continue
        if review.smiles != proposal.smiles:
            llm_warnings.append(f"LLM candidate review returned wrong SMILES for {proposal.smiles}")
            continue
        review_notes.append(review)
        review_by_smiles[proposal.smiles] = review
    return review_notes, review_by_smiles


def _apply_candidate_reviews(
    candidate_proposals: list[CandidateProposal],
    reviews: Mapping[str, CandidateReview],
) -> tuple[list[CandidateProposal], dict[str, float]]:
    kept: list[CandidateProposal] = []
    review_penalty_by_smiles: dict[str, float] = {}
    for proposal in candidate_proposals:
        review = reviews.get(proposal.smiles)
        if review is None:
            kept.append(proposal)
            continue
        if review.decision == "reject":
            continue
        if review.decision == "deprioritize":
            review_penalty_by_smiles[proposal.smiles] = 0.25
        kept.append(proposal)
    return kept, review_penalty_by_smiles


def _apply_review_penalties(
    annotated_results: list[AnnotatedResult],
    review_penalty_by_smiles: Mapping[str, float],
) -> list[AnnotatedResult]:
    if not review_penalty_by_smiles:
        return list(annotated_results)
    adjusted: list[AnnotatedResult] = []
    for item in annotated_results:
        penalty = review_penalty_by_smiles.get(item.result.curve.smiles_b)
        if penalty is None:
            adjusted.append(item)
            continue
        adjusted.append(replace(item, ranking_score=max(0.0, item.ranking_score - penalty)))
    return rank_annotated_results(adjusted)


def _predict_viscosity_predictions(
    component_a: str,
    proposals: list[CandidateProposal],
    model_path: str | None,
    llm_warnings: list[str],
) -> list[ViscosityPrediction]:
    predictions: list[ViscosityPrediction] = []
    if model_path is None:
        return predictions
    for proposal in proposals:
        try:
            predictions.append(predict_viscosity(component_a, proposal.smiles, model_path=model_path, allow_fallback=True))
        except Exception as exc:
            llm_warnings.append(f"Viscosity prediction failed for {proposal.smiles}: {exc}")
    return predictions


def _build_des_export_payload(
    outcome: SearchOutcome,
    component_a: str,
    n: int,
    checkpoint_path: str,
    config_path: str,
) -> dict[str, object]:
    proposal_by_smiles = {item.smiles: item for item in outcome.candidate_proposals}
    ranked_results: list[dict[str, object]] = []
    for rank, annotated in enumerate(outcome.annotated_results, start=1):
        result = annotated.result
        proposal = proposal_by_smiles.get(result.curve.smiles_b)
        ranked_results.append(
            {
                "rank": rank,
                "smiles_a": result.curve.smiles_a,
                "smiles_b": result.curve.smiles_b,
                "is_des": result.is_des,
                "absolute_pass": result.absolute_pass,
                "relative_pass": result.relative_pass,
                "min_tm_k": result.min_tm_k,
                "rationale": result.rationale,
                "source": proposal.source if proposal is not None else "heuristic",
                "source_id": proposal.source_id if proposal is not None else "",
                "similarity_score": proposal.similarity_score if proposal is not None else None,
                "reference_note": proposal.reference_note if proposal is not None else "",
                "trust_score": annotated.trust_score,
                "uncertainty_flag": annotated.uncertainty.uncertainty_flag,
                "ranking_score": annotated.ranking_score,
                "uncertainty": asdict(annotated.uncertainty),
            }
        )
    return {
        "workflow": "des",
        "component_a": component_a,
        "n": n,
        "checkpoint_path": checkpoint_path,
        "config_path": config_path,
        "results": ranked_results,
        "candidate_proposals": [asdict(item) for item in outcome.candidate_proposals],
        "candidate_reviews": [asdict(item) for item in outcome.candidate_reviews],
        "brainstorm_candidates": [asdict(item) for item in outcome.brainstorm_candidates],
        "explanation_notes": [asdict(item) for item in outcome.explanation_notes],
        "critique_notes": [asdict(item) for item in outcome.critique_notes],
        "viscosity_predictions": [asdict(item) for item in outcome.viscosity_predictions],
        "memory_notes": list(outcome.memory_notes),
        "warnings": list(outcome.llm_warnings),
    }


def _resolve_des_output_dir(output_dir: str | None, save_run_memory_path: str | None) -> Path:
    if output_dir:
        return Path(output_dir)
    if save_run_memory_path:
        return Path(save_run_memory_path).parent
    return Path.cwd()


def run_search_report(
    component_a: str,
    n: int,
    checkpoint_path: str,
    config_path: str = "ml_des_mp/config.yaml",
    thresholds: DesThresholds | None = None,
    uncertainty_policy: UncertaintyPolicy | None = None,
    llm_cfg: Mapping[str, object] | None = None,
    llm_request_fn=None,
    discovery_path: str | None = None,
    viscosity_model_path: str | None = None,
    save_run_memory_path: str | None = None,
    reuse_run_path: str | None = None,
    output_dir: str | None = None,
):
    checkpoint_path = resolve_existing_path(checkpoint_path)
    config_path = resolve_existing_path(config_path)
    heuristic_candidates = generate_candidates(component_a, n=n, constraints=None)
    llm_warnings: list[str] = []
    discovery_candidates = _build_discovery_candidates(component_a, n, discovery_path, llm_warnings)
    candidate_proposals = _merge_candidates(discovery_candidates, heuristic_candidates)
    provider = build_llm_provider(llm_cfg, request_fn=llm_request_fn) if llm_cfg else None
    llm_candidates: list[CandidateBrainstorm] = []
    candidate_reviews: list[CandidateReview] = []
    review_penalties: dict[str, float] = {}
    if provider is not None:
        try:
            llm_candidates = provider.brainstorm_candidates(
                component_a,
                None,
                _search_context(component_a, n, str(checkpoint_path), str(config_path)),
            )
        except Exception as exc:
            llm_warnings.append(f"LLM brainstorming failed: {exc}")
            llm_candidates = []
    candidate_proposals = _merge_candidates(candidate_proposals, _promote_brainstorm_candidates(llm_candidates))
    filtered = filter_candidates(component_a, candidate_proposals)
    thresholds = thresholds or DesThresholds(
        absolute_tm_max_k=DEFAULT_ABSOLUTE_TM_MAX_K,
        relative_drop_min=DEFAULT_RELATIVE_DROP_MIN,
    )
    review_context = _search_context(component_a, n, str(checkpoint_path), str(config_path))
    if provider is not None and filtered:
        try:
            review_candidates, review_by_smiles = _review_top_candidates(
                provider,
                component_a,
                filtered,
                review_context,
                min(max(n, 0), len(filtered)),
                llm_warnings,
            )
            candidate_reviews = review_candidates
            filtered, review_penalties = _apply_candidate_reviews(filtered, review_by_smiles)
        except Exception as exc:
            llm_warnings.append(f"LLM candidate review pipeline failed: {exc}")
    component_a_tp = resolve_melting_point(component_a)
    results = []
    for proposal in filtered:
        component_b_tp = resolve_melting_point(proposal.smiles)
        curve = predict_curve(
            component_a,
            proposal.smiles,
            t1_k=component_a_tp.tm_k,
            t2_k=component_b_tp.tm_k,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
        )
        result = classify_des(curve, thresholds)
        results.append(result)
    ranked = rank_results(results)
    policy = uncertainty_policy or UncertaintyPolicy()
    uncertainty_by_smiles: dict[str, MinimumTmUncertainty] = {}
    for result in ranked:
        smiles_b = result.curve.smiles_b
        try:
            uncertainty_by_smiles[smiles_b] = estimate_min_tm_uncertainty(
                component_a,
                smiles_b,
                str(checkpoint_path),
                str(config_path),
            )
        except Exception as exc:
            llm_warnings.append(f"Uncertainty estimation failed for {smiles_b}: {exc}")
            uncertainty_by_smiles[smiles_b] = _fallback_uncertainty(
                component_a,
                smiles_b,
                str(checkpoint_path),
                str(config_path),
                f"Uncertainty estimation failed: {exc}",
            )
    annotated_results = apply_uncertainty_policy(ranked, uncertainty_by_smiles, policy)
    annotated_results = _apply_review_penalties(annotated_results, review_penalties)
    memory_notes: list[str] = []
    if reuse_run_path:
        reuse_memories = load_run_memory_history(reuse_run_path)
        annotated_results, reuse_notes = apply_run_memory_preferences(
            annotated_results=annotated_results,
            memory=reuse_memories,
            component_a=component_a,
        )
        memory_notes.extend(reuse_notes)
        memory_notes.insert(0, f"Loaded reuse memory from {reuse_run_path} ({len(reuse_memories)} run memory file(s)).")
    viscosity_predictions = _predict_viscosity_predictions(component_a, filtered, viscosity_model_path, llm_warnings)
    final_results = [item.result for item in annotated_results]
    explanation_notes: list[ExplanationNote] = []
    critique_notes: list[CritiqueNote] = []
    if provider is not None:
        try:
            explanation_notes = provider.generate_explanations(final_results, review_context)
        except Exception as exc:
            llm_warnings.append(f"LLM explanation generation failed: {exc}")
        try:
            critique_notes = provider.critique_results(final_results, review_context)
        except Exception as exc:
            llm_warnings.append(f"LLM critique generation failed: {exc}")
    if save_run_memory_path:
        memory = build_run_memory(
            component_a=component_a,
            n=n,
            annotated_results=annotated_results,
            candidate_proposals=candidate_proposals,
        )
        write_run_memory(save_run_memory_path, memory)
        memory_notes.append(f"Wrote run memory to {save_run_memory_path}.")
    export_outcome = SearchOutcome(
        results=final_results,
        annotated_results=annotated_results,
        candidate_proposals=candidate_proposals,
        candidate_reviews=candidate_reviews,
        brainstorm_candidates=llm_candidates,
        explanation_notes=explanation_notes,
        critique_notes=critique_notes,
        llm_warnings=llm_warnings,
        memory_notes=memory_notes,
        viscosity_predictions=viscosity_predictions,
    )
    report_text = format_report(
        final_results,
        annotated_results=annotated_results,
        candidate_proposals=candidate_proposals,
        candidate_reviews=candidate_reviews,
        explanation_notes=explanation_notes,
        critique_notes=critique_notes,
        brainstorm_candidates=llm_candidates,
        llm_warnings=llm_warnings,
        memory_notes=memory_notes,
        viscosity_predictions=viscosity_predictions,
    )
    export_output_dir = _resolve_des_output_dir(output_dir, save_run_memory_path)
    try:
        export_des_run_bundle(
            export_output_dir,
            _build_des_export_payload(
                export_outcome,
                component_a=component_a,
                n=n,
                checkpoint_path=str(checkpoint_path),
                config_path=str(config_path),
            ),
            report_text,
        )
    except (OSError, TypeError, ValueError, KeyError) as exc:
        raise ValueError(f"DES export failed for {export_output_dir}: {exc}") from exc
    return export_outcome


def run_search(
    component_a: str,
    n: int,
    checkpoint_path: str,
    config_path: str = "ml_des_mp/config.yaml",
    thresholds: DesThresholds | None = None,
    uncertainty_policy: UncertaintyPolicy | None = None,
):
    return run_search_report(
        component_a=component_a,
        n=n,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        thresholds=thresholds,
        uncertainty_policy=uncertainty_policy,
    ).results
