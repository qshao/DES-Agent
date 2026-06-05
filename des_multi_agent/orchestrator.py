from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

from .candidate_generation import generate_candidates
from .chemistry_filter import canonicalize_smiles, filter_candidates
from .config import DEFAULT_ABSOLUTE_TM_MAX_K, DEFAULT_RELATIVE_DROP_MIN
from .discovery import load_discovery_library, literature_lookup, merge_discovery_candidates, similarity_search
from .evaluation import DesResult, classify_des
from .llm.factory import build_llm_provider
from .llm.schemas import CandidateBrainstorm, CritiqueNote, ExplanationNote
from .paths import resolve_existing_path
from .prediction import predict_curve
from .property_resolution import resolve_melting_point
from .ranking import rank_results
from .schemas import CandidateProposal, DesThresholds
from .uncertainty import AnnotatedResult, MinimumTmUncertainty, UncertaintyPolicy, apply_uncertainty_policy, estimate_min_tm_uncertainty


@dataclass(frozen=True)
class SearchOutcome:
    results: list[DesResult]
    annotated_results: list[AnnotatedResult]
    candidate_proposals: list[CandidateProposal]
    brainstorm_candidates: list[CandidateBrainstorm]
    explanation_notes: list[ExplanationNote]
    critique_notes: list[CritiqueNote]
    llm_warnings: list[str]


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
):
    checkpoint_path = resolve_existing_path(checkpoint_path)
    config_path = resolve_existing_path(config_path)
    heuristic_candidates = generate_candidates(component_a, n=n, constraints=None)
    llm_warnings: list[str] = []
    discovery_candidates = _build_discovery_candidates(component_a, n, discovery_path, llm_warnings)
    candidate_proposals = _merge_candidates(discovery_candidates, heuristic_candidates)
    provider = build_llm_provider(llm_cfg, request_fn=llm_request_fn) if llm_cfg else None
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
    else:
        llm_candidates = []
    candidate_proposals = _merge_candidates(candidate_proposals, _promote_brainstorm_candidates(llm_candidates))
    filtered = filter_candidates(component_a, candidate_proposals)
    thresholds = thresholds or DesThresholds(
        absolute_tm_max_k=DEFAULT_ABSOLUTE_TM_MAX_K,
        relative_drop_min=DEFAULT_RELATIVE_DROP_MIN,
    )
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
    final_results = [item.result for item in annotated_results]
    explanation_notes: list[ExplanationNote] = []
    critique_notes: list[CritiqueNote] = []
    if provider is not None:
        context = _search_context(component_a, n, str(checkpoint_path), str(config_path))
        try:
            explanation_notes = provider.generate_explanations(final_results, context)
        except Exception as exc:
            llm_warnings.append(f"LLM explanation generation failed: {exc}")
        try:
            critique_notes = provider.critique_results(final_results, context)
        except Exception as exc:
            llm_warnings.append(f"LLM critique generation failed: {exc}")
    return SearchOutcome(
        results=final_results,
        annotated_results=annotated_results,
        candidate_proposals=candidate_proposals,
        brainstorm_candidates=llm_candidates,
        explanation_notes=explanation_notes,
        critique_notes=critique_notes,
        llm_warnings=llm_warnings,
    )


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
