from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

from .candidate_generation import generate_candidates
from .chemistry_filter import filter_candidates
from .config import DEFAULT_ABSOLUTE_TM_MAX_K, DEFAULT_RELATIVE_DROP_MIN
from .evaluation import DesResult, classify_des
from .llm.factory import build_llm_provider
from .llm.schemas import CandidateBrainstorm, CritiqueNote, ExplanationNote
from .paths import resolve_existing_path
from .prediction import predict_curve
from .property_resolution import resolve_melting_point
from .ranking import rank_results
from .schemas import DesThresholds


@dataclass(frozen=True)
class SearchOutcome:
    results: list[DesResult]
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
            if not smiles or smiles in seen:
                continue
            seen.add(smiles)
            merged.append(candidate)
    return merged


def _search_context(component_a: str, n: int, checkpoint_path: str, config_path: str) -> str:
    return (
        f"Component A: {component_a}\n"
        f"Requested deterministic candidates: {n}\n"
        f"Checkpoint: {checkpoint_path}\n"
        f"Config: {config_path}"
    )


def run_search_report(
    component_a: str,
    n: int,
    checkpoint_path: str,
    config_path: str = "ml_des_mp/config.yaml",
    thresholds: DesThresholds | None = None,
    llm_cfg: Mapping[str, object] | None = None,
    llm_request_fn=None,
):
    checkpoint_path = resolve_existing_path(checkpoint_path)
    config_path = resolve_existing_path(config_path)
    proposals = generate_candidates(component_a, n=n, constraints=None)
    llm_candidates: list[CandidateBrainstorm] = []
    llm_warnings: list[str] = []
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
    merged = _merge_candidates(proposals, llm_candidates)
    filtered = filter_candidates(component_a, merged)
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
    explanation_notes: list[ExplanationNote] = []
    critique_notes: list[CritiqueNote] = []
    if provider is not None:
        context = _search_context(component_a, n, str(checkpoint_path), str(config_path))
        try:
            explanation_notes = provider.generate_explanations(ranked, context)
        except Exception as exc:
            llm_warnings.append(f"LLM explanation generation failed: {exc}")
        try:
            critique_notes = provider.critique_results(ranked, context)
        except Exception as exc:
            llm_warnings.append(f"LLM critique generation failed: {exc}")
    return SearchOutcome(
        results=ranked,
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
):
    return run_search_report(
        component_a=component_a,
        n=n,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        thresholds=thresholds,
    ).results
