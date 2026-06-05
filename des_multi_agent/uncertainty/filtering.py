from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..evaluation import DesResult
from ..property_resolution import MeltingPointEstimate, resolve_melting_point
from .heuristics import _clamp01, score_candidate_trust
from .policy import UncertaintyPolicy
from .schemas import AnnotatedResult, MinimumTmUncertainty


def _lookup_neat_status(component_a: str, component_b: str) -> tuple[MeltingPointEstimate, MeltingPointEstimate]:
    return (
        resolve_melting_point(component_a),
        resolve_melting_point(component_b),
    )


def _std_band_multiplier(std_tm_k: float, policy: UncertaintyPolicy) -> float:
    if std_tm_k >= policy.std_high_threshold_k:
        return 0.5
    if std_tm_k >= policy.std_medium_threshold_k:
        return 0.8
    return 1.0


def _ranking_score(
    trust_score: float,
    index: int,
    total: int,
    policy: UncertaintyPolicy,
) -> float:
    if policy.mode == "report_only":
        return float(total - index)
    shortfall = max(0.0, policy.min_trust_score - trust_score)
    score = trust_score - policy.soft_penalty_weight * shortfall
    return _clamp01(score)


def apply_uncertainty_policy(
    results: Sequence[DesResult],
    uncertainty_by_smiles: Mapping[str, MinimumTmUncertainty],
    policy: UncertaintyPolicy,
) -> list[AnnotatedResult]:
    annotated: list[AnnotatedResult] = []

    total = len(results)
    for index, result in enumerate(results):
        uncertainty = uncertainty_by_smiles.get(result.curve.smiles_b)
        fallback_uncertainty = False
        if uncertainty is None:
            fallback_uncertainty = True
            uncertainty = MinimumTmUncertainty(
                component_a=result.curve.smiles_a,
                component_b=result.curve.smiles_b,
                repeated_values=(),
                mean_tm_k=float("inf"),
                std_tm_k=float("inf"),
                min_tm_k=float("inf"),
                max_tm_k=float("inf"),
                trust_score=0.0,
                uncertainty_flag="high",
                explanation="Missing uncertainty estimate; defaulting to conservative low trust.",
                checkpoint_path="",
                config_path="",
            )
        elif uncertainty.explanation.startswith("Uncertainty estimation failed"):
            fallback_uncertainty = True
        neat_status = _lookup_neat_status(result.curve.smiles_a, result.curve.smiles_b)
        base_trust = score_candidate_trust(
            result.curve.smiles_a,
            result.curve.smiles_b,
            uncertainty,
            neat_status,
        )
        trust_score = 0.0 if fallback_uncertainty else _clamp01(base_trust)
        trust_score = _clamp01(trust_score * _std_band_multiplier(uncertainty.std_tm_k, policy))

        if policy.mode == "filter" and trust_score < policy.min_trust_score:
            continue

        annotated.append(
            AnnotatedResult(
                result=result,
                uncertainty=uncertainty,
                trust_score=trust_score,
                ranking_score=_ranking_score(trust_score, index, total, policy),
            )
        )

    if policy.mode == "report_only":
        return list(annotated)
    return rank_annotated_results(annotated)


def rank_annotated_results(annotated_results: Sequence[AnnotatedResult]) -> list[AnnotatedResult]:
    return sorted(
        annotated_results,
        key=lambda item: (
            not item.result.is_des,
            -item.ranking_score,
            item.result.min_tm_k,
            item.uncertainty.std_tm_k,
            item.result.curve.smiles_b,
        ),
    )
