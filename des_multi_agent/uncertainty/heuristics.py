from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..property_resolution import MeltingPointEstimate
from .schemas import MinimumTmUncertainty


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _status_confidence(
    component: str,
    neat_component_status: Mapping[str, MeltingPointEstimate] | Sequence[MeltingPointEstimate],
) -> float:
    if isinstance(neat_component_status, Mapping):
        status = neat_component_status.get(component)
        return _clamp01(status.confidence) if status is not None else 0.0

    for status in neat_component_status:
        if status.component == component:
            return _clamp01(status.confidence)

    return 0.0


def score_candidate_trust(
    component_a: str,
    component_b: str,
    tm_uncertainty: MinimumTmUncertainty,
    neat_component_status: Mapping[str, MeltingPointEstimate] | Sequence[MeltingPointEstimate],
) -> float:
    """Combine repeatability and neat-component confidence into a normalized trust score."""

    uncertainty_score = _clamp01(tm_uncertainty.trust_score)
    neat_confidence = (
        _status_confidence(component_a, neat_component_status)
        + _status_confidence(component_b, neat_component_status)
    ) / 2.0
    combined = 0.7 * uncertainty_score + 0.3 * neat_confidence
    return _clamp01(combined)
