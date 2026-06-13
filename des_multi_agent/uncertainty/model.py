from __future__ import annotations

from statistics import mean, pstdev
from typing import Final

import torch

from ..prediction import predict_curve
from ..property_resolution import resolve_melting_point
from .schemas import MinimumTmUncertainty


_N_REPEATS: Final[int] = 3
# Seed the MC-dropout sampling so the uncertainty estimate is reproducible across
# runs (the draws are fixed); the global RNG state is saved and restored so this
# does not perturb other randomness.
_MC_SEED: Final[int] = 12345


def _trust_score(std_tm_k: float) -> float:
    trust = 1.0 / (1.0 + (std_tm_k / 10.0))
    return max(0.0, min(1.0, trust))


def _uncertainty_flag(std_tm_k: float) -> str:
    if std_tm_k <= 2.0:
        return "low"
    if std_tm_k <= 8.0:
        return "medium"
    return "high"


def _explanation(repeated_values: list[float], std_tm_k: float, trust_score: float, uncertainty_flag: str) -> str:
    spread = ", ".join(f"{value:.2f}" for value in repeated_values)
    return (
        f"Repeated minimum-Tm predictions were {spread}. "
        f"Std={std_tm_k:.2f} K, trust={trust_score:.2f}, flag={uncertainty_flag}."
    )


def estimate_min_tm_uncertainty(
    component_a: str,
    component_b: str,
    checkpoint_path: str,
    config_path: str,
) -> MinimumTmUncertainty:
    est_a = resolve_melting_point(component_a)
    est_b = resolve_melting_point(component_b)
    t1 = est_a.tm_k
    t2 = est_b.tm_k
    # The eutectic prediction can be no more trustworthy than the pure-component
    # melting points it is built on. Use the weakest of the two input estimates.
    input_confidence = min(est_a.confidence, est_b.confidence)

    repeated_values: list[float] = []
    _rng_state = torch.get_rng_state()
    torch.manual_seed(_MC_SEED)
    try:
        for _ in range(_N_REPEATS):
            curve = predict_curve(
                component_a=component_a,
                component_b=component_b,
                t1_k=t1,
                t2_k=t2,
                checkpoint_path=checkpoint_path,
                config_path=config_path,
                mc_dropout=True,
            )
            if not curve.tm_pred_k:
                raise ValueError("Predicted curve did not contain any melting-temperature values")
            repeated_values.append(min(curve.tm_pred_k))
    finally:
        torch.set_rng_state(_rng_state)

    mean_tm_k = mean(repeated_values)
    std_tm_k = pstdev(repeated_values) if len(repeated_values) > 1 else 0.0
    min_tm_k = min(repeated_values)
    max_tm_k = max(repeated_values)
    spread_trust = _trust_score(std_tm_k)
    trust_score = spread_trust * input_confidence
    uncertainty_flag = _uncertainty_flag(std_tm_k)
    explanation = _explanation(repeated_values, std_tm_k, trust_score, uncertainty_flag)
    explanation += (
        f" Input Tm confidence={input_confidence:.2f} "
        f"({est_a.source}/{est_b.source}); spread-trust={spread_trust:.2f}."
    )

    return MinimumTmUncertainty(
        component_a=component_a,
        component_b=component_b,
        repeated_values=tuple(repeated_values),
        mean_tm_k=float(mean_tm_k),
        std_tm_k=float(std_tm_k),
        min_tm_k=float(min_tm_k),
        max_tm_k=float(max_tm_k),
        trust_score=float(trust_score),
        uncertainty_flag=uncertainty_flag,
        explanation=explanation,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
    )
