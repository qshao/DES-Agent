from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MinimumTmUncertainty:
    component_a: str
    component_b: str
    repeated_values: list[float]
    mean_tm_k: float
    std_tm_k: float
    min_tm_k: float
    max_tm_k: float
    trust_score: float
    uncertainty_flag: str
    explanation: str
    checkpoint_path: str
    config_path: str
