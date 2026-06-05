from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


PolicyMode = Literal["filter", "penalize", "report_only"]


@dataclass(frozen=True)
class UncertaintyPolicy:
    mode: PolicyMode = "penalize"
    min_trust_score: float = 0.55
    soft_penalty_weight: float = 0.35
    std_high_threshold_k: float = 15.0
    std_medium_threshold_k: float = 5.0

    def __post_init__(self) -> None:
        if self.mode not in {"filter", "penalize", "report_only"}:
            raise ValueError(f"Unsupported uncertainty policy mode: {self.mode}")
        if not 0.0 <= self.min_trust_score <= 1.0:
            raise ValueError("min_trust_score must be within [0.0, 1.0]")
        if self.soft_penalty_weight < 0.0:
            raise ValueError("soft_penalty_weight must be non-negative")
        if self.std_medium_threshold_k < 0.0:
            raise ValueError("std_medium_threshold_k must be non-negative")
        if self.std_high_threshold_k <= self.std_medium_threshold_k:
            raise ValueError("std_high_threshold_k must be greater than std_medium_threshold_k")
