from __future__ import annotations

from dataclasses import dataclass

from .prediction import CurvePrediction
from .schemas import DesThresholds


@dataclass(frozen=True)
class DesResult:
    curve: CurvePrediction
    absolute_pass: bool
    relative_pass: bool
    is_des: bool
    rationale: str
    min_tm_k: float
    eutectic_ratio_b: float = 0.5
    # False when the predicted eutectic Tm exceeds both pure-component melting
    # points, which is physically impossible (a model/input red flag).
    eutectic_physical: bool = True
    # Provenance of the pure-component melting points the curve was anchored on
    # (attached by the orchestrator). None when not resolved through the layered
    # resolver, e.g. in programmatic/test construction.
    t1_source: str | None = None
    t2_source: str | None = None
    t1_confidence: float | None = None
    t2_confidence: float | None = None


def classify_des(curve: CurvePrediction, thresholds: DesThresholds) -> DesResult:
    min_idx = min(range(len(curve.tm_pred_k)), key=lambda i: curve.tm_pred_k[i])
    min_tm = curve.tm_pred_k[min_idx]
    eutectic_ratio_b = curve.ratios[min_idx]
    absolute_pass = min_tm <= thresholds.absolute_tm_max_k
    baseline = min(curve.t1_k, curve.t2_k)
    relative_drop = (baseline - min_tm) / baseline if baseline else 0.0
    relative_pass = relative_drop >= thresholds.relative_drop_min
    is_des = absolute_pass and relative_pass
    eutectic_physical = min_tm <= baseline
    rationale = (
        f"min Tm={min_tm:.2f} K, absolute<= {thresholds.absolute_tm_max_k:.2f} K, "
        f"relative_drop={relative_drop:.3f}"
    )
    return DesResult(
        curve=curve,
        absolute_pass=absolute_pass,
        relative_pass=relative_pass,
        is_des=is_des,
        rationale=rationale,
        min_tm_k=min_tm,
        eutectic_ratio_b=eutectic_ratio_b,
        eutectic_physical=eutectic_physical,
    )
