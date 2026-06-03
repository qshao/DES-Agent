from __future__ import annotations

from .evaluation import DesResult


def rank_results(results: list[DesResult]) -> list[DesResult]:
    return sorted(
        results,
        key=lambda r: (
            not r.is_des,
            r.min_tm_k,
            -sum(r.curve.tm_pred_k) / len(r.curve.tm_pred_k),
        ),
    )
