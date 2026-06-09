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


def rank_results_composite(
    results: list[DesResult],
    visc_by_smiles_b: dict[str, float],
    *,
    viscosity_weight: float = 0.3,
    viscosity_threshold_cp: float | None = None,
) -> list[DesResult]:
    """Rank by a composite of Tm-drop and viscosity.

    When visc_by_smiles_b is empty, falls back to rank_results ordering.
    When viscosity_threshold_cp is set, candidates above the threshold are
    sorted after those that pass it (regardless of Tm).
    """
    if not visc_by_smiles_b:
        return rank_results(results)

    des = [r for r in results if r.is_des]
    non_des = sorted([r for r in results if not r.is_des], key=lambda r: r.min_tm_k)

    def _composite_score(r: DesResult) -> float:
        baseline = min(r.curve.t1_k, r.curve.t2_k)
        tm_score = (baseline - r.min_tm_k) / baseline if baseline else 0.0
        visc_cp = visc_by_smiles_b.get(r.curve.smiles_b)
        visc_score = 1.0 / (1.0 + visc_cp / 100.0) if visc_cp is not None else 0.5
        return (1.0 - viscosity_weight) * tm_score + viscosity_weight * visc_score

    if viscosity_threshold_cp is not None:
        passing = [r for r in des if visc_by_smiles_b.get(r.curve.smiles_b, 0.0) <= viscosity_threshold_cp]
        failing = [r for r in des if visc_by_smiles_b.get(r.curve.smiles_b, 0.0) > viscosity_threshold_cp]
        ranked_des = (
            sorted(passing, key=_composite_score, reverse=True)
            + sorted(failing, key=_composite_score, reverse=True)
        )
    else:
        ranked_des = sorted(des, key=_composite_score, reverse=True)

    return ranked_des + non_des
