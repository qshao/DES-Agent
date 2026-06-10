from __future__ import annotations

from dataclasses import dataclass, field

from ..evaluation import DesResult
from ..multi_cycle import run_multi_cycle_search
from .metal_binding_selectivity import (
    SelectivityResult,
    SelectivityScreenOutcome,
)


@dataclass(frozen=True)
class LigandDesResult:
    ligand: SelectivityResult
    des_results: list[DesResult]
    n_des_screened: int
    des_compatible: bool


@dataclass
class SelectivityDesPipelineOutcome:
    target_metal: str
    competitor_metal: str
    selectivity_outcome: SelectivityScreenOutcome
    ligand_des_results: list[LigandDesResult]
    n_outer_cycles_run: int
    converged: bool
    warnings: list[str] = field(default_factory=list)


def _bridge_filter(
    results: list[SelectivityResult],
    min_delta_log_k: float,
    top_n: int,
    warnings: list[str],
) -> list[SelectivityResult]:
    if not results:
        return []
    filtered = [r for r in results if r.delta_log_k >= min_delta_log_k]
    if not filtered:
        warnings.append(
            f"Bridge filter found 0 ligands above min_delta_log_k={min_delta_log_k}; "
            f"using top-{top_n} unconditionally."
        )
        filtered = results
    return filtered[:top_n]
