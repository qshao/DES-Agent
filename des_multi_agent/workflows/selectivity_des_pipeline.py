from __future__ import annotations

import sys
from dataclasses import dataclass, field

from ..evaluation import DesResult
from ..multi_cycle import run_multi_cycle_search
from ..trajectory import CycleSnapshot, SearchTrajectory, TopEntry, shortlist_delta
from .metal_binding_selectivity import (
    SelectivityResult,
    SelectivityScreenOutcome,
    run_metal_selectivity_screen,
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
    competitor_metals: list[str]
    selectivity_outcome: SelectivityScreenOutcome
    ligand_des_results: list[LigandDesResult]
    n_outer_cycles_run: int
    converged: bool
    warnings: list[str] = field(default_factory=list)
    trajectory: object = None   # SearchTrajectory | None


def _as_list(competitor_metal: str | list[str]) -> list[str]:
    return [competitor_metal] if isinstance(competitor_metal, str) else list(competitor_metal)


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


def run_selectivity_des_pipeline(
    target_metal: str,
    competitor_metal: str | list[str],
    checkpoint_path: str,
    config_path: str = "ml_des_mp/config.yaml",
    n_ligands: int = 20,
    n_des_candidates: int = 20,
    n_selectivity_cycles: int = 3,
    n_des_cycles: int = 3,
    n_outer_cycles: int = 2,
    min_delta_log_k: float = 0.0,
    top_ligands: int = 3,
    w_affinity: float = 0.5,
    w_selectivity: float = 0.5,
    stability_model_path=None,
    llm_cfg=None,
    constraints: dict | None = None,
    des_thresholds=None,
    viscosity_model_path: str | None = None,
    viscosity_weight: float = 0.3,
    viscosity_threshold_cp: float | None = None,
    chemical_pattern_memory_mode: str = "adaptive",
    pattern_memory_max_examples: int = 3,
) -> SelectivityDesPipelineOutcome:
    all_warnings: list[str] = []
    des_compatible_smiles: set[str] = set()
    des_incompatible_smiles: set[str] = set()
    prev_compatible: set[str] = set()
    final_selectivity_outcome: SelectivityScreenOutcome | None = None
    final_ligand_des_results: list[LigandDesResult] = []
    converged = False
    outer_cycle_count = 0
    pipe_snapshots: list[CycleSnapshot] = []
    pipe_prev_labels: list[str] = []

    llm_provider = None
    if llm_cfg is not None:
        from ..llm.factory import build_llm_provider
        llm_provider = build_llm_provider(llm_cfg)

    for outer_cycle in range(1, n_outer_cycles + 1):
        outer_cycle_count = outer_cycle
        print(
            f"[outer {outer_cycle}/{n_outer_cycles}] phase 1: selectivity screening",
            file=sys.stderr, flush=True,
        )

        sel_outcome = run_metal_selectivity_screen(
            target_metal=target_metal,
            competitor_metal=competitor_metal,
            n=n_ligands,
            model_path=stability_model_path,
            llm_provider=llm_provider,
            constraints=constraints,
            n_cycles=n_selectivity_cycles,
            w_affinity=w_affinity,
            w_selectivity=w_selectivity,
            des_compatible_hints=list(des_compatible_smiles) if des_compatible_smiles else None,
            des_incompatible_hints=list(des_incompatible_smiles) if des_incompatible_smiles else None,
        )
        final_selectivity_outcome = sel_outcome
        all_warnings.extend(sel_outcome.warnings)

        shortlisted = _bridge_filter(
            sel_outcome.results, min_delta_log_k, top_ligands, all_warnings
        )

        print(
            f"[outer {outer_cycle}/{n_outer_cycles}] phase 2: DES search for "
            f"{len(shortlisted)} ligand(s)",
            file=sys.stderr, flush=True,
        )

        ligand_des_results: list[LigandDesResult] = []
        new_compatible: set[str] = set()
        new_incompatible: set[str] = set()

        for ligand_idx, ligand_result in enumerate(shortlisted, 1):
            print(
                f"[outer {outer_cycle}/{n_outer_cycles}] phase 2: ligand {ligand_idx}/{len(shortlisted)}"
                f" — {ligand_result.ligand_smiles}",
                file=sys.stderr, flush=True,
            )
            try:
                des_mco = run_multi_cycle_search(
                    component_a=ligand_result.ligand_smiles,
                    n=n_des_candidates,
                    checkpoint_path=checkpoint_path,
                    config_path=config_path,
                    n_cycles=n_des_cycles,
                    llm_cfg=llm_cfg,
                    thresholds=des_thresholds,
                    viscosity_model_path=viscosity_model_path,
                    viscosity_weight=viscosity_weight,
                    viscosity_threshold_cp=viscosity_threshold_cp,
                    chemical_pattern_memory_mode=chemical_pattern_memory_mode,
                    pattern_memory_max_examples=pattern_memory_max_examples,
                )
                des_compat = any(r.is_des for r in des_mco.final_outcome.results)
                n_screened = sum(d.n_screened for d in des_mco.cycle_deltas)
                ldr = LigandDesResult(
                    ligand=ligand_result,
                    des_results=des_mco.final_outcome.results,
                    n_des_screened=n_screened,
                    des_compatible=des_compat,
                )
            except Exception as exc:
                all_warnings.append(
                    f"DES search failed for {ligand_result.ligand_smiles}: {exc}"
                )
                ldr = LigandDesResult(
                    ligand=ligand_result,
                    des_results=[],
                    n_des_screened=0,
                    des_compatible=False,
                )

            ligand_des_results.append(ldr)
            if ldr.des_compatible:
                new_compatible.add(ligand_result.ligand_smiles)
            else:
                new_incompatible.add(ligand_result.ligand_smiles)

        final_ligand_des_results = ligand_des_results
        des_compatible_smiles = new_compatible
        des_incompatible_smiles = new_incompatible

        this_converged = outer_cycle > 1 and new_compatible == prev_compatible
        try:
            compatible_ldrs = sorted(
                (ldr for ldr in ligand_des_results if ldr.des_compatible),
                key=lambda ldr: ldr.ligand.composite_score, reverse=True,
            )
            entries = [
                TopEntry(
                    label=ldr.ligand.ligand_smiles, metric_name="composite_score",
                    metric_value=ldr.ligand.composite_score,
                    secondary=f"ΔlogK={ldr.ligand.delta_log_k:.2f}, DES-compatible",
                )
                for ldr in compatible_ldrs
            ]
            curr_labels = sorted(new_compatible)
            snap_new, snap_drop = shortlist_delta(pipe_prev_labels, curr_labels)
            pipe_snapshots.append(CycleSnapshot(
                cycle=outer_cycle,
                n_screened=len(shortlisted),
                n_hits=len(new_compatible),
                top_entries=entries,
                new_entrants=snap_new if outer_cycle > 1 else [],
                dropouts=snap_drop if outer_cycle > 1 else [],
                converged=this_converged,
                convergence_reason="DES-compatible ligand set stable vs previous outer cycle" if this_converged else "",
            ))
            pipe_prev_labels = curr_labels
        except Exception as exc:
            all_warnings.append(f"[TRAJECTORY] snapshot capture failed (outer {outer_cycle}): {exc}")

        if outer_cycle > 1 and new_compatible == prev_compatible:
            converged = True
            print(
                f"[outer {outer_cycle}/{n_outer_cycles}] DES-compatible set stable — converged early",
                file=sys.stderr, flush=True,
            )
            break

        prev_compatible = new_compatible

    pipe_trajectory = SearchTrajectory(
        workflow="selectivity-des",
        headline=f"{target_metal} over {', '.join(_as_list(competitor_metal))} — selectivity-DES",
        metric_label="composite score",
        snapshots=pipe_snapshots,
        total_cycles=outer_cycle_count,
        converged=converged,
        convergence_reason=(pipe_snapshots[-1].convergence_reason if pipe_snapshots else ""),
        final_summary=(pipe_snapshots[-1].top_entries if pipe_snapshots else []),
    )
    return SelectivityDesPipelineOutcome(
        target_metal=target_metal,
        competitor_metals=_as_list(competitor_metal),
        selectivity_outcome=final_selectivity_outcome,
        ligand_des_results=final_ligand_des_results,
        n_outer_cycles_run=outer_cycle_count,
        converged=converged,
        warnings=all_warnings,
        trajectory=pipe_trajectory,
    )
