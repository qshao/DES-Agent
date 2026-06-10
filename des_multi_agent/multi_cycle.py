"""H1 + H5 — multi-cycle DES screening with convergence detection."""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

from .orchestrator import run_search_report
from .schemas import DesThresholds
from .uncertainty import UncertaintyPolicy


@dataclass
class CycleDelta:
    cycle: int
    n_screened: int
    n_des: int
    top_smiles: frozenset
    new_entrants: list[str]
    dropouts: list[str]
    converged: bool
    family_ledger: dict[str, int] = field(default_factory=dict)


@dataclass
class MultiCycleOutcome:
    final_outcome: object   # SearchOutcome — avoid circular import
    cycle_deltas: list[CycleDelta]
    total_cycles: int
    converged: bool


def run_multi_cycle_search(
    component_a: str,
    n: int,
    checkpoint_path: str,
    config_path: str = "ml_des_mp/config.yaml",
    *,
    n_cycles: int = 3,
    top_k_convergence: int = 5,
    thresholds: DesThresholds | None = None,
    uncertainty_policy: UncertaintyPolicy | None = None,
    llm_cfg=None,
    llm_request_fn=None,
    discovery_path: str | None = None,
    viscosity_model_path: str | None = None,
    viscosity_weight: float = 0.3,
    viscosity_threshold_cp: float | None = None,
    output_dir: str | None = None,
    ensemble_checkpoints: list[str] | None = None,
    candidates_file: str | None = None,
) -> MultiCycleOutcome:
    """Run up to n_cycles iterations, passing top hits forward as brainstorm context.

    Stops early (H5) when the top-K canonical SMILES set is identical across
    two consecutive cycles.
    """
    cycle_deltas: list[CycleDelta] = []
    prev_top: frozenset = frozenset()
    last_outcome = None
    accumulated_ledger: dict[str, int] = {}

    for cycle in range(1, n_cycles + 1):
        prior_results = last_outcome.results[:top_k_convergence] if last_outcome else None

        per_cycle_dir: str | None = None
        if output_dir is not None:
            per_cycle_dir = str(pathlib.Path(output_dir) / f"cycle_{cycle:02d}")

        outcome = run_search_report(
            component_a=component_a,
            n=n,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            thresholds=thresholds,
            uncertainty_policy=uncertainty_policy,
            llm_cfg=llm_cfg,
            llm_request_fn=llm_request_fn,
            discovery_path=discovery_path,
            viscosity_model_path=viscosity_model_path,
            viscosity_weight=viscosity_weight,
            viscosity_threshold_cp=viscosity_threshold_cp,
            output_dir=per_cycle_dir,
            ensemble_checkpoints=ensemble_checkpoints,
            candidates_file=candidates_file,
            prior_cycle_top_results=prior_results,
            prior_family_ledger=accumulated_ledger if cycle > 1 else None,
        )

        # H6 — build family ledger: DES-positive hit count per chemical family
        smiles_to_family = {
            bc.smiles: bc.family for bc in getattr(outcome, "brainstorm_candidates", [])
        }
        family_ledger: dict[str, int] = {}
        for r in outcome.results:
            if r.is_des:
                fam = smiles_to_family.get(r.curve.smiles_b, "unknown")
                family_ledger[fam] = family_ledger.get(fam, 0) + 1

        for fam, n_hits in family_ledger.items():
            accumulated_ledger[fam] = accumulated_ledger.get(fam, 0) + n_hits

        top_k = frozenset(
            r.curve.smiles_b for r in outcome.results[:top_k_convergence] if r.is_des
        )
        new_entrants = sorted(top_k - prev_top)
        dropouts = sorted(prev_top - top_k)
        converged = (top_k == prev_top) and cycle > 1 and bool(top_k)

        cycle_deltas.append(CycleDelta(
            cycle=cycle,
            n_screened=len(outcome.results),
            n_des=sum(1 for r in outcome.results if r.is_des),
            top_smiles=top_k,
            new_entrants=new_entrants,
            dropouts=dropouts,
            converged=converged,
            family_ledger=family_ledger,
        ))

        last_outcome = outcome
        prev_top = top_k

        if converged:
            break

    return MultiCycleOutcome(
        final_outcome=last_outcome,
        cycle_deltas=cycle_deltas,
        total_cycles=len(cycle_deltas),
        converged=cycle_deltas[-1].converged if cycle_deltas else False,
    )
