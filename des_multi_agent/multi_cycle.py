"""H1 + H5 — multi-cycle DES screening with convergence detection."""
from __future__ import annotations

import pathlib
from collections import Counter
from dataclasses import dataclass, field

from .chemical_lesson_summary import ChemistryLessonSummary
from .chemical_pattern_memory import ChemicalPatternMemory
from .chemistry_filter import canonicalize_smiles
from .orchestrator import run_search_report
from .reporting import _confidence_label
from .schemas import DesThresholds
from .smiles_names import display_name
from .trajectory import CycleSnapshot, SearchTrajectory, TopEntry, shortlist_delta
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
    family_ledger: dict[str, int] = field(default_factory=dict)  # family → DES-positive hit count this cycle


@dataclass
class MultiCycleOutcome:
    final_outcome: object   # SearchOutcome — avoid circular import
    cycle_deltas: list[CycleDelta]
    total_cycles: int
    converged: bool
    accumulated_family_ledger: dict[str, int] = field(default_factory=dict)  # cross-cycle totals
    trajectory: object = None   # SearchTrajectory | None


def _des_top_entries(outcome, top_k: int) -> tuple[list[TopEntry], list[str]]:
    """Build best-first TopEntry list + display labels for a DES cycle outcome."""
    ann_by_smiles = {a.result.curve.smiles_b: a for a in getattr(outcome, "annotated_results", [])}
    entries: list[TopEntry] = []
    labels: list[str] = []
    hits = [r for r in outcome.results if r.is_des][:top_k]
    for r in hits:
        label = display_name(r.curve.smiles_b)
        t1, t2 = getattr(r.curve, "t1_k", None), getattr(r.curve, "t2_k", None)
        secondary = ""
        if t1 is not None and t2 is not None:
            baseline = min(t1, t2)
            if baseline:
                secondary = f"Δ{(baseline - r.min_tm_k) / baseline * 100:.1f}%"
        ann = ann_by_smiles.get(r.curve.smiles_b)
        if ann is not None:
            conf = _confidence_label(ann.trust_score, ann.uncertainty.uncertainty_flag)
            secondary = f"{secondary}, {conf}" if secondary else conf
        entries.append(TopEntry(label, "min_tm_k", r.min_tm_k, secondary))
        labels.append(label)
    return entries, labels


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
    proposal_diversity_cfg=None,
    prior_pattern_memory: ChemicalPatternMemory | None = None,
    prior_chemistry_lesson_summary: ChemistryLessonSummary | None = None,
    chemical_pattern_memory_mode: str = "adaptive",
    pattern_memory_max_examples: int = 3,
) -> MultiCycleOutcome:
    """Run up to n_cycles iterations, passing top hits forward as brainstorm context.

    Stops early (H5) when the top-K canonical SMILES set is identical across
    two consecutive cycles.
    """
    cycle_deltas: list[CycleDelta] = []
    prev_top: frozenset = frozenset()
    last_outcome = None
    accumulated_ledger: Counter[str] = Counter()
    cycle_pattern_memory = prior_pattern_memory
    cycle_lesson_summary = prior_chemistry_lesson_summary
    snapshots: list[CycleSnapshot] = []
    prev_labels: list[str] = []
    # Cross-cycle caches: skip re-running predictions and uncertainty estimation
    # for SMILES already evaluated in a prior cycle (predictions are deterministic).
    accumulated_results: dict = {}     # canonical smiles → DesResult
    accumulated_uncertainty: dict = {} # canonical smiles → MinimumTmUncertainty

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
            proposal_diversity_cfg=proposal_diversity_cfg,
            prior_pattern_memory=cycle_pattern_memory,
            prior_chemistry_lesson_summary=cycle_lesson_summary,
            chemical_pattern_memory_mode=chemical_pattern_memory_mode,
            pattern_memory_max_examples=pattern_memory_max_examples,
            prior_results_by_smiles=accumulated_results,
            prior_uncertainty_by_smiles=accumulated_uncertainty,
        )

        # Update cross-cycle caches with this cycle's fresh evaluations.
        for r in outcome.results:
            try:
                accumulated_results[canonicalize_smiles(r.curve.smiles_b)] = r
            except ValueError:
                pass
        for item in outcome.annotated_results:
            try:
                accumulated_uncertainty[canonicalize_smiles(item.result.curve.smiles_b)] = item.uncertainty
            except ValueError:
                pass

        # H6 — build family ledger: DES-positive hit count per chemical family
        smiles_to_family = {bc.smiles: bc.family for bc in outcome.brainstorm_candidates}
        family_ledger: Counter[str] = Counter(
            smiles_to_family.get(r.curve.smiles_b, "unknown")
            for r in outcome.results if r.is_des
        )
        accumulated_ledger.update(family_ledger)

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

        try:
            top_entries, curr_labels = _des_top_entries(outcome, top_k_convergence)
            snap_new, snap_drop = shortlist_delta(prev_labels, curr_labels)
            notable = [w for w in outcome.llm_warnings
                       if w.startswith("[GROUNDING]") or w.startswith("[REALITY]")][:3]
            reason = "top-{0} shortlist identical to previous cycle".format(top_k_convergence) if converged else ""
            snapshots.append(CycleSnapshot(
                cycle=cycle,
                n_screened=len(outcome.results),
                n_hits=sum(1 for r in outcome.results if r.is_des),
                top_entries=top_entries,
                new_entrants=snap_new if cycle > 1 else [],
                dropouts=snap_drop if cycle > 1 else [],
                family_ledger=dict(family_ledger),
                converged=converged,
                convergence_reason=reason,
                notable_warnings=notable,
            ))
            prev_labels = curr_labels
        except Exception as exc:  # capture is best-effort
            outcome.llm_warnings.append(f"[TRAJECTORY] snapshot capture failed (cycle {cycle}): {exc}")

        last_outcome = outcome
        prev_top = top_k
        cycle_pattern_memory = getattr(outcome, "chemical_pattern_memory", None)
        cycle_lesson_summary = getattr(outcome, "chemistry_lesson_summary", None)

        if converged:
            break

    final_converged = cycle_deltas[-1].converged if cycle_deltas else False
    trajectory = SearchTrajectory(
        workflow="des",
        headline=f"DES partners for {display_name(component_a)}",
        metric_label="min Tm (K)",
        snapshots=snapshots,
        total_cycles=len(cycle_deltas),
        converged=final_converged,
        convergence_reason=(snapshots[-1].convergence_reason if snapshots else ""),
        final_summary=(snapshots[-1].top_entries if snapshots else []),
    )
    return MultiCycleOutcome(
        final_outcome=last_outcome,
        cycle_deltas=cycle_deltas,
        total_cycles=len(cycle_deltas),
        converged=final_converged,
        accumulated_family_ledger=dict(accumulated_ledger),
        trajectory=trajectory,
    )
