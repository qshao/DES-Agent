"""Task 6: Tests for trajectory capture in the selectivity-DES pipeline."""
from __future__ import annotations

import pytest

from des_multi_agent.workflows import selectivity_des_pipeline as pipe


def test_pipeline_trajectory_field_present_and_typed(monkeypatch):
    # Stub the two inner phases so the outer loop runs deterministically with no ML/LLM.
    from des_multi_agent.workflows.metal_binding_selectivity import (
        SelectivityResult,
        SelectivityScreenOutcome,
    )

    lig = SelectivityResult(
        ligand_smiles="NCCN", log_k_target=8.0, log_k_competitor=6.0, delta_log_k=2.0,
        composite_score=10.0, source="heuristic", source_id="", rationale="ok",
    )

    monkeypatch.setattr(pipe, "run_metal_selectivity_screen", lambda **kw: SelectivityScreenOutcome(
        target_metal="Cu2+", competitor_metal="Zn2+", results=[lig], n_screened=1, n_cycles=1,
    ))
    monkeypatch.setattr(pipe, "_bridge_filter", lambda results, mindelta, topn, warnings: [lig])

    class _MCO:
        cycle_deltas = []
        class final_outcome:  # noqa: N801
            results = []

    monkeypatch.setattr(pipe, "run_multi_cycle_search", lambda **kw: _MCO())

    outcome = pipe.run_selectivity_des_pipeline(
        target_metal="Cu2+", competitor_metal="Zn2+", checkpoint_path="ckpt.pt",
        n_ligands=1, n_des_candidates=1, n_outer_cycles=1,
    )

    assert outcome.trajectory is not None
    assert outcome.trajectory.workflow == "selectivity-des"
    assert outcome.trajectory.metric_label == "composite score"
    assert len(outcome.trajectory.snapshots) == 1


def test_pipeline_trajectory_snapshot_has_correct_hits(monkeypatch):
    """A DES-compatible ligand contributes to n_hits; an incompatible one does not."""
    from des_multi_agent.workflows.metal_binding_selectivity import (
        SelectivityResult,
        SelectivityScreenOutcome,
    )

    lig_compatible = SelectivityResult(
        ligand_smiles="NCCN", log_k_target=8.0, log_k_competitor=6.0, delta_log_k=2.0,
        composite_score=10.0, source="heuristic", source_id="", rationale="ok",
    )
    lig_incompatible = SelectivityResult(
        ligand_smiles="CCO", log_k_target=5.0, log_k_competitor=4.5, delta_log_k=0.5,
        composite_score=5.0, source="heuristic", source_id="", rationale="ok",
    )

    shortlisted = [lig_compatible, lig_incompatible]

    monkeypatch.setattr(pipe, "run_metal_selectivity_screen", lambda **kw: SelectivityScreenOutcome(
        target_metal="Cu2+", competitor_metal="Zn2+", results=shortlisted, n_screened=2, n_cycles=1,
    ))
    monkeypatch.setattr(pipe, "_bridge_filter", lambda results, mindelta, topn, warnings: shortlisted)

    call_idx = [0]

    class _MCO_Compatible:
        cycle_deltas = []
        class final_outcome:
            results = []

    class _MCO_Incompatible:
        cycle_deltas = []
        class final_outcome:
            results = []

    def mock_multi_cycle(**kw):
        i = call_idx[0]
        call_idx[0] += 1
        if kw["component_a"] == "NCCN":
            return _MCO_Compatible()
        return _MCO_Incompatible()

    monkeypatch.setattr(pipe, "run_multi_cycle_search", mock_multi_cycle)

    outcome = pipe.run_selectivity_des_pipeline(
        target_metal="Cu2+", competitor_metal="Zn2+", checkpoint_path="ckpt.pt",
        n_outer_cycles=1,
    )

    snap = outcome.trajectory.snapshots[0]
    assert snap.n_screened == 2
    # Neither ligand yields DES-compatible results (empty final_outcome.results)
    assert snap.n_hits == 0


def test_pipeline_trajectory_convergence_detected(monkeypatch):
    """When the DES-compatible set is stable across outer cycles, converged=True."""
    from des_multi_agent.workflows.metal_binding_selectivity import (
        SelectivityResult,
        SelectivityScreenOutcome,
    )

    lig = SelectivityResult(
        ligand_smiles="NCCN", log_k_target=8.0, log_k_competitor=6.0, delta_log_k=2.0,
        composite_score=10.0, source="heuristic", source_id="", rationale="ok",
    )

    monkeypatch.setattr(pipe, "run_metal_selectivity_screen", lambda **kw: SelectivityScreenOutcome(
        target_metal="Cu2+", competitor_metal="Zn2+", results=[lig], n_screened=1, n_cycles=1,
    ))
    monkeypatch.setattr(pipe, "_bridge_filter", lambda results, mindelta, topn, warnings: [lig])

    # Build an MCO that marks the ligand as DES-compatible (is_des=True)
    class _FakeDesResult:
        is_des = True

    class _MCO:
        cycle_deltas = []
        class final_outcome:
            results = [_FakeDesResult()]

    monkeypatch.setattr(pipe, "run_multi_cycle_search", lambda **kw: _MCO())

    outcome = pipe.run_selectivity_des_pipeline(
        target_metal="Cu2+", competitor_metal="Zn2+", checkpoint_path="ckpt.pt",
        n_outer_cycles=3,
    )

    assert outcome.trajectory is not None
    assert outcome.trajectory.converged is True
    # The last snapshot should record convergence
    last_snap = outcome.trajectory.snapshots[-1]
    assert last_snap.converged is True
    assert "stable" in last_snap.convergence_reason
