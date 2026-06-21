"""Tests for CLI trajectory wiring: _emit_trajectory helper, console output to stderr, and
trajectory.md artifact written to --output-dir.  These use monkeypatching for speed rather
than subprocess so the full test suite stays fast.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import des_multi_agent.cli as cli_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trajectory():
    """Return a minimal SearchTrajectory for use in stubs."""
    from des_multi_agent.trajectory import CycleSnapshot, SearchTrajectory, TopEntry

    snap = CycleSnapshot(
        cycle=1,
        n_screened=5,
        n_hits=2,
        top_entries=[TopEntry("acetamide (CC(=O)N)", "min_tm_k", 230.0, "")],
        new_entrants=[],
        dropouts=[],
    )
    return SearchTrajectory(
        workflow="des",
        headline="DES partners for ethanol (CCO)",
        metric_label="min Tm (K)",
        snapshots=[snap],
        total_cycles=1,
        converged=False,
        convergence_reason="",
        final_summary=[TopEntry("acetamide (CC(=O)N)", "min_tm_k", 230.0, "")],
    )


def _make_multi_outcome(trajectory=None):
    final = SimpleNamespace(
        results=[],
        annotated_results=[],
        candidate_proposals=[],
        candidate_reviews=[],
        explanation_notes=[],
        critique_notes=[],
        brainstorm_candidates=[],
        llm_warnings=[],
        memory_notes=[],
        viscosity_predictions=[],
    )
    return SimpleNamespace(
        final_outcome=final,
        cycle_deltas=[],
        total_cycles=2,
        converged=False,
        trajectory=trajectory,
    )


# ---------------------------------------------------------------------------
# _emit_trajectory unit tests
# ---------------------------------------------------------------------------

def test_emit_trajectory_none_is_noop(capsys):
    """_emit_trajectory with traj=None must not print or write anything."""
    cli_module._emit_trajectory(None, None)
    out, err = capsys.readouterr()
    assert out == "" and err == ""


def test_emit_trajectory_prints_to_stderr(capsys, tmp_path):
    """_emit_trajectory must print the console summary to stderr."""
    traj = _make_trajectory()
    cli_module._emit_trajectory(traj, None)
    out, err = capsys.readouterr()
    assert out == "", "trajectory output must NOT go to stdout (keeps json/csv clean)"
    assert "Trajectory" in err
    assert "ethanol (CCO)" in err


def test_emit_trajectory_writes_artifact(tmp_path, capsys):
    """_emit_trajectory must write trajectory.md when output_dir is given."""
    traj = _make_trajectory()
    cli_module._emit_trajectory(traj, str(tmp_path))
    artifact = tmp_path / "trajectory.md"
    assert artifact.exists(), "trajectory.md not written"
    text = artifact.read_text()
    assert "# Search Trajectory" in text
    assert "## Cycle 1" in text


def test_emit_trajectory_no_artifact_without_output_dir(tmp_path, capsys):
    """_emit_trajectory must not raise when output_dir is None."""
    traj = _make_trajectory()
    cli_module._emit_trajectory(traj, None)   # must not raise
    # nothing written
    assert not (tmp_path / "trajectory.md").exists()


# ---------------------------------------------------------------------------
# DES multi-cycle branch wiring
# ---------------------------------------------------------------------------

def test_des_multicycle_emits_trajectory_to_stderr(monkeypatch, tmp_path, capsys):
    """Running the DES multi-cycle path must print trajectory to stderr."""
    checkpoint = tmp_path / "ckpt.pt"
    checkpoint.write_text("ckpt")
    config = tmp_path / "config.yaml"
    config.write_text("device: cpu\n")

    traj = _make_trajectory()
    multi = _make_multi_outcome(trajectory=traj)

    monkeypatch.setattr(cli_module, "run_multi_cycle_search", lambda **kw: multi)
    monkeypatch.setattr(cli_module, "format_report", lambda *a, **kw: "DES REPORT")

    cli_module.main([
        "--workflow", "des",
        "--component-a", "CCO",
        "--checkpoint-path", str(checkpoint),
        "--config-path", str(config),
        "--n-cycles", "2",
    ])

    _, err = capsys.readouterr()
    assert "Trajectory" in err
    assert "ethanol (CCO)" in err


def test_des_multicycle_writes_trajectory_md(monkeypatch, tmp_path, capsys):
    """Running the DES multi-cycle path with --output-dir writes trajectory.md."""
    checkpoint = tmp_path / "ckpt.pt"
    checkpoint.write_text("ckpt")
    config = tmp_path / "config.yaml"
    config.write_text("device: cpu\n")
    out_dir = tmp_path / "run"

    traj = _make_trajectory()
    multi = _make_multi_outcome(trajectory=traj)

    monkeypatch.setattr(cli_module, "run_multi_cycle_search", lambda **kw: multi)
    monkeypatch.setattr(cli_module, "format_report", lambda *a, **kw: "DES REPORT")

    cli_module.main([
        "--workflow", "des",
        "--component-a", "CCO",
        "--checkpoint-path", str(checkpoint),
        "--config-path", str(config),
        "--n-cycles", "2",
        "--output-dir", str(out_dir),
    ])

    artifact = out_dir / "trajectory.md"
    assert artifact.exists(), "trajectory.md not written to --output-dir"
    text = artifact.read_text()
    assert "# Search Trajectory — DES partners for ethanol (CCO)" in text
    assert "## Cycle 1" in text
    # console trajectory should also appear on stderr
    _, err = capsys.readouterr()
    assert "Trajectory — DES partners for ethanol (CCO)" in err


def test_des_multicycle_none_trajectory_does_not_crash(monkeypatch, tmp_path, capsys):
    """If multi_outcome.trajectory is None, CLI must not crash."""
    checkpoint = tmp_path / "ckpt.pt"
    checkpoint.write_text("ckpt")
    config = tmp_path / "config.yaml"
    config.write_text("device: cpu\n")

    multi = _make_multi_outcome(trajectory=None)
    monkeypatch.setattr(cli_module, "run_multi_cycle_search", lambda **kw: multi)
    monkeypatch.setattr(cli_module, "format_report", lambda *a, **kw: "DES REPORT")

    cli_module.main([
        "--workflow", "des",
        "--component-a", "CCO",
        "--checkpoint-path", str(checkpoint),
        "--config-path", str(config),
        "--n-cycles", "2",
    ])
    # Just must not raise; no trajectory in stderr
    _, err = capsys.readouterr()
    assert "Trajectory" not in err


# ---------------------------------------------------------------------------
# old per-cycle loop must be gone
# ---------------------------------------------------------------------------

def test_des_multicycle_old_loop_removed(monkeypatch, tmp_path, capsys):
    """The old per-cycle stderr loop must be replaced — the format
    '[cycle N/M] screened=X des=Y top-K changes:' must NOT appear in stderr."""
    checkpoint = tmp_path / "ckpt.pt"
    checkpoint.write_text("ckpt")
    config = tmp_path / "config.yaml"
    config.write_text("device: cpu\n")

    from des_multi_agent.multi_cycle import CycleDelta
    delta = CycleDelta(
        cycle=1, n_screened=5, n_des=2,
        top_smiles=frozenset(), new_entrants=[], dropouts=[], converged=False,
    )
    multi = _make_multi_outcome(trajectory=None)
    multi.cycle_deltas = [delta]

    monkeypatch.setattr(cli_module, "run_multi_cycle_search", lambda **kw: multi)
    monkeypatch.setattr(cli_module, "format_report", lambda *a, **kw: "DES REPORT")

    cli_module.main([
        "--workflow", "des",
        "--component-a", "CCO",
        "--checkpoint-path", str(checkpoint),
        "--config-path", str(config),
        "--n-cycles", "2",
    ])
    _, err = capsys.readouterr()
    assert "[cycle 1/2] screened=" not in err, (
        "old per-cycle stderr loop was not removed from cli.py"
    )


# ---------------------------------------------------------------------------
# metal-selectivity branch wiring
# ---------------------------------------------------------------------------

def test_metal_selectivity_emits_trajectory(monkeypatch, tmp_path, capsys):
    """metal-selectivity branch must call _emit_trajectory after the report."""
    traj = _make_trajectory()
    sel_outcome = SimpleNamespace(
        results=[],
        trajectory=traj,
    )

    monkeypatch.setattr(cli_module, "run_metal_selectivity_screen", lambda **kw: sel_outcome)
    monkeypatch.setattr(cli_module, "format_metal_selectivity_report", lambda o: "SEL REPORT")

    cli_module.main([
        "--workflow", "metal-selectivity",
        "--target-metal-ion", "Cu2+",
        "--competitor-metal-ion", "Zn2+",
    ])
    _, err = capsys.readouterr()
    assert "Trajectory" in err
    assert "ethanol (CCO)" in err


# ---------------------------------------------------------------------------
# selectivity-des branch wiring
# ---------------------------------------------------------------------------

def test_selectivity_des_emits_trajectory(monkeypatch, tmp_path, capsys):
    """selectivity-des branch must call _emit_trajectory after the report."""
    checkpoint = tmp_path / "ckpt.pt"
    checkpoint.write_text("ckpt")
    config = tmp_path / "config.yaml"
    config.write_text("device: cpu\n")

    traj = _make_trajectory()
    pipeline_outcome = SimpleNamespace(
        results=[],
        trajectory=traj,
    )

    monkeypatch.setattr(cli_module, "run_selectivity_des_pipeline", lambda **kw: pipeline_outcome)
    monkeypatch.setattr(cli_module, "format_selectivity_des_report", lambda o: "PIPELINE REPORT")

    cli_module.main([
        "--workflow", "selectivity-des",
        "--target-metal-ion", "Cu2+",
        "--competitor-metal-ion", "Zn2+",
        "--checkpoint-path", str(checkpoint),
        "--config-path", str(config),
    ])
    _, err = capsys.readouterr()
    assert "Trajectory" in err
    assert "ethanol (CCO)" in err
