import json
from pathlib import Path

from des_multi_agent.trajectory import (
    SearchTrajectory,
    write_trajectory_artifact,
    write_trajectory_json_artifact,
)


def _traj():
    return SearchTrajectory("des", "DES partners for ethanol (CCO)", "min Tm (K)",
                            [], 0, False, "", [])


def test_writes_trajectory_md(tmp_path):
    out = write_trajectory_artifact(tmp_path, _traj())
    assert out == tmp_path / "trajectory.md"
    assert out.exists()
    assert "# Search Trajectory — DES partners for ethanol (CCO)" in out.read_text()


def test_creates_missing_dir(tmp_path):
    target = tmp_path / "nested" / "run"
    out = write_trajectory_artifact(target, _traj())
    assert out.exists()
    assert out.parent == target


def test_overwrites_existing(tmp_path):
    (tmp_path / "trajectory.md").write_text("stale", encoding="utf-8")
    out = write_trajectory_artifact(tmp_path, _traj())
    assert "stale" not in out.read_text()


# ---------------------------------------------------------------------------
# JSON writer
# ---------------------------------------------------------------------------

def test_writes_trajectory_json(tmp_path):
    out = write_trajectory_json_artifact(tmp_path, _traj())
    assert out == tmp_path / "trajectory.json"
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["workflow"] == "des"
    assert data["headline"] == "DES partners for ethanol (CCO)"
    assert data["total_cycles"] == 0
    assert data["snapshots"] == []


def test_json_creates_missing_dir(tmp_path):
    target = tmp_path / "nested" / "run"
    out = write_trajectory_json_artifact(target, _traj())
    assert out.exists()
    assert out.parent == target


def test_json_overwrites_existing(tmp_path):
    (tmp_path / "trajectory.json").write_text('{"stale": true}', encoding="utf-8")
    out = write_trajectory_json_artifact(tmp_path, _traj())
    data = json.loads(out.read_text())
    assert "stale" not in data
