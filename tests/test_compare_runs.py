from pathlib import Path

import pytest

from des_multi_agent.compare_runs import compare_saved_runs, format_compare_report


def _write_memory(path: Path, payload: str) -> None:
    path.write_text(payload, encoding="utf-8")


def test_compare_saved_runs_accepts_run_folders(tmp_path: Path):
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()
    _write_memory(
        run_a / "run.memory.json",
        """{
          "workflow": "des",
          "component_a": "CCO",
          "n": 5,
          "labels": [],
          "ranked_candidates": [
            {"smiles_b": "O", "rank": 1, "min_tm_k": 208.69, "trust_score": 0.95, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""},
            {"smiles_b": "CC(=O)O", "rank": 2, "min_tm_k": 236.03, "trust_score": 0.83, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""}
          ]
        }""",
    )
    _write_memory(
        run_b / "run.memory.json",
        """{
          "workflow": "des",
          "component_a": "CCO",
          "n": 6,
          "labels": [],
          "ranked_candidates": [
            {"smiles_b": "CC(=O)O", "rank": 1, "min_tm_k": 236.03, "trust_score": 0.83, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""},
            {"smiles_b": "CN", "rank": 2, "min_tm_k": 241.11, "trust_score": 0.80, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""}
          ]
        }""",
    )

    comparison = compare_saved_runs(run_a, run_b)

    assert comparison.workflow == "des"
    assert comparison.left_path == run_a
    assert comparison.right_path == run_b
    assert comparison.left_n == 5
    assert comparison.right_n == 6


def test_compare_saved_runs_marks_new_removed_moved_and_unchanged(tmp_path: Path):
    left = tmp_path / "left.memory.json"
    right = tmp_path / "right.memory.json"
    _write_memory(
        left,
        """{
          "workflow": "des",
          "component_a": "CCO",
          "n": 5,
          "labels": [],
          "ranked_candidates": [
            {"smiles_b": "O", "rank": 1, "min_tm_k": 208.69, "trust_score": 0.95, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""},
            {"smiles_b": "CC(=O)O", "rank": 2, "min_tm_k": 236.03, "trust_score": 0.83, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""},
            {"smiles_b": "CN", "rank": 3, "min_tm_k": 241.11, "trust_score": 0.80, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""}
          ]
        }""",
    )
    _write_memory(
        right,
        """{
          "workflow": "des",
          "component_a": "CCO",
          "n": 5,
          "labels": [],
          "ranked_candidates": [
            {"smiles_b": "CC(=O)O", "rank": 1, "min_tm_k": 236.03, "trust_score": 0.83, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""},
            {"smiles_b": "CN", "rank": 2, "min_tm_k": 241.11, "trust_score": 0.80, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""},
            {"smiles_b": "OC", "rank": 3, "min_tm_k": 230.00, "trust_score": 0.75, "uncertainty_flag": "medium", "source": "heuristic", "source_id": ""}
          ]
        }""",
    )

    comparison = compare_saved_runs(left, right)
    rows = {row.smiles_b: row for row in comparison.rows}

    assert rows["CC(=O)O"].status == "moved"
    assert rows["CC(=O)O"].left_rank == 2
    assert rows["CC(=O)O"].right_rank == 1
    assert rows["O"].status == "removed"
    assert rows["O"].left_rank == 1
    assert rows["O"].right_rank is None
    assert rows["CN"].status == "moved"
    assert rows["CN"].left_rank == 3
    assert rows["CN"].right_rank == 2
    assert rows["OC"].status == "new"
    assert rows["OC"].left_rank is None
    assert rows["OC"].right_rank == 3
    assert "new" in format_compare_report(comparison)
    assert "removed" in format_compare_report(comparison)
    assert "moved" in format_compare_report(comparison)


def test_compare_saved_runs_rejects_mismatched_workflow(monkeypatch, tmp_path: Path):
    from des_multi_agent import compare_runs as compare_module
    from des_multi_agent.memory_schema import RunCandidateSummary, RunLabel, RunMemory

    def _fake_load(path):
        if path.name == "left":
            return RunMemory(
                workflow="des",
                component_a="CCO",
                n=5,
                labels=[],
                ranked_candidates=[RunCandidateSummary(smiles_b="O", rank=1)],
            )
        return RunMemory(
            workflow="metal-binding",
            component_a=None,
            n=None,
            labels=[],
            ranked_candidates=[RunCandidateSummary(smiles_b="NCCN", rank=1)],
        )

    monkeypatch.setattr(compare_module, "load_run_memory", _fake_load)

    with pytest.raises(ValueError, match="workflow"):
        compare_saved_runs(tmp_path / "left", tmp_path / "right")


def test_compare_saved_runs_rejects_missing_files(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        compare_saved_runs(tmp_path / "missing_left", tmp_path / "missing_right")


def test_compare_saved_runs_rejects_malformed_memory(tmp_path: Path):
    left = tmp_path / "left.memory.json"
    right = tmp_path / "right.memory.json"
    _write_memory(left, "{not-json}")
    _write_memory(
        right,
        """{
          "workflow": "des",
          "component_a": "CCO",
          "n": 5,
          "labels": [],
          "ranked_candidates": [
            {"smiles_b": "O", "rank": 1, "min_tm_k": 208.69, "trust_score": 0.95, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""}
          ]
        }""",
    )

    with pytest.raises(Exception):
        compare_saved_runs(left, right)
