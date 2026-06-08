from pathlib import Path

import pytest

from des_multi_agent.cli import main
from des_multi_agent.label_run import run_label_command
from des_multi_agent.run_memory import load_run_memory


def _write_memory(path: Path, payload: str) -> None:
    path.write_text(payload, encoding="utf-8")


def test_label_run_updates_memory_in_place(tmp_path: Path):
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()
    _write_memory(
        run_dir / "run.memory.json",
        """{
          "workflow": "des",
          "component_a": "CCO",
          "n": 20,
          "labels": [],
          "ranked_candidates": [
            {"smiles_b": "O", "rank": 1, "min_tm_k": 208.69, "trust_score": 0.95, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""},
            {"smiles_b": "CC(=O)O", "rank": 2, "min_tm_k": 236.03, "trust_score": 0.83, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""}
          ]
        }""",
    )

    message = run_label_command(run_dir, ["O=good", "O=bad", "CC(=O)O=good"])

    updated = load_run_memory(run_dir)
    assert [label.smiles_b for label in updated.labels] == ["O", "CC(=O)O"]
    assert updated.labels[0].label == "bad"
    assert updated.labels[1].label == "good"
    assert (run_dir / "run.memory.json").exists()
    assert "Updated" in message


def test_label_run_rejects_unknown_smiles(tmp_path: Path):
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()
    _write_memory(
        run_dir / "run.memory.json",
        """{
          "workflow": "des",
          "component_a": "CCO",
          "n": 20,
          "labels": [],
          "ranked_candidates": [
            {"smiles_b": "O", "rank": 1, "min_tm_k": 208.69, "trust_score": 0.95, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""}
          ]
        }""",
    )

    with pytest.raises(SystemExit):
        main(["label-run", "--run", str(run_dir), "--label", "N=good"])


def test_label_run_rejects_invalid_label(tmp_path: Path):
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()
    _write_memory(
        run_dir / "run.memory.json",
        """{
          "workflow": "des",
          "component_a": "CCO",
          "n": 20,
          "labels": [],
          "ranked_candidates": [
            {"smiles_b": "O", "rank": 1, "min_tm_k": 208.69, "trust_score": 0.95, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""}
          ]
        }""",
    )

    with pytest.raises(SystemExit):
        main(["label-run", "--run", str(run_dir), "--label", "O=maybe"])


def test_label_run_rejects_non_des_memory(tmp_path: Path):
    run_dir = tmp_path / "run_002"
    run_dir.mkdir()
    _write_memory(
        run_dir / "run.memory.json",
        """{
          "workflow": "metal-binding",
          "component_a": null,
          "n": null,
          "labels": [],
          "ranked_candidates": []
        }""",
    )

    with pytest.raises(SystemExit):
        main(["label-run", "--run", str(run_dir), "--label", "O=good"])


def test_label_run_rejects_malformed_memory(tmp_path: Path):
    run_dir = tmp_path / "run_003"
    run_dir.mkdir()
    _write_memory(run_dir / "run.memory.json", "{not-json}")

    with pytest.raises(SystemExit):
        main(["label-run", "--run", str(run_dir), "--label", "O=good"])
