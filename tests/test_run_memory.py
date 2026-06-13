from pathlib import Path

import pytest

from des_multi_agent.evaluation import DesResult
from des_multi_agent.memory_schema import RunCandidateSummary
from des_multi_agent.prediction import CurvePrediction
from des_multi_agent.run_memory import (
    apply_run_memory_preferences,
    build_chemistry_advisor_memory_notes,
    load_run_memory,
    load_run_memory_history,
    parse_run_memory,
    resolve_run_memory_path,
    update_run_memory_labels,
    write_run_memory,
)
from des_multi_agent.uncertainty.schemas import AnnotatedResult, MinimumTmUncertainty


def _make_annotated_result(smiles_b: str, score: float) -> AnnotatedResult:
    curve = CurvePrediction(
        smiles_a="CCO",
        smiles_b=smiles_b,
        ratios=[0.5],
        tm_pred_k=[200.0],
        t1_k=300.0,
        t2_k=250.0,
        checkpoint_path="ckpt.pt",
    )
    result = DesResult(
        curve=curve,
        absolute_pass=True,
        relative_pass=True,
        is_des=True,
        rationale="ok",
        min_tm_k=200.0,
    )
    uncertainty = MinimumTmUncertainty(
        component_a="CCO",
        component_b=smiles_b,
        repeated_values=(200.0,),
        mean_tm_k=200.0,
        std_tm_k=1.0,
        min_tm_k=200.0,
        max_tm_k=200.0,
        trust_score=score,
        uncertainty_flag="low",
        explanation="ok",
        checkpoint_path="ckpt.pt",
        config_path="config.yaml",
    )
    return AnnotatedResult(
        result=result,
        uncertainty=uncertainty,
        trust_score=score,
        ranking_score=score,
    )


def test_load_run_memory_reads_des_json_from_file(tmp_path: Path):
    memory_path = tmp_path / "run.memory.json"
    memory_path.write_text(
        """{
          "workflow": "des",
          "component_a": "CCO",
          "n": 20,
          "labels": [
            {"smiles_b": "O", "label": "good"},
            {"smiles_b": "CC(=O)O", "label": "bad"}
          ],
          "ranked_candidates": [
            {"smiles_b": "O", "rank": 1, "min_tm_k": 208.69, "trust_score": 0.95, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""}
          ]
        }""",
        encoding="utf-8",
    )

    memory = load_run_memory(memory_path)

    assert memory.workflow == "des"
    assert memory.component_a == "CCO"
    assert memory.n == 20
    assert memory.labels[0].smiles_b == "O"
    assert memory.labels[0].label == "good"
    assert memory.labels[1].smiles_b == "CC(=O)O"
    assert memory.labels[1].label == "bad"
    assert memory.ranked_candidates[0].smiles_b == "O"
    assert memory.ranked_candidates[0].rank == 1


def test_load_run_memory_reads_des_json_from_folder(tmp_path: Path):
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()
    (run_dir / "run.memory.json").write_text(
        """{
          "workflow": "des",
          "component_a": "CCO",
          "n": 10,
          "labels": [],
          "ranked_candidates": []
        }""",
        encoding="utf-8",
    )

    memory = load_run_memory(run_dir)

    assert memory.workflow == "des"
    assert memory.component_a == "CCO"
    assert memory.n == 10


def test_parse_run_memory_rejects_metal_binding_workflow():
    with pytest.raises(ValueError, match="workflow must be des"):
        parse_run_memory(
            {
                "workflow": "metal-binding",
                "component_a": None,
                "n": None,
                "labels": [],
                "ranked_candidates": [{"smiles_b": "O", "rank": 1, "min_tm_k": 208.69, "trust_score": 0.95, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""}],
            }
        )


def test_resolve_run_memory_path_accepts_folder(tmp_path: Path):
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()
    (run_dir / "run.memory.json").write_text("{}", encoding="utf-8")
    assert resolve_run_memory_path(run_dir) == run_dir / "run.memory.json"


def test_resolve_run_memory_path_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        resolve_run_memory_path(tmp_path / "missing")


def test_write_run_memory_round_trips(tmp_path: Path):
    memory = parse_run_memory(
        {
            "workflow": "des",
            "component_a": "CCO",
            "n": 20,
            "labels": [{"smiles_b": "O", "label": "good"}],
            "ranked_candidates": [],
        }
    )
    memory_path = tmp_path / "run.memory.json"
    write_run_memory(memory_path, memory)

    loaded = load_run_memory(memory_path)
    assert loaded == memory


def test_build_chemistry_advisor_memory_notes_compacts_prior_run_memory():
    memory = parse_run_memory(
        {
            "workflow": "des",
            "component_a": "CCO",
            "n": 20,
            "labels": [{"smiles_b": "O", "label": "good"}, {"smiles_b": "CC(=O)O", "label": "bad"}],
            "ranked_candidates": [
                {"smiles_b": "O", "rank": 1, "min_tm_k": 208.69, "trust_score": 0.95, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""},
                {"smiles_b": "CC(=O)O", "rank": 2, "min_tm_k": 236.03, "trust_score": 0.83, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""},
            ],
        }
    )
    notes = build_chemistry_advisor_memory_notes(memory)
    assert any(note.startswith("Prior good labels:") for note in notes)
    assert any(note.startswith("Prior bad labels:") for note in notes)
    assert any(note.startswith("Prior top ranked candidates:") for note in notes)


def test_run_memory_bumps_preferred_candidate():
    memory = parse_run_memory(
        {
            "workflow": "des",
            "component_a": "CCO",
            "n": 20,
            "labels": [{"smiles_b": "O", "label": "good"}],
            "ranked_candidates": [
                {"smiles_b": "O", "rank": 1, "min_tm_k": 208.69, "trust_score": 0.95, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""}
            ],
        }
    )
    adjusted, notes = apply_run_memory_preferences(
        annotated_results=[
            _make_annotated_result("CC(=O)O", 0.70),
            _make_annotated_result("O", 0.60),
        ],
        memory=memory,
        component_a="CCO",
    )
    assert adjusted[0].result.curve.smiles_b == "O"
    assert notes == [
        "Applied reuse memory to 1 good label(s) and 0 bad label(s) across 1 run memory file(s).",
    ]


def test_run_memory_skips_different_component_a():
    memory = parse_run_memory(
        {
            "workflow": "des",
            "component_a": "CCO",
            "n": 20,
            "labels": [{"smiles_b": "O", "label": "good"}],
            "ranked_candidates": [],
        }
    )
    original = [
        _make_annotated_result("CC(=O)O", 0.70),
        _make_annotated_result("O", 0.60),
    ]
    adjusted, notes = apply_run_memory_preferences(
        annotated_results=original,
        memory=memory,
        component_a="CCN",
    )
    assert adjusted == original
    assert notes == ["Reuse memory ignored because it was recorded for CCO, not CCN."]


def test_update_run_memory_labels_last_label_wins():
    memory = parse_run_memory(
        {
            "workflow": "des",
            "component_a": "CCO",
            "n": 20,
            "labels": [{"smiles_b": "O", "label": "good"}],
            "ranked_candidates": [
                {"smiles_b": "O", "rank": 1, "min_tm_k": 208.69, "trust_score": 0.95, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""},
                {"smiles_b": "CC(=O)O", "rank": 2, "min_tm_k": 236.03, "trust_score": 0.83, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""},
            ],
        }
    )
    updated = update_run_memory_labels(
        memory,
        [("O", "good"), ("O", "bad"), ("CC(=O)O", "good")],
    )

    assert [label.smiles_b for label in updated.labels] == ["O", "CC(=O)O"]
    assert updated.labels[0].label == "bad"
    assert updated.labels[1].label == "good"


def test_update_run_memory_labels_rejects_unknown_smiles():
    memory = parse_run_memory(
        {
            "workflow": "des",
            "component_a": "CCO",
            "n": 20,
            "labels": [],
            "ranked_candidates": [
                {"smiles_b": "O", "rank": 1, "min_tm_k": 208.69, "trust_score": 0.95, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""},
            ],
        }
    )

    with pytest.raises(ValueError, match="not found in the saved DES run"):
        update_run_memory_labels(memory, [("N", "good")])


def test_update_run_memory_labels_rejects_invalid_label():
    memory = parse_run_memory(
        {
            "workflow": "des",
            "component_a": "CCO",
            "n": 20,
            "labels": [],
            "ranked_candidates": [
                {"smiles_b": "O", "rank": 1, "min_tm_k": 208.69, "trust_score": 0.95, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""},
            ],
        }
    )

    with pytest.raises(ValueError, match="label must be good or bad"):
        update_run_memory_labels(memory, [("O", "maybe")])


def test_update_run_memory_labels_changes_reuse_bias():
    memory = parse_run_memory(
        {
            "workflow": "des",
            "component_a": "CCO",
            "n": 20,
            "labels": [],
            "ranked_candidates": [
                {"smiles_b": "O", "rank": 1, "min_tm_k": 208.69, "trust_score": 0.95, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""},
                {"smiles_b": "CC(=O)O", "rank": 2, "min_tm_k": 236.03, "trust_score": 0.83, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""},
            ],
        }
    )
    updated = update_run_memory_labels(memory, [("CC(=O)O", "good")])
    adjusted, notes = apply_run_memory_preferences(
        annotated_results=[
            _make_annotated_result("O", 0.60),
            _make_annotated_result("CC(=O)O", 0.70),
        ],
        memory=updated,
        component_a="CCO",
    )
    assert adjusted[0].result.curve.smiles_b == "CC(=O)O"
    assert notes == [
        "Applied reuse memory to 1 good label(s) and 0 bad label(s) across 1 run memory file(s).",
        "Loaded 1 prior ranked candidate(s) for ranking bias across 1 run memory file(s).",
    ]



def test_load_run_memory_history_collects_multiple_run_folders(tmp_path: Path):
    history = tmp_path / "runs"
    run_001 = history / "run_001"
    run_002 = history / "run_002"
    write_run_memory(
        run_001 / "run.memory.json",
        parse_run_memory(
            {
                "workflow": "des",
                "component_a": "CCO",
                "n": 5,
                "labels": [{"smiles_b": "O", "label": "good"}],
                "ranked_candidates": [
                    {"smiles_b": "O", "rank": 1, "min_tm_k": 208.69, "trust_score": 0.95, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""},
                ],
            }
        ),
    )
    write_run_memory(
        run_002 / "run.memory.json",
        parse_run_memory(
            {
                "workflow": "des",
                "component_a": "CCO",
                "n": 5,
                "labels": [{"smiles_b": "CC(=O)O", "label": "bad"}],
                "ranked_candidates": [
                    {"smiles_b": "CC(=O)O", "rank": 1, "min_tm_k": 236.03, "trust_score": 0.83, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""},
                ],
            }
        ),
    )

    memories = load_run_memory_history(history)

    assert len(memories) == 2
    assert [memory.labels[0].smiles_b for memory in memories] == ["O", "CC(=O)O"]
    assert [memory.labels[0].label for memory in memories] == ["good", "bad"]



def test_run_memory_aggregates_feedback_across_multiple_memories():
    memory_a = parse_run_memory(
        {
            "workflow": "des",
            "component_a": "CCO",
            "n": 20,
            "labels": [{"smiles_b": "O", "label": "good"}],
            "ranked_candidates": [
                {"smiles_b": "O", "rank": 1, "min_tm_k": 208.69, "trust_score": 0.95, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""},
                {"smiles_b": "CC(=O)O", "rank": 2, "min_tm_k": 236.03, "trust_score": 0.83, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""},
            ],
        }
    )
    memory_b = parse_run_memory(
        {
            "workflow": "des",
            "component_a": "CCO",
            "n": 20,
            "labels": [{"smiles_b": "CC(=O)O", "label": "bad"}],
            "ranked_candidates": [
                {"smiles_b": "CC(=O)O", "rank": 1, "min_tm_k": 236.03, "trust_score": 0.83, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""},
                {"smiles_b": "O", "rank": 2, "min_tm_k": 208.69, "trust_score": 0.95, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""},
            ],
        }
    )
    adjusted, notes = apply_run_memory_preferences(
        annotated_results=[
            _make_annotated_result("O", 0.60),
            _make_annotated_result("CC(=O)O", 0.60),
        ],
        memory=[memory_a, memory_b],
        component_a="CCO",
    )

    assert adjusted[0].result.curve.smiles_b == "O"
    assert notes == [
        "Applied reuse memory to 1 good label(s) and 1 bad label(s) across 2 run memory file(s).",
        "Loaded 2 prior ranked candidate(s) for ranking bias across 2 run memory file(s).",
    ]


def test_parse_run_memory_rejects_missing_candidate_smiles_with_field_path():
    with pytest.raises(ValueError, match=r"ranked_candidates\[0\] missing required field: smiles_b"):
        parse_run_memory({"workflow": "des", "labels": [], "ranked_candidates": [{"rank": 1}]})


def test_parse_run_memory_rejects_invalid_label_value_with_field_path():
    with pytest.raises(ValueError, match=r"labels\[0\]\.label must be good or bad"):
        parse_run_memory({"workflow": "des", "labels": [{"smiles_b": "O", "label": "maybe"}], "ranked_candidates": []})


def test_parse_run_memory_rejects_non_list_ranked_candidates():
    with pytest.raises(ValueError, match="ranked_candidates must be a list"):
        parse_run_memory({"workflow": "des", "labels": [], "ranked_candidates": {"smiles_b": "O"}})
