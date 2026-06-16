from __future__ import annotations

from csv import reader
from pathlib import Path
import json

import pytest

from des_multi_agent import orchestrator
import des_multi_agent.exporting as exporting_module
from des_multi_agent.evaluation import DesResult
from des_multi_agent.exporting import CSV_COLUMNS, export_des_run_bundle
from des_multi_agent.run_memory import parse_run_memory, write_run_memory
from des_multi_agent.prediction import CurvePrediction
from des_multi_agent.property_resolution import MeltingPointEstimate
from des_multi_agent.schemas import CandidateProposal, DesThresholds
from des_multi_agent.uncertainty import AnnotatedResult, MinimumTmUncertainty, UncertaintyPolicy


def test_export_des_run_bundle_writes_three_files_with_fixed_csv_header(tmp_path: Path):
    output_dir = tmp_path / "runs" / "run_001"
    run_payload = {
        "workflow": "des",
        "component_a": "CCO",
        "n": 5,
        "results": [
            {
                "smiles_b": "O",
                "is_des": True,
                "min_tm_k": 208.69,
                "rank": 1,
                "source": "heuristic",
                "source_id": "rule",
                "trust_score": 0.95,
                "uncertainty_flag": "low",
            }
        ],
        "memory_notes": ["Loaded 1 prior ranked candidates for ranking bias."],
        "warnings": ["demo warning"],
    }

    report_text = "smiles_b | is_des | min_tm_k | source | trust | mean_tm_k | spread_k | std_k | uncertainty_flag | rationale\nO | True | 208.69 | heuristic | trust=0.95 | mean=208.69 K | spread=208.19-209.19 K | std=0.50 K | flag=low | demo"
    exported = export_des_run_bundle(output_dir, run_payload, report_text)

    report_path = output_dir / "report.txt"
    json_path = output_dir / "run.json"
    csv_path = output_dir / "run.csv"
    manifest_path = output_dir / "run.manifest.json"

    assert output_dir.is_dir()
    assert exported == {"report": report_path, "json": json_path, "csv": csv_path, "manifest": manifest_path}
    assert report_path.read_text(encoding="utf-8") == report_text
    assert json.loads(json_path.read_text(encoding="utf-8")) == run_payload

    with csv_path.open(encoding="utf-8") as handle:
        csv_rows = list(reader(handle))
    assert csv_rows[0] == list(CSV_COLUMNS)
    assert csv_rows[1] == ["O", "True", "208.69", "1", "heuristic", "rule", "0.95", "low"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["workflow"] == "des"
    assert manifest["component_a"] == "CCO"
    assert manifest["n"] == 5
    assert manifest["exported_at_utc"].endswith("Z")
    assert manifest["report_filename"] == "report.txt"
    assert manifest["json_filename"] == "run.json"
    assert manifest["csv_filename"] == "run.csv"
    assert manifest["manifest_filename"] == "run.manifest.json"


def test_export_des_run_bundle_fails_when_output_path_is_not_writable(monkeypatch):
    def _raise_permission_error(*args, **kwargs):
        raise PermissionError("cannot write export bundle")

    monkeypatch.setattr(Path, "mkdir", _raise_permission_error)

    with pytest.raises(PermissionError, match="cannot write export bundle"):
        export_des_run_bundle(Path("/tmp/unwritable"), {"workflow": "des", "results": []}, "report")




def _curve(smiles_a: str, smiles_b: str, min_tm_k: float) -> CurvePrediction:
    return CurvePrediction(
        smiles_a=smiles_a,
        smiles_b=smiles_b,
        ratios=[0.1, 0.5, 0.9],
        tm_pred_k=[min_tm_k + 5.0, min_tm_k, min_tm_k + 2.0],
        t1_k=298.15,
        t2_k=300.0,
        checkpoint_path="ckpt.pt",
    )


def _result(smiles_a: str, smiles_b: str, min_tm_k: float) -> DesResult:
    curve = _curve(smiles_a, smiles_b, min_tm_k)
    return DesResult(
        curve=curve,
        absolute_pass=True,
        relative_pass=True,
        is_des=True,
        rationale="ok",
        min_tm_k=min_tm_k,
    )


def _uncertainty(smiles_b: str) -> MinimumTmUncertainty:
    return MinimumTmUncertainty(
        component_a="CCO",
        component_b=smiles_b,
        repeated_values=(238.0, 239.0, 240.0),
        mean_tm_k=239.0,
        std_tm_k=1.0,
        min_tm_k=238.0,
        max_tm_k=240.0,
        trust_score=0.88,
        uncertainty_flag="low",
        explanation="demo",
        checkpoint_path="ckpt.pt",
        config_path="config.yaml",
    )


def test_build_des_export_payload_maps_run_fields():
    outcome = orchestrator.SearchOutcome(
        results=[_result("CCO", "O", 208.69)],
        annotated_results=[AnnotatedResult(result=_result("CCO", "O", 208.69), uncertainty=_uncertainty("O"), trust_score=0.88, ranking_score=0.91)],
        candidate_proposals=[CandidateProposal(smiles="O", rationale="demo", family="alcohol", source="heuristic", source_id="rule")],
        candidate_reviews=[],
        brainstorm_candidates=[],
        explanation_notes=[],
        critique_notes=[],
        llm_warnings=[],
        memory_notes=["Loaded reuse memory from run.memory.json."],
        viscosity_predictions=[],
    )

    payload = orchestrator._build_des_export_payload(
        outcome,
        component_a="CCO",
        n=5,
        checkpoint_path="ckpt.pt",
        config_path="config.yaml",
    )

    assert payload["workflow"] == "des"
    assert payload["component_a"] == "CCO"
    assert payload["n"] == 5
    assert payload["results"][0]["smiles_b"] == "O"
    assert payload["results"][0]["source"] == "heuristic"
    assert payload["results"][0]["uncertainty"]["uncertainty_flag"] == "low"
    assert payload["memory_notes"] == ["Loaded reuse memory from run.memory.json."]
    assert payload["warnings"] == []


def test_run_search_report_exports_bundle_next_to_saved_memory(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        orchestrator,
        "generate_candidates",
        lambda component_a, n, constraints=None: [
            CandidateProposal(smiles="O", rationale="demo", family="alcohol", source="heuristic", source_id="rule"),
        ],
    )
    monkeypatch.setattr(orchestrator, "filter_candidates", lambda component_a, candidates: candidates)
    monkeypatch.setattr(
        orchestrator,
        "resolve_melting_point",
        lambda component, override_k=None: MeltingPointEstimate(component=component, tm_k=300.0, source="heuristic", confidence=0.5),
    )
    monkeypatch.setattr(
        orchestrator,
        "predict_curve",
        lambda component_a, component_b, t1_k, t2_k, checkpoint_path, config_path="ml_des_mp/config.yaml": _curve(component_a, component_b, 230.0),
    )
    monkeypatch.setattr(orchestrator, "classify_des", lambda curve, thresholds: _result(curve.smiles_a, curve.smiles_b, min(curve.tm_pred_k)))
    monkeypatch.setattr(orchestrator, "estimate_min_tm_uncertainty", lambda component_a, component_b, checkpoint_path, config_path: _uncertainty(component_b))

    captured = {}

    def fake_export(output_dir, payload, report_text):
        captured["output_dir"] = Path(output_dir)
        captured["payload"] = payload
        captured["report_text"] = report_text
        return {"report": Path(output_dir) / "report.txt", "json": Path(output_dir) / "run.json", "csv": Path(output_dir) / "run.csv", "manifest": Path(output_dir) / "run.manifest.json"}

    monkeypatch.setattr(orchestrator, "export_des_run_bundle", fake_export)

    checkpoint_path = tmp_path / "ckpt.pt"
    checkpoint_path.write_text("ckpt", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """device: cpu
embedding:
  method: morgan
  morgan:
    radius: 2
    n_bits: 16
    use_chirality: false
""",
        encoding="utf-8",
    )
    save_memory_path = tmp_path / "runs" / "run_001" / "run.memory.json"

    outcome = orchestrator.run_search_report(
        component_a="CCO",
        n=1,
        checkpoint_path=str(checkpoint_path),
        config_path=str(config_path),
        thresholds=DesThresholds(absolute_tm_max_k=260.0, relative_drop_min=0.1),
        uncertainty_policy=UncertaintyPolicy(mode="report_only"),
        save_run_memory_path=str(save_memory_path),
    )

    assert outcome.results
    assert captured["output_dir"] == save_memory_path.parent
    assert captured["payload"]["workflow"] == "des"
    assert captured["payload"]["results"][0]["smiles_b"] == "O"
    assert "compound | is_des" in captured["report_text"]


def test_export_des_run_bundle_requires_fixed_csv_fields(tmp_path: Path):
    output_dir = tmp_path / "run_002"
    run_payload = {
        "workflow": "des",
        "component_a": "CCO",
        "n": 1,
        "results": [
            {
                "smiles_b": "O",
                "is_des": True,
                "min_tm_k": 208.69,
                "rank": 1,
                "source": "heuristic",
                "source_id": "rule",
                "trust_score": 0.95,
            }
        ],
    }

    with pytest.raises(KeyError, match="uncertainty_flag"):
        export_des_run_bundle(output_dir, run_payload, "report")
    assert not output_dir.exists()



def test_run_search_report_reuses_history_directory(monkeypatch, tmp_path: Path):
    history = tmp_path / "runs"
    run_001 = history / "run_001"
    run_002 = history / "run_002"
    run_001.mkdir(parents=True)
    run_002.mkdir(parents=True)
    write_run_memory(
        run_001 / "run.memory.json",
        parse_run_memory(
            {
                "workflow": "des",
                "component_a": "CCO",
                "n": 1,
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
                "n": 1,
                "labels": [{"smiles_b": "CC(=O)O", "label": "bad"}],
                "ranked_candidates": [
                    {"smiles_b": "CC(=O)O", "rank": 1, "min_tm_k": 236.03, "trust_score": 0.83, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""},
                ],
            }
        ),
    )

    monkeypatch.setattr(
        orchestrator,
        "generate_candidates",
        lambda component_a, n, constraints=None: [
            CandidateProposal(smiles="O", rationale="demo", family="alcohol", source="heuristic", source_id="rule"),
            CandidateProposal(smiles="CC(=O)O", rationale="demo", family="acid", source="heuristic", source_id="rule"),
        ],
    )
    monkeypatch.setattr(orchestrator, "filter_candidates", lambda component_a, candidates: candidates)
    monkeypatch.setattr(
        orchestrator,
        "resolve_melting_point",
        lambda component, override_k=None: MeltingPointEstimate(component=component, tm_k=300.0, source="heuristic", confidence=0.5),
    )
    monkeypatch.setattr(
        orchestrator,
        "predict_curve",
        lambda component_a, component_b, t1_k, t2_k, checkpoint_path, config_path="ml_des_mp/config.yaml": _curve(component_a, component_b, 230.0),
    )
    monkeypatch.setattr(orchestrator, "classify_des", lambda curve, thresholds: _result(curve.smiles_a, curve.smiles_b, min(curve.tm_pred_k)))
    monkeypatch.setattr(orchestrator, "estimate_min_tm_uncertainty", lambda component_a, component_b, checkpoint_path, config_path: _uncertainty(component_b))

    captured = {}

    def fake_export(output_dir, payload, report_text):
        captured["output_dir"] = Path(output_dir)
        captured["payload"] = payload
        captured["report_text"] = report_text
        return {"report": Path(output_dir) / "report.txt", "json": Path(output_dir) / "run.json", "csv": Path(output_dir) / "run.csv", "manifest": Path(output_dir) / "run.manifest.json"}

    monkeypatch.setattr(orchestrator, "export_des_run_bundle", fake_export)

    checkpoint_path = tmp_path / "ckpt.pt"
    checkpoint_path.write_text("ckpt", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """device: cpu
embedding:
  method: morgan
  morgan:
    radius: 2
    n_bits: 16
    use_chirality: false
""",
        encoding="utf-8",
    )

    outcome = orchestrator.run_search_report(
        component_a="CCO",
        n=2,
        checkpoint_path=str(checkpoint_path),
        config_path=str(config_path),
        thresholds=DesThresholds(absolute_tm_max_k=260.0, relative_drop_min=0.1),
        uncertainty_policy=UncertaintyPolicy(mode="report_only"),
        reuse_run_path=str(history),
    )

    assert outcome.results[0].curve.smiles_b == "O"
    assert any("2 run memory file(s)" in note for note in outcome.memory_notes)
    # No output_dir passed → export_des_run_bundle should NOT be called
    assert captured == {}, "export_des_run_bundle must not be called when output_dir is not set"
    assert "\nsummary:" not in outcome.report_text  # CLI compact-summary block must not be embedded in report_text


def test_export_des_run_bundle_preserves_existing_bundle_when_replace_fails(monkeypatch, tmp_path: Path):
    output_dir = tmp_path / "run_001"
    output_dir.mkdir()
    for filename in ("report.txt", "run.json", "run.csv", "run.manifest.json"):
        (output_dir / filename).write_text("old", encoding="utf-8")

    def _raise_after_stage(staged_path, final_path):
        raise PermissionError("replace failed")

    monkeypatch.setattr(exporting_module, "_replace_export_file", _raise_after_stage)

    with pytest.raises(PermissionError, match="replace failed"):
        export_des_run_bundle(
            output_dir,
            {
                "workflow": "des",
                "component_a": "CCO",
                "n": 1,
                "results": [{
                    "smiles_b": "O",
                    "is_des": True,
                    "min_tm_k": 208.69,
                    "rank": 1,
                    "source": "heuristic",
                    "source_id": "rule",
                    "trust_score": 0.95,
                    "uncertainty_flag": "low",
                }],
            },
            "new report",
        )

    for filename in ("report.txt", "run.json", "run.csv", "run.manifest.json"):
        assert (output_dir / filename).read_text(encoding="utf-8") == "old"
