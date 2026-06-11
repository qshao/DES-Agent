"""Tests for C1 (SMILES names), X4 (ensemble prediction), X3 (FastAPI server)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

# ---------------------------------------------------------------------------
# C1 — SMILES → human-readable names
# ---------------------------------------------------------------------------

from des_multi_agent.smiles_names import resolve_name, display_name, _CANONICAL_TO_NAME


def test_resolve_name_water():
    assert resolve_name("O") == "water"


def test_resolve_name_ethanol():
    assert resolve_name("CCO") == "ethanol"


def test_resolve_name_glycerol():
    assert resolve_name("OCC(O)CO") == "glycerol"


def test_resolve_name_urea():
    assert resolve_name("NC(N)=O") == "urea"


def test_resolve_name_choline_chloride():
    # Test both orderings of the ionic SMILES
    name = resolve_name("OCC[N+](C)(C)C.[Cl-]")
    assert name is not None
    assert "choline" in name.lower()


def test_resolve_name_unknown_returns_none():
    assert resolve_name("C1CCCCC1CCCC") is None


def test_resolve_name_invalid_smiles_returns_none():
    assert resolve_name("not_a_smiles!!!") is None


def test_display_name_known_compound():
    result = display_name("CCO")
    assert result == "ethanol (CCO)"


def test_display_name_unknown_compound():
    unusual = "C1CCCCC1CCCC"
    result = display_name(unusual)
    assert result == unusual


def test_display_name_pubchem_skipped_by_default():
    # This SMILES is not in the local dict; pubchem=False (default) should return None for resolve_name
    unusual = "C1CCCCC1CCCC"
    assert resolve_name(unusual, pubchem=False) is None


def test_canonical_dict_is_nonempty():
    assert len(_CANONICAL_TO_NAME) >= 30


def test_resolve_name_normalizes_smiles_variants():
    # Different but equivalent SMILES for ethanol
    assert resolve_name("OCC") == "ethanol"   # non-canonical form


def test_report_shows_display_names(monkeypatch):
    """format_report should show display names in compound column."""
    from des_multi_agent.reporting import format_report

    curve = SimpleNamespace(
        smiles_a="CCO", smiles_b="O",
        t1_k=300.0, t2_k=290.0,
        ensemble_checkpoint_count=None, ensemble_std_k=None,
        ratios=[0.5], tm_pred_k=[220.0],
    )
    result = SimpleNamespace(curve=curve, is_des=True, min_tm_k=220.0, rationale="ok")
    report = format_report([result], resolve_names=True)
    assert "water" in report
    assert "(O)" in report


def test_report_without_names():
    from des_multi_agent.reporting import format_report

    curve = SimpleNamespace(
        smiles_a="CCO", smiles_b="O",
        t1_k=300.0, t2_k=290.0,
        ensemble_checkpoint_count=None, ensemble_std_k=None,
        ratios=[0.5], tm_pred_k=[220.0],
    )
    result = SimpleNamespace(curve=curve, is_des=True, min_tm_k=220.0, rationale="ok")
    report = format_report([result], resolve_names=False)
    # Should NOT expand "O" to "water (O)" when resolve_names=False
    lines = [l for l in report.splitlines() if "O" in l and "|" in l]
    assert any(line.startswith("O ") or line.startswith("O|") for line in lines)


# ---------------------------------------------------------------------------
# X4 — Ensemble prediction
# ---------------------------------------------------------------------------

from des_multi_agent.prediction import predict_curve_ensemble, discover_ensemble_checkpoints, CurvePrediction


def test_discover_ensemble_checkpoints_finds_five_folds(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    for i in range(1, 6):
        (runs_dir / f"model_fold0{i}of05_best.pt").write_text("fake")
    found = discover_ensemble_checkpoints(runs_dir)
    assert len(found) == 5


def test_discover_ensemble_checkpoints_empty_dir(tmp_path):
    assert discover_ensemble_checkpoints(tmp_path) == []


def test_discover_ensemble_checkpoints_missing_dir(tmp_path):
    assert discover_ensemble_checkpoints(tmp_path / "nonexistent") == []


def test_predict_curve_ensemble_averages_predictions(monkeypatch):
    """Ensemble should return the mean of per-fold tm_pred_k."""
    from des_multi_agent import prediction as pred_module

    call_count = [0]

    def fake_predict_curve(component_a, component_b, t1_k, t2_k, checkpoint_path, config_path):
        call_count[0] += 1
        # Each fake fold gives slightly different Tm values
        offset = call_count[0] * 2.0
        return CurvePrediction(
            smiles_a=component_a, smiles_b=component_b,
            ratios=[0.25, 0.50, 0.75],
            tm_pred_k=[220.0 + offset, 218.0 + offset, 222.0 + offset],
            t1_k=t1_k, t2_k=t2_k,
            checkpoint_path=str(checkpoint_path),
        )

    monkeypatch.setattr(pred_module, "predict_curve", fake_predict_curve)

    checkpoints = [f"ckpt_fold0{i}.pt" for i in range(1, 4)]
    result = pred_module.predict_curve_ensemble(
        "CCO", "O", t1_k=300.0, t2_k=290.0,
        checkpoint_paths=checkpoints,
        config_path="config.yaml",
    )

    assert result.ensemble_checkpoint_count == 3
    assert result.ensemble_std_k is not None
    assert len(result.ensemble_std_k) == 3
    # Mean of offsets 2, 4, 6 == 4 → mean tm[0] = 220 + 4 = 224
    assert abs(result.tm_pred_k[0] - 224.0) < 1e-6
    # Std of [222, 224, 226] with ddof=1 = 2.0
    assert abs(result.ensemble_std_k[0] - 2.0) < 1e-6


def test_predict_curve_ensemble_raises_on_single_checkpoint():
    with pytest.raises(ValueError, match="at least 2"):
        predict_curve_ensemble("CCO", "O", 300.0, 290.0, ["only_one.pt"])


def test_ensemble_std_shown_in_report():
    from des_multi_agent.reporting import format_report

    curve = CurvePrediction(
        smiles_a="CCO", smiles_b="O",
        ratios=[0.25, 0.50, 0.75],
        tm_pred_k=[220.0, 218.0, 222.0],
        t1_k=300.0, t2_k=290.0,
        checkpoint_path="fold1.pt",
        ensemble_std_k=[1.5, 1.2, 1.8],
        ensemble_checkpoint_count=5,
    )
    result = SimpleNamespace(curve=curve, is_des=True, min_tm_k=218.0, rationale="ok")
    report = format_report([result], resolve_names=False)
    assert "ensemble_folds=5" in report
    assert "ens_std=" in report


def test_cli_ensemble_flag_parsed():
    from des_multi_agent.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["--workflow", "des", "--component-a", "CCO", "--ensemble"])
    assert args.ensemble is True


def test_cli_ensemble_defaults_to_false():
    from des_multi_agent.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["--workflow", "des", "--component-a", "CCO"])
    assert args.ensemble is False


# ---------------------------------------------------------------------------
# X3 — FastAPI server
# ---------------------------------------------------------------------------

from fastapi.testclient import TestClient
from des_multi_agent.server import app

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_search_endpoint_missing_required_fields():
    resp = client.post("/search", json={})
    assert resp.status_code == 422


def test_search_endpoint_file_not_found_returns_422(tmp_path):
    resp = client.post("/search", json={
        "component_a": "CCO",
        "n": 1,
        "checkpoint_path": str(tmp_path / "missing.pt"),
        "config_path": str(tmp_path / "missing.yaml"),
    })
    assert resp.status_code == 422


def test_search_endpoint_runs_pipeline(monkeypatch, tmp_path):
    from des_multi_agent import server as server_module

    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_text("fake")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("device: cpu\n")

    fake_outcome = SimpleNamespace(
        results=[],
        annotated_results=[],
        candidate_proposals=[],
        candidate_reviews=[],
        explanation_notes=[],
        critique_notes=[],
        brainstorm_candidates=[],
        llm_warnings=["warn1"],
        memory_notes=[],
        viscosity_predictions=[],
        report_text="MOCK REPORT",
    )
    monkeypatch.setattr(server_module, "run_search_report", lambda **kw: fake_outcome)

    resp = client.post("/search", json={
        "component_a": "CCO",
        "n": 1,
        "checkpoint_path": str(ckpt),
        "config_path": str(cfg),
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "report" in data
    assert data["llm_warnings"] == ["warn1"]
    assert data["results"] == []


def test_metal_binding_endpoint_missing_fields():
    resp = client.post("/metal-binding", json={})
    assert resp.status_code == 422


def test_metal_binding_endpoint_runs_pipeline(monkeypatch):
    from des_multi_agent import server as server_module

    fake_pred = SimpleNamespace(value=12.5, units="log K", model_name="fallback", source="heuristic", warnings=[])
    fake_outcome = SimpleNamespace(
        metal_ion="Cu2+",
        ligand_smiles="NCC(=O)O",
        prediction=fake_pred,
        warnings=[],
    )
    monkeypatch.setattr(server_module, "run_metal_binding_workflow", lambda **kw: fake_outcome)

    resp = client.post("/metal-binding", json={
        "metal_ion": "Cu2+",
        "ligand_smiles": "NCC(=O)O",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["metal_ion"] == "Cu2+"
    assert data["value"] == pytest.approx(12.5)


def test_search_ensemble_not_enough_checkpoints(monkeypatch):
    from des_multi_agent import server as server_module
    monkeypatch.setattr(server_module, "discover_ensemble_checkpoints", lambda: [])

    resp = client.post("/search", json={
        "component_a": "CCO",
        "n": 1,
        "checkpoint_path": "/fake.pt",
        "config_path": "/fake.yaml",
        "ensemble": True,
    })
    assert resp.status_code == 422
    assert "ensemble" in resp.json()["detail"].lower()


def test_server_openapi_schema_has_all_endpoints():
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    paths = schema["paths"]
    assert "/health" in paths
    assert "/search" in paths
    assert "/metal-binding" in paths
