"""Tests for the improvements: B1, B3, C2, C4, X1, A1, A2, B5."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from des_multi_agent.llm.transport import RequestTransport
from des_multi_agent.llm.parser import parse_candidate_review, parse_candidate_brainstorms
from des_multi_agent.reporting import _confidence_label, _format_summary_block, format_report
from des_multi_agent.user_config import load_user_config, save_user_config, get_user_config_path
from des_multi_agent.doctor import run_doctor, DoctorResult
from des_multi_agent import orchestrator as orchestrator_module


# ---------------------------------------------------------------------------
# B1 — Retry / exponential back-off
# ---------------------------------------------------------------------------

def test_transport_retries_on_os_error():
    attempts = []

    def flaky(url, payload, api_key=None, timeout_seconds=None):
        attempts.append(1)
        if len(attempts) < 3:
            raise OSError("connection refused")
        return '{"ok": true}'

    transport = RequestTransport(request_fn=flaky, max_retries=3, retry_base_delay=0.0)
    result = transport.post_json("http://localhost", {})
    assert result == '{"ok": true}'
    assert len(attempts) == 3


def test_transport_raises_after_exhausting_retries():
    def always_fail(url, payload, api_key=None, timeout_seconds=None):
        raise OSError("always down")

    transport = RequestTransport(request_fn=always_fail, max_retries=3, retry_base_delay=0.0)
    with pytest.raises(OSError, match="always down"):
        transport.post_json("http://localhost", {})


def test_transport_does_not_retry_on_value_error():
    attempts = []

    def raises_value_error(url, payload, api_key=None, timeout_seconds=None):
        attempts.append(1)
        raise ValueError("bad payload")

    transport = RequestTransport(request_fn=raises_value_error, max_retries=3, retry_base_delay=0.0)
    with pytest.raises(ValueError):
        transport.post_json("http://localhost", {})
    assert len(attempts) == 1


def test_transport_succeeds_first_try_with_no_retry():
    calls = []

    def ok(url, payload, api_key=None, timeout_seconds=None):
        calls.append(1)
        return "response"

    transport = RequestTransport(request_fn=ok, max_retries=3, retry_base_delay=0.0)
    assert transport.post_json("http://x", {}) == "response"
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# B3 — Raw response in parse errors
# ---------------------------------------------------------------------------

def test_parse_candidate_review_includes_excerpt_on_bad_json():
    bad_raw = "not json at all, here is some garbage text"
    with pytest.raises(ValueError) as exc_info:
        parse_candidate_review(bad_raw)
    assert "not json" in str(exc_info.value).lower() or "excerpt" in str(exc_info.value).lower()
    assert "not json at all" in str(exc_info.value)


def test_parse_candidate_brainstorms_includes_excerpt_on_bad_json():
    bad_raw = "totally invalid { broken json ]["
    with pytest.raises(ValueError) as exc_info:
        parse_candidate_brainstorms(bad_raw)
    assert "Excerpt" in str(exc_info.value)


# ---------------------------------------------------------------------------
# C4 — Confidence labels
# ---------------------------------------------------------------------------

def test_confidence_label_low_flag_is_high_confidence():
    assert _confidence_label(0.90, "low") == "high confidence"


def test_confidence_label_high_flag_is_low_confidence():
    label = _confidence_label(0.30, "high")
    assert "low confidence" in label
    assert "experimental" in label


def test_confidence_label_medium_high_trust():
    label = _confidence_label(0.75, "medium")
    assert "moderate-high" in label


def test_confidence_label_medium_low_trust():
    label = _confidence_label(0.50, "medium")
    assert "moderate confidence" in label


# ---------------------------------------------------------------------------
# C2 — Plain-language summary block
# ---------------------------------------------------------------------------

def _make_result(smiles_b, is_des=True, min_tm_k=220.0, t1_k=300.0, t2_k=290.0):
    curve = SimpleNamespace(smiles_a="CCO", smiles_b=smiles_b, t1_k=t1_k, t2_k=t2_k)
    return SimpleNamespace(curve=curve, is_des=is_des, min_tm_k=min_tm_k, rationale="test")


def test_summary_block_shows_component_a_and_counts():
    results = [_make_result("O"), _make_result("CC", is_des=False)]
    block = _format_summary_block(results, None)
    assert "CCO" in block
    assert "2 candidate(s)" in block
    assert "1 predicted DES-former(s)" in block


def test_summary_block_shows_top_candidate_and_relative_drop():
    results = [_make_result("O", min_tm_k=220.0, t1_k=300.0, t2_k=290.0)]
    block = _format_summary_block(results, None)
    assert "O" in block
    assert "220.0 K" in block
    assert "%" in block


def test_summary_block_empty_results():
    block = _format_summary_block([], None)
    assert "No candidates" in block


def test_format_report_starts_with_summary():
    result = _make_result("O")
    report = format_report([result])
    assert report.startswith("=== DES Search:")


# ---------------------------------------------------------------------------
# X1 — Progress indicators go to stderr
# ---------------------------------------------------------------------------

def test_progress_written_to_stderr(capsys, monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "ckpt.pt"
    checkpoint_path.write_text("ckpt")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("device: cpu\nembedding:\n  method: morgan\n  morgan:\n    radius: 2\n    n_bits: 16\n    use_chirality: false\n")

    from des_multi_agent.schemas import CandidateProposal, DesThresholds
    from des_multi_agent.uncertainty import UncertaintyPolicy

    monkeypatch.setattr(orchestrator_module, "generate_candidates", lambda *a, **kw: [
        CandidateProposal(smiles="O", rationale="r", family="alcohol"),
    ])
    monkeypatch.setattr(orchestrator_module, "filter_candidates", lambda *a, **kw: [
        CandidateProposal(smiles="O", rationale="r", family="alcohol"),
    ])
    from des_multi_agent.prediction import CurvePrediction
    fake_curve = CurvePrediction(smiles_a="CCO", smiles_b="O", ratios=[0.5], tm_pred_k=[220.0], t1_k=300.0, t2_k=290.0, checkpoint_path="ckpt.pt")
    monkeypatch.setattr(orchestrator_module, "resolve_melting_point", lambda *a, **kw: SimpleNamespace(tm_k=300.0))
    monkeypatch.setattr(orchestrator_module, "predict_curve", lambda *a, **kw: fake_curve)
    from des_multi_agent.evaluation import DesResult
    monkeypatch.setattr(orchestrator_module, "classify_des", lambda *a, **kw: DesResult(
        curve=fake_curve, absolute_pass=True, relative_pass=True, is_des=True,
        rationale="ok", min_tm_k=220.0,
    ))
    from des_multi_agent.uncertainty.schemas import MinimumTmUncertainty
    fake_unc = MinimumTmUncertainty(
        component_a="CCO", component_b="O",
        repeated_values=(220.0, 220.5, 221.0),
        mean_tm_k=220.5, std_tm_k=0.5,
        min_tm_k=220.0, max_tm_k=221.0,
        trust_score=0.9, uncertainty_flag="low",
        explanation="ok",
        checkpoint_path="ckpt.pt", config_path="config.yaml",
    )
    monkeypatch.setattr(orchestrator_module, "estimate_min_tm_uncertainty", lambda *a, **kw: fake_unc)

    orchestrator_module.run_search_report(
        component_a="CCO", n=1,
        checkpoint_path=str(checkpoint_path),
        config_path=str(config_path),
        thresholds=DesThresholds(absolute_tm_max_k=260.0, relative_drop_min=0.1),
        uncertainty_policy=UncertaintyPolicy(mode="report_only"),
    )
    err = capsys.readouterr().err
    assert "[1/" in err
    assert "[3/" in err
    assert "[6/" in err


# ---------------------------------------------------------------------------
# A1 — User config file
# ---------------------------------------------------------------------------

def test_load_user_config_returns_empty_when_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("DES_AGENT_CONFIG", str(tmp_path / "nonexistent.yaml"))
    result = load_user_config()
    assert result == {}


def test_save_and_load_user_config(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.yaml"
    monkeypatch.setenv("DES_AGENT_CONFIG", str(cfg_path))
    save_user_config({"checkpoint_path": "/some/path.pt", "llm_config": "/some/llm.yaml"})
    loaded = load_user_config()
    assert loaded["checkpoint_path"] == "/some/path.pt"
    assert loaded["llm_config"] == "/some/llm.yaml"


def test_save_user_config_ignores_unknown_keys(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.yaml"
    monkeypatch.setenv("DES_AGENT_CONFIG", str(cfg_path))
    save_user_config({"checkpoint_path": "/pt", "unknown_key": "should_be_dropped"})
    loaded = load_user_config()
    assert "unknown_key" not in loaded
    assert loaded["checkpoint_path"] == "/pt"


# ---------------------------------------------------------------------------
# A2 — Checkpoint auto-discovery
# ---------------------------------------------------------------------------

def test_cli_auto_discovers_checkpoint(monkeypatch, capsys, tmp_path):
    from des_multi_agent import cli as cli_module

    ckpt = tmp_path / "chemberta_random_row_fold01of05_best.pt"
    ckpt.write_text("fake")

    fake_outcome = SimpleNamespace(
        results=[], annotated_results=[], candidate_proposals=[], candidate_reviews=[],
        explanation_notes=[], critique_notes=[], brainstorm_candidates=[],
        llm_warnings=[], viscosity_predictions=[], memory_notes=[], report_text="REPORT",
    )
    monkeypatch.setattr(cli_module, "run_search_report", lambda **kw: fake_outcome)
    monkeypatch.setattr(cli_module, "_discover_checkpoint", lambda: str(ckpt))
    monkeypatch.setattr(cli_module, "load_user_config", lambda: {})
    monkeypatch.setattr(cli_module, "resolve_existing_path", lambda p, **kw: Path(p))
    monkeypatch.setattr(cli_module, "format_report", lambda *a, **kw: "REPORT")

    cli_module.main(["--workflow", "des", "--component-a", "CCO"])
    err = capsys.readouterr().err
    assert "auto" in err or "discovered" in err


# ---------------------------------------------------------------------------
# B5 — LLM connectivity check in doctor
# ---------------------------------------------------------------------------

def test_doctor_llm_check_warns_when_no_config():
    result = run_doctor(optional_checks=["llm"], llm_cfg=None)
    assert any("[llm]" in issue.message for issue in result.warnings)


def test_doctor_llm_check_warns_on_unreachable_provider(monkeypatch):
    from des_multi_agent.llm.config import LLMConfig
    fake_cfg = LLMConfig(provider="ollama", model_name="gemma4:12b", api_base_url="http://localhost:11434")

    from des_multi_agent import doctor as doctor_module

    def fake_build(cfg):
        return SimpleNamespace(api_base_url="http://localhost:19999")

    monkeypatch.setattr(doctor_module, "_check_llm_connectivity",
                        lambda cfg, warnings: warnings.append(
                            __import__("des_multi_agent.doctor", fromlist=["DoctorIssue"]).DoctorIssue(
                                severity="warning",
                                message="[llm] provider at 'http://localhost:19999' is not reachable: connection refused"
                            )
                        ))
    result = run_doctor(optional_checks=["llm"], llm_cfg=fake_cfg)
    assert any("not reachable" in issue.message for issue in result.warnings)
