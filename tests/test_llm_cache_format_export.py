"""TDD tests for B4 (LLM caching), C6 (--format flag), C3 (ASCII curve chart),
G4 (CI columns in export), B7 (--dry-run flag)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# B4 — LLM call caching
# ---------------------------------------------------------------------------

from des_multi_agent.llm.cache import LLMCache


def test_cache_miss_calls_backend(tmp_path):
    """First call for a prompt hits the backend function."""
    calls = []
    def backend(url, payload, **kw):
        calls.append(payload)
        return '{"result": "fresh"}'

    cache = LLMCache(cache_dir=tmp_path)
    result = cache.get_or_call("http://example.com", {"messages": [{"role": "user", "content": "hi"}]}, backend)
    assert result == '{"result": "fresh"}'
    assert len(calls) == 1


def test_cache_hit_skips_backend(tmp_path):
    """Second identical call returns cached value without calling backend."""
    calls = []
    def backend(url, payload, **kw):
        calls.append(1)
        return '{"result": "cached"}'

    cache = LLMCache(cache_dir=tmp_path)
    payload = {"messages": [{"role": "user", "content": "hello"}]}
    cache.get_or_call("http://example.com", payload, backend)
    result = cache.get_or_call("http://example.com", payload, backend)

    assert result == '{"result": "cached"}'
    assert len(calls) == 1  # backend only called once


def test_different_payloads_are_separate_cache_entries(tmp_path):
    """Different prompts use different cache keys."""
    calls = []
    def backend(url, payload, **kw):
        calls.append(payload)
        return f'{{"n": {len(calls)}}}'

    cache = LLMCache(cache_dir=tmp_path)
    r1 = cache.get_or_call("http://x.com", {"msg": "a"}, backend)
    r2 = cache.get_or_call("http://x.com", {"msg": "b"}, backend)
    assert r1 != r2
    assert len(calls) == 2


def test_cache_respects_ttl(tmp_path):
    """An expired cache entry triggers a new backend call."""
    calls = []
    def backend(url, payload, **kw):
        calls.append(1)
        return f'{{"call": {len(calls)}}}'

    cache = LLMCache(cache_dir=tmp_path, ttl_seconds=0)
    payload = {"messages": [{"role": "user", "content": "ttl_test"}]}
    cache.get_or_call("http://x.com", payload, backend)
    time.sleep(0.01)  # ensure mtime is in the past
    cache.get_or_call("http://x.com", payload, backend)
    assert len(calls) == 2


def test_cache_persists_across_instances(tmp_path):
    """A second LLMCache instance with the same directory reads the prior cache."""
    calls = []
    def backend(url, payload, **kw):
        calls.append(1)
        return '{"cached": true}'

    payload = {"messages": [{"content": "persist"}]}
    LLMCache(cache_dir=tmp_path).get_or_call("http://x.com", payload, backend)
    LLMCache(cache_dir=tmp_path).get_or_call("http://x.com", payload, backend)
    assert len(calls) == 1


def test_cache_disabled_when_dir_is_none():
    """LLMCache with cache_dir=None always calls the backend (no caching)."""
    calls = []
    def backend(url, payload, **kw):
        calls.append(1)
        return '{"no_cache": true}'

    cache = LLMCache(cache_dir=None)
    payload = {"messages": [{"content": "no_cache"}]}
    cache.get_or_call("http://x.com", payload, backend)
    cache.get_or_call("http://x.com", payload, backend)
    assert len(calls) == 2


def test_transport_uses_cache_when_configured(tmp_path):
    """RequestTransport with cache_dir set caches responses."""
    from des_multi_agent.llm.transport import RequestTransport

    calls = []
    def fake_request(url, payload, **kw):
        calls.append(1)
        return '{"answer": 42}'

    transport = RequestTransport(request_fn=fake_request, cache_dir=str(tmp_path))
    payload = {"messages": [{"role": "user", "content": "cached_transport"}]}
    transport.post_json("http://x.com", payload)
    transport.post_json("http://x.com", payload)
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# C6 — --format flag
# ---------------------------------------------------------------------------

def test_cli_format_flag_parsed_table():
    from des_multi_agent.cli import build_parser
    args = build_parser().parse_args(["--workflow", "des", "--component-a", "CCO", "--format", "table"])
    assert args.format == "table"


def test_cli_format_flag_parsed_json():
    from des_multi_agent.cli import build_parser
    args = build_parser().parse_args(["--workflow", "des", "--component-a", "CCO", "--format", "json"])
    assert args.format == "json"


def test_cli_format_flag_parsed_csv():
    from des_multi_agent.cli import build_parser
    args = build_parser().parse_args(["--workflow", "des", "--component-a", "CCO", "--format", "csv"])
    assert args.format == "csv"


def test_cli_format_flag_parsed_prose():
    from des_multi_agent.cli import build_parser
    args = build_parser().parse_args(["--workflow", "des", "--component-a", "CCO", "--format", "prose"])
    assert args.format == "prose"


def test_cli_format_defaults_to_table():
    from des_multi_agent.cli import build_parser
    args = build_parser().parse_args(["--workflow", "des", "--component-a", "CCO"])
    assert args.format == "table"


def test_cli_format_invalid_rejected():
    from des_multi_agent.cli import build_parser
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--workflow", "des", "--component-a", "CCO", "--format", "xml"])


def test_format_report_json_returns_parseable_json():
    from des_multi_agent.reporting import format_report_json
    from des_multi_agent.prediction import CurvePrediction

    curve = CurvePrediction(
        smiles_a="NC(N)=O", smiles_b="O",
        ratios=[0.25, 0.5, 0.75], tm_pred_k=[220.0, 218.0, 222.0],
        t1_k=300.0, t2_k=290.0, checkpoint_path="ckpt.pt",
    )
    result = SimpleNamespace(curve=curve, is_des=True, min_tm_k=218.0, rationale="ok")
    text = format_report_json([result])
    data = json.loads(text)
    assert isinstance(data, list)
    assert data[0]["smiles_b"] == "O"
    assert data[0]["min_tm_k"] == pytest.approx(218.0)
    assert data[0]["is_des"] is True


def test_format_report_csv_returns_header_and_row():
    from des_multi_agent.reporting import format_report_csv
    from des_multi_agent.prediction import CurvePrediction

    curve = CurvePrediction(
        smiles_a="NC(N)=O", smiles_b="O",
        ratios=[0.25, 0.5, 0.75], tm_pred_k=[220.0, 218.0, 222.0],
        t1_k=300.0, t2_k=290.0, checkpoint_path="ckpt.pt",
    )
    result = SimpleNamespace(curve=curve, is_des=True, min_tm_k=218.0, rationale="ok")
    text = format_report_csv([result])
    lines = [l for l in text.splitlines() if l.strip()]
    assert len(lines) >= 2
    assert "smiles_b" in lines[0]
    assert "O" in lines[1]


def test_format_report_prose_contains_only_summary():
    from des_multi_agent.reporting import format_report_prose
    from des_multi_agent.prediction import CurvePrediction

    curve = CurvePrediction(
        smiles_a="NC(N)=O", smiles_b="O",
        ratios=[0.25, 0.5, 0.75], tm_pred_k=[220.0, 218.0, 222.0],
        t1_k=300.0, t2_k=290.0, checkpoint_path="ckpt.pt",
    )
    result = SimpleNamespace(curve=curve, is_des=True, min_tm_k=218.0, rationale="ok")
    text = format_report_prose([result])
    assert "screened" in text.lower() or "candidate" in text.lower()
    # prose should not contain raw pipe-table rows
    assert " | True | " not in text and " | False | " not in text


# ---------------------------------------------------------------------------
# C3 — ASCII melting curve chart
# ---------------------------------------------------------------------------

from des_multi_agent.reporting import format_curve_chart


def test_curve_chart_contains_ratio_labels():
    from des_multi_agent.prediction import CurvePrediction
    curve = CurvePrediction(
        smiles_a="NC(N)=O", smiles_b="O",
        ratios=[0.1, 0.3, 0.5, 0.7, 0.9],
        tm_pred_k=[280.0, 265.0, 240.0, 250.0, 270.0],
        t1_k=300.0, t2_k=290.0, checkpoint_path="ckpt.pt",
    )
    chart = format_curve_chart(curve, title="water (O)")
    assert "water (O)" in chart or "O" in chart
    assert "0.1" in chart or "0.9" in chart  # x-axis tick present


def test_curve_chart_multi_line():
    from des_multi_agent.prediction import CurvePrediction
    curve = CurvePrediction(
        smiles_a="NC(N)=O", smiles_b="O",
        ratios=[0.1, 0.5, 0.9],
        tm_pred_k=[280.0, 240.0, 270.0],
        t1_k=300.0, t2_k=290.0, checkpoint_path="ckpt.pt",
    )
    chart = format_curve_chart(curve)
    assert len(chart.splitlines()) >= 3


def test_curve_chart_marks_minimum():
    """The chart marks the minimum Tm point distinctly."""
    from des_multi_agent.prediction import CurvePrediction
    curve = CurvePrediction(
        smiles_a="NC(N)=O", smiles_b="O",
        ratios=[0.1, 0.5, 0.9],
        tm_pred_k=[280.0, 220.0, 270.0],  # minimum at 0.5
        t1_k=300.0, t2_k=290.0, checkpoint_path="ckpt.pt",
    )
    chart = format_curve_chart(curve)
    # Should show at least one marker character
    assert any(c in chart for c in ("*", "●", "▼", "v", "^", "o", "x"))


def test_format_report_includes_curve_chart_section():
    """format_report with show_curves=True includes a curve chart section."""
    from des_multi_agent.reporting import format_report
    from des_multi_agent.prediction import CurvePrediction

    curve = CurvePrediction(
        smiles_a="NC(N)=O", smiles_b="O",
        ratios=[0.1, 0.5, 0.9],
        tm_pred_k=[280.0, 220.0, 270.0],
        t1_k=300.0, t2_k=290.0, checkpoint_path="ckpt.pt",
    )
    result = SimpleNamespace(curve=curve, is_des=True, min_tm_k=220.0, rationale="ok")
    report = format_report([result], show_curves=True)
    assert "curve" in report.lower() or any(c in report for c in ("*", "●", "▼"))


# ---------------------------------------------------------------------------
# G4 — Confidence interval columns in JSON export
# ---------------------------------------------------------------------------

def test_ensemble_ci_columns_present_in_export_payload(tmp_path):
    """When ensemble_std_k is set, the export payload includes CI columns."""
    from des_multi_agent.prediction import CurvePrediction
    from des_multi_agent import orchestrator

    curve = CurvePrediction(
        smiles_a="NC(N)=O", smiles_b="O",
        ratios=[0.25, 0.5, 0.75],
        tm_pred_k=[230.0, 220.0, 228.0],
        t1_k=300.0, t2_k=290.0,
        checkpoint_path="ckpt.pt",
        ensemble_std_k=[2.0, 1.5, 2.5],
        ensemble_checkpoint_count=5,
    )
    from des_multi_agent.evaluation import DesResult
    from des_multi_agent.uncertainty import AnnotatedResult, MinimumTmUncertainty

    result = DesResult(curve=curve, absolute_pass=True, relative_pass=True,
                       is_des=True, rationale="ok", min_tm_k=220.0)
    unc = MinimumTmUncertainty(
        component_a="NC(N)=O", component_b="O",
        repeated_values=(219.0, 220.0, 221.0),
        mean_tm_k=220.0, std_tm_k=1.0, min_tm_k=219.0, max_tm_k=221.0,
        trust_score=0.9, uncertainty_flag="low", explanation="ok",
        checkpoint_path="ckpt.pt", config_path="config.yaml",
    )
    annotated = AnnotatedResult(result=result, uncertainty=unc, trust_score=0.9, ranking_score=0.95)

    outcome = orchestrator.SearchOutcome(
        results=[result], annotated_results=[annotated],
        candidate_proposals=[], candidate_reviews=[], brainstorm_candidates=[],
        explanation_notes=[], critique_notes=[], llm_warnings=[], memory_notes=[],
    )
    payload = orchestrator._build_des_export_payload(
        outcome, component_a="NC(N)=O", n=1,
        checkpoint_path="ckpt.pt", config_path="config.yaml",
    )
    r = payload["results"][0]
    assert "ensemble_ci_low_k" in r
    assert "ensemble_ci_high_k" in r
    # CI = min_tm ± 2*std_at_min_idx = 220 ± 2*1.5 = [217, 223]
    assert r["ensemble_ci_low_k"] == pytest.approx(217.0)
    assert r["ensemble_ci_high_k"] == pytest.approx(223.0)


def test_ensemble_ci_absent_when_no_ensemble(tmp_path):
    """When ensemble was not used, CI columns are absent from the export payload."""
    from des_multi_agent.prediction import CurvePrediction
    from des_multi_agent import orchestrator
    from des_multi_agent.evaluation import DesResult
    from des_multi_agent.uncertainty import AnnotatedResult, MinimumTmUncertainty

    curve = CurvePrediction(
        smiles_a="NC(N)=O", smiles_b="O",
        ratios=[0.25, 0.5, 0.75], tm_pred_k=[230.0, 220.0, 228.0],
        t1_k=300.0, t2_k=290.0, checkpoint_path="ckpt.pt",
    )
    result = DesResult(curve=curve, absolute_pass=True, relative_pass=True,
                       is_des=True, rationale="ok", min_tm_k=220.0)
    unc = MinimumTmUncertainty(
        component_a="NC(N)=O", component_b="O",
        repeated_values=(219.0, 220.0, 221.0),
        mean_tm_k=220.0, std_tm_k=1.0, min_tm_k=219.0, max_tm_k=221.0,
        trust_score=0.9, uncertainty_flag="low", explanation="ok",
        checkpoint_path="ckpt.pt", config_path="config.yaml",
    )
    annotated = AnnotatedResult(result=result, uncertainty=unc, trust_score=0.9, ranking_score=0.95)
    outcome = orchestrator.SearchOutcome(
        results=[result], annotated_results=[annotated],
        candidate_proposals=[], candidate_reviews=[], brainstorm_candidates=[],
        explanation_notes=[], critique_notes=[], llm_warnings=[], memory_notes=[],
    )
    payload = orchestrator._build_des_export_payload(
        outcome, component_a="NC(N)=O", n=1,
        checkpoint_path="ckpt.pt", config_path="config.yaml",
    )
    r = payload["results"][0]
    assert "ensemble_ci_low_k" not in r
    assert "ensemble_ci_high_k" not in r


# ---------------------------------------------------------------------------
# B7 — --dry-run flag
# ---------------------------------------------------------------------------

def test_cli_dry_run_flag_parsed():
    from des_multi_agent.cli import build_parser
    args = build_parser().parse_args(["--workflow", "des", "--component-a", "CCO", "--dry-run"])
    assert args.dry_run is True


def test_cli_dry_run_defaults_to_false():
    from des_multi_agent.cli import build_parser
    args = build_parser().parse_args(["--workflow", "des", "--component-a", "CCO"])
    assert args.dry_run is False


def test_dry_run_exits_zero_on_valid_setup(monkeypatch, tmp_path, capsys):
    """--dry-run exits 0 when checkpoint, config and compat check all pass."""
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_text("fake")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("device: cpu\nembedding:\n  method: morgan\n  morgan:\n    radius: 2\n    n_bits: 16\n    use_chirality: false\n")

    monkeypatch.setattr("des_multi_agent.prediction.check_checkpoint_config_compat", lambda *a, **kw: [])

    from des_multi_agent.cli import main
    with pytest.raises(SystemExit) as exc:
        main(["--workflow", "des", "--component-a", "CCO",
              "--checkpoint-path", str(ckpt),
              "--config-path", str(cfg),
              "--dry-run"])
    assert exc.value.code == 0


def test_dry_run_does_not_call_generate_candidates(monkeypatch, tmp_path):
    """--dry-run must not call generate_candidates (no predictions run)."""
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_text("fake")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("device: cpu\nembedding:\n  method: morgan\n  morgan:\n    radius: 2\n    n_bits: 16\n    use_chirality: false\n")

    monkeypatch.setattr("des_multi_agent.prediction.check_checkpoint_config_compat", lambda *a, **kw: [])
    generated = []
    monkeypatch.setattr("des_multi_agent.orchestrator.generate_candidates",
                        lambda *a, **kw: generated.append(1) or [])

    from des_multi_agent.cli import main
    with pytest.raises(SystemExit):
        main(["--workflow", "des", "--component-a", "CCO",
              "--checkpoint-path", str(ckpt),
              "--config-path", str(cfg),
              "--dry-run"])
    assert generated == [], "generate_candidates must not be called during --dry-run"


def test_dry_run_reports_compat_warnings(monkeypatch, tmp_path, capsys):
    """--dry-run prints any checkpoint–config mismatch warnings before exiting."""
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_text("fake")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("device: cpu\nembedding:\n  method: morgan\n  morgan:\n    radius: 2\n    n_bits: 16\n    use_chirality: false\n")

    monkeypatch.setattr("des_multi_agent.prediction.check_checkpoint_config_compat",
                        lambda *a, **kw: ["embedding method mismatch: chemberta vs morgan"])

    from des_multi_agent.cli import main
    with pytest.raises(SystemExit) as exc:
        main(["--workflow", "des", "--component-a", "CCO",
              "--checkpoint-path", str(ckpt),
              "--config-path", str(cfg),
              "--dry-run"])
    captured = capsys.readouterr()
    assert "mismatch" in (captured.out + captured.err).lower()
    assert exc.value.code == 0
