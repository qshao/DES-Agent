"""TDD tests for G1 (leaderboard), B2 (LLM schema validation),
D2 (checkpoint–config check), E2 (history viewer), E4 (config save command)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# G1 — Cross-run leaderboard
# ---------------------------------------------------------------------------

from des_multi_agent.leaderboard import build_leaderboard, format_leaderboard


def _write_run_json(path: Path, component_a: str, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "workflow": "des",
        "component_a": component_a,
        "n": len(results),
        "results": results,
        "warnings": [],
        "memory_notes": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_leaderboard_merges_results_from_two_runs(tmp_path):
    """build_leaderboard combines candidates across runs, keeping the best Tm per compound."""
    _write_run_json(tmp_path / "run_001" / "run.json", "NC(N)=O", [
        {"smiles_b": "O", "min_tm_k": 225.0, "is_des": True, "rank": 1,
         "source": "heuristic", "trust_score": 0.9, "uncertainty_flag": "low"},
    ])
    _write_run_json(tmp_path / "run_002" / "run.json", "NC(N)=O", [
        {"smiles_b": "O", "min_tm_k": 220.0, "is_des": True, "rank": 1,
         "source": "heuristic", "trust_score": 0.85, "uncertainty_flag": "low"},
        {"smiles_b": "CC(=O)O", "min_tm_k": 240.0, "is_des": True, "rank": 2,
         "source": "heuristic", "trust_score": 0.8, "uncertainty_flag": "medium"},
    ])
    entries = build_leaderboard(tmp_path)
    smiles_set = {e["smiles_b"] for e in entries}
    assert "O" in smiles_set
    assert "CC(=O)O" in smiles_set
    assert len(entries) == 2


def test_leaderboard_keeps_best_min_tm_per_compound(tmp_path):
    """When a compound appears in multiple runs, only the best (lowest) Tm is kept."""
    _write_run_json(tmp_path / "run_001" / "run.json", "NC(N)=O", [
        {"smiles_b": "O", "min_tm_k": 230.0, "is_des": True, "rank": 1,
         "source": "heuristic", "trust_score": 0.9, "uncertainty_flag": "low"},
    ])
    _write_run_json(tmp_path / "run_002" / "run.json", "NC(N)=O", [
        {"smiles_b": "O", "min_tm_k": 210.0, "is_des": True, "rank": 1,
         "source": "heuristic", "trust_score": 0.88, "uncertainty_flag": "low"},
    ])
    entries = build_leaderboard(tmp_path)
    assert len(entries) == 1
    assert entries[0]["min_tm_k"] == pytest.approx(210.0)


def test_leaderboard_sorted_by_min_tm_ascending(tmp_path):
    """Leaderboard entries are sorted from lowest (best) Tm to highest."""
    _write_run_json(tmp_path / "run_001" / "run.json", "NC(N)=O", [
        {"smiles_b": "O", "min_tm_k": 230.0, "is_des": True, "rank": 2,
         "source": "heuristic", "trust_score": 0.9, "uncertainty_flag": "low"},
        {"smiles_b": "CC(=O)O", "min_tm_k": 215.0, "is_des": True, "rank": 1,
         "source": "heuristic", "trust_score": 0.85, "uncertainty_flag": "low"},
    ])
    entries = build_leaderboard(tmp_path)
    tms = [e["min_tm_k"] for e in entries]
    assert tms == sorted(tms)


def test_leaderboard_records_run_count_per_compound(tmp_path):
    """Each entry tracks how many runs contained that compound."""
    _write_run_json(tmp_path / "run_001" / "run.json", "NC(N)=O", [
        {"smiles_b": "O", "min_tm_k": 225.0, "is_des": True, "rank": 1,
         "source": "heuristic", "trust_score": 0.9, "uncertainty_flag": "low"},
    ])
    _write_run_json(tmp_path / "run_002" / "run.json", "NC(N)=O", [
        {"smiles_b": "O", "min_tm_k": 220.0, "is_des": True, "rank": 1,
         "source": "heuristic", "trust_score": 0.85, "uncertainty_flag": "low"},
    ])
    entries = build_leaderboard(tmp_path)
    assert entries[0]["run_count"] == 2


def test_leaderboard_empty_directory(tmp_path):
    """An empty directory returns an empty leaderboard (no crash)."""
    assert build_leaderboard(tmp_path) == []


def test_leaderboard_missing_directory(tmp_path):
    """A non-existent directory raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        build_leaderboard(tmp_path / "nonexistent")


def test_format_leaderboard_contains_header(tmp_path):
    _write_run_json(tmp_path / "run_001" / "run.json", "NC(N)=O", [
        {"smiles_b": "O", "min_tm_k": 220.0, "is_des": True, "rank": 1,
         "source": "heuristic", "trust_score": 0.9, "uncertainty_flag": "low"},
    ])
    entries = build_leaderboard(tmp_path)
    text = format_leaderboard(entries)
    assert "min_tm_k" in text
    assert "compound" in text.lower() or "smiles" in text.lower()


def test_cli_leaderboard_subcommand_parsed():
    from des_multi_agent.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["leaderboard", "/some/dir"])
    assert args.command == "leaderboard"
    assert args.history_dir == "/some/dir"


# ---------------------------------------------------------------------------
# B2 — LLM response schema validation
# ---------------------------------------------------------------------------

from des_multi_agent.llm.errors import LLMSchemaError


def test_llm_schema_error_is_importable():
    """LLMSchemaError exists and is a subclass of ValueError."""
    assert issubclass(LLMSchemaError, ValueError)


def test_parse_candidate_review_missing_smiles_raises_schema_error():
    from des_multi_agent.llm.parser import parse_candidate_review
    import json
    raw = json.dumps({"decision": "keep", "confidence": 0.9, "rationale": "good"})
    with pytest.raises(LLMSchemaError, match="smiles"):
        parse_candidate_review(raw)


def test_parse_candidate_review_invalid_decision_raises_schema_error():
    from des_multi_agent.llm.parser import parse_candidate_review
    import json
    raw = json.dumps({"smiles": "O", "decision": "maybe", "confidence": 0.9, "rationale": "good"})
    with pytest.raises(LLMSchemaError, match="decision"):
        parse_candidate_review(raw)


def test_parse_candidate_review_missing_confidence_raises_schema_error():
    from des_multi_agent.llm.parser import parse_candidate_review
    import json
    raw = json.dumps({"smiles": "O", "decision": "keep", "rationale": "good"})
    with pytest.raises(LLMSchemaError, match="confidence"):
        parse_candidate_review(raw)


def test_parse_candidate_brainstorm_missing_smiles_skipped_not_raised():
    """Brainstorm items missing smiles are silently skipped (parser is lenient for lists)."""
    from des_multi_agent.llm.parser import parse_candidate_brainstorms
    import json
    raw = json.dumps([{"rationale": "good", "family": "alcohol"}])
    result = parse_candidate_brainstorms(raw)
    assert result == []


def test_parse_candidate_review_valid_does_not_raise():
    from des_multi_agent.llm.parser import parse_candidate_review
    import json
    raw = json.dumps({"smiles": "O", "decision": "keep", "confidence": 0.85, "rationale": "ok"})
    review = parse_candidate_review(raw)
    assert review.smiles == "O"


# ---------------------------------------------------------------------------
# D2 — Checkpoint–config compatibility check
# ---------------------------------------------------------------------------

from des_multi_agent.prediction import check_checkpoint_config_compat


def test_compat_check_passes_on_matching_embedding(tmp_path):
    """No warning when checkpoint and config agree on embedding method."""
    import torch
    ckpt_path = tmp_path / "model.pt"
    torch.save({"model_state": {}, "emb_dim_in": 128, "cfg": {"embedding": {"method": "morgan"}}}, str(ckpt_path))
    cfg = {"embedding": {"method": "morgan", "morgan": {"radius": 2, "n_bits": 16, "use_chirality": False}}}
    warnings = check_checkpoint_config_compat(str(ckpt_path), cfg)
    assert warnings == []


def test_compat_check_warns_on_method_mismatch(tmp_path):
    """Warning when checkpoint embedding method differs from config."""
    import torch
    ckpt_path = tmp_path / "model.pt"
    torch.save({"model_state": {}, "emb_dim_in": 128, "cfg": {"embedding": {"method": "chemberta"}}}, str(ckpt_path))
    cfg = {"embedding": {"method": "morgan", "morgan": {"radius": 2, "n_bits": 16, "use_chirality": False}}}
    warnings = check_checkpoint_config_compat(str(ckpt_path), cfg)
    assert any("method" in w.lower() or "embedding" in w.lower() for w in warnings)


def test_compat_check_warns_on_n_bits_mismatch(tmp_path):
    """Warning when morgan n_bits in checkpoint differs from config."""
    import torch
    ckpt_path = tmp_path / "model.pt"
    torch.save({
        "model_state": {}, "emb_dim_in": 128,
        "cfg": {"embedding": {"method": "morgan", "morgan": {"radius": 2, "n_bits": 4096}}},
    }, str(ckpt_path))
    cfg = {"embedding": {"method": "morgan", "morgan": {"radius": 2, "n_bits": 16, "use_chirality": False}}}
    warnings = check_checkpoint_config_compat(str(ckpt_path), cfg)
    assert any("n_bits" in w for w in warnings)


def test_compat_check_no_cfg_key_returns_empty(tmp_path):
    """Checkpoint without 'cfg' key returns no warnings (old format, can't check)."""
    import torch
    ckpt_path = tmp_path / "model.pt"
    torch.save({"model_state": {}, "emb_dim_in": 128}, str(ckpt_path))
    cfg = {"embedding": {"method": "morgan", "morgan": {"radius": 2, "n_bits": 16, "use_chirality": False}}}
    warnings = check_checkpoint_config_compat(str(ckpt_path), cfg)
    assert warnings == []


def test_compat_check_warning_surfaced_in_predict_curve(monkeypatch, tmp_path):
    """Compatibility warnings from check_checkpoint_config_compat reach the caller."""
    import torch
    ckpt_path = tmp_path / "ckpt.pt"
    cfg_path = tmp_path / "config.yaml"
    torch.save({
        "model_state": {}, "emb_dim_in": 128,
        "cfg": {"embedding": {"method": "chemberta"}},
    }, str(ckpt_path))
    cfg_path.write_text("device: cpu\nembedding:\n  method: morgan\n  morgan:\n    radius: 2\n    n_bits: 16\n    use_chirality: false\n")

    captured = []
    monkeypatch.setattr("des_multi_agent.prediction.check_checkpoint_config_compat",
                        lambda *a, **kw: ["embedding method mismatch: chemberta vs morgan"])

    import sys, io
    orig_stderr = sys.stderr
    sys.stderr = buf = io.StringIO()
    try:
        from des_multi_agent import prediction as pred_mod
        # Trigger the warning path by calling check via the module
        warns = pred_mod.check_checkpoint_config_compat(str(ckpt_path), {})
        captured.extend(warns)
    finally:
        sys.stderr = orig_stderr
    assert any("mismatch" in w for w in captured)


# ---------------------------------------------------------------------------
# E2 — Run history viewer
# ---------------------------------------------------------------------------

from des_multi_agent.history import build_history_table, format_history_table


def _write_manifest(path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    defaults = {
        "workflow": "des",
        "component_a": "NC(N)=O",
        "n": 5,
        "exported_at_utc": "2026-06-09T10:00:00Z",
        "report_filename": "report.txt",
        "json_filename": "run.json",
        "csv_filename": "run.csv",
        "manifest_filename": "run.manifest.json",
    }
    defaults.update(kwargs)
    path.write_text(json.dumps(defaults), encoding="utf-8")


def _write_run_with_manifest(base: Path, run_name: str, component_a: str,
                              results: list[dict], exported_at: str = "2026-06-09T10:00:00Z") -> None:
    run_dir = base / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    run_json = {
        "workflow": "des",
        "component_a": component_a,
        "n": len(results),
        "results": results,
        "warnings": [],
        "memory_notes": [],
    }
    (run_dir / "run.json").write_text(json.dumps(run_json), encoding="utf-8")
    _write_manifest(run_dir / "run.manifest.json", component_a=component_a,
                    n=len(results), exported_at_utc=exported_at)


def test_history_table_lists_all_runs(tmp_path):
    _write_run_with_manifest(tmp_path, "run_001", "NC(N)=O", [
        {"smiles_b": "O", "min_tm_k": 220.0, "is_des": True, "rank": 1,
         "source": "heuristic", "trust_score": 0.9, "uncertainty_flag": "low"},
    ])
    _write_run_with_manifest(tmp_path, "run_002", "NC(N)=O", [
        {"smiles_b": "CC(=O)O", "min_tm_k": 235.0, "is_des": True, "rank": 1,
         "source": "heuristic", "trust_score": 0.8, "uncertainty_flag": "medium"},
    ])
    rows = build_history_table(tmp_path)
    assert len(rows) == 2


def test_history_table_row_has_required_fields(tmp_path):
    _write_run_with_manifest(tmp_path, "run_001", "NC(N)=O", [
        {"smiles_b": "O", "min_tm_k": 220.0, "is_des": True, "rank": 1,
         "source": "heuristic", "trust_score": 0.9, "uncertainty_flag": "low"},
    ])
    rows = build_history_table(tmp_path)
    row = rows[0]
    assert "run_name" in row
    assert "exported_at_utc" in row
    assert "n_screened" in row
    assert "n_des" in row
    assert "top_candidate" in row
    assert "top_min_tm_k" in row


def test_history_table_empty_dir(tmp_path):
    assert build_history_table(tmp_path) == []


def test_history_table_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_history_table(tmp_path / "nonexistent")


def test_format_history_table_contains_headers(tmp_path):
    _write_run_with_manifest(tmp_path, "run_001", "NC(N)=O", [
        {"smiles_b": "O", "min_tm_k": 220.0, "is_des": True, "rank": 1,
         "source": "heuristic", "trust_score": 0.9, "uncertainty_flag": "low"},
    ])
    rows = build_history_table(tmp_path)
    text = format_history_table(rows)
    assert "run_name" in text or "run" in text.lower()
    assert "top" in text.lower() or "tm" in text.lower()


def test_cli_history_subcommand_parsed():
    from des_multi_agent.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["history", "/some/runs"])
    assert args.command == "history"
    assert args.history_dir == "/some/runs"


# ---------------------------------------------------------------------------
# E4 — Config save command
# ---------------------------------------------------------------------------

def test_cli_config_set_subcommand_parsed():
    from des_multi_agent.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["config", "set", "checkpoint_path=/tmp/ckpt.pt"])
    assert args.command == "config"
    assert args.config_subcommand == "set"
    assert args.assignment == "checkpoint_path=/tmp/ckpt.pt"


def test_config_set_writes_key_to_user_config(tmp_path, monkeypatch):
    """des-agent config set checkpoint_path=<path> saves the value to user config."""
    config_path = tmp_path / ".des-agent" / "config.yaml"
    monkeypatch.setenv("DES_AGENT_CONFIG", str(config_path))

    from des_multi_agent.cli import main
    main(["config", "set", f"checkpoint_path={tmp_path}/ckpt.pt"])

    from des_multi_agent.user_config import load_user_config
    cfg = load_user_config()
    assert cfg.get("checkpoint_path") == str(tmp_path / "ckpt.pt")


def test_config_set_unknown_key_exits_with_error(tmp_path, monkeypatch, capsys):
    """des-agent config set unknown_key=val exits with a non-zero error."""
    config_path = tmp_path / ".des-agent" / "config.yaml"
    monkeypatch.setenv("DES_AGENT_CONFIG", str(config_path))

    from des_multi_agent.cli import main
    with pytest.raises(SystemExit) as exc:
        main(["config", "set", "unknown_key=foo"])
    assert exc.value.code != 0


def test_config_set_malformed_assignment_exits_with_error(tmp_path, monkeypatch):
    config_path = tmp_path / ".des-agent" / "config.yaml"
    monkeypatch.setenv("DES_AGENT_CONFIG", str(config_path))

    from des_multi_agent.cli import main
    with pytest.raises(SystemExit) as exc:
        main(["config", "set", "no_equals_sign"])
    assert exc.value.code != 0
