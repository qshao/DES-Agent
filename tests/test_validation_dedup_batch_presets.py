"""TDD tests for D1 (SMILES validation), D3 (deduplication notes),
F1 (per-candidate graceful failure), D4 (batch file input), E1 (threshold presets)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from des_multi_agent.prediction import CurvePrediction
from des_multi_agent.property_resolution import MeltingPointEstimate
from des_multi_agent.schemas import CandidateProposal, DesThresholds
from des_multi_agent.uncertainty import AnnotatedResult, MinimumTmUncertainty, UncertaintyPolicy
from des_multi_agent import orchestrator


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _curve(smiles_a: str, smiles_b: str, min_tm_k: float = 230.0) -> CurvePrediction:
    return CurvePrediction(
        smiles_a=smiles_a, smiles_b=smiles_b,
        ratios=[0.1, 0.5, 0.9],
        tm_pred_k=[min_tm_k + 5.0, min_tm_k, min_tm_k + 2.0],
        t1_k=298.15, t2_k=300.0,
        checkpoint_path="ckpt.pt",
    )


def _uncertainty(smiles_b: str) -> MinimumTmUncertainty:
    return MinimumTmUncertainty(
        component_a="CCO", component_b=smiles_b,
        repeated_values=(238.0, 239.0, 240.0),
        mean_tm_k=239.0, std_tm_k=1.0,
        min_tm_k=238.0, max_tm_k=240.0,
        trust_score=0.88, uncertainty_flag="low",
        explanation="demo",
        checkpoint_path="ckpt.pt", config_path="config.yaml",
    )


def _patch_orchestrator_basics(monkeypatch, proposals: list[CandidateProposal]):
    """Patch the minimal orchestrator collaborators for unit tests."""
    from des_multi_agent.evaluation import DesResult

    monkeypatch.setattr(orchestrator, "generate_candidates", lambda *a, **kw: proposals)
    monkeypatch.setattr(orchestrator, "filter_candidates", lambda component_a, candidates: candidates)
    monkeypatch.setattr(
        orchestrator, "resolve_melting_point",
        lambda component, override_k=None: MeltingPointEstimate(component=component, tm_k=300.0, source="heuristic", confidence=0.5),
    )

    def _fake_curve(component_a, component_b, t1_k, t2_k, checkpoint_path, config_path="ml_des_mp/config.yaml"):
        return _curve(component_a, component_b)

    monkeypatch.setattr(orchestrator, "predict_curve", _fake_curve)
    monkeypatch.setattr(
        orchestrator, "classify_des",
        lambda curve, thresholds: DesResult(
            curve=curve, absolute_pass=True, relative_pass=True,
            is_des=True, rationale="ok", min_tm_k=min(curve.tm_pred_k),
        ),
    )
    monkeypatch.setattr(
        orchestrator, "estimate_min_tm_uncertainty",
        lambda component_a, component_b, checkpoint_path, config_path: _uncertainty(component_b),
    )


# ---------------------------------------------------------------------------
# D1 — Input SMILES validation
# ---------------------------------------------------------------------------

def test_invalid_component_a_raises_value_error(monkeypatch, tmp_path):
    """run_search_report raises ValueError with a clear message for an invalid component_a SMILES."""
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_text("fake")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("device: cpu\nembedding:\n  method: morgan\n  morgan:\n    radius: 2\n    n_bits: 16\n    use_chirality: false\n")

    with pytest.raises(ValueError, match="component_a"):
        orchestrator.run_search_report(
            component_a="not_valid_smiles!!!",
            n=1,
            checkpoint_path=str(ckpt),
            config_path=str(cfg),
        )


def test_invalid_component_a_message_contains_smiles(monkeypatch, tmp_path):
    """Error message for invalid component_a includes the offending SMILES."""
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_text("fake")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("device: cpu\nembedding:\n  method: morgan\n  morgan:\n    radius: 2\n    n_bits: 16\n    use_chirality: false\n")

    with pytest.raises(ValueError, match="INVALID_SMILES_XYZ"):
        orchestrator.run_search_report(
            component_a="INVALID_SMILES_XYZ",
            n=1,
            checkpoint_path=str(ckpt),
            config_path=str(cfg),
        )


def test_valid_component_a_does_not_raise(monkeypatch, tmp_path):
    """A valid SMILES passes validation and continues normally."""
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_text("fake")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("device: cpu\nembedding:\n  method: morgan\n  morgan:\n    radius: 2\n    n_bits: 16\n    use_chirality: false\n")

    proposals = [CandidateProposal(smiles="O", rationale="water", family="alcohol", source="heuristic", source_id="")]
    _patch_orchestrator_basics(monkeypatch, proposals)

    outcome = orchestrator.run_search_report(
        component_a="CCO",  # valid SMILES for ethanol
        n=1,
        checkpoint_path=str(ckpt),
        config_path=str(cfg),
    )
    assert outcome.results  # ran successfully


# ---------------------------------------------------------------------------
# D3 — Candidate deduplication memory note
# ---------------------------------------------------------------------------

def test_dedup_note_appears_when_candidates_collapse(monkeypatch, tmp_path):
    """When two proposals have equivalent SMILES (canonical vs non-canonical),
    a memory note records the deduplication."""
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_text("fake")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("device: cpu\nembedding:\n  method: morgan\n  morgan:\n    radius: 2\n    n_bits: 16\n    use_chirality: false\n")

    # CCO and OCC are both ethanol — should collapse to one
    proposals = [
        CandidateProposal(smiles="CCO", rationale="ethanol canonical", family="alcohol", source="heuristic", source_id="rule1"),
        CandidateProposal(smiles="OCC", rationale="ethanol non-canonical", family="alcohol", source="llm", source_id="brainstorm"),
    ]
    _patch_orchestrator_basics(monkeypatch, proposals)

    outcome = orchestrator.run_search_report(
        component_a="NC(N)=O",
        n=2,
        checkpoint_path=str(ckpt),
        config_path=str(cfg),
    )
    dedup_notes = [n for n in outcome.memory_notes if "dedup" in n.lower() or "duplicate" in n.lower()]
    assert dedup_notes, f"Expected a deduplication memory note, got: {outcome.memory_notes}"


def test_dedup_note_absent_when_no_duplicates(monkeypatch, tmp_path):
    """No deduplication note when all candidates are distinct."""
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_text("fake")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("device: cpu\nembedding:\n  method: morgan\n  morgan:\n    radius: 2\n    n_bits: 16\n    use_chirality: false\n")

    proposals = [
        CandidateProposal(smiles="O", rationale="water", family="alcohol", source="heuristic", source_id=""),
        CandidateProposal(smiles="CC(=O)O", rationale="acetic acid", family="acid", source="heuristic", source_id=""),
    ]
    _patch_orchestrator_basics(monkeypatch, proposals)

    outcome = orchestrator.run_search_report(
        component_a="NC(N)=O",
        n=2,
        checkpoint_path=str(ckpt),
        config_path=str(cfg),
    )
    dedup_notes = [n for n in outcome.memory_notes if "dedup" in n.lower() or "duplicate" in n.lower()]
    assert not dedup_notes


# ---------------------------------------------------------------------------
# F1 — Per-candidate graceful failure
# ---------------------------------------------------------------------------

def test_prediction_failure_skips_candidate_and_continues(monkeypatch, tmp_path):
    """If predict_curve raises for one candidate the run continues with the others."""
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_text("fake")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("device: cpu\nembedding:\n  method: morgan\n  morgan:\n    radius: 2\n    n_bits: 16\n    use_chirality: false\n")

    proposals = [
        CandidateProposal(smiles="O", rationale="water", family="alcohol", source="heuristic", source_id=""),
        CandidateProposal(smiles="CC(=O)O", rationale="acetic acid", family="acid", source="heuristic", source_id=""),
    ]
    _patch_orchestrator_basics(monkeypatch, proposals)

    from des_multi_agent.evaluation import DesResult

    call_count = [0]

    def _failing_predict(component_a, component_b, t1_k, t2_k, checkpoint_path, config_path="ml_des_mp/config.yaml"):
        call_count[0] += 1
        if component_b == "O":
            raise RuntimeError("Simulated model crash for water")
        return _curve(component_a, component_b)

    monkeypatch.setattr(orchestrator, "predict_curve", _failing_predict)

    outcome = orchestrator.run_search_report(
        component_a="NC(N)=O",
        n=2,
        checkpoint_path=str(ckpt),
        config_path=str(cfg),
    )
    # The run must not raise; acetic acid should still be in results
    assert any("CC(=O)O" in r.curve.smiles_b for r in outcome.results)
    # "O" should be absent from results
    assert not any("smiles_b" in str(r) and r.curve.smiles_b == "O" for r in outcome.results)


def test_prediction_failure_adds_warning(monkeypatch, tmp_path):
    """A failed prediction emits a warning in llm_warnings."""
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_text("fake")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("device: cpu\nembedding:\n  method: morgan\n  morgan:\n    radius: 2\n    n_bits: 16\n    use_chirality: false\n")

    proposals = [
        CandidateProposal(smiles="O", rationale="water", family="alcohol", source="heuristic", source_id=""),
    ]
    _patch_orchestrator_basics(monkeypatch, proposals)

    def _failing_predict(component_a, component_b, t1_k, t2_k, checkpoint_path, config_path="ml_des_mp/config.yaml"):
        raise RuntimeError("Simulated crash")

    monkeypatch.setattr(orchestrator, "predict_curve", _failing_predict)

    outcome = orchestrator.run_search_report(
        component_a="NC(N)=O",
        n=1,
        checkpoint_path=str(ckpt),
        config_path=str(cfg),
    )
    assert any("prediction" in w.lower() or "failed" in w.lower() for w in outcome.llm_warnings)


def test_all_predictions_fail_returns_empty_results(monkeypatch, tmp_path):
    """If every candidate fails prediction, results is empty (no crash)."""
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_text("fake")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("device: cpu\nembedding:\n  method: morgan\n  morgan:\n    radius: 2\n    n_bits: 16\n    use_chirality: false\n")

    proposals = [
        CandidateProposal(smiles="O", rationale="water", family="alcohol", source="heuristic", source_id=""),
        CandidateProposal(smiles="CC(=O)O", rationale="acetic acid", family="acid", source="heuristic", source_id=""),
    ]
    _patch_orchestrator_basics(monkeypatch, proposals)
    monkeypatch.setattr(orchestrator, "predict_curve", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("crash")))

    outcome = orchestrator.run_search_report(
        component_a="NC(N)=O",
        n=2,
        checkpoint_path=str(ckpt),
        config_path=str(cfg),
    )
    assert outcome.results == []
    assert len(outcome.llm_warnings) >= 2


# ---------------------------------------------------------------------------
# D4 — Batch screening from file
# ---------------------------------------------------------------------------

def test_candidates_file_bypasses_generate_candidates(monkeypatch, tmp_path):
    """When --candidates-file is given, generate_candidates is NOT called."""
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_text("fake")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("device: cpu\nembedding:\n  method: morgan\n  morgan:\n    radius: 2\n    n_bits: 16\n    use_chirality: false\n")

    candidates_file = tmp_path / "candidates.txt"
    candidates_file.write_text("O\nCC(=O)O\n")

    generate_called = [False]

    def _fake_generate(*a, **kw):
        generate_called[0] = True
        return []

    _patch_orchestrator_basics(monkeypatch, [])
    monkeypatch.setattr(orchestrator, "generate_candidates", _fake_generate)

    outcome = orchestrator.run_search_report(
        component_a="NC(N)=O",
        n=5,
        checkpoint_path=str(ckpt),
        config_path=str(cfg),
        candidates_file=str(candidates_file),
    )
    assert not generate_called[0], "generate_candidates should be skipped when candidates_file is provided"


def test_candidates_file_screens_listed_smiles(monkeypatch, tmp_path):
    """Candidates from the file appear in results."""
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_text("fake")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("device: cpu\nembedding:\n  method: morgan\n  morgan:\n    radius: 2\n    n_bits: 16\n    use_chirality: false\n")

    candidates_file = tmp_path / "candidates.txt"
    candidates_file.write_text("O\nCC(=O)O\n")

    _patch_orchestrator_basics(monkeypatch, [])  # generate returns nothing

    outcome = orchestrator.run_search_report(
        component_a="NC(N)=O",
        n=5,
        checkpoint_path=str(ckpt),
        config_path=str(cfg),
        candidates_file=str(candidates_file),
    )
    screened = {r.curve.smiles_b for r in outcome.results}
    assert "O" in screened or "CC(=O)O" in screened


def test_candidates_file_skips_blank_lines(monkeypatch, tmp_path):
    """Blank lines and comment lines in the candidates file are ignored."""
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_text("fake")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("device: cpu\nembedding:\n  method: morgan\n  morgan:\n    radius: 2\n    n_bits: 16\n    use_chirality: false\n")

    candidates_file = tmp_path / "candidates.txt"
    candidates_file.write_text("\n# a comment\nO\n\n  \nCC(=O)O\n")

    _patch_orchestrator_basics(monkeypatch, [])

    outcome = orchestrator.run_search_report(
        component_a="NC(N)=O",
        n=5,
        checkpoint_path=str(ckpt),
        config_path=str(cfg),
        candidates_file=str(candidates_file),
    )
    assert len(outcome.results) == 2


def test_candidates_file_missing_raises_file_not_found(monkeypatch, tmp_path):
    """A non-existent candidates file raises FileNotFoundError."""
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_text("fake")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("device: cpu\nembedding:\n  method: morgan\n  morgan:\n    radius: 2\n    n_bits: 16\n    use_chirality: false\n")

    with pytest.raises(FileNotFoundError):
        orchestrator.run_search_report(
            component_a="NC(N)=O",
            n=5,
            checkpoint_path=str(ckpt),
            config_path=str(cfg),
            candidates_file=str(tmp_path / "nonexistent.txt"),
        )


def test_cli_candidates_file_flag_parsed():
    """--candidates-file argument is present in the CLI parser."""
    from des_multi_agent.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["--workflow", "des", "--component-a", "CCO", "--candidates-file", "smiles.txt"])
    assert args.candidates_file == "smiles.txt"


def test_cli_candidates_file_defaults_to_none():
    from des_multi_agent.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["--workflow", "des", "--component-a", "CCO"])
    assert args.candidates_file is None


# ---------------------------------------------------------------------------
# E1 — Threshold presets
# ---------------------------------------------------------------------------

def test_cli_preset_strict_parsed():
    from des_multi_agent.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["--workflow", "des", "--component-a", "CCO", "--preset", "strict"])
    assert args.preset == "strict"


def test_cli_preset_relaxed_parsed():
    from des_multi_agent.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["--workflow", "des", "--component-a", "CCO", "--preset", "relaxed"])
    assert args.preset == "relaxed"


def test_cli_preset_standard_parsed():
    from des_multi_agent.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["--workflow", "des", "--component-a", "CCO", "--preset", "standard"])
    assert args.preset == "standard"


def test_cli_preset_defaults_to_none():
    from des_multi_agent.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["--workflow", "des", "--component-a", "CCO"])
    assert args.preset is None


def test_cli_preset_invalid_rejected():
    from des_multi_agent.cli import build_parser
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--workflow", "des", "--component-a", "CCO", "--preset", "ultra"])


def test_preset_strict_tighter_than_standard():
    """Strict preset has lower absolute_tm_max_k and higher relative_drop_min than standard."""
    from des_multi_agent.cli import THRESHOLD_PRESETS
    strict = THRESHOLD_PRESETS["strict"]
    standard = THRESHOLD_PRESETS["standard"]
    assert strict.absolute_tm_max_k < standard.absolute_tm_max_k
    assert strict.relative_drop_min > standard.relative_drop_min


def test_preset_relaxed_looser_than_standard():
    """Relaxed preset has higher absolute_tm_max_k and lower relative_drop_min than standard."""
    from des_multi_agent.cli import THRESHOLD_PRESETS
    relaxed = THRESHOLD_PRESETS["relaxed"]
    standard = THRESHOLD_PRESETS["standard"]
    assert relaxed.absolute_tm_max_k > standard.absolute_tm_max_k
    assert relaxed.relative_drop_min < standard.relative_drop_min


def test_preset_standard_matches_defaults():
    """Standard preset matches the existing DEFAULT constants."""
    from des_multi_agent.cli import THRESHOLD_PRESETS
    from des_multi_agent.config import DEFAULT_ABSOLUTE_TM_MAX_K, DEFAULT_RELATIVE_DROP_MIN
    standard = THRESHOLD_PRESETS["standard"]
    assert standard.absolute_tm_max_k == DEFAULT_ABSOLUTE_TM_MAX_K
    assert standard.relative_drop_min == DEFAULT_RELATIVE_DROP_MIN
