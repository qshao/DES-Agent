from __future__ import annotations
import pytest


# ── H3: ContradictionNote schema ─────────────────────────────────────────────

def test_contradiction_note_fields():
    from des_multi_agent.llm.schemas import ContradictionNote
    note = ContradictionNote(smiles="CCO", agreement="conflict", explanation="High polarity mismatch.")
    assert note.smiles == "CCO"
    assert note.agreement == "conflict"
    assert note.explanation == "High polarity mismatch."


def test_parse_contradiction_notes_agree():
    from des_multi_agent.llm.parser import parse_contradiction_notes
    raw = '[{"smiles": "CCO", "agreement": "agree", "explanation": "Fits HBD/HBA profile."}]'
    notes = parse_contradiction_notes(raw)
    assert len(notes) == 1
    assert notes[0].smiles == "CCO"
    assert notes[0].agreement == "agree"


def test_parse_contradiction_notes_conflict():
    from des_multi_agent.llm.parser import parse_contradiction_notes
    raw = '[{"smiles": "c1ccccc1", "agreement": "conflict", "explanation": "Aromatic ring unlikely to form HBD."}]'
    notes = parse_contradiction_notes(raw)
    assert notes[0].agreement == "conflict"


def test_parse_contradiction_notes_skips_invalid():
    """Items missing smiles or agreement are silently skipped."""
    from des_multi_agent.llm.parser import parse_contradiction_notes
    raw = '[{"smiles": "CCO"}, {"agreement": "agree"}, {"smiles": "OCC", "agreement": "uncertain", "explanation": "OK"}]'
    notes = parse_contradiction_notes(raw)
    assert len(notes) == 1
    assert notes[0].smiles == "OCC"


def test_parse_contradiction_notes_empty_list():
    from des_multi_agent.llm.parser import parse_contradiction_notes
    assert parse_contradiction_notes("[]") == []


def test_parse_contradiction_notes_wrapped():
    """Handles LLM wrapping the array under a key."""
    from des_multi_agent.llm.parser import parse_contradiction_notes
    raw = '{"items": [{"smiles": "CC(=O)O", "agreement": "agree", "explanation": "Acid fits."}]}'
    notes = parse_contradiction_notes(raw)
    assert len(notes) == 1


def test_contradiction_prompt_contains_smiles(fake_des_result):
    from des_multi_agent.llm.prompts import contradiction_prompt
    text = contradiction_prompt([fake_des_result], "context")
    assert fake_des_result.curve.smiles_b in text


def test_contradiction_prompt_instructs_json(fake_des_result):
    from des_multi_agent.llm.prompts import contradiction_prompt
    text = contradiction_prompt([fake_des_result], "context")
    assert "JSON" in text
    assert "agreement" in text


# ── H3: provider detect_contradictions ───────────────────────────────────────

def test_detect_contradictions_returns_notes(monkeypatch, fake_des_result):
    """BaseLLMProvider.detect_contradictions calls the LLM and parses results."""
    from des_multi_agent.llm.base import BaseLLMProvider

    class _StubProvider(BaseLLMProvider):
        request_profile = None
        def extract_text(self, raw):
            return raw

    provider = _StubProvider.__new__(_StubProvider)
    monkeypatch.setattr(
        provider, "_request",
        lambda prompt: '[{"smiles": "CCO", "agreement": "agree", "explanation": "Good HBD donor."}]'
    )
    notes = provider.detect_contradictions([fake_des_result], "ctx")
    assert len(notes) == 1
    assert notes[0].agreement == "agree"


def test_detect_contradictions_provider_is_abstract():
    """LLMProvider declares detect_contradictions as abstract."""
    import inspect
    from des_multi_agent.llm.provider import LLMProvider
    assert "detect_contradictions" in {
        name for name, _ in inspect.getmembers(LLMProvider, predicate=inspect.isfunction)
    }


# ── H2: viscosity-aware composite ranking ────────────────────────────────────

def _make_result(smiles_b, min_tm_k, is_des=True, t1=330.0, t2=289.0):
    from dataclasses import dataclass
    from des_multi_agent.evaluation import DesResult

    @dataclass(frozen=True)
    class _Curve:
        smiles_a: str = "Cc1ccc(O)cc1"
        smiles_b: str = ""
        tm_pred_k: list = None
        ratios: list = None
        t1_k: float = 330.0
        t2_k: float = 289.0
        def __post_init__(self):
            if self.tm_pred_k is None:
                object.__setattr__(self, "tm_pred_k", [min_tm_k + 10, min_tm_k, min_tm_k + 5])
            if self.ratios is None:
                object.__setattr__(self, "ratios", [0.1, 0.5, 0.9])

    return DesResult(
        curve=_Curve(smiles_b=smiles_b, t1_k=t1, t2_k=t2),
        absolute_pass=is_des,
        relative_pass=is_des,
        is_des=is_des,
        rationale="test",
        min_tm_k=min_tm_k,
    )


def test_composite_ranking_prefers_low_viscosity():
    """A DES-former with lower viscosity ranks above one with higher viscosity
    even when both have the same Tm."""
    from des_multi_agent.ranking import rank_results_composite
    high_visc = _make_result("CCO", 240.0)
    low_visc = _make_result("OCCO", 240.0)
    visc = {"CCO": 500.0, "OCCO": 50.0}
    ranked = rank_results_composite([high_visc, low_visc], visc, viscosity_weight=0.5)
    assert ranked[0].curve.smiles_b == "OCCO"


def test_composite_ranking_threshold_moves_high_visc_below_passing():
    """With a viscosity threshold, high-viscosity DES-formers appear after
    low-viscosity ones regardless of Tm."""
    from des_multi_agent.ranking import rank_results_composite
    good = _make_result("CCO", 230.0)
    sticky = _make_result("OCCO", 235.0)
    visc = {"CCO": 80.0, "OCCO": 600.0}
    ranked = rank_results_composite([good, sticky], visc, viscosity_threshold_cp=500.0)
    assert ranked[0].curve.smiles_b == "CCO"


def test_composite_ranking_no_visc_falls_back_to_min_tm():
    """When no viscosity data is available, sort by min_tm_k ascending."""
    from des_multi_agent.ranking import rank_results_composite
    r1 = _make_result("CCO", 250.0)
    r2 = _make_result("OCCO", 230.0)
    ranked = rank_results_composite([r1, r2], {})
    assert ranked[0].curve.smiles_b == "OCCO"


def test_composite_ranking_non_des_always_last():
    """Non-DES candidates always sort after DES-formers."""
    from des_multi_agent.ranking import rank_results_composite
    des = _make_result("CCO", 240.0, is_des=True)
    non_des = _make_result("OCCO", 200.0, is_des=False)
    ranked = rank_results_composite([non_des, des], {})
    assert ranked[0].is_des is True


# ── shared fixtures ───────────────────────────────────────────────────────────

# ── H3 + H4: orchestrator wiring ────────────────────────────────────────────

def test_search_outcome_has_contradiction_notes():
    """SearchOutcome exposes contradiction_notes field."""
    from des_multi_agent.orchestrator import SearchOutcome
    outcome = SearchOutcome(
        results=[], annotated_results=[], candidate_proposals=[],
        candidate_reviews=[], brainstorm_candidates=[],
        explanation_notes=[], critique_notes=[], llm_warnings=[],
        contradiction_notes=[],
    )
    assert outcome.contradiction_notes == []


def test_run_search_report_accepts_prior_cycle_top_results(monkeypatch, tmp_path):
    """run_search_report accepts prior_cycle_top_results without error."""
    from des_multi_agent.orchestrator import run_search_report

    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_bytes(b"")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("device: cpu\nembedding:\n  method: morgan\n  morgan:\n    radius: 2\n    n_bits: 2048\n    use_chirality: false\n")

    monkeypatch.setattr("des_multi_agent.orchestrator.generate_candidates", lambda *a, **kw: [])
    monkeypatch.setattr("des_multi_agent.orchestrator.filter_candidates", lambda *a, **kw: [])
    monkeypatch.setattr("des_multi_agent.orchestrator.rank_results", lambda r: r)
    monkeypatch.setattr("des_multi_agent.orchestrator.rank_results_composite", lambda *a, **kw: [])

    outcome = run_search_report(
        "CCO", 1, str(ckpt), str(cfg),
        prior_cycle_top_results=[],
    )
    assert outcome.contradiction_notes == []


def test_contradiction_notes_appear_in_format_report(fake_des_result):
    """format_report includes a contradiction notes section when notes are present."""
    from des_multi_agent.llm.schemas import ContradictionNote
    from des_multi_agent.reporting import format_report

    notes = [ContradictionNote(smiles="CCO", agreement="conflict", explanation="Polarity mismatch.")]
    text = format_report([fake_des_result], contradiction_notes=notes)
    assert "conflict" in text
    assert "Polarity mismatch" in text


def test_iterative_context_includes_prior_results(fake_des_result):
    """When prior_cycle_top_results is set, the brainstorm context mentions prior top hits."""
    from des_multi_agent.orchestrator import _build_iterative_context
    ctx = _build_iterative_context("base context", [fake_des_result])
    assert "Prior cycle" in ctx
    assert fake_des_result.curve.smiles_b in ctx


# ── H1 + H5: multi-cycle runner ──────────────────────────────────────────────

def test_multi_cycle_outcome_fields():
    """MultiCycleOutcome and CycleDelta are importable dataclasses."""
    from des_multi_agent.multi_cycle import CycleDelta, MultiCycleOutcome
    delta = CycleDelta(
        cycle=1, n_screened=5, n_des=2,
        top_smiles=frozenset(["CCO"]),
        new_entrants=["CCO"], dropouts=[], converged=False,
    )
    assert delta.cycle == 1
    outcome = MultiCycleOutcome(
        final_outcome=None,
        cycle_deltas=[delta],
        total_cycles=1,
        converged=False,
    )
    assert outcome.total_cycles == 1


def test_run_multi_cycle_search_runs_n_cycles(monkeypatch, tmp_path):
    """run_multi_cycle_search calls run_search_report exactly n_cycles times."""
    call_count = {"n": 0}

    def _fake_search(*args, **kwargs):
        call_count["n"] += 1
        from unittest.mock import MagicMock
        outcome = MagicMock()
        outcome.results = []
        return outcome

    monkeypatch.setattr("des_multi_agent.multi_cycle.run_search_report", _fake_search)

    from des_multi_agent.multi_cycle import run_multi_cycle_search
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_bytes(b"")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("")

    result = run_multi_cycle_search("CCO", 2, str(ckpt), str(cfg), n_cycles=3)
    assert call_count["n"] == 3
    assert result.total_cycles == 3


def test_run_multi_cycle_search_stops_on_convergence(monkeypatch, tmp_path):
    """Convergence is detected when top-K set is unchanged across two cycles."""
    from dataclasses import dataclass
    from des_multi_agent.evaluation import DesResult

    @dataclass(frozen=True)
    class _Curve:
        smiles_b: str
        smiles_a: str = "Cc1ccc(O)cc1"
        tm_pred_k: list = None
        ratios: list = None
        t1_k: float = 330.0
        t2_k: float = 289.0
        def __post_init__(self):
            if self.tm_pred_k is None:
                object.__setattr__(self, "tm_pred_k", [240.0, 230.0])
            if self.ratios is None:
                object.__setattr__(self, "ratios", [0.1, 0.9])

    fixed_result = DesResult(
        curve=_Curve(smiles_b="CCO"), absolute_pass=True,
        relative_pass=True, is_des=True, rationale="t", min_tm_k=230.0,
    )

    def _stable_search(*args, **kwargs):
        from unittest.mock import MagicMock
        outcome = MagicMock()
        outcome.results = [fixed_result]
        return outcome

    monkeypatch.setattr("des_multi_agent.multi_cycle.run_search_report", _stable_search)

    from des_multi_agent.multi_cycle import run_multi_cycle_search
    ckpt = tmp_path / "ckpt.pt"; ckpt.write_bytes(b"")
    cfg = tmp_path / "config.yaml"; cfg.write_text("")

    result = run_multi_cycle_search("CCO", 2, str(ckpt), str(cfg), n_cycles=5, top_k_convergence=1)
    assert result.converged is True
    assert result.total_cycles == 2


def test_cycle_delta_tracks_new_entrants_and_dropouts(monkeypatch, tmp_path):
    """CycleDelta records which SMILES entered and left the top-K between cycles."""
    from dataclasses import dataclass
    from des_multi_agent.evaluation import DesResult

    @dataclass(frozen=True)
    class _Curve:
        smiles_b: str
        smiles_a: str = "Cc1ccc(O)cc1"
        tm_pred_k: list = None
        ratios: list = None
        t1_k: float = 330.0
        t2_k: float = 289.0
        def __post_init__(self):
            if self.tm_pred_k is None:
                object.__setattr__(self, "tm_pred_k", [230.0])
            if self.ratios is None:
                object.__setattr__(self, "ratios", [0.5])

    def _r(smiles):
        return DesResult(
            curve=_Curve(smiles_b=smiles), absolute_pass=True,
            relative_pass=True, is_des=True, rationale="t", min_tm_k=230.0,
        )

    cycle_results = [
        [_r("CCO"), _r("OCCO")],
        [_r("OCCO"), _r("CC(=O)O")],
    ]
    call_idx = {"n": 0}

    def _rotating_search(*args, **kwargs):
        from unittest.mock import MagicMock
        outcome = MagicMock()
        outcome.results = cycle_results[call_idx["n"]]
        call_idx["n"] += 1
        return outcome

    monkeypatch.setattr("des_multi_agent.multi_cycle.run_search_report", _rotating_search)

    from des_multi_agent.multi_cycle import run_multi_cycle_search
    ckpt = tmp_path / "ckpt.pt"; ckpt.write_bytes(b"")
    cfg = tmp_path / "config.yaml"; cfg.write_text("")

    result = run_multi_cycle_search("CCO", 2, str(ckpt), str(cfg), n_cycles=2, top_k_convergence=2)
    delta2 = result.cycle_deltas[1]
    assert "CC(=O)O" in delta2.new_entrants
    assert "CCO" in delta2.dropouts


# ── CLI: H1 + H2 flags ───────────────────────────────────────────────────────

def test_cli_n_cycles_parsed():
    from des_multi_agent.cli import build_parser
    args = build_parser().parse_args(["--n-cycles", "4"])
    assert args.n_cycles == 4


def test_cli_n_cycles_default_is_one():
    from des_multi_agent.cli import build_parser
    args = build_parser().parse_args([])
    assert args.n_cycles == 1


def test_cli_viscosity_threshold_parsed():
    from des_multi_agent.cli import build_parser
    args = build_parser().parse_args(["--viscosity-threshold", "300"])
    assert args.viscosity_threshold == 300.0


def test_cli_viscosity_threshold_default_is_none():
    from des_multi_agent.cli import build_parser
    args = build_parser().parse_args([])
    assert args.viscosity_threshold is None


def test_cli_viscosity_weight_parsed():
    from des_multi_agent.cli import build_parser
    args = build_parser().parse_args(["--viscosity-weight", "0.5"])
    assert args.viscosity_weight == pytest.approx(0.5)


def test_cli_viscosity_weight_default():
    from des_multi_agent.cli import build_parser
    args = build_parser().parse_args([])
    assert args.viscosity_weight == pytest.approx(0.3)


@pytest.fixture
def fake_des_result():
    from dataclasses import dataclass
    from des_multi_agent.evaluation import DesResult

    @dataclass(frozen=True)
    class _Curve:
        smiles_a: str = "Cc1ccc(O)cc1"
        smiles_b: str = "CCO"
        tm_pred_k: list = None
        ratios: list = None
        t1_k: float = 330.0
        t2_k: float = 289.0
        def __post_init__(self):
            if self.tm_pred_k is None:
                object.__setattr__(self, "tm_pred_k", [300.0, 240.0, 250.0])
            if self.ratios is None:
                object.__setattr__(self, "ratios", [0.1, 0.5, 0.9])

    return DesResult(
        curve=_Curve(),
        absolute_pass=True,
        relative_pass=True,
        is_des=True,
        rationale="test",
        min_tm_k=240.0,
    )
