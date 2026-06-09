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
