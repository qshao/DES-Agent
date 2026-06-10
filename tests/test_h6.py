from __future__ import annotations
import pytest


# ── H6: CandidateFamily schema ───────────────────────────────────────────────

def test_candidate_family_fields():
    from des_multi_agent.llm.schemas import CandidateFamily
    f = CandidateFamily(name="polyols", rationale="Multiple OH groups donate H-bonds.", hbd_hba_role="HBD")
    assert f.name == "polyols"
    assert f.hbd_hba_role == "HBD"


def test_parse_candidate_families_basic():
    from des_multi_agent.llm.parser import parse_candidate_families
    raw = '[{"name": "polyols", "rationale": "Strong HBD.", "hbd_hba_role": "HBD"}]'
    families = parse_candidate_families(raw)
    assert len(families) == 1
    assert families[0].name == "polyols"


def test_parse_candidate_families_skips_incomplete():
    from des_multi_agent.llm.parser import parse_candidate_families
    raw = '[{"name": "polyols"}, {"name": "amides", "rationale": "ok", "hbd_hba_role": "HBA"}]'
    families = parse_candidate_families(raw)
    assert len(families) == 1
    assert families[0].name == "amides"


def test_parse_candidate_families_empty():
    from des_multi_agent.llm.parser import parse_candidate_families
    assert parse_candidate_families("[]") == []


def test_parse_candidate_families_wrapped():
    from des_multi_agent.llm.parser import parse_candidate_families
    raw = '{"items": [{"name": "amines", "rationale": "HBA.", "hbd_hba_role": "HBA"}]}'
    families = parse_candidate_families(raw)
    assert len(families) == 1


def test_family_selection_prompt_contains_component_a():
    from des_multi_agent.llm.prompts import family_selection_prompt
    text = family_selection_prompt("CC(=O)O", None, "ctx")
    assert "CC(=O)O" in text
    assert "JSON" in text
    assert "hbd_hba_role" in text


def test_candidate_brainstorm_prompt_includes_families():
    from des_multi_agent.llm.schemas import CandidateFamily
    from des_multi_agent.llm.prompts import candidate_brainstorm_prompt
    families = [CandidateFamily(name="polyols", rationale="HBD.", hbd_hba_role="HBD")]
    text = candidate_brainstorm_prompt("CC(=O)O", None, "ctx", families=families)
    assert "polyols" in text


def test_select_candidate_families_is_abstract():
    from des_multi_agent.llm.provider import LLMProvider
    assert "select_candidate_families" in LLMProvider.__abstractmethods__


# ── H6: two-stage brainstorm ─────────────────────────────────────────────────

def test_brainstorm_calls_family_selection_first(monkeypatch):
    """BaseLLMProvider.brainstorm_candidates calls select_candidate_families before the brainstorm."""
    from des_multi_agent.llm.base import BaseLLMProvider
    from des_multi_agent.llm.schemas import CandidateFamily

    call_order = []

    class _StubProvider(BaseLLMProvider):
        request_profile = None
        def extract_text(self, raw): return raw

    provider = _StubProvider.__new__(_StubProvider)

    def _fake_families(component_a, constraints, context):
        call_order.append("families")
        return [CandidateFamily(name="polyols", rationale="HBD.", hbd_hba_role="HBD")]

    def _fake_request(prompt):
        call_order.append("brainstorm")
        assert "polyols" in prompt
        return '[{"smiles": "OCCO", "rationale": "diol", "family": "polyols"}]'

    monkeypatch.setattr(provider, "select_candidate_families", _fake_families)
    monkeypatch.setattr(provider, "_request", _fake_request)
    object.__setattr__(provider, "max_candidates", 10)

    results = provider.brainstorm_candidates("CC(=O)O", None, "ctx")
    assert call_order == ["families", "brainstorm"]
    assert results[0].smiles == "OCCO"


def test_brainstorm_falls_back_when_family_selection_raises(monkeypatch):
    """If select_candidate_families raises, brainstorm_candidates continues without families."""
    from des_multi_agent.llm.base import BaseLLMProvider

    class _StubProvider(BaseLLMProvider):
        request_profile = None
        def extract_text(self, raw): return raw

    provider = _StubProvider.__new__(_StubProvider)

    def _failing_families(*a, **kw):
        raise RuntimeError("LLM timeout")

    def _fake_request(prompt):
        return '[{"smiles": "OCCO", "rationale": "ok", "family": "polyols"}]'

    monkeypatch.setattr(provider, "select_candidate_families", _failing_families)
    monkeypatch.setattr(provider, "_request", _fake_request)
    object.__setattr__(provider, "max_candidates", 10)

    results = provider.brainstorm_candidates("CC(=O)O", None, "ctx")
    assert len(results) == 1


def test_select_candidate_families_implemented_in_base(monkeypatch):
    """BaseLLMProvider.select_candidate_families calls the LLM and parses families."""
    from des_multi_agent.llm.base import BaseLLMProvider

    class _StubProvider(BaseLLMProvider):
        request_profile = None
        def extract_text(self, raw): return raw

    provider = _StubProvider.__new__(_StubProvider)
    monkeypatch.setattr(
        provider, "_request",
        lambda prompt: '[{"name": "amides", "rationale": "HBA.", "hbd_hba_role": "HBA"}]'
    )
    families = provider.select_candidate_families("CC(=O)O", None, "ctx")
    assert len(families) == 1
    assert families[0].name == "amides"


# ── H6: family ledger in CycleDelta ─────────────────────────────────────────

def test_cycle_delta_has_family_ledger():
    from des_multi_agent.multi_cycle import CycleDelta
    delta = CycleDelta(
        cycle=1, n_screened=5, n_des=2,
        top_smiles=frozenset(["CCO"]),
        new_entrants=["CCO"], dropouts=[], converged=False,
        family_ledger={"polyols": 2},
    )
    assert delta.family_ledger["polyols"] == 2


def test_multi_cycle_builds_family_ledger(monkeypatch, tmp_path):
    """run_multi_cycle_search builds a family_ledger from brainstorm_candidates + DES results."""
    from dataclasses import dataclass
    from des_multi_agent.evaluation import DesResult
    from des_multi_agent.llm.schemas import CandidateBrainstorm

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

    des_result = DesResult(
        curve=_Curve(smiles_b="OCCO"), absolute_pass=True,
        relative_pass=True, is_des=True, rationale="t", min_tm_k=230.0,
    )
    brainstorm = CandidateBrainstorm(smiles="OCCO", rationale="diol", family="polyols")

    from unittest.mock import MagicMock
    outcome = MagicMock()
    outcome.results = [des_result]
    outcome.brainstorm_candidates = [brainstorm]

    monkeypatch.setattr("des_multi_agent.multi_cycle.run_search_report", lambda *a, **kw: outcome)

    from des_multi_agent.multi_cycle import run_multi_cycle_search
    ckpt = tmp_path / "ckpt.pt"; ckpt.write_bytes(b"")
    cfg = tmp_path / "config.yaml"; cfg.write_text("")

    result = run_multi_cycle_search("CCO", 2, str(ckpt), str(cfg), n_cycles=1)
    assert result.cycle_deltas[0].family_ledger.get("polyols", 0) >= 1


# ── H6: enriched iterative context ──────────────────────────────────────────

def test_build_iterative_context_includes_family_ledger(monkeypatch):
    from des_multi_agent.orchestrator import _build_iterative_context
    from dataclasses import dataclass
    from des_multi_agent.evaluation import DesResult

    @dataclass(frozen=True)
    class _Curve:
        smiles_b: str = "OCCO"
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

    result = DesResult(
        curve=_Curve(), absolute_pass=True, relative_pass=True,
        is_des=True, rationale="t", min_tm_k=230.0,
    )
    ledger = {"polyols": 3, "amides": 1}
    ctx = _build_iterative_context("base", [result], family_ledger=ledger)
    assert "polyols" in ctx
    assert "3" in ctx
