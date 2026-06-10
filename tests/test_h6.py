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
    import inspect
    from des_multi_agent.llm.provider import LLMProvider
    assert "select_candidate_families" in {
        name for name, _ in inspect.getmembers(LLMProvider, predicate=inspect.isfunction)
    }
