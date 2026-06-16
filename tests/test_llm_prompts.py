"""Tests for facts_block injection in LLM prompt builders (Phase 3)."""
from __future__ import annotations

import pytest


def test_candidate_review_prompt_with_facts_block():
    from des_multi_agent.llm.prompts import candidate_review_prompt
    result = candidate_review_prompt("CCO", "NC(N)=O", "ctx", facts_block="HBD=2, HBA=2")
    assert "Computed facts:" in result
    assert "HBD=2" in result


def test_candidate_review_prompt_no_facts_block_unchanged():
    from des_multi_agent.llm.prompts import candidate_review_prompt
    without = candidate_review_prompt("CCO", "NC(N)=O", "ctx")
    with_empty = candidate_review_prompt("CCO", "NC(N)=O", "ctx", facts_block="")
    assert without == with_empty
    assert "Computed facts:" not in without


def test_candidate_review_prompt_facts_block_before_candidate_line():
    from des_multi_agent.llm.prompts import candidate_review_prompt
    result = candidate_review_prompt("CCO", "NC(N)=O", "ctx", facts_block="HBD=2")
    facts_pos = result.index("Computed facts:")
    candidate_pos = result.index("Candidate:")
    assert facts_pos < candidate_pos


def test_candidate_brainstorm_prompt_with_facts_block():
    from des_multi_agent.llm.prompts import candidate_brainstorm_prompt
    result = candidate_brainstorm_prompt("CCO", None, "ctx", facts_block="HBD=1")
    assert "Computed facts:" in result


def test_candidate_brainstorm_prompt_no_facts_block_unchanged():
    from des_multi_agent.llm.prompts import candidate_brainstorm_prompt
    without = candidate_brainstorm_prompt("CCO", None, "ctx")
    with_empty = candidate_brainstorm_prompt("CCO", None, "ctx", facts_block="")
    assert without == with_empty
    assert "Computed facts:" not in without


def test_candidate_brainstorm_prompt_facts_block_before_component_a():
    from des_multi_agent.llm.prompts import candidate_brainstorm_prompt
    result = candidate_brainstorm_prompt("CCO", None, "ctx", facts_block="HBD=1")
    facts_pos = result.index("Computed facts:")
    comp_pos = result.index("Component A:")
    assert facts_pos < comp_pos


def test_chemistry_assessment_prompt_with_facts_block():
    from des_multi_agent.llm.prompts import chemistry_assessment_prompt
    result = chemistry_assessment_prompt("NC(N)=O", "ctx", facts_block="HBD=2")
    assert "Computed facts:" in result


def test_chemistry_assessment_prompt_no_facts_block_unchanged():
    from des_multi_agent.llm.prompts import chemistry_assessment_prompt
    without = chemistry_assessment_prompt("NC(N)=O", "ctx")
    with_empty = chemistry_assessment_prompt("NC(N)=O", "ctx", facts_block="")
    assert without == with_empty
    assert "Computed facts:" not in without


def test_chemistry_assessment_prompt_facts_block_before_candidate_line():
    from des_multi_agent.llm.prompts import chemistry_assessment_prompt
    result = chemistry_assessment_prompt("NC(N)=O", "ctx", facts_block="HBD=2")
    facts_pos = result.index("Computed facts:")
    candidate_pos = result.index("Candidate:")
    assert facts_pos < candidate_pos


def test_contradiction_prompt_with_facts_block():
    from des_multi_agent.llm.prompts import contradiction_prompt
    result = contradiction_prompt([], "ctx", facts_block="HBD=1")
    assert "Computed facts:" in result


def test_contradiction_prompt_no_facts_block_unchanged():
    from des_multi_agent.llm.prompts import contradiction_prompt
    without = contradiction_prompt([], "ctx")
    with_empty = contradiction_prompt([], "ctx", facts_block="")
    assert without == with_empty
    assert "Computed facts:" not in without


def test_ligand_brainstorm_prompt_with_facts_block():
    from des_multi_agent.llm.prompts import ligand_brainstorm_prompt
    result = ligand_brainstorm_prompt("Cu2+", None, "ctx", facts_block="HBD=0")
    assert "Computed facts:" in result


def test_ligand_brainstorm_prompt_no_facts_block_unchanged():
    from des_multi_agent.llm.prompts import ligand_brainstorm_prompt
    without = ligand_brainstorm_prompt("Cu2+", None, "ctx")
    with_empty = ligand_brainstorm_prompt("Cu2+", None, "ctx", facts_block="")
    assert without == with_empty
    assert "Computed facts:" not in without


def test_ligand_selectivity_brainstorm_prompt_with_facts_block():
    from des_multi_agent.llm.prompts import ligand_selectivity_brainstorm_prompt
    result = ligand_selectivity_brainstorm_prompt("Cu2+", "Zn2+", None, "ctx", facts_block="denticity=2")
    assert "Computed facts:" in result


def test_ligand_selectivity_brainstorm_prompt_no_facts_block_unchanged():
    from des_multi_agent.llm.prompts import ligand_selectivity_brainstorm_prompt
    without = ligand_selectivity_brainstorm_prompt("Cu2+", "Zn2+", None, "ctx")
    with_empty = ligand_selectivity_brainstorm_prompt("Cu2+", "Zn2+", None, "ctx", facts_block="")
    assert without == with_empty
    assert "Computed facts:" not in without
