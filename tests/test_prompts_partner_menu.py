"""Tests for known-partner anchor block in brainstorm prompt."""
from __future__ import annotations

from des_multi_agent.chemistry.partner_registry import MenuEntry
from des_multi_agent.llm.prompts import candidate_brainstorm_prompt


def test_prompt_without_menu_has_no_anchor_block():
    p = candidate_brainstorm_prompt("CCO", None, "ctx")
    assert "known, real molecules" not in p


def test_prompt_with_menu_renders_anchor_block():
    menu = [
        MenuEntry("NC(N)=O", "urea", "HBD"),
        MenuEntry("OCC(O)CO", "glycerol", "amphoteric"),
    ]
    p = candidate_brainstorm_prompt("CCO", None, "ctx", known_partner_menu=menu)
    assert "known, real molecules" in p
    assert "urea [HBD]" in p
    assert "glycerol [amphoteric]" in p


def test_prompt_with_empty_menu_has_no_anchor_block():
    p = candidate_brainstorm_prompt("CCO", None, "ctx", known_partner_menu=[])
    assert "known, real molecules" not in p
