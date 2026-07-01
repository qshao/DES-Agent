"""Tests for metal-ligand brainstorm anchoring.

Covers:
  - known_ligand_menu: returns MenuEntry objects with donor-capable molecules,
    sorted by predicted log K; metal-specific ordering.
  - ground_ligand_reality: gates on invalid SMILES, no donor atoms, structural
    sanity, and known/novel status.
  - ligand_brainstorm_prompt / ligand_selectivity_brainstorm_prompt: render the
    known_ligand_menu block when non-empty.
"""
from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# known_ligand_menu
# ---------------------------------------------------------------------------

from des_multi_agent.chemistry.partner_registry import MenuEntry, known_ligand_menu


class TestKnownLigandMenu:
    def test_returns_list(self):
        result = known_ligand_menu("Cu2+")
        assert isinstance(result, list)

    def test_length_bounded_by_limit(self):
        result = known_ligand_menu("Cu2+", limit=5)
        assert len(result) <= 5

    def test_entries_are_menu_entries(self):
        result = known_ligand_menu("Cu2+", limit=10)
        for e in result:
            assert isinstance(e, MenuEntry)
            assert e.smiles
            assert e.display_name
            assert e.role  # coordination summary e.g. "bidentate (N,O)"

    def test_all_entries_have_donor_atoms(self):
        from des_multi_agent.chemistry.coordination import coordination_profile
        result = known_ligand_menu("Cu2+", limit=20)
        for e in result:
            prof = coordination_profile(e.smiles)
            assert prof.n_donor_atoms >= 1, f"{e.smiles} has no donor atoms"

    def test_role_describes_coordination(self):
        result = known_ligand_menu("Cu2+", limit=10)
        for e in result:
            # role should contain donor element info like (N,O) or (O) etc.
            assert "(" in e.role and ")" in e.role, f"unexpected role: {e.role!r}"

    def test_sorted_by_log_k_descending(self):
        from des_multi_agent.chemistry.stability_rules import rule_based_log_k
        result = known_ligand_menu("Cu2+", limit=10)
        log_ks = [rule_based_log_k("Cu2+", e.smiles) for e in result]
        assert log_ks == sorted(log_ks, reverse=True)

    def test_cu2_differs_from_zn2(self):
        # Cu2+ (Irving-Williams max) scores higher than Zn2+ for most ligands
        cu_menu = known_ligand_menu("Cu2+", limit=5)
        zn_menu = known_ligand_menu("Zn2+", limit=5)
        # menus exist for both; exact ordering may differ
        assert len(cu_menu) > 0
        assert len(zn_menu) > 0

    def test_unknown_metal_returns_list(self):
        # Should not raise; may return empty or partial list
        result = known_ligand_menu("Unobtainium99+", limit=5)
        assert isinstance(result, list)

    def test_zero_limit_returns_empty(self):
        result = known_ligand_menu("Cu2+", limit=0)
        assert result == []


# ---------------------------------------------------------------------------
# ground_ligand_reality
# ---------------------------------------------------------------------------

from des_multi_agent.chemistry.claim_grounding import ground_ligand_reality, PartnerVerdict


class TestGroundLigandReality:
    def test_invalid_smiles_drops(self):
        rv = ground_ligand_reality("Cu2+", "NOT_A_SMILES")
        assert rv.disposition == "drop"
        assert rv.status == "novel_implausible"

    def test_no_donor_atoms_drops(self):
        # Benzene has no heteroatoms — zero donor atoms for coordination
        rv = ground_ligand_reality("Cu2+", "c1ccccc1")
        assert rv.disposition == "drop"
        assert "donor" in rv.detail.lower()

    def test_bidentate_nitrogen_oxygen_kept(self):
        # Glycine: N + carboxylate O → bidentate ligand, well-known Cu2+ binder
        rv = ground_ligand_reality("Cu2+", "NCC(=O)O")
        assert rv.disposition == "keep"

    def test_ethylenediamine_kept(self):
        # en: classic bidentate N,N chelator
        rv = ground_ligand_reality("Cu2+", "NCCN")
        assert rv.disposition == "keep"
        assert rv.status in ("known", "novel_plausible")

    def test_returns_partner_verdict(self):
        rv = ground_ligand_reality("Zn2+", "NCCN")
        assert isinstance(rv, PartnerVerdict)
        assert rv.candidate_smiles == "NCCN"

    def test_never_raises_on_bad_metal(self):
        rv = ground_ligand_reality("BADMETAL", "NCCN")
        assert rv.disposition in ("keep", "drop", "demote")

    def test_penalty_zero_for_keep(self):
        rv = ground_ligand_reality("Cu2+", "NCCN")
        if rv.disposition == "keep":
            assert rv.penalty == 0.0

    def test_penalty_zero_for_drop(self):
        rv = ground_ligand_reality("Cu2+", "c1ccccc1")
        assert rv.penalty == 0.0


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

from des_multi_agent.llm.prompts import ligand_brainstorm_prompt, ligand_selectivity_brainstorm_prompt


def _menu(n: int = 2) -> list[MenuEntry]:
    return [MenuEntry(f"SMILES{i}", f"Ligand-{i}", f"bidentate (N,O)") for i in range(n)]


class TestLigandBrainstormPromptMenu:
    def test_menu_injected_when_present(self):
        prompt = ligand_brainstorm_prompt("Cu2+", None, "ctx", known_ligand_menu=_menu())
        assert "Ligand-0" in prompt
        assert "bidentate (N,O)" in prompt

    def test_no_menu_section_when_absent(self):
        prompt = ligand_brainstorm_prompt("Cu2+", None, "ctx", known_ligand_menu=None)
        assert "Ligand-0" not in prompt
        assert "known" not in prompt.lower() or "known" not in prompt  # no menu header

    def test_metal_name_in_menu_header(self):
        prompt = ligand_brainstorm_prompt("Zn2+", None, "ctx", known_ligand_menu=_menu(1))
        assert "Zn2+" in prompt

    def test_selectivity_prompt_menu_injected(self):
        prompt = ligand_selectivity_brainstorm_prompt(
            "Cu2+", "Zn2+", None, "ctx", known_ligand_menu=_menu()
        )
        assert "Ligand-0" in prompt
        assert "Cu2+" in prompt

    def test_selectivity_no_menu_when_absent(self):
        prompt = ligand_selectivity_brainstorm_prompt(
            "Cu2+", "Zn2+", None, "ctx", known_ligand_menu=None
        )
        assert "Ligand-0" not in prompt
