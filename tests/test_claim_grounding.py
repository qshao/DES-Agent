"""Deterministic unit tests for the chemistry grounding layer.

All assertions are grounded in known structural chemistry — no mocks, no LLM
calls, same result every run.

Molecule key:
  Glycine          NCC(=O)O             bidentate N,O ligand; denticity=2
  Glycerol         OCC(O)CO             HBD=3, HBA=3, amphoteric, polyol
  ChCl (HBA part)  C[N+](C)(C)CCO.[Cl-] amphoteric per RDKit (HBD=1,HBA=1)
  TBACl            CCCC[N+](CCCC)(CCCC)CCCC.[Cl-]  pure HBA (no HBD)
  Urea             NC(N)=O              amide family
  Acetic acid      CC(=O)O              carboxylic acid family
  Aniline          Nc1ccccc1            amine family (primary amine on ring)
  Phenol           Oc1ccccc1            phenol family
"""
from __future__ import annotations

import pytest

from des_multi_agent.chemistry.claim_grounding import (
    GroundingVerdict,
    StructuralFacts,
    ground_coordination,
    ground_des_plausibility,
    ground_family,
    ground_selectivity,
    structural_facts,
)

# ---------------------------------------------------------------------------
# SMILES constants
# ---------------------------------------------------------------------------
GLYCINE     = "NCC(=O)O"
GLYCEROL    = "OCC(O)CO"
CHCL        = "C[N+](C)(C)CCO.[Cl-]"
TBACL       = "CCCC[N+](CCCC)(CCCC)CCCC.[Cl-]"
UREA        = "NC(N)=O"
ACETIC_ACID = "CC(=O)O"
ANILINE     = "Nc1ccccc1"
PHENOL      = "Oc1ccccc1"
BAD_SMILES  = "not_a_smiles"

# ---------------------------------------------------------------------------
# Group 1 — structural_facts()
# ---------------------------------------------------------------------------

def test_structural_facts_glycine_denticity():
    facts = structural_facts(GLYCINE)
    assert facts.denticity == 2


def test_structural_facts_glycine_donor_atoms():
    facts = structural_facts(GLYCINE)
    # Glycine has 1 N + 2 O donor atoms (carboxylate has 2 O)
    assert facts.n_donor_atoms >= 2


def test_structural_facts_glycerol_hbd():
    facts = structural_facts(GLYCEROL)
    assert facts.n_hbd == 3


def test_structural_facts_glycerol_role():
    facts = structural_facts(GLYCEROL)
    assert facts.hbond_role == "amphoteric"


def test_structural_facts_invalid_smiles_safe():
    """structural_facts must never raise — it returns a safe sentinel object."""
    facts = structural_facts(BAD_SMILES)
    assert facts.smiles == "unparseable"


def test_structural_facts_as_prompt_block():
    facts = structural_facts(GLYCEROL)
    block = facts.as_prompt_block()
    assert isinstance(block, str)
    assert len(block) > 0
    # Block must mention HBD in some form
    assert "hbd" in block.lower()


# ---------------------------------------------------------------------------
# Group 2 — GroundingVerdict dataclass
# ---------------------------------------------------------------------------

def test_grounding_verdict_verified_has_zero_penalty():
    v = GroundingVerdict("claim", "verified", "detail", 0.0)
    assert v.penalty == 0.0


def test_grounding_verdict_contradicted_has_penalty():
    v = GroundingVerdict("claim", "contradicted", "detail", 0.25)
    assert v.penalty == 0.25


# ---------------------------------------------------------------------------
# Group 3 — ground_coordination()
# ---------------------------------------------------------------------------

def test_ground_coordination_verified():
    verdict = ground_coordination(GLYCINE, "bidentate N,O-chelator")
    assert verdict.status == "verified"


def test_ground_coordination_contradicted():
    # Hexadentate claim is far outside the ±1 tolerance for glycine (denticity=2)
    verdict = ground_coordination(GLYCINE, "hexadentate N,N,N,N,N,N-chelator")
    assert verdict.status == "contradicted"


def test_ground_coordination_unverifiable():
    # "good binding" carries no denticity or donor-element information
    verdict = ground_coordination(GLYCINE, "good binding")
    assert verdict.status == "unverifiable"


# ---------------------------------------------------------------------------
# Group 4 — ground_selectivity()
# ---------------------------------------------------------------------------

def test_ground_selectivity_target_selective():
    """Cu(II) > Zn(II) for glycine per Irving-Williams — should be verified."""
    verdict = ground_selectivity("Cu", "Zn", GLYCINE, "target_selective")
    assert verdict.status == "verified"


def test_ground_selectivity_contradicted():
    """Claiming Zn > Cu with glycine contradicts the Irving-Williams order."""
    verdict = ground_selectivity("Cu", "Zn", GLYCINE, "competitor_selective")
    assert verdict.status == "contradicted"


def test_ground_selectivity_unverifiable():
    """Unknown element 'Xy' cannot be looked up — must return unverifiable."""
    verdict = ground_selectivity("Xy", "Zn", GLYCINE, "target_selective")
    assert verdict.status == "unverifiable"


# ---------------------------------------------------------------------------
# Group 5 — ground_family()
# ---------------------------------------------------------------------------

def test_ground_family_polyol_verified():
    verdict = ground_family(GLYCEROL, "polyol")
    assert verdict.status == "verified"


def test_ground_family_polyol_contradicted():
    # Urea has no hydroxyl groups
    verdict = ground_family(UREA, "polyol")
    assert verdict.status == "contradicted"


def test_ground_family_amide_verified():
    verdict = ground_family(UREA, "amide")
    assert verdict.status == "verified"


def test_ground_family_carboxylic_acid_verified():
    verdict = ground_family(ACETIC_ACID, "carboxylic acid")
    assert verdict.status == "verified"


def test_ground_family_amine_verified():
    # Aniline has a primary amine ([NX3;H1,H2])
    verdict = ground_family(ANILINE, "amine")
    assert verdict.status == "verified"


def test_ground_family_phenol_verified():
    verdict = ground_family(PHENOL, "phenol")
    assert verdict.status == "verified"


def test_ground_family_unknown_label_unverifiable():
    # "crystal" is not in the deterministic family table
    verdict = ground_family(GLYCEROL, "crystal")
    assert verdict.status == "unverifiable"


# ---------------------------------------------------------------------------
# Group 6 — ground_des_plausibility()
# ---------------------------------------------------------------------------

def test_ground_des_plausibility_strong():
    """ChCl (HBA) paired with glycerol (HBD) is a canonical DES pair."""
    verdict = ground_des_plausibility(CHCL, GLYCEROL)
    # Label must be "strong" or "moderate" — anything but "none"
    assert verdict.status == "verified"


def test_ground_des_plausibility_contradicted():
    """Two pure-HBA quaternary salts (TBACl) provide no H-bond donors,
    so des_hbond_complementarity returns label='none'."""
    verdict = ground_des_plausibility(TBACL, TBACL)
    assert verdict.status == "contradicted"


# ---------------------------------------------------------------------------
# Group 7 — pH-aware structural_facts() and ground_coordination()
# ---------------------------------------------------------------------------

def test_structural_facts_default_ph_none_is_unchanged_for_glycine():
    # Regression guard: pH=None must reproduce the as-drawn Phase 1 result.
    facts = structural_facts("NCC(=O)O")
    assert facts.denticity == 2
    assert facts.n_donor_atoms >= 2
    assert facts.net_charge == 0
    assert facts.protonation_summary == ""


def test_structural_facts_ph7_glycine_is_species_aware():
    # At pH 7 glycine is a zwitterion: N is protonated (not a donor),
    # only the carboxylate site remains → denticity 1.
    facts = structural_facts("NCC(=O)O", pH=7.0)
    assert facts.denticity == 1
    assert facts.donor_element_counts.get("N", 0) == 0
    assert facts.net_charge == 0
    assert facts.protonation_summary != ""


def test_as_prompt_block_species_clause_only_when_ph_applied():
    drawn = structural_facts("NCC(=O)O").as_prompt_block()
    species = structural_facts("NCC(=O)O", pH=7.0).as_prompt_block()
    assert "species @ pH" not in drawn
    assert "species @ pH" in species


def test_ground_coordination_ph_aware_demotes_protonated_amine_claim():
    # "bidentate N,O-chelator" is true as-drawn but not for the pH-7 zwitterion
    # whose amine N is protonated.
    drawn = ground_coordination("NCC(=O)O", "bidentate N,O-chelator")
    species = ground_coordination("NCC(=O)O", "bidentate N,O-chelator", pH=7.0)
    assert drawn.status == "verified"
    assert species.status != "verified"


def test_ground_coordination_default_ph_none_unchanged():
    v = ground_coordination("NCC(=O)O", "bidentate N,O-chelator")
    assert v.status == "verified"
