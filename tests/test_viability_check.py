"""Tests for viability_check — stable, non-toxic, synthesizable DES candidates."""
from __future__ import annotations

import pytest
from rdkit import Chem

from des_multi_agent.chemistry_filter import viability_check
from des_multi_agent.chemistry.claim_grounding import ground_partner_reality


def _mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid test SMILES: {smiles!r}"
    return mol


# ---------------------------------------------------------------------------
# Reactive / unstable groups — must be rejected
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,reason_fragment", [
    ("CCOOC",                       "peroxide"),
    ("CCN=[N+]=[N-]",               "azide"),
    ("c1ccc([N+]#N)cc1",            "diazonium"),
    ("CC(=O)Cl",                    "acyl halide"),
    ("CS(=O)(=O)Cl",                "sulfonyl chloride"),
    ("c1cc([N+](=O)[O-])ccc1",      "nitro"),
    ("C[N+](=O)[O-]",               "nitro"),
])
def test_viability_check_rejects_reactive(smiles: str, reason_fragment: str) -> None:
    ok, reason = viability_check(_mol(smiles))
    assert not ok, f"Expected rejection of {smiles!r}"
    assert reason_fragment.lower() in reason.lower(), f"Expected '{reason_fragment}' in {reason!r}"


# ---------------------------------------------------------------------------
# Valid DES components — must be accepted
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,name", [
    ("OCCO",                                "ethylene glycol"),
    ("OCC(O)CO",                            "glycerol"),
    ("NCCO",                                "ethanolamine"),
    ("CC(=O)O",                             "acetic acid"),
    ("CC(=O)N",                             "acetamide"),
    ("c1ccccc1O",                           "phenol"),
    ("CC(C)[C@@H]1CC[C@H](C)C[C@H]1O",    "menthol (3 stereocenters)"),
    ("CC1CC(=O)NC1=O",                     "5-methylhydantoin"),
    ("NC(=O)N",                             "urea"),
    ("c1ccc2ccccc2c1",                     "naphthalene (2 rings, valid)"),
    ("O=C1NC(=O)N1",                       "hydantoin"),
    ("c1cc(O)ccc1O",                       "catechol"),
])
def test_viability_check_passes_valid_des(smiles: str, name: str) -> None:
    ok, reason = viability_check(_mol(smiles))
    assert ok, f"Valid DES component {name!r} ({smiles!r}) rejected: {reason!r}"


# ---------------------------------------------------------------------------
# Synthesizability limits
# ---------------------------------------------------------------------------

def test_viability_check_rejects_excess_rings() -> None:
    # 5-ring PAH: far too complex for a DES component
    coronene = "c1cc2ccc3cccc4ccc5cccc6ccc1c1c2c3c4c5c61"
    mol = Chem.MolFromSmiles(coronene)
    if mol is None:
        pytest.skip("coronene SMILES not parseable")
    ok, reason = viability_check(mol)
    assert not ok
    assert "ring" in reason.lower()


def test_viability_check_passes_four_ring_compound() -> None:
    # Pyrene (4 rings): at the limit — should pass viability (H-bond check handles DES suitability)
    ok, _ = viability_check(_mol("c1cc2cccc3ccc4cccc1c4c23"))
    assert ok


def test_viability_check_rejects_excess_stereocenters() -> None:
    # D-sorbitol (4 stereocenters): too many for a typical DES component
    sorbitol = "OC[C@H](O)[C@@H](O)[C@H](O)[C@@H](O)CO"
    ok, reason = viability_check(_mol(sorbitol))
    assert not ok
    assert "stereocenter" in reason.lower()


def test_viability_check_accepts_three_stereocenters() -> None:
    # Menthol has exactly 3 stereocenters — at the limit, should pass
    ok, _ = viability_check(_mol("CC(C)[C@@H]1CC[C@H](C)C[C@H]1O"))
    assert ok


# ---------------------------------------------------------------------------
# Integration: ground_partner_reality drops reactive novel compounds
# ---------------------------------------------------------------------------

def test_ground_partner_reality_drops_nitro_compound() -> None:
    # Nitrobenzene: valid SMILES, passes structural_sanity (MW~123, common atoms),
    # but viability_check should reject it for the nitro group.
    # It's also not in the known registry, so novel path runs.
    verdict = ground_partner_reality("OCCO", "c1ccc([N+](=O)[O-])cc1")
    assert verdict.disposition == "drop"
    assert "nitro" in verdict.detail.lower()


def test_ground_partner_reality_drops_acyl_chloride() -> None:
    verdict = ground_partner_reality("OCCO", "CC(=O)Cl")
    assert verdict.disposition == "drop"
    assert "acyl halide" in verdict.detail.lower()


def test_ground_partner_reality_keeps_valid_novel_with_hbond() -> None:
    # A novel but viable DES candidate with good H-bond capacity: propanediol
    verdict = ground_partner_reality("OCCO", "OCCO")
    # ethylene glycol vs ethylene glycol — same as component_a, will depend on is_known
    # Use a closely related but distinct molecule
    verdict = ground_partner_reality("CC(=O)O", "OCCO")
    # ethylene glycol should be novel_plausible or known
    assert verdict.disposition in ("keep",)
