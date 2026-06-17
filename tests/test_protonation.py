"""Tests for the deterministic protonation / dominant-species engine."""
from __future__ import annotations

import pytest
from rdkit import Chem

from des_multi_agent.chemistry.protonation import (
    IonizedGroup,
    ProtonationResult,
    dominant_species,
    _IONIZABLE_SMARTS,
)


def _net(smiles: str, pH: float) -> int:
    return dominant_species(smiles, pH).net_charge


def test_carboxylic_acid_deprotonates_at_ph7():
    res = dominant_species("CC(=O)O", 7.0)
    assert res.net_charge == -1
    assert any(g.state == "deprotonated" and g.group_name == "carboxylic acid" for g in res.groups)


def test_aliphatic_amine_protonates_at_ph7():
    res = dominant_species("CCN", 7.0)
    assert res.net_charge == +1
    assert any(g.state == "protonated" for g in res.groups)


def test_glycine_is_zwitterion_at_ph7():
    res = dominant_species("NCC(=O)O", 7.0)
    assert res.net_charge == 0
    states = sorted(g.state for g in res.groups if g.state != "neutral")
    assert states == ["deprotonated", "protonated"]


def test_glycine_cation_at_ph1():
    assert _net("NCC(=O)O", 1.0) == +1


def test_glycine_anion_at_ph12():
    assert _net("NCC(=O)O", 12.0) == -1


def test_glycerol_unchanged_at_ph7():
    res = dominant_species("OCC(O)CO", 7.0)
    assert res.net_charge == 0
    assert all(g.state == "neutral" for g in res.groups)
    # Canonical species equals canonical input (no edits).
    assert res.species_smiles == Chem.MolToSmiles(Chem.MolFromSmiles("OCC(O)CO"))


def test_imidazole_state_flips_across_its_pka():
    # imidazole basic N, pKa ~7: protonated below, neutral above
    assert _net("c1c[nH]cn1", 6.0) == +1
    assert _net("c1c[nH]cn1", 8.0) == 0


def test_invalid_smiles_passthrough_never_raises():
    res = dominant_species("not_a_smiles", 7.0)
    assert res.species_smiles == "not_a_smiles"
    assert res.groups == []
    assert res.net_charge == 0


def test_untabulated_ionizable_group_left_as_drawn():
    # ethane has no ionizable group → unchanged, net 0
    res = dominant_species("CC", 7.0)
    assert res.net_charge == 0
    assert all(g.state == "neutral" for g in res.groups) or res.groups == []


def test_table_integrity_every_smarts_compiles_and_is_well_formed():
    for smarts, name, pka, kind in _IONIZABLE_SMARTS:
        assert Chem.MolFromSmarts(smarts) is not None, f"bad SMARTS: {smarts!r} ({name})"
        assert isinstance(pka, (int, float))
        assert kind in ("acid", "base")


def test_precompiled_patterns_all_non_none():
    from des_multi_agent.chemistry.protonation import _IONIZABLE_PATTERNS
    for patt, name, pka, kind in _IONIZABLE_PATTERNS:
        assert patt is not None, f"precompiled SMARTS is None for {name!r}"
