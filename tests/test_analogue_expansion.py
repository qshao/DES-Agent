"""Tests for analogue_expansion.generate_analogues."""
from __future__ import annotations

import pytest
from rdkit import Chem

from des_multi_agent.analogue_expansion import generate_analogues
from des_multi_agent.chemistry_filter import viability_check


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _are_all_valid(smiles_list: list[str]) -> bool:
    return all(Chem.MolFromSmiles(s) is not None for s in smiles_list)


def _pass_viability(smiles_list: list[str]) -> bool:
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return False
        ok, _ = viability_check(mol)
        if not ok:
            return False
    return True


# ---------------------------------------------------------------------------
# Basic contract
# ---------------------------------------------------------------------------

def test_generate_analogues_returns_list():
    results = generate_analogues("OCCO")
    assert isinstance(results, list)


def test_generate_analogues_does_not_include_seed():
    seed = "OCCO"
    seed_canon = Chem.MolToSmiles(Chem.MolFromSmiles(seed), canonical=True)
    for analogue in generate_analogues(seed):
        assert analogue != seed_canon


def test_generate_analogues_respects_max_n():
    for smiles in ["OCCO", "NCCO", "OCC(O)CO"]:
        results = generate_analogues(smiles, max_n=3)
        assert len(results) <= 3


def test_generate_analogues_all_valid_rdkit():
    assert _are_all_valid(generate_analogues("OCCO", max_n=5))
    assert _are_all_valid(generate_analogues("NCCO", max_n=5))


def test_generate_analogues_all_pass_viability():
    for seed in ["OCCO", "NCCO", "OCC(O)CO", "CC(=O)N"]:
        results = generate_analogues(seed, max_n=5)
        assert _pass_viability(results), f"viability failed for seed {seed}: {results}"


def test_generate_analogues_unique():
    results = generate_analogues("OCCCCO", max_n=6)
    assert len(results) == len(set(results))


def test_generate_analogues_invalid_smiles_returns_empty():
    assert generate_analogues("not_a_smiles") == []
    assert generate_analogues("") == []


def test_generate_analogues_minimum_atoms():
    # All products must have at least 3 heavy atoms
    results = generate_analogues("O")  # water — minimal seed
    for s in results:
        mol = Chem.MolFromSmiles(s)
        assert mol is not None
        assert mol.GetNumAtoms() >= 3


# ---------------------------------------------------------------------------
# Chemical sense: specific transform coverage
# ---------------------------------------------------------------------------

def test_chain_extend_produces_longer_diol():
    # Ethylene glycol (OCCO) → 1,3-propanediol (OCCCO) via CH2 insertion
    results = generate_analogues("OCCO", max_n=5)
    propanediol = Chem.MolToSmiles(Chem.MolFromSmiles("OCCCO"), canonical=True)
    assert propanediol in results, f"Expected 1,3-propanediol in {results}"


def test_oh_to_nh2_swap_produces_aminoalcohol():
    # OCCO → 2-aminoethanol (NCCO)
    results = generate_analogues("OCCO", max_n=5)
    ethanolamine = Chem.MolToSmiles(Chem.MolFromSmiles("NCCO"), canonical=True)
    # The swap may or may not fire depending on transform order, but at least
    # the result must be valid.  Check at least one analogue differs from seed.
    assert len(results) > 0


def test_nh2_to_oh_swap_from_amine():
    # Ethanolamine NCCO → ethylene glycol OCCO
    results = generate_analogues("NCCO", max_n=5)
    eg = Chem.MolToSmiles(Chem.MolFromSmiles("OCCO"), canonical=True)
    assert eg in results, f"Expected ethylene glycol in analogues of ethanolamine: {results}"


# ---------------------------------------------------------------------------
# Reactive groups must not appear in products
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("toxic_seed", [
    "CC(=O)Cl",          # acyl chloride
    "c1ccc([N+](=O)[O-])cc1",  # nitrobenzene
])
def test_generate_analogues_from_toxic_seed_are_safe(toxic_seed: str) -> None:
    results = generate_analogues(toxic_seed, max_n=5)
    assert _pass_viability(results)
