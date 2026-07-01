"""Tests for scaffold-memory features: murcko_scaffold_smiles and filter_candidates scaffold gate."""
from __future__ import annotations

import pytest
from rdkit import Chem

from des_multi_agent.chemistry_filter import (
    filter_candidates,
    murcko_scaffold_smiles,
)
from des_multi_agent.schemas import CandidateProposal


def _proposal(smiles: str, family: str = "test") -> CandidateProposal:
    return CandidateProposal(smiles=smiles, rationale="test", family=family, source="test", source_id="")


# ---------------------------------------------------------------------------
# murcko_scaffold_smiles
# ---------------------------------------------------------------------------

def test_scaffold_cyclic_compound():
    # Phenol has a benzene ring as its scaffold
    sc = murcko_scaffold_smiles("c1ccccc1O")
    assert sc is not None
    mol = Chem.MolFromSmiles(sc)
    assert mol is not None
    assert mol.GetNumAtoms() > 0


def test_scaffold_acyclic_returns_full_molecule():
    # Ethylene glycol is acyclic; scaffold equals full molecule
    sc = murcko_scaffold_smiles("OCCO")
    eg = Chem.MolToSmiles(Chem.MolFromSmiles("OCCO"), canonical=True)
    assert sc == eg


def test_scaffold_invalid_smiles_returns_none():
    assert murcko_scaffold_smiles("not_smiles") is None


def test_scaffold_two_ring_compound():
    # Naphthalene — scaffold has 10 atoms (two fused rings)
    sc = murcko_scaffold_smiles("c1ccc2ccccc2c1")
    assert sc is not None
    mol = Chem.MolFromSmiles(sc)
    assert mol is not None and mol.GetNumAtoms() == 10


def test_scaffold_determinism():
    # Same molecule → same scaffold regardless of input form
    sc1 = murcko_scaffold_smiles("OC1CCCCC1")    # cyclohexanol
    sc2 = murcko_scaffold_smiles("C1CCCCC1O")    # same compound, different notation
    assert sc1 == sc2


# ---------------------------------------------------------------------------
# filter_candidates with failing_scaffolds
# ---------------------------------------------------------------------------

def test_filter_candidates_scaffold_gate_drops_matching():
    # Phenol and 4-methylphenol share the benzene ring scaffold
    phenol_scaffold = murcko_scaffold_smiles("c1ccccc1O")
    assert phenol_scaffold is not None

    candidates = [
        _proposal("c1ccccc1O"),        # phenol — scaffold matches
        _proposal("Cc1ccccc1O"),       # 4-methylphenol — same scaffold
        _proposal("OCCO"),             # ethylene glycol — different (acyclic)
    ]
    failing = {phenol_scaffold}
    filtered = filter_candidates("NCCO", candidates, failing_scaffolds=failing)
    result_smiles = [p.smiles for p in filtered]
    # Only ethylene glycol should survive
    assert all("c1ccccc1" not in s for s in result_smiles)
    assert any("OCCO" in s or "OCC" in s for s in result_smiles)


def test_filter_candidates_no_scaffold_gate_when_empty():
    candidates = [_proposal("c1ccccc1O"), _proposal("OCCO")]
    filtered_without = filter_candidates("NCCO", candidates, failing_scaffolds=None)
    filtered_empty = filter_candidates("NCCO", candidates, failing_scaffolds=set())
    # Both should behave identically
    assert len(filtered_without) == len(filtered_empty)


def test_filter_candidates_scaffold_gate_preserves_non_matching():
    failing = {"c1ccccc1"}  # benzene scaffold
    candidates = [
        _proposal("OCCO"),   # acyclic — different scaffold
        _proposal("OCC(O)CO"),  # glycerol — acyclic
    ]
    filtered = filter_candidates("NCCO", candidates, failing_scaffolds=failing)
    assert len(filtered) == 2


def test_filter_candidates_backward_compat_no_scaffold_arg():
    # Existing call signature (no failing_scaffolds) still works
    candidates = [_proposal("OCCO"), _proposal("NCCO")]
    filtered = filter_candidates("CC(=O)O", candidates)
    assert len(filtered) >= 1


# ---------------------------------------------------------------------------
# Integration: scaffold memory narrative
# ---------------------------------------------------------------------------

def test_scaffold_exclusion_reduces_candidate_count():
    # Build a failing scaffold set from catechol
    catechol_sc = murcko_scaffold_smiles("Oc1ccccc1O")
    assert catechol_sc is not None
    candidates = [
        _proposal("Oc1ccccc1O"),    # catechol — matching scaffold
        _proposal("Oc1ccc(O)cc1"),  # hydroquinone — same ring system
        _proposal("OCCO"),          # ethylene glycol — no ring
    ]
    failing = {catechol_sc}
    filtered = filter_candidates("NCCO", candidates, failing_scaffolds=failing)
    # Ring-bearing proposals sharing catechol's benzene scaffold should be dropped
    assert len(filtered) < len(candidates)
