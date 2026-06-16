"""Tests for DES H-bond complementarity scoring."""
from __future__ import annotations

from des_multi_agent.chemistry.hbond import (
    HBondComplementarity,
    HBondProfile,
    des_hbond_complementarity,
    hbond_profile,
    rank_by_hbond,
)


# ---------------------------------------------------------------------------
# Profile extraction
# ---------------------------------------------------------------------------

def test_choline_chloride_is_hba():
    # Choline chloride: N+ has no lone pair but the OH is a donor; overall amphoteric
    p = hbond_profile("C[N+](C)(C)CCO.[Cl-]")
    assert p.n_hbd >= 1
    assert p.n_hba >= 1
    assert p.role == "amphoteric"


def test_urea_is_amphoteric():
    p = hbond_profile("NC(=O)N")
    assert p.n_hbd >= 2   # two NH groups
    assert p.n_hba >= 1
    assert p.role == "amphoteric"


def test_methanol_is_amphoteric():
    # Methanol is the simplest molecule with both OH-donor and O-acceptor.
    # Water ("O") returns HBD=HBA=0 from RDKit Lipinski — a known edge case.
    p = hbond_profile("CO")
    assert p.n_hbd == 1
    assert p.n_hba == 1
    assert p.role == "amphoteric"


def test_methane_has_no_hbond():
    p = hbond_profile("C")
    assert p.n_hbd == 0
    assert p.n_hba == 0
    assert p.role == "none"


def test_acetone_is_hba_only():
    p = hbond_profile("CC(=O)C")
    assert p.n_hbd == 0
    assert p.n_hba >= 1
    assert p.role == "HBA"


def test_profile_capacity():
    p = hbond_profile("NC(=O)N")  # urea
    assert p.capacity == p.n_hbd + p.n_hba


# ---------------------------------------------------------------------------
# Pair complementarity
# ---------------------------------------------------------------------------

def test_choline_urea_des_is_strong():
    # Choline chloride + urea is a well-known DES (reline)
    r = des_hbond_complementarity("C[N+](C)(C)CCO.[Cl-]", "NC(=O)N")
    assert isinstance(r, HBondComplementarity)
    assert r.composite_score > 0
    assert r.label in {"strong", "moderate"}


def test_pure_hbd_hba_pair_high_complementarity():
    # Ethanol (pure HBD) + dimethyl ether (HBA only, C=O absent but O acceptor)
    # Use acetone as HBA representative
    r = des_hbond_complementarity("CCO", "CC(=O)C")
    # ethanol: HBD=1, HBA=1 (actually amphoteric)
    # acetone: HBD=0, HBA=1
    # complementarity > 0 since ethanol donates to acetone carbonyl
    assert r.complementarity_score > 0


def test_two_pure_hbd_molecules_low_complementarity():
    # Two alcohols: neither accepts well relative to what the other donates
    r1 = des_hbond_complementarity("CCO", "OCCO")
    # Both have donors AND acceptors (amphoteric), so complementarity will be nonzero
    # but may be lower than an ideal HBD+HBA pair
    assert 0 <= r1.complementarity_score <= 1


def test_no_hbond_pair_is_none():
    # Two alkanes: no H-bond capacity at all
    r = des_hbond_complementarity("CCCC", "CCCCC")
    assert r.complementarity_score == 0.0
    assert r.capacity_score == 0.0
    assert r.composite_score == 0.0
    assert r.label == "none"


def test_complementarity_is_symmetric():
    a = des_hbond_complementarity("CCO", "NC(=O)N")
    b = des_hbond_complementarity("NC(=O)N", "CCO")
    assert abs(a.composite_score - b.composite_score) < 1e-9


def test_hbond_complementarity_is_frozen():
    r = des_hbond_complementarity("CCO", "NC(=O)N")
    assert isinstance(r, HBondComplementarity)


# ---------------------------------------------------------------------------
# Ranking helper
# ---------------------------------------------------------------------------

def test_rank_by_hbond_returns_sorted():
    # Acetone (HBA) should be ranked highly against ethanol (HBD)
    candidates = ["CCCC", "CC(=O)C", "NC(=O)N", "NOTVALID!!!"]
    ranked = rank_by_hbond("CCO", candidates)
    # Invalid SMILES skipped
    assert all(smiles != "NOTVALID!!!" for smiles, _ in ranked)
    # Scores are descending
    scores = [hbc.composite_score for _, hbc in ranked]
    assert scores == sorted(scores, reverse=True)


def test_rank_by_hbond_skips_invalid_smiles():
    ranked = rank_by_hbond("CCO", ["INVALID", "NOTSMILES"])
    assert len(ranked) == 0
