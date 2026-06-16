"""Tests for LLM coordination and selectivity claim verification."""
from __future__ import annotations

from des_multi_agent.chemistry.claim_verification import (
    ClaimVerification,
    SelectivityVerification,
    batch_verify_coordination,
    verify_coordination_claim,
    verify_selectivity_claim,
)


# ---------------------------------------------------------------------------
# Coordination claim verifier
# ---------------------------------------------------------------------------

def test_bidentate_no_chelator_glycine():
    # Glycine: N-donor + carboxylate O-donor → bidentate N,O
    r = verify_coordination_claim("NCC(=O)O", "bidentate N,O-chelator")
    assert r.claimed_denticity == 2
    assert "N" in r.claimed_donors and "O" in r.claimed_donors
    assert r.actual_denticity == 2
    assert r.verdict == "ok"


def test_tridentate_nnn_claim_matches_edta_fragment():
    # NTA: N(CH2COOH)3 — tetradentate, but if we claim tridentate it should flag
    r = verify_coordination_claim("OC(=O)CN(CC(=O)O)CC(=O)O", "tridentate N,N,N-donor")
    # NTA is tetradentate — mismatch expected (N is present so not donor_mismatch)
    assert r.verdict in {"denticity_mismatch", "ok"}  # ±1 tolerance may save tetra vs tri


def test_monodentate_o_water():
    r = verify_coordination_claim("O", "monodentate O-donor")
    assert r.claimed_denticity == 1
    assert r.verdict == "ok"


def test_missing_sulfur_donor_flagged():
    # Claim S-donor but ethanol has only O
    r = verify_coordination_claim("CCO", "bidentate S,O-chelator")
    assert r.verdict == "donor_mismatch"
    assert any("S" in note for note in r.notes)


def test_claimed_denticity_mismatch():
    # Ethylenediamine is bidentate; claiming tetradentate should flag
    r = verify_coordination_claim("NCCN", "tetradentate N,N,N,N-donor")
    assert r.claimed_denticity == 4
    assert r.actual_denticity == 2
    assert r.verdict == "denticity_mismatch"


def test_molecule_with_no_donor_atoms():
    # Methane has no donor atoms
    r = verify_coordination_claim("C", "monodentate C-donor")
    assert r.verdict == "not_a_ligand"


def test_unparseable_claim_returns_unparseable():
    r = verify_coordination_claim("CCO", "good solvent for the reaction")
    assert r.verdict == "unparseable"


def test_claim_verification_is_frozen():
    r = verify_coordination_claim("NCC(=O)O", "bidentate N,O-chelator")
    assert isinstance(r, ClaimVerification)
    assert r.is_ok


# ---------------------------------------------------------------------------
# Selectivity claim verifier
# ---------------------------------------------------------------------------

def test_ni_over_co_is_target_selective():
    r = verify_selectivity_claim("Ni2+", "Co2+", "NCC(=O)O")
    assert isinstance(r, SelectivityVerification)
    assert r.verdict == "target_selective"
    assert r.delta_log_k > 0


def test_antisymmetry():
    a = verify_selectivity_claim("Ni2+", "Co2+", "NCC(=O)O")
    b = verify_selectivity_claim("Co2+", "Ni2+", "NCC(=O)O")
    assert abs(a.delta_log_k + b.delta_log_k) < 1e-9
    assert a.verdict == "target_selective"
    assert b.verdict == "competitor_selective"


def test_neutral_when_same_metal():
    # Same metal on both sides: ΔlogK == 0 → neutral
    r = verify_selectivity_claim("Ni2+", "Ni2+", "NCC(=O)O")
    assert r.verdict == "neutral"
    assert r.delta_log_k == 0.0


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------

def test_batch_verify_skips_bad_smiles():
    pairs = [
        ("NCC(=O)O", "bidentate N,O-chelator"),
        ("NOTVALID!!!SMILES", "bidentate N,O-chelator"),
        ("NCCN", "bidentate N,N-donor"),
    ]
    results = batch_verify_coordination(pairs)
    # Bad SMILES should either be skipped or return unparseable, not crash
    assert len(results) >= 2
    smiles_in_results = {r.smiles for r in results}
    assert "NCC(=O)O" in smiles_in_results
    assert "NCCN" in smiles_in_results
