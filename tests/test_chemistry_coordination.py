"""Coordination-site perception from SMILES."""
from __future__ import annotations

from des_multi_agent.chemistry.coordination import coordination_profile


def test_water_is_monodentate_oxygen():
    p = coordination_profile("O")
    assert p.donor_element_counts == {"O": 1}
    assert p.n_donor_atoms == 1
    assert p.denticity == 1


def test_methane_has_no_donors():
    p = coordination_profile("C")
    assert p.n_donor_atoms == 0
    assert p.denticity == 0


def test_ethylenediamine_is_bidentate_nn_with_5ring():
    p = coordination_profile("NCCN")
    assert p.donor_element_counts == {"N": 2}
    assert p.denticity == 2
    assert 5 in p.chelate_ring_sizes  # M-N-C-C-N 5-membered chelate


def test_glycine_carboxylate_collapses_to_one_site():
    # amine N + carboxylate (two O on one C count as a single binding site)
    p = coordination_profile("NCC(=O)O")
    assert p.donor_element_counts["O"] == 2     # two donor oxygen atoms
    assert p.denticity == 2                      # but two coordinating sites (N, COO)
    assert set(p.donor_site_elements) == {"N", "O"}


def test_catechol_is_bidentate_oo():
    p = coordination_profile("Oc1ccccc1O")
    assert p.donor_element_counts == {"O": 2}
    assert p.denticity == 2
    assert set(p.donor_site_elements) == {"O"}


def test_donor_softness_orders_S_above_N_above_O():
    s = coordination_profile("CCS").mean_donor_softness   # thiol (soft)
    n = coordination_profile("CCN").mean_donor_softness   # amine (borderline)
    o = coordination_profile("CCO").mean_donor_softness   # alcohol (hard)
    assert s > n > o


def test_quaternary_ammonium_nitrogen_is_not_a_donor():
    # choline cation: the N+ has no lone pair to donate
    p = coordination_profile("C[N+](C)(C)CCO")
    assert p.donor_element_counts.get("N", 0) == 0
    assert p.donor_element_counts.get("O", 0) == 1
