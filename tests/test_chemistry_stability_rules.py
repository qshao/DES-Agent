"""Rule-based stability scoring: Irving-Williams series + HSAB + chelate effect."""
from __future__ import annotations

from des_multi_agent.chemistry.stability_rules import (
    irving_williams_offset,
    rule_based_log_k,
    selectivity_delta_log_k,
    hsab_match,
)


def test_irving_williams_ordering():
    # Mn < Fe < Co < Ni < Cu > Zn (the universal divalent first-row order)
    o = {m: irving_williams_offset(m) for m in ("Mn2+", "Fe2+", "Co2+", "Ni2+", "Cu2+", "Zn2+")}
    assert o["Mn2+"] < o["Fe2+"] < o["Co2+"] < o["Ni2+"] < o["Cu2+"]
    assert o["Zn2+"] < o["Ni2+"]


def test_ni_over_co_is_favoured_by_irving_williams():
    d = selectivity_delta_log_k("Ni2+", "Co2+", "NCC(=O)O")  # glycine
    assert d > 0  # Ni2+ selective over Co2+


def test_selectivity_is_antisymmetric():
    a = selectivity_delta_log_k("Ni2+", "Co2+", "NCC(=O)O")
    b = selectivity_delta_log_k("Co2+", "Ni2+", "NCC(=O)O")
    assert abs(a + b) < 1e-9


def test_chelate_effect_increases_stability():
    mono = rule_based_log_k("Ni2+", "O")               # water, monodentate
    multi = rule_based_log_k("Ni2+", "OC(=O)CN(CC(=O)O)CC(=O)O")  # NTA, tetradentate
    assert multi > mono


def test_hsab_soft_metal_prefers_soft_donor():
    # soft metal (Hg2+, softness 1.0) matches a soft S-donor better than a hard O-donor
    soft_lig = hsab_match("Hg2+", "CCS")   # thiol
    hard_lig = hsab_match("Hg2+", "CCO")   # alcohol
    assert soft_lig > hard_lig


def test_unknown_metal_is_handled_gracefully():
    # a metal not in the Irving-Williams set still returns a finite score
    val = rule_based_log_k("Ca2+", "NCC(=O)O")
    assert val == val and abs(val) < 1e6  # finite, not NaN
