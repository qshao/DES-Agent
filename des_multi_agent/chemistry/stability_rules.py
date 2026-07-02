"""Rule-based metal-ligand stability scoring grounded in real coordination
chemistry — the Irving-Williams series, HSAB donor/acceptor matching, and the
chelate effect.

This exists to give the metal-selectivity screen genuine discrimination (e.g.
Ni2+ vs Co2+), which the uniform-output heuristic stability model cannot. It is a
transparent, citable baseline — not a substitute for measured or QM stability
constants, but far better than "every ligand scores the same".
"""
from __future__ import annotations

from ..predictors.stability_constants import _METAL_IDENTITY, _parse_metal_charge
from .coordination import coordination_profile

# Relative log-stability offsets for the divalent first-row transition metals.
# The *order* (Mn<Fe<Co<Ni<Cu>Zn) is universal across ligands; the magnitudes are
# representative relative values.
_IRVING_WILLIAMS: dict[str, float] = {
    "Mn2+": 0.0,
    "Fe2+": 0.6,
    "Co2+": 1.0,
    "Ni2+": 1.5,
    "Cu2+": 2.3,
    "Zn2+": 0.9,
}

_BASE_LOG_K = 2.0
_W_HSAB = 1.5
_W_CHELATE = 0.8
_W_CHARGE = 0.5
_W_DONOR = 0.1


def irving_williams_offset(metal_ion: str) -> float:
    return _IRVING_WILLIAMS.get(metal_ion.strip(), 0.0)


def _metal_softness(metal_ion: str) -> float:
    return float(_METAL_IDENTITY.get(metal_ion.strip(), (0, 0, 0, 0.0))[3])


def metal_softness(metal_ion: str) -> float:
    """Public accessor for a metal's HSAB softness in [0, 1] (0=hard, 1=soft)."""
    return _metal_softness(metal_ion)


def hsab_match(metal_ion: str, ligand_smiles: str) -> float:
    """0..1 HSAB compatibility: 1 when metal and donor softness coincide.

    Hard metals favour hard (O/N) donors; soft metals favour soft (S/P) donors.
    """
    prof = coordination_profile(ligand_smiles)
    return 1.0 - abs(_metal_softness(metal_ion) - prof.mean_donor_softness)


def _rule_based_log_k_from_profile(metal_ion: str, prof) -> float:
    """Compute log K from a pre-computed coordination profile."""
    if prof.n_donor_atoms == 0:
        return 0.0
    iw = irving_williams_offset(metal_ion)
    hsab = 1.0 - abs(_metal_softness(metal_ion) - prof.mean_donor_softness)
    chelate = max(0, prof.denticity - 1) * _W_CHELATE
    charge = abs(_parse_metal_charge(metal_ion)) * _W_CHARGE
    donor = prof.n_donor_atoms * _W_DONOR
    return _BASE_LOG_K + iw + _W_HSAB * hsab + chelate + charge + donor


def rule_based_log_k(metal_ion: str, ligand_smiles: str) -> float:
    """A transparent stability-constant estimate (relative log K units)."""
    return _rule_based_log_k_from_profile(metal_ion, coordination_profile(ligand_smiles))


def selectivity_delta_log_k(target_metal: str, competitor_metal: str, ligand_smiles: str) -> float:
    """log K(target) - log K(competitor) for the same ligand (positive = target
    selective). The chelate/donor terms cancel, leaving the Irving-Williams and
    HSAB metal differences."""
    prof = coordination_profile(ligand_smiles)
    return (
        _rule_based_log_k_from_profile(target_metal, prof)
        - _rule_based_log_k_from_profile(competitor_metal, prof)
    )
