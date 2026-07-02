"""Converts DFT free-ligand HOMO energy to a metal-selectivity ranking adjustment.

Entry point: dft_selectivity_adjustment(dft_result, target_metal, competitor_metal) -> float.
Returns a value in [-0.05, +0.05] to add to composite_score.
"""
from __future__ import annotations

from .dft_validator import DFTResult

# HOMO energy calibration anchors (eV) → donor softness in [0, 1].
# −9.5 eV ≈ hard donors (carboxylate O, amide N);
# −7.5 eV ≈ soft donors (thiolate S, phosphine P).
_HOMO_HARD_EV: float = -9.5   # softness = 0.0
_HOMO_SOFT_EV: float = -7.5   # softness = 1.0

_MAX_ADJ: float = 0.05         # half the H-bond bias magnitude (±0.10)


def _homo_to_softness(homo_ev: float) -> float:
    """Linearly map HOMO energy to donor softness in [0, 1]."""
    t = (homo_ev - _HOMO_HARD_EV) / (_HOMO_SOFT_EV - _HOMO_HARD_EV)
    return max(0.0, min(1.0, t))


def dft_selectivity_adjustment(
    dft_result: DFTResult,
    target_metal: str,
    competitor_metal: str,
) -> float:
    """±0.05 composite-score nudge based on HOMO energy vs HSAB metal softness.

    Positive → ligand HOMO profile matches target better than competitor.
    Returns 0.0 if DFT did not succeed or HOMO is unavailable.
    """
    if not dft_result.success or dft_result.homo_ev is None:
        return 0.0

    from ..chemistry.stability_rules import metal_softness

    s_target = metal_softness(target_metal)
    s_comp = metal_softness(competitor_metal)

    if s_target == s_comp:
        return 0.0

    s_ligand = _homo_to_softness(dft_result.homo_ev)

    # delta > 0 → ligand softness closer to competitor; negate so target-match = positive
    delta = abs(s_ligand - s_target) - abs(s_ligand - s_comp)
    scale = abs(s_target - s_comp)           # normalise by the metal-pair separation
    raw = -delta / scale * _MAX_ADJ if scale > 0 else 0.0
    return max(-_MAX_ADJ, min(_MAX_ADJ, raw))
