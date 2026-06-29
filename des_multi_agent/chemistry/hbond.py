"""DES H-bond complementarity scoring.

A deep-eutectic solvent forms through hydrogen-bond interaction between a
hydrogen-bond donor (HBD) component and a hydrogen-bond acceptor (HBA)
component.  Strong DES formation requires complementarity:

* a high HBD count on one component
* a high HBA count on the other

This module quantifies that complementarity and the overall H-bond capacity
of a pair, without requiring any ML model.  The scores serve as lightweight
structural filters or ranking signals in the DES screening pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors


@dataclass(frozen=True)
class HBondProfile:
    """H-bond count profile for a single component."""
    smiles: str
    n_hbd: int   # Lipinski H-bond donors
    n_hba: int   # Lipinski H-bond acceptors
    role: str    # "HBD" | "HBA" | "amphoteric" | "none"

    @property
    def capacity(self) -> float:
        """Total H-bond capacity (donors + acceptors)."""
        return float(self.n_hbd + self.n_hba)


@dataclass(frozen=True)
class HBondComplementarity:
    """Complementarity score for a DES pair (component_a, component_b)."""
    smiles_a: str
    smiles_b: str
    profile_a: HBondProfile
    profile_b: HBondProfile
    complementarity_score: float  # 0..1; higher = more complementary
    capacity_score: float         # (n_hbd_total + n_hba_total) / normaliser
    composite_score: float        # weighted combination
    label: str                    # "strong" | "moderate" | "weak" | "none"


# ---------------------------------------------------------------------------
# H-bond profile extraction
# ---------------------------------------------------------------------------

_ROLE_THRESHOLD_HBD = 1   # molecule with ≥1 donor is at least partly HBD
_ROLE_THRESHOLD_HBA = 1   # molecule with ≥1 acceptor is at least partly HBA


def hbond_profile(smiles_or_mol) -> HBondProfile:
    """Return the H-bond profile for a SMILES string or RDKit molecule."""
    mol = (
        smiles_or_mol
        if isinstance(smiles_or_mol, Chem.Mol)
        else Chem.MolFromSmiles(smiles_or_mol)
    )
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles_or_mol!r}")

    n_hbd = int(Lipinski.NumHDonors(mol))
    n_hba = int(Lipinski.NumHAcceptors(mol))
    smiles = Chem.MolToSmiles(mol)

    if n_hbd >= _ROLE_THRESHOLD_HBD and n_hba >= _ROLE_THRESHOLD_HBA:
        role = "amphoteric"
    elif n_hbd >= _ROLE_THRESHOLD_HBD:
        role = "HBD"
    elif n_hba >= _ROLE_THRESHOLD_HBA:
        role = "HBA"
    else:
        role = "none"

    return HBondProfile(smiles=smiles, n_hbd=n_hbd, n_hba=n_hba, role=role)


# ---------------------------------------------------------------------------
# Pair complementarity scoring
# ---------------------------------------------------------------------------

_CAPACITY_NORM = 10.0   # expected max total H-bond count for score normalisation
_W_COMPLEMENTARITY = 0.6
_W_CAPACITY = 0.4


def _complementarity(profile_a: HBondProfile, profile_b: HBondProfile) -> float:
    """Asymmetric complementarity: how well A's donors match B's acceptors and vice versa.

    Score is 1 when one molecule is a pure HBD and the other is a pure HBA,
    0 when both have no H-bond groups.
    """
    hbd_a, hba_a = profile_a.n_hbd, profile_a.n_hba
    hbd_b, hba_b = profile_b.n_hbd, profile_b.n_hba

    # Forward direction: A donors ↔ B acceptors
    if hbd_a == 0 and hba_b == 0:
        forward = 0.0
    else:
        forward = min(hbd_a, hba_b) / max(hbd_a, hba_b, 1)

    # Reverse direction: B donors ↔ A acceptors
    if hbd_b == 0 and hba_a == 0:
        reverse = 0.0
    else:
        reverse = min(hbd_b, hba_a) / max(hbd_b, hba_a, 1)

    # Pure asymmetric pair (A=HBD only, B=HBA only, or vice versa): the reverse
    # guard above set the non-applicable direction to 0.0, so averaging would
    # halve a perfect forward score. Return the dominant direction directly.
    if forward == 0.0 or reverse == 0.0:
        return max(forward, reverse)
    return (forward + reverse) / 2.0


def _label(composite: float) -> str:
    if composite >= 0.6:
        return "strong"
    if composite >= 0.35:
        return "moderate"
    if composite > 0.0:
        return "weak"
    return "none"


def des_hbond_complementarity(
    smiles_a: str,
    smiles_b: str,
    *,
    w_complementarity: float = _W_COMPLEMENTARITY,
    w_capacity: float = _W_CAPACITY,
) -> HBondComplementarity:
    """Compute H-bond complementarity for a DES candidate pair.

    Returns a :class:`HBondComplementarity` with:

    * ``complementarity_score`` — how well the donor/acceptor roles match (0–1).
    * ``capacity_score`` — total H-bond count normalised to [0–1].
    * ``composite_score`` — weighted combination.
    * ``label`` — human-readable strength ("strong" | "moderate" | "weak" | "none").
    """
    pa = hbond_profile(smiles_a)
    pb = hbond_profile(smiles_b)

    comp = _complementarity(pa, pb)
    cap = min((pa.n_hbd + pa.n_hba + pb.n_hbd + pb.n_hba) / _CAPACITY_NORM, 1.0)
    composite = w_complementarity * comp + w_capacity * cap

    return HBondComplementarity(
        smiles_a=smiles_a,
        smiles_b=smiles_b,
        profile_a=pa,
        profile_b=pb,
        complementarity_score=comp,
        capacity_score=cap,
        composite_score=composite,
        label=_label(composite),
    )


def rank_by_hbond(
    component_a: str,
    candidates: list[str],
    *,
    w_complementarity: float = _W_COMPLEMENTARITY,
    w_capacity: float = _W_CAPACITY,
) -> list[tuple[str, HBondComplementarity]]:
    """Rank a list of candidate SMILES against *component_a* by composite H-bond score.

    Returns a list of (smiles, HBondComplementarity) pairs sorted best-first.
    Invalid SMILES are silently skipped.
    """
    results: list[tuple[str, HBondComplementarity]] = []
    for smiles_b in candidates:
        try:
            hbc = des_hbond_complementarity(
                component_a, smiles_b,
                w_complementarity=w_complementarity,
                w_capacity=w_capacity,
            )
            results.append((smiles_b, hbc))
        except ValueError:
            pass
    return sorted(results, key=lambda t: t[1].composite_score, reverse=True)
