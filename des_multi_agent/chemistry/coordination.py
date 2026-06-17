"""Perceive metal-coordination structure from a SMILES string.

Turns any molecule into a structured ``CoordinationProfile`` — donor atoms and
their HSAB softness, an estimated denticity (number of coordinating *sites*, with
carboxylate/nitro oxygens collapsed to one site), and the chelate ring sizes its
donor pairs would form. This is the structural primitive the HSAB/Irving-Williams
stability rules and the LLM claim verifier build on.
"""
from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem

# HSAB softness of donor elements: 0 = hard, 1 = soft.
DONOR_SOFTNESS: dict[str, float] = {"O": 0.0, "F": 0.0, "N": 0.5, "Cl": 0.6, "S": 1.0, "P": 1.0}
_DONOR_ELEMENTS = set(DONOR_SOFTNESS)


@dataclass(frozen=True)
class CoordinationProfile:
    smiles: str
    donor_atom_indices: tuple[int, ...]
    donor_element_counts: dict[str, int]   # by donor *atom*
    n_donor_atoms: int
    denticity: int                          # number of coordinating *sites*
    donor_site_elements: tuple[str, ...]    # representative element per site
    chelate_ring_sizes: tuple[int, ...]     # ring sizes for donor pairs (M + path)
    mean_donor_softness: float


def _is_donor(atom) -> bool:
    sym = atom.GetSymbol()
    if sym not in _DONOR_ELEMENTS:
        return False
    # A positively charged atom with a full valence (e.g. quaternary ammonium)
    # has no lone pair available to donate.
    if atom.GetFormalCharge() > 0 and atom.GetTotalNumHs() == 0 and atom.GetDegree() >= 4:
        return False
    if atom.GetFormalCharge() > 0 and sym == "N" and atom.GetDegree() == 4:
        return False
    # Any positively charged N/O has no lone pair available to donate,
    # regardless of H count (covers ammonium, oxonium, etc.).
    if atom.GetFormalCharge() > 0 and sym in ("N", "O"):
        return False
    return True


def _carboxylate_like_partner(atom) -> int | None:
    """For a donor O, return the carbon index if it belongs to a carboxylate/
    nitro-like group (two donor O on one C/N), so the pair collapses to one site."""
    if atom.GetSymbol() != "O":
        return None
    for nbr in atom.GetNeighbors():
        if nbr.GetSymbol() in ("C", "N", "S", "P"):
            o_count = sum(1 for a in nbr.GetNeighbors() if a.GetSymbol() == "O")
            if o_count >= 2:
                return nbr.GetIdx()
    return None


def coordination_profile(smiles_or_mol) -> CoordinationProfile:
    mol = smiles_or_mol if isinstance(smiles_or_mol, Chem.Mol) else Chem.MolFromSmiles(smiles_or_mol)
    if mol is None:
        raise ValueError("Invalid SMILES")
    smiles = Chem.MolToSmiles(mol)

    donors = [a for a in mol.GetAtoms() if _is_donor(a)]
    donor_indices = tuple(a.GetIdx() for a in donors)

    element_counts: dict[str, int] = {}
    for a in donors:
        element_counts[a.GetSymbol()] = element_counts.get(a.GetSymbol(), 0) + 1

    # Collapse multi-oxygen sites (carboxylate, nitro, sulfonate) into one site.
    sites: list[list[int]] = []
    grouped: dict[int, int] = {}  # central atom idx -> site index
    for a in donors:
        central = _carboxylate_like_partner(a)
        if central is not None:
            if central in grouped:
                sites[grouped[central]].append(a.GetIdx())
            else:
                grouped[central] = len(sites)
                sites.append([a.GetIdx()])
        else:
            sites.append([a.GetIdx()])

    site_elements = tuple(mol.GetAtomWithIdx(s[0]).GetSymbol() for s in sites)

    # Chelate ring sizes for donor pairs (metal + through-bond path).
    ring_sizes: list[int] = []
    for i in range(len(donor_indices)):
        for j in range(i + 1, len(donor_indices)):
            path = Chem.GetShortestPath(mol, donor_indices[i], donor_indices[j])
            if path:
                size = len(path) + 1  # the two donors + intervening atoms + metal
                if 4 <= size <= 7:
                    ring_sizes.append(size)

    softness = [DONOR_SOFTNESS[a.GetSymbol()] for a in donors]
    mean_soft = sum(softness) / len(softness) if softness else 0.0

    return CoordinationProfile(
        smiles=smiles,
        donor_atom_indices=donor_indices,
        donor_element_counts=element_counts,
        n_donor_atoms=len(donors),
        denticity=len(sites),
        donor_site_elements=site_elements,
        chelate_ring_sizes=tuple(sorted(ring_sizes)),
        mean_donor_softness=mean_soft,
    )
