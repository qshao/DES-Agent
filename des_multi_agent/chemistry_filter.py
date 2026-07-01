from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from .schemas import CandidateProposal


def murcko_scaffold_smiles(smiles: str) -> str | None:
    """Return the canonical Murcko scaffold SMILES for a molecule.

    For acyclic compounds the full canonical SMILES is returned (the molecule
    itself is its own scaffold).  Returns None on any error; never raises.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold.GetNumAtoms() == 0:
            return Chem.MolToSmiles(mol, canonical=True)
        return Chem.MolToSmiles(scaffold, canonical=True)
    except Exception:
        return None


_ALLOWED_ATOMS = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}

# Precompiled SMARTS for reactive/unstable groups and toxicophores.
# Molecules matching any of these are rejected as DES component candidates.
_VIABILITY_SMARTS: list[tuple[str, str]] = [
    ("[OX2][OX2]",                 "peroxide bond"),
    ("[N]=[N+]=[N-]",              "organic azide"),
    ("[c,C][N+]#N",                "diazonium salt"),
    ("C(=[OX1])[F,Cl,Br,I]",      "acyl halide"),
    ("[SX4](=[OX1])(=[OX1])[Cl]", "sulfonyl chloride"),
    ("[N+](=O)[O-]",               "nitro group"),
]
_VIABILITY_PATTERNS: list[tuple[object, str]] = [
    (Chem.MolFromSmarts(smarts), name)
    for smarts, name in _VIABILITY_SMARTS
]

# Synthesizability thresholds for DES components.
_MAX_RINGS = 4
_MAX_STEREOCENTERS = 3


@dataclass(frozen=True)
class FilteredCandidate:
    proposal: CandidateProposal
    reason: str


def canonicalize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True)


def _is_chemically_plausible(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in _ALLOWED_ATOMS:
            return False
    return True


def viability_check(mol: Chem.Mol) -> tuple[bool, str]:
    """Check that a molecule is stable, non-toxic, and synthesizable as a DES component.

    Rejects reactive/unstable groups (peroxides, azides, diazonium, acyl halides,
    sulfonyl chlorides, nitro groups) and structures too complex to synthesize
    (ring count > _MAX_RINGS or stereocenters > _MAX_STEREOCENTERS).

    Returns (ok, reason). Never raises; on any internal error returns (True, "").
    """
    try:
        for patt, name in _VIABILITY_PATTERNS:
            if patt is None:
                continue
            if mol.HasSubstructMatch(patt):
                return False, f"contains {name}"
        n_rings = rdMolDescriptors.CalcNumRings(mol)
        if n_rings > _MAX_RINGS:
            return False, f"too complex ({n_rings} rings; DES max {_MAX_RINGS})"
        stereo = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        if len(stereo) > _MAX_STEREOCENTERS:
            return False, f"too many stereocenters ({len(stereo)}; DES max {_MAX_STEREOCENTERS})"
    except Exception:
        pass
    return True, ""


def filter_candidates(
    component_a: str,
    candidates: Iterable[CandidateProposal],
    failing_scaffolds: set[str] | None = None,
):
    mol_a = Chem.MolFromSmiles(component_a)
    if mol_a is None:
        raise ValueError("Invalid component A SMILES")

    canonical_component_a = canonicalize_smiles(component_a)
    filtered: list[CandidateProposal] = []
    seen_smiles: set[str] = set()
    for proposal in candidates:
        mol = Chem.MolFromSmiles(proposal.smiles)
        if mol is None:
            continue
        if any(atom.GetAtomicNum() not in _ALLOWED_ATOMS for atom in mol.GetAtoms()):
            continue
        ok, _ = viability_check(mol)
        if not ok:
            continue
        canonical_candidate = Chem.MolToSmiles(mol, canonical=True)
        if canonical_candidate == canonical_component_a:
            continue
        if canonical_candidate in seen_smiles:
            continue
        if failing_scaffolds:
            sc = murcko_scaffold_smiles(canonical_candidate)
            if sc and sc in failing_scaffolds:
                continue
        seen_smiles.add(canonical_candidate)
        filtered.append(proposal)
    return filtered
