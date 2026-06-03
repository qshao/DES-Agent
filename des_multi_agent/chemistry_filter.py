from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from rdkit import Chem

from .schemas import CandidateProposal


_ALLOWED_ATOMS = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}


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


def filter_candidates(component_a: str, candidates: Iterable[CandidateProposal]):
    mol_a = Chem.MolFromSmiles(component_a)
    if mol_a is None:
        raise ValueError("Invalid component A SMILES")

    canonical_component_a = canonicalize_smiles(component_a)
    filtered: list[CandidateProposal] = []
    seen_smiles: set[str] = set()
    for proposal in candidates:
        if not _is_chemically_plausible(proposal.smiles):
            continue
        canonical_candidate = canonicalize_smiles(proposal.smiles)
        if canonical_candidate == canonical_component_a:
            continue
        if canonical_candidate in seen_smiles:
            continue
        seen_smiles.add(canonical_candidate)
        filtered.append(proposal)
    return filtered
