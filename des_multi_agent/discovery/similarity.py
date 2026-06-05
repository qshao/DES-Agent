from __future__ import annotations

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from ..schemas import CandidateProposal
from .library import DiscoveryLibrary


def similarity_search(component_a: str, library: DiscoveryLibrary, limit: int) -> list[CandidateProposal]:
    mol_a = Chem.MolFromSmiles(component_a)
    if mol_a is None:
        raise ValueError("Invalid component A SMILES")
    fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, radius=2, nBits=2048)
    scored: list[tuple[float, CandidateProposal]] = []
    for record in library.candidate_library:
        mol_b = Chem.MolFromSmiles(record.smiles)
        if mol_b is None:
            continue
        fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, radius=2, nBits=2048)
        score = DataStructs.TanimotoSimilarity(fp_a, fp_b)
        scored.append(
            (
                score,
                CandidateProposal(
                    smiles=record.smiles,
                    rationale=record.note or f"Local similarity hit from {record.source}",
                    family=record.family,
                    source="similarity",
                    source_id=record.source,
                    similarity_score=float(score),
                    reference_note=record.note,
                ),
            )
        )
    scored.sort(key=lambda item: (-item[0], item[1].smiles))
    return [candidate for _, candidate in scored[:limit]]
