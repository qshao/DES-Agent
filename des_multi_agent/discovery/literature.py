from __future__ import annotations

from ..chemistry_filter import canonicalize_smiles
from ..schemas import CandidateProposal
from .library import DiscoveryLibrary


def literature_lookup(component_a: str, library: DiscoveryLibrary) -> list[CandidateProposal]:
    canonical_a = canonicalize_smiles(component_a)
    hits: list[CandidateProposal] = []
    for record in library.literature:
        if canonicalize_smiles(record.component_a) != canonical_a:
            continue
        hits.append(
            CandidateProposal(
                smiles=record.component_b,
                rationale=record.note or f"Local literature match from {record.reference_id or record.source}",
                family="literature",
                source="literature",
                source_id=record.reference_id or record.source,
                reference_note=record.note,
            )
        )
    return hits
