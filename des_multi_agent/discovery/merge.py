from __future__ import annotations

from ..chemistry_filter import canonicalize_smiles
from ..schemas import CandidateProposal


def merge_discovery_candidates(*candidate_groups) -> list[CandidateProposal]:
    merged: list[CandidateProposal] = []
    seen: set[str] = set()
    for group in candidate_groups:
        for candidate in group:
            canonical = canonicalize_smiles(candidate.smiles)
            if canonical in seen:
                continue
            seen.add(canonical)
            merged.append(candidate)
    return merged
