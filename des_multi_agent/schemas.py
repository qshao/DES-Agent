from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateProposal:
    smiles: str
    rationale: str
    family: str
    source: str = "heuristic"
    source_id: str = ""
    similarity_score: float | None = None
    reference_note: str = ""


@dataclass(frozen=True)
class MeltingPointEstimate:
    component: str
    tm_k: float
    source: str
    confidence: float


@dataclass(frozen=True)
class DesThresholds:
    absolute_tm_max_k: float
    relative_drop_min: float
