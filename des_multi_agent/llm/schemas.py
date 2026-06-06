from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CandidateBrainstorm:
    smiles: str
    rationale: str
    family: str


@dataclass(frozen=True)
class CandidateReview:
    smiles: str
    decision: str
    confidence: float
    rationale: str
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ExplanationNote:
    smiles: str
    summary: str
    evidence: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CritiqueNote:
    smiles: str
    assessment: str
    concerns: list[str] = field(default_factory=list)
