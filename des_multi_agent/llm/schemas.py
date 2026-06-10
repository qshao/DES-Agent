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


@dataclass(frozen=True)
class ContradictionNote:
    smiles: str
    agreement: str   # "agree" | "conflict" | "uncertain"
    explanation: str


@dataclass(frozen=True)
class CandidateFamily:
    name: str           # e.g., "polyols", "amides", "imidazolium salts"
    rationale: str      # why this family suits DES formation with component A
    hbd_hba_role: str   # "HBD", "HBA", or "both"


@dataclass(frozen=True)
class LigandFamily:
    name: str            # e.g., "aminocarboxylates", "catecholates"
    rationale: str       # why this family binds well to the target metal
    coordination_mode: str  # e.g., "bidentate N,O-chelator"
