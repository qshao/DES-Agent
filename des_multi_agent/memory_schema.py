from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunLabel:
    smiles_b: str
    label: str


@dataclass(frozen=True)
class RunCandidateSummary:
    smiles_b: str
    rank: int
    min_tm_k: float | None = None
    trust_score: float | None = None
    uncertainty_flag: str = ""
    source: str = ""
    source_id: str = ""


@dataclass(frozen=True)
class RunMemory:
    workflow: str
    component_a: str | None
    n: int | None
    labels: list[RunLabel]
    ranked_candidates: list[RunCandidateSummary]
