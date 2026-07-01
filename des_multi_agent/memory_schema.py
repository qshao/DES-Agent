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
    # Cross-run learned signals persisted for reuse on subsequent searches
    accumulated_family_scores: dict[str, list[float]] | None = None  # family → [min_tm_k] for hits
    accumulated_family_hit_counts: dict[str, int] | None = None      # family → DES-positive count
    accumulated_family_fail_counts: dict[str, int] | None = None     # family → DES-negative count
    scaffold_counts: dict[str, dict] | None = None                   # scaffold_smi → {"hit": int, "fail": int}
    fg_hit_counts: dict[str, int] | None = None                      # fg_tag → count in DES hits
    fg_fail_counts: dict[str, int] | None = None                     # fg_tag → count in DES failures
