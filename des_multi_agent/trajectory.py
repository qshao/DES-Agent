"""Readable iteration-trajectory model, renderers, and artifact writer.

Workflow-agnostic: this module is imported BY workflows, never the reverse.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TopEntry:
    """One ranked candidate as it stood in a given cycle."""
    label: str            # display: "acetamide (CC(=O)N)"
    metric_name: str      # "min_tm_k" | "composite_score"
    metric_value: float
    secondary: str        # short context, e.g. "Δ11.9%, high confidence"


@dataclass(frozen=True)
class CycleSnapshot:
    """Render-oriented record of one iteration."""
    cycle: int
    n_screened: int
    n_hits: int
    top_entries: list[TopEntry]
    new_entrants: list[str] = field(default_factory=list)
    dropouts: list[str] = field(default_factory=list)
    family_ledger: dict[str, int] = field(default_factory=dict)
    converged: bool = False
    convergence_reason: str = ""
    notable_warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SearchTrajectory:
    """Full readable record of an iterative run."""
    workflow: str
    headline: str
    metric_label: str
    snapshots: list[CycleSnapshot]
    total_cycles: int
    converged: bool
    convergence_reason: str
    final_summary: list[TopEntry]


def shortlist_delta(
    prev_labels: list[str], curr_labels: list[str]
) -> tuple[list[str], list[str]]:
    """Return (new_entrants, dropouts) between two shortlists of display labels."""
    prev_set, curr_set = set(prev_labels), set(curr_labels)
    return sorted(curr_set - prev_set), sorted(prev_set - curr_set)
