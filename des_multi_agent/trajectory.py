"""Readable iteration-trajectory model, renderers, and artifact writer.

Workflow-agnostic: this module is imported BY workflows, never the reverse.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile


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


def _converged_phrase(converged: bool, reason: str) -> str:
    if converged:
        return f"yes ({reason})" if reason else "yes"
    return "no"


def _format_top_list(snapshot: CycleSnapshot, metric_label: str) -> list[str]:
    if not snapshot.top_entries:
        return ["_No hits this cycle._"]
    out = [f"Top by {metric_label}:"]
    for i, e in enumerate(snapshot.top_entries, 1):
        tail = f"  ({e.secondary})" if e.secondary else ""
        out.append(f"{i}. {e.label} — {e.metric_value:.1f}{tail}")
    return out


def _format_change_line(snapshot: CycleSnapshot) -> str | None:
    if snapshot.cycle <= 1:
        return None
    if not snapshot.new_entrants and not snapshot.dropouts:
        return "Shortlist change vs previous cycle: none — shortlist identical"
    parts = []
    if snapshot.new_entrants:
        parts.append(f"+{len(snapshot.new_entrants)} entered ({', '.join(snapshot.new_entrants)})")
    if snapshot.dropouts:
        parts.append(f"-{len(snapshot.dropouts)} left ({', '.join(snapshot.dropouts)})")
    return "Shortlist change vs previous cycle: " + ", ".join(parts)


def format_trajectory_report(traj: SearchTrajectory) -> str:
    lines = [f"# Search Trajectory — {traj.headline}", ""]
    lines.append(
        f"Workflow: {traj.workflow}  ·  Cycles run: {traj.total_cycles}  ·  "
        f"Converged: {_converged_phrase(traj.converged, traj.convergence_reason)}"
    )
    if not traj.snapshots:
        lines += ["", "_No cycles recorded._"]
        return "\n".join(lines)
    for s in traj.snapshots:
        lines.append("")
        suffix = "  ✓ converged" if s.converged else ""
        lines.append(f"## Cycle {s.cycle} — {s.n_screened} screened, {s.n_hits} hits{suffix}")
        change = _format_change_line(s)
        if change:
            lines.append(change)
        lines += _format_top_list(s, traj.metric_label)
        if s.family_ledger:
            fam = ", ".join(f"{k} ({v})" for k, v in s.family_ledger.items())
            lines.append(f"Families reinforced: {fam}")
        if s.converged and s.convergence_reason:
            lines.append(f"Converged: {s.convergence_reason}")
        if s.notable_warnings:
            lines.append("> warnings:")
            lines += [f"> - {w}" for w in s.notable_warnings]
    lines += ["", "## Final shortlist"]
    if traj.final_summary:
        for i, e in enumerate(traj.final_summary, 1):
            lines.append(f"{i}. {e.label} — {traj.metric_label} {e.metric_value:.1f}")
    else:
        lines.append("_No final results._")
    return "\n".join(lines) + "\n"


def format_trajectory_console(traj: SearchTrajectory) -> str:
    conv = "converged" if traj.converged else "ran to budget"
    out = [f"Trajectory — {traj.headline}  ({traj.total_cycles} cycles, {conv})"]
    for s in traj.snapshots:
        top = ""
        if s.top_entries:
            e = s.top_entries[0]
            top = f" · top: {e.label} {e.metric_value:.1f}"
        if s.cycle <= 1:
            change = f"{s.n_screened} screened, {s.n_hits} hits"
        elif not s.new_entrants and not s.dropouts:
            change = "stable"
        else:
            change = f"+{len(s.new_entrants)}/-{len(s.dropouts)} shortlist"
        fam = ""
        if s.family_ledger:
            fam = " · families: " + ", ".join(s.family_ledger.keys())
        conv_tag = " ✓ converged" if s.converged else ""
        out.append(f"  cycle {s.cycle}: {change}{top}{fam}{conv_tag}")
    return "\n".join(out)


def write_trajectory_artifact(output_dir: str | Path, traj: SearchTrajectory) -> Path:
    """Atomically write trajectory.md into output_dir; return its path."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / "trajectory.md"
    content = format_trajectory_report(traj)
    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=out_dir, prefix=".trajectory-", suffix=".md", delete=False
    ) as fh:
        fh.write(content)
        staged = Path(fh.name)
    staged.replace(final_path)
    return final_path


def write_trajectory_json_artifact(output_dir: str | Path, traj: SearchTrajectory) -> Path:
    """Atomically write trajectory.json into output_dir; return its path."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / "trajectory.json"
    content = json.dumps(asdict(traj), indent=2, sort_keys=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=out_dir, prefix=".trajectory-", suffix=".json", delete=False
    ) as fh:
        fh.write(content)
        staged = Path(fh.name)
    staged.replace(final_path)
    return final_path
