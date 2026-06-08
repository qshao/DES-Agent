from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from .run_memory import load_run_memory

TOP_RANK_LIMIT = 5


@dataclass(frozen=True)
class CompareRow:
    smiles_b: str
    left_rank: int | None
    right_rank: int | None
    status: str


@dataclass(frozen=True)
class CompareResult:
    workflow: str
    left_path: Path
    right_path: Path
    left_component_a: str | None
    right_component_a: str | None
    left_n: int | None
    right_n: int | None
    rows: list[CompareRow]


def _rank_map(memory) -> dict[str, int]:
    return {candidate.smiles_b: candidate.rank for candidate in memory.ranked_candidates}


def _top_ranked_smiles(memory, limit: int = TOP_RANK_LIMIT) -> set[str]:
    return {candidate.smiles_b for candidate in sorted(memory.ranked_candidates, key=lambda item: item.rank)[:limit]}


def _format_rank(rank: int | None) -> str:
    return "-" if rank is None else str(rank)


def compare_saved_runs(left: str | Path, right: str | Path) -> CompareResult:
    left_path = Path(left)
    right_path = Path(right)
    left_memory = load_run_memory(left_path)
    right_memory = load_run_memory(right_path)
    if left_memory.workflow != right_memory.workflow:
        raise ValueError("compare-runs requires both saved runs to have the same workflow")
    if not left_memory.ranked_candidates or not right_memory.ranked_candidates:
        raise ValueError("compare-runs requires saved runs with ranked candidates")

    left_ranks = _rank_map(left_memory)
    right_ranks = _rank_map(right_memory)
    left_top = _top_ranked_smiles(left_memory)
    right_top = _top_ranked_smiles(right_memory)
    all_top_smiles = sorted(
        left_top | right_top,
        key=lambda smiles: (
            min(
                left_ranks.get(smiles, TOP_RANK_LIMIT + 1),
                right_ranks.get(smiles, TOP_RANK_LIMIT + 1),
            ),
            smiles,
        ),
    )

    rows: list[CompareRow] = []
    for smiles_b in all_top_smiles:
        left_rank = left_ranks.get(smiles_b)
        right_rank = right_ranks.get(smiles_b)
        if left_rank is None:
            status = "new"
        elif right_rank is None:
            status = "removed"
        elif left_rank == right_rank:
            status = "unchanged"
        else:
            status = "moved"
        rows.append(
            CompareRow(
                smiles_b=smiles_b,
                left_rank=left_rank,
                right_rank=right_rank,
                status=status,
            )
        )

    return CompareResult(
        workflow=left_memory.workflow,
        left_path=left_path,
        right_path=right_path,
        left_component_a=left_memory.component_a,
        right_component_a=right_memory.component_a,
        left_n=left_memory.n,
        right_n=right_memory.n,
        rows=rows,
    )


def format_compare_report(result: CompareResult) -> str:
    lines = [f"compare-runs: {result.workflow}"]
    lines.append(f"left: {result.left_path}")
    lines.append(f"right: {result.right_path}")
    if result.left_component_a is not None or result.right_component_a is not None:
        lines.append(
            f"component_a: {result.left_component_a or '-'} -> {result.right_component_a or '-'}"
        )
    if result.left_n is not None or result.right_n is not None:
        lines.append(f"n: {result.left_n if result.left_n is not None else '-'} -> {result.right_n if result.right_n is not None else '-'}")
    lines.append("")
    lines.append("smiles_b | status | left_rank | right_rank")
    for row in result.rows:
        lines.append(
            f"{row.smiles_b} | {row.status} | {_format_rank(row.left_rank)} | {_format_rank(row.right_rank)}"
        )
    return "\n".join(lines)


def format_compare_json(result: CompareResult) -> dict[str, object]:
    counts = {"new": 0, "removed": 0, "moved": 0, "unchanged": 0}
    changed_candidates: list[dict[str, object]] = []
    for row in result.rows:
        counts[row.status] += 1
        if row.status != "unchanged":
            changed_candidates.append(
                {
                    "smiles_b": row.smiles_b,
                    "status": row.status,
                    "left_rank": row.left_rank,
                    "right_rank": row.right_rank,
                }
            )
    return {
        "workflow": result.workflow,
        "left": {
            "path": str(result.left_path),
            "component_a": result.left_component_a,
            "n": result.left_n,
        },
        "right": {
            "path": str(result.right_path),
            "component_a": result.right_component_a,
            "n": result.right_n,
        },
        "counts": counts,
        "changed_candidates": changed_candidates,
    }


def format_compare_json_text(result: CompareResult) -> str:
    return json.dumps(format_compare_json(result), indent=2, sort_keys=True)
