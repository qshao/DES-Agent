"""G1 — Cross-run leaderboard.

Reads run.json files from a history directory, deduplicates compounds by
canonical SMILES, and returns a ranked list of the best prediction per compound
across all runs.
"""
from __future__ import annotations

import json
from pathlib import Path

from .chemistry_filter import canonicalize_smiles
from .smiles_names import display_name


def build_leaderboard(history_dir: str | Path) -> list[dict]:
    """Scan *history_dir* for run.json files and build a compound leaderboard.

    Each entry in the returned list represents a unique compound (canonical
    SMILES) and carries the best (lowest) min_tm_k seen across all runs, plus
    how many runs contained that compound.

    Returns entries sorted by min_tm_k ascending (best DES-formers first).
    Raises FileNotFoundError if history_dir does not exist.
    """
    root = Path(history_dir)
    if not root.exists():
        raise FileNotFoundError(f"History directory not found: {history_dir}")

    # canonical_smiles → {"min_tm_k", "is_des", "run_count", "trust_score",
    #                     "uncertainty_flag", "smiles_b", "source"}
    best: dict[str, dict] = {}

    for run_json in sorted(root.rglob("run.json")):
        try:
            payload = json.loads(run_json.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if payload.get("workflow") != "des":
            continue
        for result in payload.get("results", []):
            smiles_b = str(result.get("smiles_b", "")).strip()
            if not smiles_b:
                continue
            try:
                canonical = canonicalize_smiles(smiles_b)
            except ValueError:
                canonical = smiles_b
            min_tm_k = float(result.get("min_tm_k", float("inf")))
            prev_count = best[canonical]["run_count"] if canonical in best else 0
            if canonical not in best or min_tm_k < best[canonical]["min_tm_k"]:
                best[canonical] = {
                    "smiles_b": smiles_b,
                    "canonical": canonical,
                    "min_tm_k": min_tm_k,
                    "is_des": bool(result.get("is_des", False)),
                    "source": str(result.get("source", "heuristic")),
                    "trust_score": result.get("trust_score"),
                    "uncertainty_flag": str(result.get("uncertainty_flag", "unknown")),
                    "run_count": prev_count,
                }
            best[canonical]["run_count"] += 1

    entries = sorted(best.values(), key=lambda e: e["min_tm_k"])
    for i, entry in enumerate(entries, start=1):
        entry["rank"] = i
    return entries


def format_leaderboard(entries: list[dict], *, resolve_names: bool = True) -> str:
    """Render leaderboard entries as a pipe-delimited text table."""
    if not entries:
        return "No results found."

    lines = ["rank | compound | min_tm_k | is_des | trust | uncertainty | runs"]
    for e in entries:
        smiles_b = e["smiles_b"]
        label = display_name(smiles_b) if resolve_names else smiles_b
        trust = f"{e['trust_score']:.2f}" if e.get("trust_score") is not None else "—"
        lines.append(
            f"{e['rank']} | {label} | {e['min_tm_k']:.2f} K | {e['is_des']} | "
            f"{trust} | {e['uncertainty_flag']} | {e['run_count']}"
        )
    return "\n".join(lines)
