"""E2 — Run history viewer.

Reads run.manifest.json + run.json pairs from a history directory and builds
a summary table of past runs.
"""
from __future__ import annotations

import json
from pathlib import Path


def build_history_table(history_dir: str | Path) -> list[dict]:
    """Scan *history_dir* for run.manifest.json files and build a history table.

    Each row summarises one run: name, date, candidates screened, DES-formers
    found, and the top candidate with its min Tm.

    Raises FileNotFoundError if history_dir does not exist.
    """
    root = Path(history_dir)
    if not root.exists():
        raise FileNotFoundError(f"History directory not found: {history_dir}")

    rows: list[dict] = []
    for manifest_path in sorted(root.rglob("run.manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        run_dir = manifest_path.parent
        run_name = run_dir.name

        # Read run.json for detailed result counts
        results: list[dict] = []
        run_json_path = run_dir / manifest.get("json_filename", "run.json")
        if run_json_path.exists():
            try:
                payload = json.loads(run_json_path.read_text(encoding="utf-8"))
                results = payload.get("results", [])
            except (json.JSONDecodeError, OSError):
                pass

        n_screened = len(results)
        n_des = sum(1 for r in results if r.get("is_des", False))
        des_results = [r for r in results if r.get("is_des", False)]
        if des_results:
            best = min(des_results, key=lambda r: float(r.get("min_tm_k", float("inf"))))
            top_candidate = best.get("smiles_b", "—")
            top_min_tm_k: float | None = float(best.get("min_tm_k", 0.0))
        else:
            top_candidate = "—"
            top_min_tm_k = None

        rows.append({
            "run_name": run_name,
            "exported_at_utc": manifest.get("exported_at_utc", "unknown"),
            "component_a": manifest.get("component_a", "?"),
            "n_screened": n_screened,
            "n_des": n_des,
            "top_candidate": top_candidate,
            "top_min_tm_k": top_min_tm_k,
        })

    return rows


def format_history_table(rows: list[dict]) -> str:
    """Render history rows as a pipe-delimited text table."""
    if not rows:
        return "No run history found."

    lines = ["run_name | exported_at_utc | component_a | n_screened | n_des | top_candidate | top_min_tm_k"]
    for row in rows:
        tm = f"{row['top_min_tm_k']:.2f} K" if row["top_min_tm_k"] is not None else "—"
        lines.append(
            f"{row['run_name']} | {row['exported_at_utc']} | {row['component_a']} | "
            f"{row['n_screened']} | {row['n_des']} | {row['top_candidate']} | {tm}"
        )
    return "\n".join(lines)
