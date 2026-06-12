from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed {label}: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return data


def build_run_view(run_dir: str | Path, top_n: int = 5) -> dict[str, Any]:
    root = Path(run_dir)
    if not root.exists():
        raise FileNotFoundError(f"Run directory not found: {root}")
    if not root.is_dir():
        raise ValueError(f"Run path must be a directory: {root}")
    manifest = _read_json(root / "run.manifest.json", "run manifest")
    run_json = _read_json(root / str(manifest.get("json_filename", "run.json")), "run json")
    results = run_json.get("results", [])
    if not isinstance(results, list):
        raise ValueError("run.json field 'results' must be a list")
    memory_path = root / "run.memory.json"
    label_count = 0
    if memory_path.exists():
        memory = _read_json(memory_path, "run memory")
        labels = memory.get("labels", [])
        if isinstance(labels, list):
            label_count = len(labels)
    return {
        "run_dir": root,
        "workflow": manifest.get("workflow", run_json.get("workflow", "unknown")),
        "component_a": manifest.get("component_a", run_json.get("component_a", "unknown")),
        "n": manifest.get("n", run_json.get("n")),
        "candidate_count": len(results),
        "label_count": label_count,
        "top_candidates": results[:top_n],
        "report_path": root / str(manifest.get("report_filename", "report.txt")),
        "json_path": root / str(manifest.get("json_filename", "run.json")),
        "csv_path": root / str(manifest.get("csv_filename", "run.csv")),
        "manifest_path": root / "run.manifest.json",
    }


def format_run_view(view: dict[str, Any]) -> str:
    lines = [
        f"run: {view['run_dir']}",
        f"workflow: {view['workflow']}",
        f"component_a: {view['component_a']}",
        f"requested_n: {view['n']}",
        f"candidates: {view['candidate_count']}",
        f"memory_labels: {view['label_count']}",
        "top_candidates:",
    ]
    top_candidates = view.get("top_candidates", [])
    if not top_candidates:
        lines.append("- none")
    for item in top_candidates:
        if not isinstance(item, dict):
            continue
        rank = item.get("rank", "?")
        smiles = item.get("smiles_b", "?")
        is_des = item.get("is_des", "?")
        min_tm = item.get("min_tm_k", "?")
        lines.append(f"- rank={rank} smiles_b={smiles} is_des={is_des} min_tm_k={min_tm}")
    lines.extend([
        "artifacts:",
        f"- report: {view['report_path']}",
        f"- json: {view['json_path']}",
        f"- csv: {view['csv_path']}",
        f"- manifest: {view['manifest_path']}",
    ])
    return "\n".join(lines)
