from __future__ import annotations

from pathlib import Path

from .run_memory import load_run_memory, resolve_run_memory_path, update_run_memory_labels, write_run_memory


def parse_label_specs(label_specs: list[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for spec in label_specs:
        if "=" not in spec:
            raise ValueError("label must be in the form SMILES=good or SMILES=bad")
        smiles_b, label = spec.rsplit("=", 1)
        smiles_b = smiles_b.strip()
        label = label.strip()
        if not smiles_b or not label:
            raise ValueError("label must be in the form SMILES=good or SMILES=bad")
        parsed.append((smiles_b, label))
    return parsed


def run_label_command(run_path: str | Path, label_specs: list[str]) -> str:
    if not label_specs:
        raise ValueError("label-run requires at least one --label")
    memory = load_run_memory(run_path)
    updated = update_run_memory_labels(memory, parse_label_specs(label_specs))
    memory_path = resolve_run_memory_path(run_path)
    write_run_memory(memory_path, updated)
    return f"Updated {memory_path} with {len(label_specs)} labels."
