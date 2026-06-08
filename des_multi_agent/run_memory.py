from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
import json

from .memory_schema import RunCandidateSummary, RunLabel, RunMemory
from .schemas import CandidateProposal
from .uncertainty import rank_annotated_results
from .uncertainty.schemas import AnnotatedResult


def resolve_run_memory_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "run.memory.json"
    if not candidate.exists():
        raise FileNotFoundError(f"Run memory file not found: {candidate}")
    return candidate


def parse_run_memory(data: Mapping[str, object]) -> RunMemory:
    if data.get("workflow") != "des":
        raise ValueError("run memory workflow must be des")
    labels: list[RunLabel] = []
    for item in data.get("labels", []):
        labels.append(RunLabel(smiles_b=item["smiles_b"], label=item["label"]))
    ranked_candidates: list[RunCandidateSummary] = []
    for item in data.get("ranked_candidates", []):
        ranked_candidates.append(
            RunCandidateSummary(
                smiles_b=item["smiles_b"],
                rank=int(item["rank"]),
                min_tm_k=item.get("min_tm_k"),
                trust_score=item.get("trust_score"),
                uncertainty_flag=item.get("uncertainty_flag", ""),
                source=item.get("source", ""),
                source_id=item.get("source_id", ""),
            )
        )
    return RunMemory(
        workflow="des",
        component_a=data.get("component_a"),
        n=data.get("n"),
        labels=labels,
        ranked_candidates=ranked_candidates,
    )


def load_run_memory(path: str | Path) -> RunMemory:
    memory_path = resolve_run_memory_path(path)
    data = json.loads(memory_path.read_text(encoding="utf-8"))
    return parse_run_memory(data)


def write_run_memory(path: str | Path, memory: RunMemory) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(memory)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def build_run_memory(
    component_a: str,
    n: int,
    annotated_results: list[AnnotatedResult],
    candidate_proposals: list[CandidateProposal],
    labels: list[RunLabel] | None = None,
) -> RunMemory:
    proposal_by_smiles = {proposal.smiles: proposal for proposal in candidate_proposals}
    ranked_candidates: list[RunCandidateSummary] = []
    for rank, item in enumerate(annotated_results, start=1):
        proposal = proposal_by_smiles.get(item.result.curve.smiles_b)
        ranked_candidates.append(
            RunCandidateSummary(
                smiles_b=item.result.curve.smiles_b,
                rank=rank,
                min_tm_k=item.result.min_tm_k,
                trust_score=item.trust_score,
                uncertainty_flag=item.uncertainty.uncertainty_flag,
                source=proposal.source if proposal is not None else "",
                source_id=proposal.source_id if proposal is not None else "",
            )
        )
    return RunMemory(
        workflow="des",
        component_a=component_a,
        n=n,
        labels=list(labels or []),
        ranked_candidates=ranked_candidates,
    )


def apply_run_memory_preferences(
    annotated_results: list[AnnotatedResult],
    memory: RunMemory | None,
    component_a: str,
) -> tuple[list[AnnotatedResult], list[str]]:
    if memory is None:
        return list(annotated_results), []
    if memory.component_a is not None and memory.component_a != component_a:
        return list(annotated_results), [
            f"Reuse memory ignored because it was recorded for {memory.component_a}, not {component_a}."
        ]
    preferred = {item.smiles_b for item in memory.labels if item.label == "good"}
    penalized = {item.smiles_b for item in memory.labels if item.label == "bad"}
    ranked_bonus = {
        item.smiles_b: max(0.0, 0.08 - 0.01 * (item.rank - 1)) for item in memory.ranked_candidates
    }
    adjusted: list[AnnotatedResult] = []
    for item in annotated_results:
        smiles_b = item.result.curve.smiles_b
        bonus = 0.15 if smiles_b in preferred else ranked_bonus.get(smiles_b, 0.0)
        penalty = 0.15 if smiles_b in penalized else 0.0
        adjusted.append(replace(item, ranking_score=item.ranking_score + bonus - penalty))
    note_parts = []
    if preferred or penalized:
        note_parts.append(f"Applied reuse memory to {len(preferred)} preferred candidate and {len(penalized)} penalized candidates.")
    if memory.ranked_candidates:
        note_parts.append(f"Loaded {len(memory.ranked_candidates)} prior ranked candidates for ranking bias.")
    return rank_annotated_results(adjusted), note_parts


def update_run_memory_labels(memory: RunMemory, label_specs: list[tuple[str, str]]) -> RunMemory:
    if memory.workflow != "des":
        raise ValueError("run memory workflow must be des")
    valid_smiles = {candidate.smiles_b for candidate in memory.ranked_candidates}
    labels_by_smiles = {label.smiles_b: label.label for label in memory.labels}
    new_smiles_order: list[str] = []
    for smiles_b, label in label_specs:
        normalized_label = label.strip().lower()
        if normalized_label not in {"good", "bad"}:
            raise ValueError("label must be good or bad")
        if smiles_b not in valid_smiles:
            raise ValueError(f"SMILES {smiles_b} not found in the saved DES run")
        if smiles_b not in labels_by_smiles and smiles_b not in new_smiles_order:
            new_smiles_order.append(smiles_b)
        labels_by_smiles[smiles_b] = normalized_label

    merged_labels: list[RunLabel] = []
    seen: set[str] = set()
    for label in memory.labels:
        if label.smiles_b in seen:
            continue
        seen.add(label.smiles_b)
        merged_labels.append(RunLabel(smiles_b=label.smiles_b, label=labels_by_smiles[label.smiles_b]))
    for smiles_b in new_smiles_order:
        if smiles_b in seen:
            continue
        merged_labels.append(RunLabel(smiles_b=smiles_b, label=labels_by_smiles[smiles_b]))
    return replace(memory, labels=merged_labels)
