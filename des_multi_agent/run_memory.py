from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import NamedTemporaryFile
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


def resolve_run_memory_history_paths(path: str | Path) -> list[Path]:
    candidate = Path(path)
    if candidate.is_file():
        return [candidate]
    if candidate.is_dir():
        direct = candidate / "run.memory.json"
        if direct.exists():
            return [direct]
        files = sorted(candidate.rglob("run.memory.json"))
        if files:
            return files
    raise FileNotFoundError(f"Run memory file not found: {candidate}")


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping, got {type(value).__name__}")
    return value


def _require_field(data: Mapping[str, object], field: str, label: str) -> object:
    if field not in data:
        raise ValueError(f"{label} missing required field: {field}")
    return data[field]


def _require_list(data: Mapping[str, object], field: str) -> list[object]:
    value = data.get(field, [])
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return value


def _optional_float(value: object, label: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number or null") from exc


def parse_run_memory(data: Mapping[str, object]) -> RunMemory:
    if not isinstance(data, Mapping):
        raise ValueError("run memory must be a mapping")
    if data.get("workflow") != "des":
        raise ValueError("run memory workflow must be des")
    labels: list[RunLabel] = []
    for index, raw_item in enumerate(_require_list(data, "labels")):
        item = _require_mapping(raw_item, f"labels[{index}]")
        smiles_b = _require_field(item, "smiles_b", f"labels[{index}]")
        label = _require_field(item, "label", f"labels[{index}]")
        if not isinstance(smiles_b, str) or not smiles_b.strip():
            raise ValueError(f"labels[{index}].smiles_b must be a non-empty string")
        if label not in {"good", "bad"}:
            raise ValueError(f"labels[{index}].label must be good or bad")
        labels.append(RunLabel(smiles_b=smiles_b, label=str(label)))
    ranked_candidates: list[RunCandidateSummary] = []
    for index, raw_item in enumerate(_require_list(data, "ranked_candidates")):
        item = _require_mapping(raw_item, f"ranked_candidates[{index}]")
        smiles_b = _require_field(item, "smiles_b", f"ranked_candidates[{index}]")
        rank_value = _require_field(item, "rank", f"ranked_candidates[{index}]")
        if not isinstance(smiles_b, str) or not smiles_b.strip():
            raise ValueError(f"ranked_candidates[{index}].smiles_b must be a non-empty string")
        try:
            rank = int(rank_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"ranked_candidates[{index}].rank must be an integer") from exc
        ranked_candidates.append(
            RunCandidateSummary(
                smiles_b=smiles_b,
                rank=rank,
                min_tm_k=_optional_float(item.get("min_tm_k"), f"ranked_candidates[{index}].min_tm_k"),
                trust_score=_optional_float(item.get("trust_score"), f"ranked_candidates[{index}].trust_score"),
                uncertainty_flag=str(item.get("uncertainty_flag", "")),
                source=str(item.get("source", "")),
                source_id=str(item.get("source_id", "")),
            )
        )
    raw_n = data.get("n")
    try:
        parsed_n = int(raw_n) if raw_n is not None else None
    except (TypeError, ValueError) as exc:
        raise ValueError("n must be an integer or null") from exc
    raw_component_a = data.get("component_a")
    component_a = raw_component_a if isinstance(raw_component_a, str) or raw_component_a is None else str(raw_component_a)
    return RunMemory(
        workflow="des",
        component_a=component_a,
        n=parsed_n,
        labels=labels,
        ranked_candidates=ranked_candidates,
    )


def load_run_memory(path: str | Path) -> RunMemory:
    memory_path = resolve_run_memory_path(path)
    data = json.loads(memory_path.read_text(encoding="utf-8"))
    return parse_run_memory(data)


def load_run_memory_history(path: str | Path) -> list[RunMemory]:
    return [load_run_memory(memory_path) for memory_path in resolve_run_memory_history_paths(path)]


def write_run_memory(path: str | Path, memory: RunMemory) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(asdict(memory), indent=2, sort_keys=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=output_path.parent, delete=False) as handle:
        temp_path = Path(handle.name)
        try:
            handle.write(content)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
    try:
        temp_path.replace(output_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
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


def _iter_run_memories(memory: RunMemory | Sequence[RunMemory] | None) -> list[RunMemory]:
    if memory is None:
        return []
    if isinstance(memory, RunMemory):
        return [memory]
    return list(memory)


def build_chemistry_advisor_memory_notes(memory: RunMemory | Sequence[RunMemory] | None) -> list[str]:
    notes: list[str] = []
    for item in _iter_run_memories(memory):
        if item.labels:
            good = [label.smiles_b for label in item.labels if label.label == "good"]
            bad = [label.smiles_b for label in item.labels if label.label == "bad"]
            if good:
                notes.append(f"Prior good labels: {', '.join(good[:5])}")
            if bad:
                notes.append(f"Prior bad labels: {', '.join(bad[:5])}")
        if item.ranked_candidates:
            top = ", ".join(candidate.smiles_b for candidate in item.ranked_candidates[:3])
            notes.append(f"Prior top ranked candidates: {top}")
    return notes


def apply_run_memory_preferences(
    annotated_results: list[AnnotatedResult],
    memory: RunMemory | Sequence[RunMemory] | None,
    component_a: str,
) -> tuple[list[AnnotatedResult], list[str]]:
    memories = _iter_run_memories(memory)
    if not memories:
        return list(annotated_results), []

    good_counts: dict[str, int] = defaultdict(int)
    bad_counts: dict[str, int] = defaultdict(int)
    rank_bias_by_smiles: dict[str, float] = defaultdict(float)
    note_parts: list[str] = []
    matched_memories = 0

    for item in memories:
        if item.component_a is not None and item.component_a != component_a:
            note_parts.append(f"Reuse memory ignored because it was recorded for {item.component_a}, not {component_a}.")
            continue
        matched_memories += 1
        labels_by_smiles = {label.smiles_b: label.label for label in item.labels}
        for smiles_b, label in labels_by_smiles.items():
            if label == "good":
                good_counts[smiles_b] += 1
            elif label == "bad":
                bad_counts[smiles_b] += 1
        for candidate in item.ranked_candidates:
            if candidate.smiles_b in labels_by_smiles:
                continue
            rank_bias_by_smiles[candidate.smiles_b] += max(0.0, 0.08 - 0.01 * (candidate.rank - 1))

    if matched_memories == 0:
        return list(annotated_results), note_parts

    adjusted: list[AnnotatedResult] = []
    for item in annotated_results:
        smiles_b = item.result.curve.smiles_b
        good_bonus = min(0.30, 0.15 * good_counts.get(smiles_b, 0))
        bad_penalty = min(0.30, 0.15 * bad_counts.get(smiles_b, 0))
        rank_bonus = min(0.20, rank_bias_by_smiles.get(smiles_b, 0.0))
        adjusted.append(replace(item, ranking_score=item.ranking_score + good_bonus + rank_bonus - bad_penalty))

    ranked_adjusted = rank_annotated_results(adjusted)
    if good_counts or bad_counts:
        note_parts.append(
            f"Applied reuse memory to {sum(good_counts.values())} good label(s) and {sum(bad_counts.values())} bad label(s) across {matched_memories} run memory file(s)."
        )
    if rank_bias_by_smiles:
        note_parts.append(
            f"Loaded {len(rank_bias_by_smiles)} prior ranked candidate(s) for ranking bias across {matched_memories} run memory file(s)."
        )
    return ranked_adjusted, note_parts


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
