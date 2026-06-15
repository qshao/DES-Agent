from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

from .chemical_pattern_memory import ChemicalPatternMemory
from .memory_schema import RunMemory
from .run_memory import build_chemistry_advisor_memory_notes
from .schemas import CandidateProposal
from .uncertainty import AnnotatedResult


@dataclass(frozen=True)
class ChemistryLessonSummaryConfig:
    mode: str = "adaptive"
    max_examples: int = 3
    max_next_steps: int = 2
    strong_label_bonus: float = 0.20
    weak_pattern_bonus: float = 0.05

    def __post_init__(self) -> None:
        mode = self.mode.strip().lower()
        if mode not in {"off", "soft", "adaptive"}:
            raise ValueError("chemistry lesson summary mode must be off, soft, or adaptive")
        if self.max_examples < 0:
            raise ValueError("lesson summary max examples must be non-negative")
        if self.max_next_steps < 0:
            raise ValueError("lesson summary max next steps must be non-negative")
        object.__setattr__(self, "mode", mode)


@dataclass(frozen=True)
class ChemistryLessonSummary:
    productive_patterns: dict[str, int] = field(default_factory=dict)
    avoid_patterns: dict[str, int] = field(default_factory=dict)
    cycle_summary: list[str] = field(default_factory=list)
    run_summary: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    representative_examples: list[str] = field(default_factory=list)
    confidence: str = "low"
    notes: list[str] = field(default_factory=list)


def _proposal_family_map(candidate_proposals: Sequence[CandidateProposal]) -> dict[str, str]:
    return {
        proposal.smiles: proposal.family.strip()
        for proposal in candidate_proposals
        if proposal.smiles and proposal.family and proposal.family.strip()
    }


def _bounded_unique(values: Sequence[str], max_examples: int) -> list[str]:
    seen: set[str] = set()
    examples: list[str] = []
    for value in values:
        cleaned = value.strip()
        if not cleaned or cleaned in seen:
            continue
        examples.append(cleaned)
        seen.add(cleaned)
        if len(examples) >= max_examples:
            break
    return examples


def _matching_run_memories(run_memories: Sequence[RunMemory] | None, component_a: str) -> list[RunMemory]:
    matched: list[RunMemory] = []
    for memory in run_memories or []:
        if memory.component_a is None or memory.component_a == component_a:
            matched.append(memory)
    return matched


def _pattern_note(prefix: str, counts: Counter[str], limit: int = 3) -> str | None:
    if not counts:
        return None
    parts = [f"{family} ({count})" for family, count in counts.most_common(limit)]
    return f"{prefix}: {', '.join(parts)}."


def _build_next_steps(
    productive: Counter[str],
    avoid: Counter[str],
    max_next_steps: int,
) -> list[str]:
    steps: list[str] = []
    if productive:
        steps.append("Stay near the productive families and probe close analogs first.")
    if avoid and len(steps) < max_next_steps:
        steps.append("Try adjacent chemistry away from the repeated failure families.")
    if not steps:
        steps.append("Broaden the search or relax one constraint to gather stronger evidence.")
    return steps[:max_next_steps] if max_next_steps > 0 else []


def _build_warnings(
    evidence_count: int,
    low_trust_count: int,
    productive: Counter[str],
    avoid: Counter[str],
) -> list[str]:
    warnings: list[str] = []
    if evidence_count == 0:
        warnings.append("Evidence is sparse; the lesson is tentative.")
        return warnings
    if low_trust_count:
        warnings.append("Some supporting results were low-trust; treat the lesson cautiously.")
    if productive and avoid:
        warnings.append("Evidence is mixed; keep the next cycle balanced.")
    return warnings


def _confidence(evidence_strength: float, labels_seen: bool) -> str:
    if labels_seen and evidence_strength >= 4.0:
        return "high"
    if evidence_strength >= 2.0 or labels_seen:
        return "medium"
    return "low"


def _unique_extend(target: list[str], values: Sequence[str], max_items: int) -> None:
    for value in values:
        cleaned = value.strip()
        if not cleaned or cleaned in target:
            continue
        target.append(cleaned)
        if len(target) >= max_items:
            break


def build_chemistry_lesson_summary(
    *,
    component_a: str,
    annotated_results: Sequence[AnnotatedResult],
    candidate_proposals: Sequence[CandidateProposal],
    run_memories: Sequence[RunMemory] | None,
    prior_pattern_memory: ChemicalPatternMemory | None,
    prior_lesson_summary: ChemistryLessonSummary | None,
    config: ChemistryLessonSummaryConfig,
) -> ChemistryLessonSummary:
    if config.mode == "off":
        return ChemistryLessonSummary()

    family_by_smiles = _proposal_family_map(candidate_proposals)
    productive: Counter[str] = Counter()
    avoid: Counter[str] = Counter()
    good_examples: list[str] = []
    bad_examples: list[str] = []
    cycle_notes: list[str] = []
    run_notes: list[str] = []
    labels_seen = False
    low_trust_count = 0

    for item in annotated_results:
        smiles = item.result.curve.smiles_b
        family = family_by_smiles.get(smiles, "")
        if getattr(item, "trust_score", 1.0) < 0.5:
            low_trust_count += 1
        if item.result.is_des:
            if family:
                productive[family] += 1
            good_examples.append(smiles)
        else:
            if family:
                avoid[family] += 1
            bad_examples.append(smiles)

    matched_memories = _matching_run_memories(run_memories, component_a)
    advisor_notes = build_chemistry_advisor_memory_notes(matched_memories)
    for memory in matched_memories:
        if memory.labels:
            labels_seen = True
        for label in memory.labels:
            if label.label == "good":
                good_examples.append(label.smiles_b)
            elif label.label == "bad":
                bad_examples.append(label.smiles_b)
        if memory.ranked_candidates:
            top_ranked = ", ".join(candidate.smiles_b for candidate in memory.ranked_candidates[:3])
            if top_ranked:
                run_notes.append(f"Prior top ranked candidates: {top_ranked}")

    if prior_pattern_memory is not None:
        productive.update(prior_pattern_memory.productive_families)
        avoid.update(prior_pattern_memory.avoid_families)
        good_examples.extend(prior_pattern_memory.good_examples)
        bad_examples.extend(prior_pattern_memory.bad_examples)
        run_notes.extend(prior_pattern_memory.prompt_notes)
        run_notes.extend(prior_pattern_memory.notes)
    if prior_lesson_summary is not None:
        run_notes.extend(prior_lesson_summary.run_summary)
        run_notes.extend(prior_lesson_summary.notes)

    bounded_good = _bounded_unique(good_examples, config.max_examples)
    bounded_bad = _bounded_unique(bad_examples, config.max_examples)
    representative_examples = _bounded_unique(bounded_good + bounded_bad, config.max_examples)

    if not productive and not avoid and not representative_examples and not matched_memories and prior_pattern_memory is None and prior_lesson_summary is None:
        return ChemistryLessonSummary()

    cycle_note = _pattern_note("Productive patterns", productive)
    if cycle_note:
        cycle_notes.append(cycle_note)
    avoid_note = _pattern_note("Avoid patterns", avoid)
    if avoid_note:
        cycle_notes.append(avoid_note)
    if representative_examples:
        cycle_notes.append("Representative examples: " + ", ".join(representative_examples) + ".")
    if not cycle_notes:
        cycle_notes.append("No strong chemistry pattern emerged yet.")

    next_steps = _build_next_steps(productive, avoid, config.max_next_steps)
    warnings = _build_warnings(
        evidence_count=sum(productive.values()) + sum(avoid.values()) + len(bounded_good) + len(bounded_bad),
        low_trust_count=low_trust_count,
        productive=productive,
        avoid=avoid,
    )

    if bounded_good:
        run_notes.append("Representative good examples: " + ", ".join(bounded_good) + ".")
    if bounded_bad:
        run_notes.append("Representative bad examples: " + ", ".join(bounded_bad) + ".")
    run_notes.extend(advisor_notes)
    run_notes.extend(next_steps)
    run_notes.extend(warnings)
    run_notes.extend(cycle_notes)

    evidence_strength = (sum(productive.values()) + sum(avoid.values())) * (1.0 + config.weak_pattern_bonus)
    evidence_strength += config.strong_label_bonus * len(matched_memories)
    confidence = _confidence(evidence_strength, labels_seen)

    return ChemistryLessonSummary(
        productive_patterns=dict(productive),
        avoid_patterns=dict(avoid),
        cycle_summary=cycle_notes,
        run_summary=_bounded_unique(run_notes, max(6, config.max_examples * 3)),
        next_steps=next_steps,
        warnings=warnings,
        representative_examples=representative_examples,
        confidence=confidence,
        notes=_bounded_unique(run_notes, max(6, config.max_examples * 3)),
    )
