from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Sequence

from .memory_schema import RunMemory
from .schemas import CandidateProposal
from .uncertainty import AnnotatedResult, rank_annotated_results


@dataclass(frozen=True)
class ChemicalPatternMemoryConfig:
    mode: str = "adaptive"
    max_examples: int = 3
    good_label_bonus: float = 0.20
    bad_label_penalty: float = 0.20
    family_bonus_cap: float = 0.10
    family_penalty_cap: float = 0.10

    def __post_init__(self) -> None:
        mode = self.mode.strip().lower()
        if mode not in {"off", "soft", "adaptive"}:
            raise ValueError("chemical pattern memory mode must be off, soft, or adaptive")
        if self.max_examples < 0:
            raise ValueError("pattern memory max examples must be non-negative")
        object.__setattr__(self, "mode", mode)


@dataclass(frozen=True)
class ChemicalPatternMemory:
    productive_families: dict[str, int] = field(default_factory=dict)
    avoid_families: dict[str, int] = field(default_factory=dict)
    good_examples: list[str] = field(default_factory=list)
    bad_examples: list[str] = field(default_factory=list)
    prompt_notes: list[str] = field(default_factory=list)
    ranking_bias_by_smiles: dict[str, float] = field(default_factory=dict)
    ranking_bias_by_family: dict[str, float] = field(default_factory=dict)
    confidence: str = "low"
    notes: list[str] = field(default_factory=list)


def _proposal_family_map(candidate_proposals: Sequence[CandidateProposal]) -> dict[str, str]:
    return {
        proposal.smiles: proposal.family.strip()
        for proposal in candidate_proposals
        if proposal.smiles and proposal.family and proposal.family.strip()
    }


def _bounded_examples(values: Sequence[str], max_examples: int) -> list[str]:
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


def _confidence(evidence_count: int, has_label: bool) -> str:
    if has_label and evidence_count >= 3:
        return "high"
    if evidence_count >= 2 or has_label:
        return "medium"
    return "low"


def build_pattern_memory(
    *,
    component_a: str,
    annotated_results: Sequence[AnnotatedResult],
    candidate_proposals: Sequence[CandidateProposal],
    run_memories: Sequence[RunMemory] | None,
    config: ChemicalPatternMemoryConfig,
    contradicted_family_smiles: set[str] | None = None,
) -> ChemicalPatternMemory:
    if config.mode == "off":
        return ChemicalPatternMemory()

    family_by_smiles = _proposal_family_map(candidate_proposals)
    productive: Counter[str] = Counter()
    avoid: Counter[str] = Counter()
    good_examples: list[str] = []
    bad_examples: list[str] = []
    bias_by_smiles: dict[str, float] = {}
    label_seen = False

    _skip_family = contradicted_family_smiles or set()
    for item in annotated_results:
        smiles = item.result.curve.smiles_b
        family = family_by_smiles.get(smiles, "")
        family_trusted = family and smiles not in _skip_family
        low_trust = getattr(item, "trust_score", 1.0) < 0.5
        multiplier = 0.5 if low_trust and config.mode == "adaptive" else 1.0
        if item.result.is_des:
            if family_trusted:
                productive[family] += 1
            good_examples.append(smiles)
            bias_by_smiles[smiles] = max(bias_by_smiles.get(smiles, 0.0), 0.04 * multiplier)
        else:
            if family_trusted:
                avoid[family] += 1
            bad_examples.append(smiles)
            bias_by_smiles[smiles] = min(bias_by_smiles.get(smiles, 0.0), -0.04 * multiplier)

    for memory in run_memories or []:
        if memory.component_a is not None and memory.component_a != component_a:
            continue
        labels = {label.smiles_b: label.label for label in memory.labels}
        if labels:
            label_seen = True
        for smiles, label in labels.items():
            if label == "good":
                good_examples.append(smiles)
                bias_by_smiles[smiles] = config.good_label_bonus
            elif label == "bad":
                bad_examples.append(smiles)
                bias_by_smiles[smiles] = -config.bad_label_penalty

    ranking_bias_by_family: dict[str, float] = {}
    for family, count in productive.items():
        ranking_bias_by_family[family] = min(config.family_bonus_cap, 0.04 * count)
    for family, count in avoid.items():
        ranking_bias_by_family[family] = -min(config.family_penalty_cap, 0.04 * count)

    notes: list[str] = []
    prompt_notes: list[str] = []
    if productive:
        families = ", ".join(family for family, _ in productive.most_common(3))
        prompt_notes.append(f"Prior predictions found these productive DES families: {families}.")
        notes.append(f"Chemical pattern memory favored productive families: {families}.")
    if avoid:
        families = ", ".join(family for family, _ in avoid.most_common(3))
        prompt_notes.append(f"Prior predictions suggest caution with these families: {families}.")
        notes.append(f"Chemical pattern memory penalized caution families: {families}.")
    bounded_good = _bounded_examples(good_examples, config.max_examples)
    bounded_bad = _bounded_examples(bad_examples, config.max_examples)
    if bounded_good:
        prompt_notes.append("Representative good examples: " + ", ".join(bounded_good) + ".")
    if bounded_bad:
        prompt_notes.append("Representative bad examples: " + ", ".join(bounded_bad) + ".")

    evidence_count = len(good_examples) + len(bad_examples) + sum(productive.values()) + sum(avoid.values())
    return ChemicalPatternMemory(
        productive_families=dict(productive),
        avoid_families=dict(avoid),
        good_examples=bounded_good,
        bad_examples=bounded_bad,
        prompt_notes=prompt_notes[:6],
        ranking_bias_by_smiles=bias_by_smiles,
        ranking_bias_by_family=ranking_bias_by_family,
        confidence=_confidence(evidence_count, label_seen),
        notes=notes,
    )


def merge_pattern_memories(*memories: ChemicalPatternMemory | None) -> ChemicalPatternMemory:
    productive: Counter[str] = Counter()
    avoid: Counter[str] = Counter()
    good_examples: list[str] = []
    bad_examples: list[str] = []
    prompt_notes: list[str] = []
    ranking_bias_by_smiles: dict[str, float] = {}
    ranking_bias_by_family: dict[str, float] = {}
    notes: list[str] = []
    confidence_rank = {"low": 0, "medium": 1, "high": 2}
    confidence = "low"

    for memory in memories:
        if memory is None:
            continue
        productive.update(memory.productive_families)
        avoid.update(memory.avoid_families)
        good_examples.extend(memory.good_examples)
        bad_examples.extend(memory.bad_examples)
        prompt_notes.extend(memory.prompt_notes)
        notes.extend(memory.notes)
        for smiles, bias in memory.ranking_bias_by_smiles.items():
            ranking_bias_by_smiles[smiles] = ranking_bias_by_smiles.get(smiles, 0.0) + bias
        for family, bias in memory.ranking_bias_by_family.items():
            ranking_bias_by_family[family] = ranking_bias_by_family.get(family, 0.0) + bias
        if confidence_rank.get(memory.confidence, 0) > confidence_rank.get(confidence, 0):
            confidence = memory.confidence

    return ChemicalPatternMemory(
        productive_families=dict(productive),
        avoid_families=dict(avoid),
        good_examples=_bounded_examples(good_examples, max(len(good_examples), 0)),
        bad_examples=_bounded_examples(bad_examples, max(len(bad_examples), 0)),
        prompt_notes=_bounded_examples(prompt_notes, max(len(prompt_notes), 0)),
        ranking_bias_by_smiles=ranking_bias_by_smiles,
        ranking_bias_by_family=ranking_bias_by_family,
        confidence=confidence,
        notes=_bounded_examples(notes, max(len(notes), 0)),
    )



def _confidence_multiplier(memory: ChemicalPatternMemory) -> float:
    if memory.confidence == "high":
        return 1.0
    if memory.confidence == "medium":
        return 0.75
    return 0.5


def apply_pattern_memory_bias(
    annotated_results: Sequence[AnnotatedResult],
    candidate_proposals: Sequence[CandidateProposal],
    memory: ChemicalPatternMemory,
) -> tuple[list[AnnotatedResult], list[str]]:
    if not annotated_results or not (memory.ranking_bias_by_smiles or memory.ranking_bias_by_family):
        return list(annotated_results), []

    family_by_smiles = _proposal_family_map(candidate_proposals)
    multiplier = _confidence_multiplier(memory)
    adjusted: list[AnnotatedResult] = []
    affected = 0

    for item in annotated_results:
        smiles = item.result.curve.smiles_b
        family = family_by_smiles.get(smiles, "")
        bias = memory.ranking_bias_by_smiles.get(smiles, 0.0)
        bias += memory.ranking_bias_by_family.get(family, 0.0)
        if bias:
            affected += 1
        adjusted.append(replace(item, ranking_score=item.ranking_score + bias * multiplier))

    if affected == 0:
        return adjusted, []
    return rank_annotated_results(adjusted), [f"Applied chemical pattern memory ranking bias to {affected} candidate(s)."]
