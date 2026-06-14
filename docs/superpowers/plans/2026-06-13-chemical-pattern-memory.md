# Chemical Pattern Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `ChemicalPatternMemory` layer that converts prior DES predictions and user labels into compact chemical lessons for later LLM proposals, advisor prompts, and bounded ranking bias.

**Architecture:** Add a focused `des_multi_agent/chemical_pattern_memory.py` module that extracts pattern evidence from current-cycle results and saved run memory. Thread its summary through `orchestrator.py` and `multi_cycle.py`, while keeping the current predictive models and proposal-diversity controls as the primary decision mechanisms.

**Tech Stack:** Python dataclasses, existing DES result and run-memory schemas, pytest, existing CLI/orchestrator patterns.

---

## File Structure

- Create `des_multi_agent/chemical_pattern_memory.py`: dataclasses, extraction helpers, prompt-note builder, ranking-bias applier.
- Modify `des_multi_agent/orchestrator.py`: accept memory mode/max examples, build pattern memory from reuse runs and prior cycle memory, add prompt/advisor notes, apply bounded ranking bias, return pattern memory in `SearchOutcome`.
- Modify `des_multi_agent/multi_cycle.py`: carry accumulated pattern memory between cycles.
- Modify `des_multi_agent/cli.py`: expose `--chemical-pattern-memory` and `--pattern-memory-max-examples`, and pass them into single-cycle and multi-cycle DES runs.
- Leave `des_multi_agent/llm/prompts.py` unchanged for the first implementation; append pattern notes through existing context strings in `orchestrator.py`.
- Modify `docs/tutorial.md` and selected examples after behavior is verified.
- Create `tests/test_chemical_pattern_memory.py`: focused unit tests.
- Modify `tests/test_llm_orchestrator.py`, `tests/test_cli.py`, and `tests/test_validation_dedup_batch_presets.py`: integration coverage.

---

### Task 1: Add Pattern Memory Data Model and Extraction

**Files:**
- Create: `des_multi_agent/chemical_pattern_memory.py`
- Test: `tests/test_chemical_pattern_memory.py`

- [ ] **Step 1: Write failing tests for empty memory and short-term extraction**

Add these tests to `tests/test_chemical_pattern_memory.py`:

```python
from des_multi_agent.chemical_pattern_memory import (
    ChemicalPatternMemory,
    ChemicalPatternMemoryConfig,
    build_pattern_memory,
)
from des_multi_agent.schemas import CandidateProposal


class DummyCurve:
    smiles_b = "OCCO"


class DummyResult:
    curve = DummyCurve()
    is_des = True
    min_tm_k = 210.0
    relative_drop = 0.22


class DummyUncertainty:
    uncertainty_flag = "ok"


class DummyAnnotated:
    result = DummyResult()
    trust_score = 0.9
    uncertainty = DummyUncertainty()
    ranking_score = 1.0


def test_empty_pattern_memory_is_inactive():
    memory = build_pattern_memory(
        component_a="CCO",
        annotated_results=[],
        candidate_proposals=[],
        run_memories=[],
        config=ChemicalPatternMemoryConfig(mode="adaptive"),
    )

    assert memory == ChemicalPatternMemory()
    assert memory.prompt_notes == []
    assert memory.ranking_bias_by_smiles == {}


def test_productive_family_from_des_hit_becomes_prompt_note():
    memory = build_pattern_memory(
        component_a="CCO",
        annotated_results=[DummyAnnotated()],
        candidate_proposals=[
            CandidateProposal(
                smiles="OCCO",
                rationale="short diol",
                family="diol",
                source="llm",
                source_id="brainstorm",
            )
        ],
        run_memories=[],
        config=ChemicalPatternMemoryConfig(mode="adaptive", max_examples=3),
    )

    assert memory.productive_families == {"diol": 1}
    assert memory.good_examples == ["OCCO"]
    assert any("diol" in note for note in memory.prompt_notes)
    assert memory.confidence in {"low", "medium"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_chemical_pattern_memory.py -q
```

Expected: fail because `des_multi_agent.chemical_pattern_memory` does not exist.

- [ ] **Step 3: Implement minimal data model and extractor**

Create `des_multi_agent/chemical_pattern_memory.py`:

```python
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import Sequence

from .memory_schema import RunMemory
from .schemas import CandidateProposal
from .uncertainty.schemas import AnnotatedResult


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
        if not value or value in seen:
            continue
        examples.append(value)
        seen.add(value)
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

    for item in annotated_results:
        smiles = item.result.curve.smiles_b
        family = family_by_smiles.get(smiles, "")
        low_trust = getattr(item, "trust_score", 1.0) < 0.5
        multiplier = 0.5 if low_trust and config.mode == "adaptive" else 1.0
        if item.result.is_des:
            if family:
                productive[family] += 1
            good_examples.append(smiles)
            bias_by_smiles[smiles] = max(bias_by_smiles.get(smiles, 0.0), 0.04 * multiplier)
        else:
            if family:
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_chemical_pattern_memory.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/chemical_pattern_memory.py tests/test_chemical_pattern_memory.py
git commit -m "feat: add chemical pattern memory model"
```

---

### Task 2: Add Ranking Bias Application

**Files:**
- Modify: `des_multi_agent/chemical_pattern_memory.py`
- Test: `tests/test_chemical_pattern_memory.py`

- [ ] **Step 1: Write failing tests for exact, family, and low-trust effects**

Append:

```python
from dataclasses import replace

from des_multi_agent.chemical_pattern_memory import apply_pattern_memory_bias


def test_pattern_memory_bias_is_capped_and_family_based():
    item = DummyAnnotated()
    memory = ChemicalPatternMemory(
        ranking_bias_by_smiles={"OCCO": 0.20},
        ranking_bias_by_family={"diol": 0.10},
        confidence="high",
    )
    proposals = [
        CandidateProposal(
            smiles="OCCO",
            rationale="short diol",
            family="diol",
            source="llm",
            source_id="brainstorm",
        )
    ]

    adjusted, notes = apply_pattern_memory_bias([item], proposals, memory)

    assert adjusted[0].ranking_score == item.ranking_score + 0.30
    assert any("Applied chemical pattern memory" in note for note in notes)


def test_low_confidence_memory_has_smaller_effect():
    item = DummyAnnotated()
    memory = ChemicalPatternMemory(
        ranking_bias_by_smiles={"OCCO": 0.20},
        confidence="low",
    )

    adjusted, _ = apply_pattern_memory_bias([item], [], memory)

    assert adjusted[0].ranking_score == item.ranking_score + 0.10
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_chemical_pattern_memory.py -q
```

Expected: fail because `apply_pattern_memory_bias` does not exist.

- [ ] **Step 3: Implement ranking bias**

Add to `des_multi_agent/chemical_pattern_memory.py`:

```python
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
    if not annotated_results or not (
        memory.ranking_bias_by_smiles or memory.ranking_bias_by_family
    ):
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
    return adjusted, [f"Applied chemical pattern memory ranking bias to {affected} candidate(s)."]
```

- [ ] **Step 4: Run tests**

Run:

```bash
python -m pytest tests/test_chemical_pattern_memory.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/chemical_pattern_memory.py tests/test_chemical_pattern_memory.py
git commit -m "feat: apply chemical pattern ranking bias"
```

---

### Task 3: Thread Pattern Memory Through Orchestrator

**Files:**
- Modify: `des_multi_agent/orchestrator.py`
- Test: `tests/test_llm_orchestrator.py`

- [ ] **Step 1: Write failing orchestrator tests**

Add these tests to `tests/test_llm_orchestrator.py`:

```python
from des_multi_agent.chemical_pattern_memory import ChemicalPatternMemory


def test_pattern_memory_notes_reach_llm_brainstorm_context(monkeypatch):
    captured_contexts = []

    class _CapturingLLM(_FakeLLM):
        def brainstorm_candidates(self, component_a, constraints, context, **kwargs):
            captured_contexts.append(context)
            return [CandidateBrainstorm(smiles="OCCO", rationale="short diol", family="diol")]

    monkeypatch.setattr(orchestrator, "build_llm_provider", lambda cfg, request_fn=None: _CapturingLLM())
    monkeypatch.setattr(
        orchestrator,
        "generate_candidates",
        lambda component_a, n, constraints=None: [CandidateProposal(smiles="O", rationale="baseline", family="alcohol")],
    )
    monkeypatch.setattr(orchestrator, "filter_candidates", lambda component_a, candidates: candidates)
    monkeypatch.setattr(
        orchestrator,
        "resolve_melting_point",
        lambda component, override_k=None: MeltingPointEstimate(component=component, tm_k=300.0, source="heuristic", confidence=0.5),
    )
    monkeypatch.setattr(
        orchestrator,
        "predict_curve",
        lambda component_a, component_b, **kwargs: CurvePrediction(
            smiles_a=component_a,
            smiles_b=component_b,
            ratios=[0.1],
            tm_pred_k=[250.0],
            t1_k=300.0,
            t2_k=300.0,
            checkpoint_path="ckpt.pt",
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "classify_des",
        lambda curve, thresholds: DesResult(curve=curve, absolute_pass=True, relative_pass=True, is_des=True, rationale="ok", min_tm_k=250.0),
    )
    monkeypatch.setattr(orchestrator, "rank_results", lambda results: results)

    outcome = orchestrator.run_search_report(
        component_a="CCO",
        n=1,
        checkpoint_path="ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt",
        llm_cfg={"enabled": True, "provider": "ollama", "model_name": "llama3.1", "api_base_url": "http://localhost:11434"},
        prior_pattern_memory=ChemicalPatternMemory(
            prompt_notes=["Prior predictions found these productive DES families: diol."],
            good_examples=["OCCO"],
            confidence="medium",
        ),
        chemical_pattern_memory_mode="adaptive",
    )

    assert any("Prior predictions found these productive DES families: diol." in context for context in captured_contexts)
    assert outcome.chemical_pattern_memory.good_examples
```

```python
def test_pattern_memory_ranking_bias_is_applied_after_uncertainty(monkeypatch):
    monkeypatch.setattr(orchestrator, "build_llm_provider", lambda cfg, request_fn=None: None)
    monkeypatch.setattr(
        orchestrator,
        "generate_candidates",
        lambda component_a, n, constraints=None: [CandidateProposal(smiles="OCCO", rationale="baseline", family="diol")],
    )
    monkeypatch.setattr(orchestrator, "filter_candidates", lambda component_a, candidates: candidates)
    monkeypatch.setattr(
        orchestrator,
        "resolve_melting_point",
        lambda component, override_k=None: MeltingPointEstimate(component=component, tm_k=300.0, source="heuristic", confidence=0.5),
    )
    monkeypatch.setattr(
        orchestrator,
        "predict_curve",
        lambda component_a, component_b, **kwargs: CurvePrediction(
            smiles_a=component_a,
            smiles_b=component_b,
            ratios=[0.1],
            tm_pred_k=[250.0],
            t1_k=300.0,
            t2_k=300.0,
            checkpoint_path="ckpt.pt",
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "classify_des",
        lambda curve, thresholds: DesResult(curve=curve, absolute_pass=True, relative_pass=True, is_des=True, rationale="ok", min_tm_k=250.0),
    )
    monkeypatch.setattr(orchestrator, "rank_results", lambda results: results)

    outcome = orchestrator.run_search_report(
        component_a="CCO",
        n=1,
        checkpoint_path="ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt",
        prior_pattern_memory=ChemicalPatternMemory(
            ranking_bias_by_smiles={"OCCO": 0.20},
            confidence="high",
        ),
        chemical_pattern_memory_mode="adaptive",
    )

    assert any("Applied chemical pattern memory ranking bias" in note for note in outcome.memory_notes)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_llm_orchestrator.py -q
```

Expected: fail because `run_search_report` does not accept `prior_pattern_memory` or `chemical_pattern_memory_mode`.

- [ ] **Step 3: Modify `SearchOutcome` and `run_search_report` signature**

In `des_multi_agent/orchestrator.py`, import:

```python
from .chemical_pattern_memory import (
    ChemicalPatternMemory,
    ChemicalPatternMemoryConfig,
    apply_pattern_memory_bias,
    build_pattern_memory,
)
```

Add to `SearchOutcome`:

```python
    chemical_pattern_memory: ChemicalPatternMemory = field(default_factory=ChemicalPatternMemory)
```

Add parameters to `run_search_report`:

```python
    prior_pattern_memory: ChemicalPatternMemory | None = None,
    chemical_pattern_memory_mode: str = "adaptive",
    pattern_memory_max_examples: int = 3,
```

- [ ] **Step 4: Add prompt context helper**

Add:

```python
def _append_pattern_memory_context(context: str, memory: ChemicalPatternMemory | None) -> str:
    if memory is None or not memory.prompt_notes:
        return context
    lines = [context, "", "Chemical lessons from prior predictions:"]
    lines.extend(f"- {note}" for note in memory.prompt_notes[:6])
    return "\n".join(lines)
```

Use this before `provider.brainstorm_candidates(...)` and in advisor context creation.

- [ ] **Step 5: Build and apply pattern memory**

After `annotated_results = _apply_review_penalties(...)` and after reuse memory is loaded, build current memory:

```python
pattern_cfg = ChemicalPatternMemoryConfig(
    mode=chemical_pattern_memory_mode,
    max_examples=pattern_memory_max_examples,
)
current_pattern_memory = build_pattern_memory(
    component_a=component_a,
    annotated_results=annotated_results,
    candidate_proposals=candidate_proposals,
    run_memories=reuse_memories,
    config=pattern_cfg,
)
if prior_pattern_memory is not None:
    pattern_prompt_notes = prior_pattern_memory.prompt_notes + current_pattern_memory.prompt_notes
else:
    pattern_prompt_notes = current_pattern_memory.prompt_notes
```

Apply prior and current memory bias:

```python
for memory in [prior_pattern_memory, current_pattern_memory]:
    if memory is None:
        continue
    annotated_results, pattern_notes = apply_pattern_memory_bias(
        annotated_results,
        candidate_proposals,
        memory,
    )
    memory_notes.extend(pattern_notes)
memory_notes.extend(current_pattern_memory.notes)
```

When creating the returned `SearchOutcome`, set `chemical_pattern_memory=current_pattern_memory`.

- [ ] **Step 6: Run orchestrator tests**

Run:

```bash
python -m pytest tests/test_llm_orchestrator.py tests/test_chemical_pattern_memory.py -q
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add des_multi_agent/orchestrator.py tests/test_llm_orchestrator.py
git commit -m "feat: thread chemical pattern memory through DES runs"
```

---

### Task 4: Carry Pattern Memory Across Multi-Cycle Runs

**Files:**
- Modify: `des_multi_agent/multi_cycle.py`
- Test: `tests/test_validation_dedup_batch_presets.py`

- [ ] **Step 1: Write failing multi-cycle test**

Add a test that monkeypatches `run_search_report` and asserts cycle 2 receives cycle 1 memory:

```python
from des_multi_agent.chemical_pattern_memory import ChemicalPatternMemory


def test_multi_cycle_passes_pattern_memory_between_cycles(monkeypatch):
    calls = []

    class Outcome:
        def __init__(self, cycle):
            self.results = []
            self.brainstorm_candidates = []
            self.chemical_pattern_memory = ChemicalPatternMemory(
                prompt_notes=[f"cycle {cycle} lesson"],
                confidence="medium",
            )

    def fake_run_search_report(**kwargs):
        calls.append(kwargs)
        return Outcome(len(calls))

    monkeypatch.setattr("des_multi_agent.multi_cycle.run_search_report", fake_run_search_report)

    run_multi_cycle_search(
        component_a="CCO",
        n=2,
        checkpoint_path="checkpoint.pt",
        n_cycles=2,
        chemical_pattern_memory_mode="adaptive",
    )

    assert calls[0]["prior_pattern_memory"] is None
    assert calls[1]["prior_pattern_memory"].prompt_notes == ["cycle 1 lesson"]
```

- [ ] **Step 2: Run test to verify it fails**

Run the chosen test file:

```bash
python -m pytest tests/test_validation_dedup_batch_presets.py -q
```

Expected: fail because `run_multi_cycle_search` does not accept or pass pattern memory.

- [ ] **Step 3: Modify multi-cycle signature and handoff**

In `run_multi_cycle_search`, add:

```python
    chemical_pattern_memory_mode: str = "adaptive",
    pattern_memory_max_examples: int = 3,
```

Before the loop:

```python
    prior_pattern_memory = None
```

Pass into `run_search_report`:

```python
            prior_pattern_memory=prior_pattern_memory,
            chemical_pattern_memory_mode=chemical_pattern_memory_mode,
            pattern_memory_max_examples=pattern_memory_max_examples,
```

After `last_outcome = outcome`, add:

```python
        prior_pattern_memory = getattr(outcome, "chemical_pattern_memory", None)
```

- [ ] **Step 4: Run multi-cycle tests**

Run:

```bash
python -m pytest tests/test_validation_dedup_batch_presets.py tests/test_chemical_pattern_memory.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/multi_cycle.py tests/test_validation_dedup_batch_presets.py
git commit -m "feat: carry chemical pattern memory across cycles"
```

---

### Task 5: Add CLI Controls

**Files:**
- Modify: `des_multi_agent/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write failing CLI parser tests**

Add these tests to `tests/test_cli.py`:

```python
def test_des_cli_accepts_chemical_pattern_memory_controls():
    parser = build_parser()
    args = parser.parse_args([
        "--workflow", "des",
        "--component-a", "CCO",
        "--checkpoint-path", "checkpoint.pt",
        "--chemical-pattern-memory", "soft",
        "--pattern-memory-max-examples", "4",
    ])

    assert args.chemical_pattern_memory == "soft"
    assert args.pattern_memory_max_examples == 4


def test_pattern_memory_max_examples_must_be_positive():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--workflow", "des",
            "--component-a", "CCO",
            "--checkpoint-path", "checkpoint.pt",
            "--pattern-memory-max-examples", "0",
        ])
```

- [ ] **Step 2: Run CLI tests to verify they fail**

Run:

```bash
python -m pytest tests/test_cli.py -q
```

Expected: fail because the flags are missing.

- [ ] **Step 3: Add arguments and pass them into workflows**

In `des_multi_agent/cli.py`, add arguments to DES-compatible parser paths:

```python
    parser.add_argument(
        "--chemical-pattern-memory",
        choices=["off", "soft", "adaptive"],
        default="adaptive",
        help="Use prior prediction patterns as off, soft, or adaptive memory guidance",
    )
    parser.add_argument(
        "--pattern-memory-max-examples",
        type=_positive_int,
        default=3,
        help="Maximum good and bad example structures to include in pattern memory prompts",
    )
```

Pass to `run_search_report` and `run_multi_cycle_search`:

```python
chemical_pattern_memory_mode=args.chemical_pattern_memory,
pattern_memory_max_examples=args.pattern_memory_max_examples,
```

- [ ] **Step 4: Run CLI and integration tests**

Run:

```bash
python -m pytest tests/test_cli.py tests/test_llm_orchestrator.py tests/test_validation_dedup_batch_presets.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/cli.py tests/test_cli.py
git commit -m "feat: expose chemical pattern memory controls"
```

---

### Task 6: Document and Refresh Examples

**Files:**
- Modify: `docs/tutorial.md`
- Modify: `examples/des_run_memory_feedback/README.md`
- Modify: `examples/plain_language_gemma4_12b/README.md`
- Modify: `examples/README.md`
- Modify mirrored files under `tests/fixtures/example_benchmark_baselines/`

- [ ] **Step 1: Update tutorial text**

Add a short section to `docs/tutorial.md`:

```markdown
### Chemical pattern memory

DES runs can use prior predictions as compact chemistry lessons for later cycles. Use `--chemical-pattern-memory adaptive` to let repeated productive families, repeated failures, user labels, and uncertainty adjust the next brainstorm and apply a capped ranking bias. Use `--chemical-pattern-memory off` when you want each run to be independent.

Use `--pattern-memory-max-examples N` to limit how many representative good and bad structures are included in LLM prompt context.
```

- [ ] **Step 2: Update example READMEs**

Add concise notes:

```markdown
This example also demonstrates chemical pattern memory: prior predictions and labels can be summarized into compact chemistry lessons for the next cycle while proposal-diversity controls keep the candidate pool broad.
```

Use this in `examples/des_run_memory_feedback/README.md` and a shorter pointer in `examples/plain_language_gemma4_12b/README.md`.

- [ ] **Step 3: Sync benchmark mirrors**

Copy the changed README wording into matching files under `tests/fixtures/example_benchmark_baselines/`.

- [ ] **Step 4: Run example benchmark tests**

Run:

```bash
python -m pytest tests/test_benchmarks_examples.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add docs/tutorial.md examples tests/fixtures/example_benchmark_baselines
git commit -m "docs: document chemical pattern memory"
```

---

### Task 7: Final Consistency Checks

**Files:**
- All touched files

- [ ] **Step 1: Run focused tests**

Run:

```bash
python -m pytest tests/test_chemical_pattern_memory.py tests/test_llm_orchestrator.py tests/test_cli.py tests/test_validation_dedup_batch_presets.py tests/test_benchmarks_examples.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run diff whitespace check**

Run:

```bash
git diff --check
```

Expected: no output and exit code 0.

- [ ] **Step 3: Inspect status**

Run:

```bash
git status -sb
```

Expected: only intentionally untracked planning docs remain, or a clean tree if the user asks to commit them too.

- [ ] **Step 4: Commit any final test/doc adjustment**

If final verification required small fixes, commit them:

```bash
git add des_multi_agent tests docs examples
git commit -m "test: cover chemical pattern memory integration"
```

Skip this commit if there are no changes after Task 6.

---

## Self-Review

- Spec coverage: evidence extraction, short-term and long-term memory, prompt notes, ranking bias, guardrails, user controls, tests, and docs are covered by Tasks 1-7.
- Scope: the plan is limited to DES workflows, as required by the spec. Metal-binding reuse is left out intentionally.
- Type consistency: `ChemicalPatternMemory`, `ChemicalPatternMemoryConfig`, `build_pattern_memory`, and `apply_pattern_memory_bias` are introduced before being used by orchestrator and multi-cycle code.
- Guardrails: mode `off`, cross-component skip, bounded examples, confidence multipliers, and capped ranking effects are explicitly included.
