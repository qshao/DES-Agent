# Chemistry Lesson Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a compact `ChemistryLessonSummary` layer that turns prior DES evidence into short chemistry lessons for the next cycle and final report, while reusing the existing pattern-memory and chemistry-advisor plumbing.

**Architecture:** Create a focused summary module that derives short cycle/run lessons from annotated results, candidate proposals, and saved run labels. Thread those lessons into the DES search outcome, report formatting, and LLM prompt context so the system can explain what it learned without changing deterministic ranking.

**Tech Stack:** Python dataclasses, existing DES result and run-memory schemas, existing LLM provider/context strings, pytest, current reporting and orchestrator modules.

---

## File Structure

- Create `des_multi_agent/chemical_lesson_summary.py`: summary dataclasses, extraction helpers, and note-builder functions.
- Modify `des_multi_agent/orchestrator.py`: build cycle/run lesson summaries, attach them to `SearchOutcome`, and thread lesson notes into brainstorm/advisor context.
- Modify `des_multi_agent/multi_cycle.py`: carry the previous cycle's lesson summary into the next cycle.
- Modify `des_multi_agent/reporting.py`: render the lesson summary in the human-readable report without duplicating the candidate table.
- Modify `des_multi_agent/workflows/selectivity_des_pipeline.py` only if the summary needs to be surfaced in the selectivity-DES report path.
- Modify `docs/tutorial.md` and the relevant example READMEs after behavior is verified.
- Create `tests/test_chemical_lesson_summary.py`: focused unit tests.
- Modify `tests/test_llm_orchestrator.py`, `tests/test_reporting.py`, and `tests/test_llm_candidate_families.py` for integration coverage.

---

### Task 1: Add Chemistry Lesson Summary Data Model and Extraction

**Files:**
- Create: `des_multi_agent/chemical_lesson_summary.py`
- Test: `tests/test_chemical_lesson_summary.py`

- [ ] **Step 1: Write failing tests for empty summary and simple cycle lessons**

Add these tests to `tests/test_chemical_lesson_summary.py`:

```python
from des_multi_agent.chemical_lesson_summary import (
    ChemistryLessonSummary,
    ChemistryLessonSummaryConfig,
    build_chemistry_lesson_summary,
)
from des_multi_agent.schemas import CandidateProposal


class DummyCurve:
    smiles_b = "OCCO"


class DummyResult:
    curve = DummyCurve()
    is_des = True
    min_tm_k = 210.0


class DummyUncertainty:
    uncertainty_flag = "ok"


class DummyAnnotated:
    result = DummyResult()
    trust_score = 0.9
    uncertainty = DummyUncertainty()
    ranking_score = 1.0


def test_empty_lesson_summary_is_blank():
    summary = build_chemistry_lesson_summary(
        component_a="CCO",
        annotated_results=[],
        candidate_proposals=[],
        run_memories=[],
        prior_pattern_memory=None,
        config=ChemistryLessonSummaryConfig(mode="adaptive"),
    )

    assert summary == ChemistryLessonSummary()
    assert summary.cycle_summary == []
    assert summary.run_summary == []


def test_cycle_lesson_from_des_hit_mentions_productive_pattern():
    summary = build_chemistry_lesson_summary(
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
        prior_pattern_memory=None,
        config=ChemistryLessonSummaryConfig(mode="adaptive", max_examples=3),
    )

    assert summary.productive_patterns == {"diol": 1}
    assert summary.representative_examples == ["OCCO"]
    assert any("productive" in note.lower() or "diol" in note for note in summary.cycle_summary)
    assert summary.confidence in {"low", "medium"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_chemical_lesson_summary.py -q
```

Expected: fail because `des_multi_agent.chemical_lesson_summary` does not exist.

- [ ] **Step 3: Implement the minimal summary model and extractor**

Create `des_multi_agent/chemical_lesson_summary.py`:

```python
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

from .chemical_pattern_memory import ChemicalPatternMemory
from .memory_schema import RunMemory
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


```

- [ ] **Step 4: Run the focused tests and confirm they pass**

Run:

```bash
python -m pytest tests/test_chemical_lesson_summary.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/chemical_lesson_summary.py tests/test_chemical_lesson_summary.py
git commit -m "feat: add chemistry lesson summary extraction"
```

### Task 2: Thread Lesson Summaries Through DES Runs and Reports

**Files:**
- Modify: `des_multi_agent/orchestrator.py`
- Modify: `des_multi_agent/multi_cycle.py`
- Modify: `des_multi_agent/reporting.py`
- Test: `tests/test_llm_orchestrator.py`
- Test: `tests/test_reporting.py`
- Test: `tests/test_llm_candidate_families.py`

- [ ] **Step 1: Write failing integration tests for report and cycle handoff**

Add tests that check:

```python
from types import SimpleNamespace

from des_multi_agent.chemical_lesson_summary import ChemistryLessonSummary
from des_multi_agent.reporting import format_report
from des_multi_agent.multi_cycle import run_multi_cycle_search
from des_multi_agent.orchestrator import run_search_report


def test_orchestrator_attaches_lesson_summary_to_outcome(monkeypatch, tmp_path):
    def fake_build_lesson_summary(**kwargs):
        return ChemistryLessonSummary(
            cycle_summary=["Short diols looked productive."],
            run_summary=["Repeat productive diols in the next cycle."],
            next_steps=["Stay near productive families."],
            warnings=["Evidence is still sparse."],
        )

    monkeypatch.setattr("des_multi_agent.orchestrator.build_chemistry_lesson_summary", fake_build_lesson_summary)
    outcome = run_search_report(
        component_a="CCO",
        n=5,
        checkpoint_path=str(tmp_path / "ckpt.pt"),
        config_path=str(tmp_path / "config.yaml"),
        llm_cfg=None,
        proposal_diversity_cfg=None,
    )
    assert outcome.chemistry_lesson_summary.cycle_summary
    assert outcome.chemistry_lesson_summary.run_summary


def test_report_includes_lesson_summary_block():
    summary = ChemistryLessonSummary(
        cycle_summary=["Short diols looked productive."],
        run_summary=["Repeat productive diols in the next cycle."],
        next_steps=["Stay near productive families."],
        warnings=["Evidence is still sparse."],
    )
    report = format_report([], chemistry_lesson_summary=summary)
    assert "Chemistry lessons" in report
    assert "Stay near productive families." in report


def test_multi_cycle_passes_lesson_summary_forward(monkeypatch, tmp_path):
    calls = []

    def fake_run_search_report(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            results=[],
            annotated_results=[],
            candidate_proposals=[],
            candidate_reviews=[],
            explanation_notes=[],
            critique_notes=[],
            brainstorm_candidates=[],
            llm_warnings=[],
            memory_notes=[],
            viscosity_predictions=[],
            chemistry_lesson_summary=ChemistryLessonSummary(run_summary=[f"cycle {len(calls)} lesson"]),
        )

    monkeypatch.setattr("des_multi_agent.multi_cycle.run_search_report", fake_run_search_report)
    run_multi_cycle_search(
        component_a="CCO",
        n=5,
        checkpoint_path=str(tmp_path / "ckpt.pt"),
        config_path=str(tmp_path / "config.yaml"),
        n_cycles=2,
    )
    assert calls[0]["prior_chemistry_lesson_summary"] is None
    assert calls[1]["prior_chemistry_lesson_summary"].run_summary == ["cycle 1 lesson"]
```

- [ ] **Step 2: Run the focused tests and confirm they fail first**

Run:

```bash
python -m pytest tests/test_llm_orchestrator.py tests/test_reporting.py tests/test_llm_candidate_families.py -q
```

Expected: FAIL because the report and orchestrator do not yet expose lesson summaries.

- [ ] **Step 3: Add the minimal wiring and report rendering**

In `des_multi_agent/orchestrator.py`, extend `SearchOutcome` and `run_search_report(...)` so the function builds a `ChemistryLessonSummary` from the current results and saved memory, then appends compact lesson notes to the existing LLM/advisor context strings.

In `des_multi_agent/multi_cycle.py`, carry the previous cycle's lesson summary into the next call.

In `des_multi_agent/reporting.py`, add an optional `chemistry_lesson_summary` argument to `format_report(...)`, `format_report_prose(...)`, and any downstream report formatters that should display the lesson block. Render it as a short section after the main summary and before the detailed tables, with separate lines for:
- productive patterns
- avoid patterns
- next steps
- warnings

Keep the existing candidate table unchanged.

- [ ] **Step 4: Run the focused tests and confirm they pass**

Run:

```bash
python -m pytest tests/test_llm_orchestrator.py tests/test_reporting.py tests/test_llm_candidate_families.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/orchestrator.py des_multi_agent/multi_cycle.py des_multi_agent/reporting.py tests/test_llm_orchestrator.py tests/test_reporting.py tests/test_llm_candidate_families.py
git commit -m "feat: thread chemistry lesson summaries through DES runs"
```

### Task 3: Update Docs and Example Guidance

**Files:**
- Modify: `docs/tutorial.md`
- Modify: `examples/README.md`
- Modify: `examples/des_run_memory_feedback/README.md`
- Modify: `tests/fixtures/example_benchmark_baselines/README.md`
- Test: `tests/test_benchmarks_examples.py`

- [ ] **Step 1: Write failing doc regression expectations**

Add an assertion that the top-level examples index mentions the chemistry lesson summary layer in the shared LLM-backfilled examples paragraph, and mirror the same wording in the benchmark baseline README.

- [ ] **Step 2: Run the example benchmark test and confirm the docs mismatch**

Run:

```bash
python -m pytest tests/test_benchmarks_examples.py -q
```

Expected: fail or expose mismatches until the docs are updated.

- [ ] **Step 3: Update the tutorial and example READMEs**

Add a short tutorial subsection that explains:
- what the chemistry lesson summary is
- how it differs from proposal diversity and chemical pattern memory
- where users will see it in the report
- that it feeds the next cycle automatically when enough evidence exists

Update the examples index and the run-memory demo README to point users at the lesson-summary behavior in the multi-cycle DES examples.

- [ ] **Step 4: Rerun the example benchmark tests and confirm they pass**

Run:

```bash
python -m pytest tests/test_benchmarks_examples.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add docs/tutorial.md examples/README.md examples/des_run_memory_feedback/README.md tests/fixtures/example_benchmark_baselines/README.md
git commit -m "docs: document chemistry lesson summaries"
```

### Task 4: Final Verification and Cleanup

**Files:**
- Modify: any files touched by Tasks 1-3 if test failures expose mismatches
- Test: full targeted pytest subset

- [ ] **Step 1: Run the full targeted verification suite**

Run:

```bash
python -m pytest tests/test_chemical_lesson_summary.py tests/test_llm_orchestrator.py tests/test_reporting.py tests/test_llm_candidate_families.py tests/test_benchmarks_examples.py -q
```

Expected: PASS, with only the existing PyTorch/PyG warnings if they still appear.

- [ ] **Step 2: Run diff formatting checks**

Run:

```bash
git diff --check
```

Expected: no whitespace or patch-format issues.

- [ ] **Step 3: Commit any final fixes**

```bash
git add -A
git commit -m "feat: finish chemistry lesson summary feature"
```

---

## Spec Coverage Check

- Summary built from current-cycle results and saved labels: Task 1
- Cycle and run summaries: Task 1 and Task 2
- Next-step suggestions and warnings: Task 1 and Task 2
- Reuse in report, memory, and advisor context: Task 2
- Minimal user controls: Task 1 and Task 2 keep it automatic
- Testing across extraction, report formatting, and cycle handoff: Tasks 1-4
- No new autonomous planner or model training: preserved as non-goals

## Placeholder Scan

No TBD/TODO placeholders are included.

## Type Consistency Check

The plan uses one new module name, `chemical_lesson_summary.py`, consistently across tasks. `ChemistryLessonSummary` and `ChemistryLessonSummaryConfig` are the only new public summary types, and the orchestrator/reporting steps refer to those exact names.
