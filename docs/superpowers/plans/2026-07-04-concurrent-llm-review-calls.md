# Concurrent LLM Review Calls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parallelize DES-Agent's 4 sequential per-candidate/per-ligand LLM review call sites with a shared `ThreadPoolExecutor`-based helper, so a `vllm`-backed server can actually batch concurrent requests instead of seeing them one at a time.

**Architecture:** One new generic helper, `run_concurrent(items, call, max_workers=8)` in a new top-level module `des_multi_agent/concurrency.py`, wraps each item's call so it never raises (captured as `CallResult.error` instead) and returns results in the same order as the input items via `ThreadPoolExecutor.map`. Each of the 4 call sites swaps its sequential `for` loop for one `run_concurrent(...)` call plus a `zip(items, results)` loop that keeps its exact existing warning strings, collection variables, and post-processing checks.

**Tech Stack:** Python stdlib `concurrent.futures.ThreadPoolExecutor`, `dataclasses`, `typing.Generic`/`TypeVar`. No new dependencies.

## Global Constraints

- Full spec: `docs/superpowers/specs/2026-07-04-concurrent-llm-review-calls-design.md` — read it if anything below is ambiguous.
- Concurrency mechanism is `ThreadPoolExecutor`, not asyncio/httpx — no changes to `des_multi_agent/llm/transport.py`, `des_multi_agent/llm/cache.py`, or any provider class.
- Fixed `max_workers=8` default — no new `LLMConfig`/YAML concurrency setting.
- `run_concurrent` must never raise from a single item's failure — always returns a `CallResult` with `.value` or `.error` set.
- `run_concurrent`'s results are returned in the same order as the input `items` list (so `zip(items, results)` pairs correctly).
- `des_multi_agent/workflows/selectivity_des_pipeline.py`'s per-ligand nested-search loop is out of scope — do not touch it.
- Every call site's existing warning message text, collection variable names, and post-processing logic (e.g. the wrong-SMILES check in `_review_top_candidates`) must stay byte-identical — only the loop's execution strategy changes from sequential to concurrent.

---

### Task 1: `run_concurrent` helper + unit tests

**Files:**
- Create: `des_multi_agent/concurrency.py`
- Test: `tests/test_concurrency.py`

**Interfaces:**
- Consumes: nothing from this codebase — pure stdlib (`concurrent.futures.ThreadPoolExecutor`, `dataclasses.dataclass`, `typing.Generic`/`TypeVar`/`Callable`/`Any`).
- Produces: `des_multi_agent.concurrency.CallResult` (a `@dataclass` with fields `value: T | None`, `error: Exception | None`) and `des_multi_agent.concurrency.run_concurrent(items: list[Any], call: Callable[[Any], T], max_workers: int = 8) -> list[CallResult[T]]`. Tasks 2-5 import both names from this module.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_concurrency.py`:

```python
import threading
import time

from des_multi_agent.concurrency import CallResult, run_concurrent


def test_run_concurrent_empty_list_returns_empty_list():
    assert run_concurrent([], lambda x: x) == []


def test_run_concurrent_preserves_input_order():
    items = [5, 1, 4, 2, 3]

    def _slow_identity(x):
        time.sleep(0.01 * (5 - x))
        return x

    results = run_concurrent(items, _slow_identity)
    assert [r.value for r in results] == items


def test_run_concurrent_captures_single_item_failure_without_aborting_others():
    def _call(x):
        if x == 2:
            raise ValueError("boom")
        return x * 10

    results = run_concurrent([1, 2, 3], _call)

    assert results[0] == CallResult(value=10, error=None)
    assert results[1].value is None
    assert isinstance(results[1].error, ValueError)
    assert str(results[1].error) == "boom"
    assert results[2] == CallResult(value=30, error=None)


def test_run_concurrent_caps_workers_at_item_count():
    max_seen_concurrent = []
    lock = threading.Lock()
    active = {"count": 0}

    def _call(x):
        with lock:
            active["count"] += 1
            max_seen_concurrent.append(active["count"])
        time.sleep(0.05)
        with lock:
            active["count"] -= 1
        return x

    run_concurrent([1, 2, 3], _call, max_workers=8)

    assert max(max_seen_concurrent) <= 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_concurrency.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'des_multi_agent.concurrency'`

- [ ] **Step 3: Write minimal implementation**

Create `des_multi_agent/concurrency.py`:

```python
from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass
class CallResult(Generic[T]):
    value: T | None
    error: Exception | None


def run_concurrent(items: list[Any], call: Callable[[Any], T], max_workers: int = 8) -> list[CallResult[T]]:
    """Run call(item) for every item concurrently; never raises.

    Results are returned in the same order as items (ThreadPoolExecutor.map
    preserves input order), so callers can zip(items, run_concurrent(...))
    to line results up with what produced them. Each call is wrapped so a
    single item's exception is captured as CallResult.error instead of
    aborting the others.
    """
    if not items:
        return []

    def _safe_call(item: Any) -> CallResult[T]:
        try:
            return CallResult(value=call(item), error=None)
        except Exception as exc:  # noqa: BLE001 - intentionally broad, mirrors existing per-item try/except
            return CallResult(value=None, error=exc)

    workers = min(len(items), max_workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_safe_call, items))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_concurrency.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/concurrency.py tests/test_concurrency.py
git commit -m "feat: add run_concurrent helper for parallel per-item calls"
```

---

### Task 2: Parallelize `_review_top_candidates` (candidate review)

**Files:**
- Modify: `des_multi_agent/orchestrator.py:1-30` (add import), `:334-354` (`_review_top_candidates` body)
- Test: `tests/test_llm_orchestrator.py`

**Interfaces:**
- Consumes: `run_concurrent`, `CallResult` from Task 1 (`des_multi_agent.concurrency`).
- Produces: `_review_top_candidates`'s signature, return type (`tuple[list[CandidateReview], dict[str, CandidateReview]]`), and every existing warning message stay identical — only its internal execution strategy changes. Nothing later in this plan depends on this task.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_llm_orchestrator.py` (near `test_review_top_candidates_ignores_wrong_smiles`, which is at line 554):

`orchestrator` is already imported at the top of this test file (`from
des_multi_agent import orchestrator`), as are `CandidateReview` and
`CandidateProposal` — no new imports needed besides `threading`.

```python
def test_review_top_candidates_runs_calls_concurrently():
    import threading

    n_candidates = 3
    barrier = threading.Barrier(n_candidates, timeout=2.0)

    class BarrierProvider:
        def review_candidate(self, component_a, candidate_smiles, context):
            barrier.wait()  # blocks until n_candidates calls are simultaneously waiting
            return CandidateReview(
                smiles=candidate_smiles,
                decision="keep",
                confidence=0.9,
                rationale="ok",
                notes=[],
            )

    proposals = [
        CandidateProposal(smiles=f"C{i}", rationale="demo", family="alcohol", source="heuristic", source_id="rule")
        for i in range(n_candidates)
    ]
    warnings = []

    reviews, review_map = orchestrator._review_top_candidates(
        BarrierProvider(), "CCO", proposals, "context", n_candidates, warnings,
    )

    assert len(reviews) == n_candidates
    assert set(review_map.keys()) == {f"C{i}" for i in range(n_candidates)}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_orchestrator.py::test_review_top_candidates_runs_calls_concurrently -v`
Expected: FAIL — the `threading.Barrier(3, timeout=2.0)` times out and raises `BrokenBarrierError`, because today's sequential `for` loop calls `review_candidate` one at a time, so only 1 of the 3 calls is ever waiting on the barrier at once (never reaches 3 simultaneous waiters within the 2s timeout).

- [ ] **Step 3: Write minimal implementation**

In `des_multi_agent/orchestrator.py`, add the import after the existing `from .config import DEFAULT_ABSOLUTE_TM_MAX_K, DEFAULT_RELATIVE_DROP_MIN` line:

```python
from .concurrency import run_concurrent
```

Then replace the body of `_review_top_candidates` (currently lines 334-354):

```python
def _review_top_candidates(
    provider,
    component_a: str,
    candidate_proposals: list[CandidateProposal],
    context: str,
    top_n: int,
    llm_warnings: list[str],
) -> tuple[list[CandidateReview], dict[str, CandidateReview]]:
    review_notes: list[CandidateReview] = []
    review_by_smiles: dict[str, CandidateReview] = {}
    top_proposals = candidate_proposals[: max(0, top_n)]
    results = run_concurrent(top_proposals, lambda p: provider.review_candidate(component_a, p.smiles, context))
    for proposal, res in zip(top_proposals, results):
        if res.error is not None:
            llm_warnings.append(f"LLM candidate review failed for {proposal.smiles}: {res.error}")
            continue
        review = res.value
        if review.smiles != proposal.smiles:
            llm_warnings.append(f"LLM candidate review returned wrong SMILES for {proposal.smiles}")
            continue
        review_notes.append(review)
        review_by_smiles[proposal.smiles] = review
    return review_notes, review_by_smiles
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm_orchestrator.py -v`
Expected: All tests PASS, including the new `test_review_top_candidates_runs_calls_concurrently` and the pre-existing `test_review_top_candidates_ignores_wrong_smiles`.

Then run the full suite to confirm no regressions elsewhere:

Run: `pytest tests/ -q --tb=short --ignore=tests/test_benchmarks_examples.py`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/orchestrator.py tests/test_llm_orchestrator.py
git commit -m "feat: run candidate LLM reviews concurrently"
```

---

### Task 3: Parallelize the chemistry advisor loop

**Files:**
- Modify: `des_multi_agent/orchestrator.py:1074-1078`
- Test: `tests/test_llm_orchestrator.py`

**Interfaces:**
- Consumes: `run_concurrent` (imported in Task 2, already available in this file).
- Produces: no change to `advisor_assessments`'s type (`list[ChemistryAssessment]`) or the warning message format. Nothing later in this plan depends on this task.

- [ ] **Step 1: Write the failing test**

This test builds `AnnotatedResult` instances using the exact same
`CurvePrediction`/`DesResult`/`MinimumTmUncertainty` construction as the
existing `_annotated` helper inside
`test_apply_review_penalties_reorders_results`
(`tests/test_llm_orchestrator.py:514-547`), copied verbatim below. It
exercises the `run_concurrent` pattern directly rather than the real
`orchestrator.py` loop, for the same reason as Task 4/5: this loop is
inline inside a larger function, not a standalone callable.

Append to `tests/test_llm_orchestrator.py`:

`DesResult`, `AnnotatedResult`, `MinimumTmUncertainty`, `CurvePrediction`,
and `ChemistryAssessment` are all already imported at the top of this test
file — only `threading` and `run_concurrent` are new for this test.

```python
def test_chemistry_advisor_loop_runs_calls_concurrently():
    import threading

    from des_multi_agent.concurrency import run_concurrent

    n_items = 3
    barrier = threading.Barrier(n_items, timeout=2.0)

    class BarrierAssessProvider:
        def assess_candidate_chemistry(self, candidate_smiles, context, memory_notes):
            barrier.wait()
            return [
                ChemistryAssessment(
                    smiles=candidate_smiles,
                    decision="stable",
                    confidence=0.8,
                    rationale="ok",
                )
            ]

    def _annotated(smiles: str) -> AnnotatedResult:
        curve = CurvePrediction(
            smiles_a="CCO", smiles_b=smiles, ratios=[0.1], tm_pred_k=[250.0],
            t1_k=300.0, t2_k=300.0, checkpoint_path="ckpt.pt",
        )
        result = DesResult(curve=curve, absolute_pass=True, relative_pass=True, is_des=True, rationale="ok", min_tm_k=250.0)
        uncertainty = MinimumTmUncertainty(
            component_a="CCO", component_b=smiles, repeated_values=(250.0,), mean_tm_k=250.0,
            std_tm_k=0.0, min_tm_k=250.0, max_tm_k=250.0, trust_score=0.9, uncertainty_flag="low",
            explanation="demo", checkpoint_path="ckpt.pt", config_path="cfg.yaml",
        )
        return AnnotatedResult(result=result, uncertainty=uncertainty, trust_score=0.9, ranking_score=0.9)

    annotated_results = [_annotated(f"C{i}") for i in range(n_items)]
    llm_warnings: list[str] = []
    provider = BarrierAssessProvider()
    advisor_items = annotated_results[: min(5, len(annotated_results))]

    results = run_concurrent(
        advisor_items,
        lambda item: provider.assess_candidate_chemistry(item.result.curve.smiles_b, "context", []),
    )
    advisor_assessments: list[ChemistryAssessment] = []
    for item, res in zip(advisor_items, results):
        if res.error is not None:
            llm_warnings.append(f"LLM chemistry assessment failed for {item.result.curve.smiles_b}: {res.error}")
            continue
        advisor_assessments.extend(res.value)

    assert len(advisor_assessments) == n_items
    assert llm_warnings == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_orchestrator.py::test_chemistry_advisor_loop_runs_calls_concurrently -v`
Expected: this test calls `run_concurrent` directly (already implemented
in Task 1), so it PASSES immediately — it proves the pattern works, not
that the real call site uses it yet. There is no isolated RED state for
this loop specifically, since it lives inline inside the larger orchestrator
function rather than a standalone callable. The real regression coverage
that the *call site* changed correctly is the full-suite run in Step 4
below, which exercises the actual modified loop inside the orchestrator's
full multi-cycle flow.

- [ ] **Step 3: Write minimal implementation**

In `des_multi_agent/orchestrator.py`, replace the loop at lines 1074-1078:

```python
        for item in annotated_results[: min(5, len(annotated_results))]:
            try:
                advisor_assessments.extend(
                    provider.assess_candidate_chemistry(item.result.curve.smiles_b, advisor_context, advisor_memory_notes)
                )
            except Exception as exc:
                llm_warnings.append(f"LLM chemistry assessment failed for {item.result.curve.smiles_b}: {exc}")
```

with:

```python
        advisor_items = annotated_results[: min(5, len(annotated_results))]
        advisor_results = run_concurrent(
            advisor_items,
            lambda item: provider.assess_candidate_chemistry(item.result.curve.smiles_b, advisor_context, advisor_memory_notes),
        )
        for item, res in zip(advisor_items, advisor_results):
            if res.error is not None:
                llm_warnings.append(f"LLM chemistry assessment failed for {item.result.curve.smiles_b}: {res.error}")
                continue
            advisor_assessments.extend(res.value)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm_orchestrator.py -v`
Expected: All PASS.

Then run the full suite:

Run: `pytest tests/ -q --tb=short --ignore=tests/test_benchmarks_examples.py`
Expected: All PASS — this is the regression check for the actual modified call site inside the orchestrator's full multi-cycle flow.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/orchestrator.py tests/test_llm_orchestrator.py
git commit -m "feat: run chemistry advisor assessments concurrently"
```

---

### Task 4: Parallelize `review_ligand` loop in `metal_binding_screen.py`

**Files:**
- Modify: `des_multi_agent/workflows/metal_binding_screen.py:1-20` (add import), `:265-270` (loop body)
- Test: `tests/test_metal_binding_screen.py`

**Interfaces:**
- Consumes: `run_concurrent` from Task 1 (`des_multi_agent.concurrency`), imported here as `from ..concurrency import run_concurrent` (workflows subpackage uses double-dot relative imports for parent-package modules — see the existing `from ..chemistry_filter import canonicalize_smiles` line in this file for the pattern).
- Produces: no change to `all_reviews`'s type (`list[CandidateReview]`) or warning message format. Nothing later in this plan depends on this task.

- [ ] **Step 1: Write the failing test**

`cycle_results` items are `LigandScreenResult` (defined in this file,
`des_multi_agent/workflows/metal_binding_screen.py:28-35`, fields:
`metal_ion, ligand_smiles, prediction, log_k, source, source_id,
rationale`), where `prediction` is a `StabilityConstantPrediction`
(defined in `des_multi_agent/predictors/stability_constants.py:16-25`,
fields: `task, value, units, model_name, source, warnings, metadata,
metal_ion, ligand`). This test builds `cycle_results` directly rather than
running the full `run_metal_binding_screen` pipeline (whose rule-based
candidate count isn't guaranteed ahead of time), so the barrier size is
deterministic. It exercises the exact `run_concurrent` call shape Step 3
inlines into the real loop — the same strategy Task 3 uses for the
advisor loop, since neither loop is a standalone function that can be
called and driven to a guaranteed-`BrokenBarrierError` RED state in
isolation.

Append to `tests/test_metal_binding_screen.py`:

```python
def test_metal_screen_llm_reviews_run_concurrently():
    import threading

    from des_multi_agent.concurrency import run_concurrent
    from des_multi_agent.predictors.stability_constants import StabilityConstantPrediction
    from des_multi_agent.workflows.metal_binding_screen import LigandScreenResult

    n_ligands = 3
    barrier = threading.Barrier(n_ligands, timeout=2.0)

    class BarrierProvider:
        def review_ligand(self, metal_ion, ligand_smiles, context):
            barrier.wait()
            return CandidateReview(
                smiles=ligand_smiles, decision="keep", confidence=0.9, rationale="ok", notes=[],
            )

    def _cycle_result(smiles: str) -> LigandScreenResult:
        prediction = StabilityConstantPrediction(
            task="log_k", value=8.0, units="log_k", model_name="heuristic",
            source="heuristic", warnings=(), metadata={}, metal_ion="Cu2+", ligand=smiles,
        )
        return LigandScreenResult(
            metal_ion="Cu2+", ligand_smiles=smiles, prediction=prediction,
            log_k=8.0, source="heuristic", source_id="rule", rationale="demo",
        )

    cycle_results = [_cycle_result(f"C{i}") for i in range(n_ligands)]
    llm_provider = BarrierProvider()
    context = "context"
    all_reviews: list[CandidateReview] = []
    all_warnings: list[str] = []

    review_results = run_concurrent(cycle_results, lambda r: llm_provider.review_ligand("Cu2+", r.ligand_smiles, context))
    for r, res in zip(cycle_results, review_results):
        if res.error is not None:
            all_warnings.append(f"LLM review failed for {r.ligand_smiles}: {res.error}")
            continue
        all_reviews.append(res.value)

    assert len(all_reviews) == n_ligands
    assert all_warnings == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_metal_binding_screen.py::test_metal_screen_llm_reviews_run_concurrently -v`
Expected: this test calls `run_concurrent` directly (already implemented
in Task 1), so it PASSES immediately — it proves the pattern works, not
that the real call site uses it yet. There is no isolated RED state for
this loop specifically, since it lives inline inside
`run_metal_binding_screen`'s cycle loop rather than a standalone function.
The real regression coverage that the *call site* changed correctly is
the full-suite run in Step 4 below.

- [ ] **Step 3: Write minimal implementation**

In `des_multi_agent/workflows/metal_binding_screen.py`, add the import after the existing `from ._metal_helpers import _apply_ligand_reality_gate` line:

```python
from ..concurrency import run_concurrent
```

Then replace the loop at lines 265-270:

```python
            for r in cycle_results:
                try:
                    review = llm_provider.review_ligand(metal_ion, r.ligand_smiles, context)
                    all_reviews.append(review)
                except Exception as exc:
                    all_warnings.append(f"LLM review failed for {r.ligand_smiles}: {exc}")
```

with:

```python
            review_results = run_concurrent(cycle_results, lambda r: llm_provider.review_ligand(metal_ion, r.ligand_smiles, context))
            for r, res in zip(cycle_results, review_results):
                if res.error is not None:
                    all_warnings.append(f"LLM review failed for {r.ligand_smiles}: {res.error}")
                    continue
                all_reviews.append(res.value)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_metal_binding_screen.py -v`
Expected: All PASS.

Then run the full suite:

Run: `pytest tests/ -q --tb=short --ignore=tests/test_benchmarks_examples.py`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/workflows/metal_binding_screen.py tests/test_metal_binding_screen.py
git commit -m "feat: run metal-binding ligand LLM reviews concurrently"
```

---

### Task 5: Parallelize `review_ligand` loop in `metal_binding_selectivity.py`

**Files:**
- Modify: `des_multi_agent/workflows/metal_binding_selectivity.py:1-20` (add import), `:441-446` (loop body)
- Test: `tests/test_metal_selectivity_screen.py`

**Interfaces:**
- Consumes: `run_concurrent` from Task 1 (`des_multi_agent.concurrency`), imported as `from ..concurrency import run_concurrent` (same pattern as Task 4).
- Produces: no change to `all_reviews`'s type or warning message format. This is the last task in this plan.

- [ ] **Step 1: Write the failing test**

This module has its own result type, `SelectivityResult`
(`des_multi_agent/workflows/metal_binding_selectivity.py:42-63`, fields:
`ligand_smiles, log_k_target, log_k_competitor, delta_log_k,
composite_score, source, source_id, rationale`, plus
`log_k_competitors: dict[str, float] = field(default_factory=dict)` and
`worst_competitor_metal: str = ""` which default, so they don't need to be
passed explicitly) — distinct from `metal_binding_screen.py`'s
`LigandScreenResult`. Same rationale as Task 4: this loop is inline inside
`run_metal_selectivity_screen`'s cycle loop, not a standalone function, so
this test exercises the `run_concurrent` pattern directly.

Append to `tests/test_metal_selectivity_screen.py`:

```python
def test_metal_selectivity_llm_reviews_run_concurrently():
    import threading

    from des_multi_agent.concurrency import run_concurrent
    from des_multi_agent.workflows.metal_binding_selectivity import SelectivityResult

    n_ligands = 3
    barrier = threading.Barrier(n_ligands, timeout=2.0)

    class BarrierProvider:
        def review_ligand(self, metal_ion, ligand_smiles, context):
            barrier.wait()
            return CandidateReview(
                smiles=ligand_smiles, decision="keep", confidence=0.9, rationale="ok", notes=[],
            )

    def _cycle_result(smiles: str) -> SelectivityResult:
        return SelectivityResult(
            ligand_smiles=smiles, log_k_target=8.0, log_k_competitor=6.0,
            delta_log_k=2.0, composite_score=0.8, source="heuristic",
            source_id="rule", rationale="demo",
        )

    cycle_results = [_cycle_result(f"C{i}") for i in range(n_ligands)]
    llm_provider = BarrierProvider()
    context = "context"
    target_metal = "Ni2+"
    all_reviews: list[CandidateReview] = []
    all_warnings: list[str] = []

    review_results = run_concurrent(cycle_results, lambda r: llm_provider.review_ligand(target_metal, r.ligand_smiles, context))
    for r, res in zip(cycle_results, review_results):
        if res.error is not None:
            all_warnings.append(f"LLM review failed for {r.ligand_smiles}: {res.error}")
            continue
        all_reviews.append(res.value)

    assert len(all_reviews) == n_ligands
    assert all_warnings == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_metal_selectivity_screen.py::test_metal_selectivity_llm_reviews_run_concurrently -v`
Expected: this test calls `run_concurrent` directly (already implemented
in Task 1), so it PASSES immediately — same rationale as Task 4's Step 2.
The real regression coverage for this call site is the full-suite run in
Step 4 below.

- [ ] **Step 3: Write minimal implementation**

In `des_multi_agent/workflows/metal_binding_selectivity.py`, add the import after the existing `from ._metal_helpers import _apply_ligand_reality_gate` line:

```python
from ..concurrency import run_concurrent
```

Then replace the loop at lines 441-446:

```python
            for r in cycle_results:
                try:
                    review = llm_provider.review_ligand(target_metal, r.ligand_smiles, context)
                    all_reviews.append(review)
                except Exception as exc:
                    all_warnings.append(f"LLM review failed for {r.ligand_smiles}: {exc}")
```

with:

```python
            review_results = run_concurrent(cycle_results, lambda r: llm_provider.review_ligand(target_metal, r.ligand_smiles, context))
            for r, res in zip(cycle_results, review_results):
                if res.error is not None:
                    all_warnings.append(f"LLM review failed for {r.ligand_smiles}: {res.error}")
                    continue
                all_reviews.append(res.value)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_metal_selectivity_screen.py -v`
Expected: All PASS.

Then run the full suite:

Run: `pytest tests/ -q --tb=short --ignore=tests/test_benchmarks_examples.py`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/workflows/metal_binding_selectivity.py tests/test_metal_selectivity_screen.py
git commit -m "feat: run metal-selectivity ligand LLM reviews concurrently"
```

---

## Final Verification

After all five tasks are complete:

```bash
pytest tests/ -q --tb=short --ignore=tests/test_benchmarks_examples.py
```

Expected: all tests pass, including the 4 new concurrency-proof tests
(`test_run_concurrent_*` x4 in Task 1, plus one barrier-based test per
call site in Tasks 2, 4, 5, and the pattern-validation test in Task 3).

As a final live sanity check (optional, not required for merge — needs a
running vLLM or Ollama server), re-run the same benchmark from
`docs/future-improvements.md` item 21 (`examples.demo_des_search
--component-a ethanol --n 10`) against a vLLM server and confirm its
request log now shows `Running: N reqs` with `N > 1` at some point during
the candidate-review stage, proving the fix actually changes runtime
behavior end to end.
