# Selectivity-DES Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-phase pipeline that screens ligands for metal-ion selectivity (Phase 1) then searches for DES partners for the top selective ligands (Phase 2), with an outer feedback loop that steers Phase 1 toward DES-compatible scaffolds.

**Architecture:** Three nested loops — outer (convergence-guarded, max `n_outer_cycles`), Phase 1 (`run_metal_selectivity_screen`, `n_selectivity_cycles`), and Phase 2 (`run_multi_cycle_search` per shortlisted ligand, `n_des_cycles`). Phase 1 → Phase 2 bridge applies a `min_delta_log_k` threshold + `top_ligands` cap. After Phase 2, DES-compatible and DES-incompatible ligand SMILES are fed back into the Phase 1 LLM context in the next outer cycle.

**Tech Stack:** Python 3.11+, RDKit, existing `run_metal_selectivity_screen`, `run_multi_cycle_search`, `reporting.py` pattern, `argparse`, `pytest` + `unittest.mock`.

---

## File Map

| File | Change |
|------|--------|
| `des_multi_agent/workflows/metal_binding_selectivity.py` | Add `des_compatible_hints` + `des_incompatible_hints` params to `_build_selectivity_context` and `run_metal_selectivity_screen` |
| `des_multi_agent/workflows/selectivity_des_pipeline.py` | **NEW** — `LigandDesResult`, `SelectivityDesPipelineOutcome`, `_bridge_filter`, `run_selectivity_des_pipeline` |
| `des_multi_agent/reporting.py` | Add `format_selectivity_des_report` |
| `des_multi_agent/cli.py` | Add `selectivity-des` to `--workflow` choices, 5 new args, routing branch, updated import |
| `tests/test_selectivity_des_pipeline.py` | **NEW** — full test suite |
| `tests/test_metal_selectivity_screen.py` | Add 2 tests for hint params |

---

## Task 1: Add DES Hints to Selectivity Context Builder

**Files:**
- Modify: `des_multi_agent/workflows/metal_binding_selectivity.py`
- Test: `tests/test_metal_selectivity_screen.py`

- [ ] **Step 1: Write the failing tests**

Open `tests/test_metal_selectivity_screen.py` and append:

```python
from des_multi_agent.workflows.metal_binding_selectivity import _build_selectivity_context


def test_build_context_includes_des_hints_when_provided():
    ctx = _build_selectivity_context(
        "Cu2+", "Zn2+", [], 2, 0.5, 0.5,
        des_compatible_hints=["NCC(=O)O"],
        des_incompatible_hints=["c1ccncc1"],
    )
    assert "formed DES" in ctx
    assert "NCC(=O)O" in ctx
    assert "NOT form DES" in ctx
    assert "c1ccncc1" in ctx


def test_build_context_omits_hint_sections_when_none():
    ctx = _build_selectivity_context("Cu2+", "Zn2+", [], 1, 0.5, 0.5)
    assert "formed DES" not in ctx
    assert "NOT form DES" not in ctx
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_metal_selectivity_screen.py::test_build_context_includes_des_hints_when_provided tests/test_metal_selectivity_screen.py::test_build_context_omits_hint_sections_when_none -v
```

Expected: FAIL — `_build_selectivity_context` does not yet accept `des_compatible_hints`.

- [ ] **Step 3: Update `_build_selectivity_context` signature and body**

In `des_multi_agent/workflows/metal_binding_selectivity.py`, replace the existing `_build_selectivity_context` function with:

```python
def _build_selectivity_context(
    target_metal: str,
    competitor_metal: str,
    prev_results: list[SelectivityResult],
    cycle: int,
    w_affinity: float,
    w_selectivity: float,
    des_compatible_hints: list[str] | None = None,
    des_incompatible_hints: list[str] | None = None,
) -> str:
    lines = [
        f"Target metal: {target_metal}",
        f"Competitor metal: {competitor_metal}",
        f"Selectivity weight: {w_selectivity} | Affinity weight: {w_affinity}",
        f"Cycle: {cycle}",
    ]
    if prev_results:
        lines.append("Top ligands from previous cycle (highest composite score first):")
        for r in prev_results[:5]:
            lines.append(
                f"  - {r.ligand_smiles}: log_K({target_metal})={r.log_k_target:.2f}, "
                f"log_K({competitor_metal})={r.log_k_competitor:.2f}, "
                f"ΔlogK={r.delta_log_k:.2f}, score={r.composite_score:.2f}"
            )
    if des_compatible_hints:
        lines.append("Ligands that formed DES in previous pass (prefer similar scaffolds):")
        for smiles in des_compatible_hints:
            lines.append(f"  - {smiles}")
    if des_incompatible_hints:
        lines.append("Ligands that did NOT form DES (avoid similar scaffolds):")
        for smiles in des_incompatible_hints:
            lines.append(f"  - {smiles}")
    return "\n".join(lines)
```

- [ ] **Step 4: Update `run_metal_selectivity_screen` signature**

Add the two new parameters to `run_metal_selectivity_screen` (keep all existing params unchanged, append at the end):

```python
def run_metal_selectivity_screen(
    target_metal: str,
    competitor_metal: str,
    n: int = 20,
    model_path=None,
    llm_provider=None,
    constraints: dict | None = None,
    n_cycles: int = 1,
    w_affinity: float = 0.5,
    w_selectivity: float = 0.5,
    des_compatible_hints: list[str] | None = None,
    des_incompatible_hints: list[str] | None = None,
) -> SelectivityScreenOutcome:
```

Inside the loop, both calls to `_build_selectivity_context` must pass the new params. Find the two calls (one before LLM brainstorm, one before LLM review) and update each:

```python
context = _build_selectivity_context(
    target_metal, competitor_metal, prev_cycle_results, cycle, w_affinity, w_selectivity,
    des_compatible_hints=des_compatible_hints,
    des_incompatible_hints=des_incompatible_hints,
)
```

- [ ] **Step 5: Run all tests to verify they pass and nothing regresses**

```bash
pytest tests/test_metal_selectivity_screen.py -v
```

Expected: all pass (including the 2 new tests).

- [ ] **Step 6: Commit**

```bash
git add des_multi_agent/workflows/metal_binding_selectivity.py tests/test_metal_selectivity_screen.py
git commit -m "feat(selectivity): add des_compatible_hints params to selectivity context builder"
```

---

## Task 2: Dataclasses and Bridge Filter

**Files:**
- Create: `des_multi_agent/workflows/selectivity_des_pipeline.py`
- Create: `tests/test_selectivity_des_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_selectivity_des_pipeline.py`:

```python
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from des_multi_agent.workflows.selectivity_des_pipeline import (
    LigandDesResult,
    SelectivityDesPipelineOutcome,
    _bridge_filter,
)
from des_multi_agent.workflows.metal_binding_selectivity import (
    SelectivityResult,
    SelectivityScreenOutcome,
)


def _sel_result(smiles: str, delta: float = 1.0, score: float = 5.0) -> SelectivityResult:
    return SelectivityResult(
        ligand_smiles=smiles,
        log_k_target=10.0,
        log_k_competitor=10.0 - delta,
        delta_log_k=delta,
        composite_score=score,
        source="heuristic",
        source_id="",
        rationale="",
    )


def _sel_outcome(smiles_list: list[str]) -> SelectivityScreenOutcome:
    return SelectivityScreenOutcome(
        target_metal="Cu2+",
        competitor_metal="Zn2+",
        results=[_sel_result(s) for s in smiles_list],
        n_screened=len(smiles_list),
        n_cycles=1,
    )


# --- LigandDesResult ---

def test_ligand_des_result_des_compatible_true_when_any_is_des():
    dr = MagicMock()
    dr.is_des = True
    ldr = LigandDesResult(
        ligand=_sel_result("NCC(=O)O"),
        des_results=[dr],
        n_des_screened=5,
        des_compatible=True,
    )
    assert ldr.des_compatible is True


def test_ligand_des_result_des_compatible_false_when_no_des():
    dr = MagicMock()
    dr.is_des = False
    ldr = LigandDesResult(
        ligand=_sel_result("NCC(=O)O"),
        des_results=[dr],
        n_des_screened=5,
        des_compatible=False,
    )
    assert ldr.des_compatible is False


# --- _bridge_filter ---

def test_bridge_filter_keeps_ligands_above_threshold():
    results = [_sel_result("AAA", delta=1.0), _sel_result("BBB", delta=-0.1)]
    warnings: list[str] = []
    out = _bridge_filter(results, min_delta_log_k=0.5, top_n=3, warnings=warnings)
    assert len(out) == 1
    assert out[0].ligand_smiles == "AAA"
    assert warnings == []


def test_bridge_filter_respects_top_n_cap():
    results = [_sel_result(f"S{i}", delta=float(i + 1)) for i in range(5)]
    warnings: list[str] = []
    out = _bridge_filter(results, min_delta_log_k=0.0, top_n=2, warnings=warnings)
    assert len(out) == 2


def test_bridge_filter_fallback_when_all_below_threshold():
    results = [_sel_result("AAA", delta=-0.5)]
    warnings: list[str] = []
    out = _bridge_filter(results, min_delta_log_k=1.0, top_n=3, warnings=warnings)
    assert out[0].ligand_smiles == "AAA"
    assert len(warnings) == 1
    assert "unconditionally" in warnings[0]


def test_bridge_filter_empty_results_returns_empty():
    warnings: list[str] = []
    out = _bridge_filter([], min_delta_log_k=0.0, top_n=3, warnings=warnings)
    assert out == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_selectivity_des_pipeline.py -v
```

Expected: FAIL — module not found.

- [ ] **Step 3: Create `selectivity_des_pipeline.py` with dataclasses and bridge filter**

Create `des_multi_agent/workflows/selectivity_des_pipeline.py`:

```python
from __future__ import annotations

import sys
from dataclasses import dataclass, field

from ..evaluation import DesResult
from ..multi_cycle import run_multi_cycle_search
from .metal_binding_selectivity import (
    SelectivityResult,
    SelectivityScreenOutcome,
    run_metal_selectivity_screen,
)


@dataclass(frozen=True)
class LigandDesResult:
    ligand: SelectivityResult
    des_results: list[DesResult]
    n_des_screened: int
    des_compatible: bool


@dataclass
class SelectivityDesPipelineOutcome:
    target_metal: str
    competitor_metal: str
    selectivity_outcome: SelectivityScreenOutcome
    ligand_des_results: list[LigandDesResult]
    n_outer_cycles_run: int
    converged: bool
    warnings: list[str] = field(default_factory=list)


def _bridge_filter(
    results: list[SelectivityResult],
    min_delta_log_k: float,
    top_n: int,
    warnings: list[str],
) -> list[SelectivityResult]:
    if not results:
        return []
    filtered = [r for r in results if r.delta_log_k >= min_delta_log_k]
    if not filtered:
        warnings.append(
            f"Bridge filter found 0 ligands above min_delta_log_k={min_delta_log_k}; "
            f"using top-{top_n} unconditionally."
        )
        filtered = results
    return filtered[:top_n]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_selectivity_des_pipeline.py -v
```

Expected: all 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/workflows/selectivity_des_pipeline.py tests/test_selectivity_des_pipeline.py
git commit -m "feat(selectivity-des): add dataclasses and bridge filter"
```

---

## Task 3: Outer Loop — `run_selectivity_des_pipeline`

**Files:**
- Modify: `des_multi_agent/workflows/selectivity_des_pipeline.py`
- Modify: `tests/test_selectivity_des_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_selectivity_des_pipeline.py`:

```python
from unittest.mock import patch, call
from des_multi_agent.workflows.selectivity_des_pipeline import run_selectivity_des_pipeline


def _make_multi_cycle_outcome(is_des: bool):
    """Return a minimal MultiCycleOutcome mock."""
    dr = MagicMock()
    dr.is_des = is_des
    dr.min_tm_k = 280.0
    dr.eutectic_ratio_b = 0.5
    dr.rationale = "test"
    dr.curve = MagicMock()
    dr.curve.smiles_b = "CCO"

    search_outcome = MagicMock()
    search_outcome.results = [dr]

    cycle_delta = MagicMock()
    cycle_delta.n_screened = 5

    mco = MagicMock()
    mco.final_outcome = search_outcome
    mco.cycle_deltas = [cycle_delta]
    return mco


@patch("des_multi_agent.workflows.selectivity_des_pipeline.run_multi_cycle_search")
@patch("des_multi_agent.workflows.selectivity_des_pipeline.run_metal_selectivity_screen")
def test_pipeline_returns_outcome_with_correct_shape(mock_sel, mock_des):
    mock_sel.return_value = _sel_outcome(["NCC(=O)O", "NCCN"])
    mock_des.return_value = _make_multi_cycle_outcome(is_des=True)
    outcome = run_selectivity_des_pipeline(
        target_metal="Cu2+",
        competitor_metal="Zn2+",
        checkpoint_path="/fake/ckpt.pt",
        n_outer_cycles=1,
        top_ligands=2,
    )
    assert outcome.target_metal == "Cu2+"
    assert outcome.competitor_metal == "Zn2+"
    assert len(outcome.ligand_des_results) == 2
    assert outcome.n_outer_cycles_run == 1
    assert not outcome.converged


@patch("des_multi_agent.workflows.selectivity_des_pipeline.run_multi_cycle_search")
@patch("des_multi_agent.workflows.selectivity_des_pipeline.run_metal_selectivity_screen")
def test_pipeline_converges_when_des_compatible_set_stable(mock_sel, mock_des):
    mock_sel.return_value = _sel_outcome(["NCC(=O)O"])
    mock_des.return_value = _make_multi_cycle_outcome(is_des=True)
    outcome = run_selectivity_des_pipeline(
        target_metal="Cu2+",
        competitor_metal="Zn2+",
        checkpoint_path="/fake/ckpt.pt",
        n_outer_cycles=3,
        top_ligands=1,
    )
    assert outcome.converged
    assert outcome.n_outer_cycles_run == 2  # stable after pass 2


@patch("des_multi_agent.workflows.selectivity_des_pipeline.run_multi_cycle_search")
@patch("des_multi_agent.workflows.selectivity_des_pipeline.run_metal_selectivity_screen")
def test_pipeline_runs_all_outer_cycles_when_set_changes(mock_sel, mock_des):
    # Alternate DES compatibility so the set never stabilises
    mock_sel.side_effect = [
        _sel_outcome(["NCC(=O)O"]),
        _sel_outcome(["NCCN"]),
    ]
    mock_des.return_value = _make_multi_cycle_outcome(is_des=True)
    outcome = run_selectivity_des_pipeline(
        target_metal="Cu2+",
        competitor_metal="Zn2+",
        checkpoint_path="/fake/ckpt.pt",
        n_outer_cycles=2,
        top_ligands=1,
    )
    assert not outcome.converged
    assert outcome.n_outer_cycles_run == 2


@patch("des_multi_agent.workflows.selectivity_des_pipeline.run_multi_cycle_search")
@patch("des_multi_agent.workflows.selectivity_des_pipeline.run_metal_selectivity_screen")
def test_pipeline_des_failure_adds_warning_and_continues(mock_sel, mock_des):
    mock_sel.return_value = _sel_outcome(["NCC(=O)O", "NCCN"])
    mock_des.side_effect = [
        RuntimeError("model unavailable"),
        _make_multi_cycle_outcome(is_des=True),
    ]
    outcome = run_selectivity_des_pipeline(
        target_metal="Cu2+",
        competitor_metal="Zn2+",
        checkpoint_path="/fake/ckpt.pt",
        n_outer_cycles=1,
        top_ligands=2,
    )
    assert any("DES search failed" in w for w in outcome.warnings)
    assert len(outcome.ligand_des_results) == 2
    assert outcome.ligand_des_results[0].des_compatible is False
    assert outcome.ligand_des_results[1].des_compatible is True


@patch("des_multi_agent.workflows.selectivity_des_pipeline.run_multi_cycle_search")
@patch("des_multi_agent.workflows.selectivity_des_pipeline.run_metal_selectivity_screen")
def test_pipeline_passes_des_hints_on_second_outer_cycle(mock_sel, mock_des):
    mock_sel.return_value = _sel_outcome(["NCC(=O)O"])
    mock_des.return_value = _make_multi_cycle_outcome(is_des=True)
    run_selectivity_des_pipeline(
        target_metal="Cu2+",
        competitor_metal="Zn2+",
        checkpoint_path="/fake/ckpt.pt",
        n_outer_cycles=2,
        top_ligands=1,
    )
    # outer cycle 1: no hints yet
    first_call_kwargs = mock_sel.call_args_list[0][1]
    assert first_call_kwargs.get("des_compatible_hints") is None
    # outer cycle 2: compatible hint present (set was {"NCC(=O)O"})
    second_call_kwargs = mock_sel.call_args_list[1][1]
    assert "NCC(=O)O" in second_call_kwargs.get("des_compatible_hints", [])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_selectivity_des_pipeline.py::test_pipeline_returns_outcome_with_correct_shape tests/test_selectivity_des_pipeline.py::test_pipeline_converges_when_des_compatible_set_stable -v
```

Expected: FAIL — `run_selectivity_des_pipeline` not defined.

- [ ] **Step 3: Implement `run_selectivity_des_pipeline` in `selectivity_des_pipeline.py`**

Append to `des_multi_agent/workflows/selectivity_des_pipeline.py`:

```python
def run_selectivity_des_pipeline(
    target_metal: str,
    competitor_metal: str,
    checkpoint_path: str,
    config_path: str = "ml_des_mp/config.yaml",
    n_ligands: int = 20,
    n_des_candidates: int = 20,
    n_selectivity_cycles: int = 3,
    n_des_cycles: int = 3,
    n_outer_cycles: int = 2,
    min_delta_log_k: float = 0.0,
    top_ligands: int = 3,
    w_affinity: float = 0.5,
    w_selectivity: float = 0.5,
    stability_model_path=None,
    llm_cfg=None,
    constraints: dict | None = None,
) -> SelectivityDesPipelineOutcome:
    all_warnings: list[str] = []
    des_compatible_smiles: set[str] = set()
    des_incompatible_smiles: set[str] = set()
    prev_compatible: set[str] = set()
    final_selectivity_outcome: SelectivityScreenOutcome | None = None
    final_ligand_des_results: list[LigandDesResult] = []
    converged = False
    outer_cycle_count = 0

    llm_provider = None
    if llm_cfg is not None:
        from ..llm.factory import build_llm_provider
        llm_provider = build_llm_provider(llm_cfg)

    for outer_cycle in range(1, n_outer_cycles + 1):
        outer_cycle_count = outer_cycle
        print(
            f"[outer {outer_cycle}/{n_outer_cycles}] phase 1: selectivity screening",
            file=sys.stderr, flush=True,
        )

        sel_outcome = run_metal_selectivity_screen(
            target_metal=target_metal,
            competitor_metal=competitor_metal,
            n=n_ligands,
            model_path=stability_model_path,
            llm_provider=llm_provider,
            constraints=constraints,
            n_cycles=n_selectivity_cycles,
            w_affinity=w_affinity,
            w_selectivity=w_selectivity,
            des_compatible_hints=list(des_compatible_smiles) if des_compatible_smiles else None,
            des_incompatible_hints=list(des_incompatible_smiles) if des_incompatible_smiles else None,
        )
        final_selectivity_outcome = sel_outcome
        all_warnings.extend(sel_outcome.warnings)

        shortlisted = _bridge_filter(
            sel_outcome.results, min_delta_log_k, top_ligands, all_warnings
        )

        print(
            f"[outer {outer_cycle}/{n_outer_cycles}] phase 2: DES search for "
            f"{len(shortlisted)} ligand(s)",
            file=sys.stderr, flush=True,
        )

        ligand_des_results: list[LigandDesResult] = []
        new_compatible: set[str] = set()
        new_incompatible: set[str] = set()

        for ligand_result in shortlisted:
            try:
                des_mco = run_multi_cycle_search(
                    component_a=ligand_result.ligand_smiles,
                    n=n_des_candidates,
                    checkpoint_path=checkpoint_path,
                    config_path=config_path,
                    n_cycles=n_des_cycles,
                    llm_cfg=llm_cfg,
                )
                des_compat = any(r.is_des for r in des_mco.final_outcome.results)
                n_screened = sum(d.n_screened for d in des_mco.cycle_deltas)
                ldr = LigandDesResult(
                    ligand=ligand_result,
                    des_results=des_mco.final_outcome.results,
                    n_des_screened=n_screened,
                    des_compatible=des_compat,
                )
            except Exception as exc:
                all_warnings.append(
                    f"DES search failed for {ligand_result.ligand_smiles}: {exc}"
                )
                ldr = LigandDesResult(
                    ligand=ligand_result,
                    des_results=[],
                    n_des_screened=0,
                    des_compatible=False,
                )

            ligand_des_results.append(ldr)
            if ldr.des_compatible:
                new_compatible.add(ligand_result.ligand_smiles)
            else:
                new_incompatible.add(ligand_result.ligand_smiles)

        final_ligand_des_results = ligand_des_results
        des_compatible_smiles = new_compatible
        des_incompatible_smiles = new_incompatible

        if outer_cycle > 1 and new_compatible == prev_compatible:
            converged = True
            print(
                f"[outer {outer_cycle}/{n_outer_cycles}] DES-compatible set stable — converged early",
                file=sys.stderr, flush=True,
            )
            break

        prev_compatible = new_compatible

    return SelectivityDesPipelineOutcome(
        target_metal=target_metal,
        competitor_metal=competitor_metal,
        selectivity_outcome=final_selectivity_outcome,
        ligand_des_results=final_ligand_des_results,
        n_outer_cycles_run=outer_cycle_count,
        converged=converged,
        warnings=all_warnings,
    )
```

- [ ] **Step 4: Run all pipeline tests to verify they pass**

```bash
pytest tests/test_selectivity_des_pipeline.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Run full suite to verify no regressions**

```bash
pytest tests/ -x -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add des_multi_agent/workflows/selectivity_des_pipeline.py tests/test_selectivity_des_pipeline.py
git commit -m "feat(selectivity-des): implement run_selectivity_des_pipeline outer loop"
```

---

## Task 4: Report Format

**Files:**
- Modify: `des_multi_agent/reporting.py`
- Modify: `tests/test_selectivity_des_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_selectivity_des_pipeline.py`:

```python
from des_multi_agent.reporting import format_selectivity_des_report
from des_multi_agent.workflows.selectivity_des_pipeline import (
    SelectivityDesPipelineOutcome,
)


def _make_pipeline_outcome(des_compatible: bool = True) -> SelectivityDesPipelineOutcome:
    dr = MagicMock()
    dr.is_des = des_compatible
    dr.min_tm_k = 287.1
    dr.eutectic_ratio_b = 0.33
    dr.rationale = "min Tm=287.1 K"
    dr.curve = MagicMock()
    dr.curve.smiles_b = "CC(=O)NCCO"

    ligand = _sel_result("NCC(=O)O", delta=1.0, score=7.5)
    ldr = LigandDesResult(
        ligand=ligand,
        des_results=[dr] if des_compatible else [],
        n_des_screened=10,
        des_compatible=des_compatible,
    )
    sel_out = _sel_outcome(["NCC(=O)O", "NCCN"])
    return SelectivityDesPipelineOutcome(
        target_metal="Cu2+",
        competitor_metal="Zn2+",
        selectivity_outcome=sel_out,
        ligand_des_results=[ldr],
        n_outer_cycles_run=2,
        converged=True,
        warnings=[],
    )


def test_report_contains_section_1_header():
    report = format_selectivity_des_report(_make_pipeline_outcome())
    assert "Section 1: Selectivity Results" in report


def test_report_contains_section_2_header():
    report = format_selectivity_des_report(_make_pipeline_outcome())
    assert "Section 2: DES Partners" in report


def test_report_section_1_has_des_compatible_column():
    report = format_selectivity_des_report(_make_pipeline_outcome(des_compatible=True))
    assert "des_compatible" in report
    assert "yes" in report


def test_report_section_2_shows_des_partner_when_compatible():
    report = format_selectivity_des_report(_make_pipeline_outcome(des_compatible=True))
    assert "CC(=O)NCCO" in report
    assert "DES-compatible: YES" in report


def test_report_section_2_shows_no_partners_when_incompatible():
    report = format_selectivity_des_report(_make_pipeline_outcome(des_compatible=False))
    assert "No DES partners found" in report
    assert "DES-compatible: NO" in report


def test_report_pipeline_outcome_none_selectivity_does_not_crash():
    outcome = SelectivityDesPipelineOutcome(
        target_metal="Cu2+",
        competitor_metal="Zn2+",
        selectivity_outcome=None,
        ligand_des_results=[],
        n_outer_cycles_run=0,
        converged=False,
    )
    report = format_selectivity_des_report(outcome)
    assert "Selectivity-DES Pipeline" in report
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_selectivity_des_pipeline.py::test_report_contains_section_1_header tests/test_selectivity_des_pipeline.py::test_report_section_2_shows_no_partners_when_incompatible -v
```

Expected: FAIL — `format_selectivity_des_report` not defined.

- [ ] **Step 3: Implement `format_selectivity_des_report` in `reporting.py`**

Append to the end of `des_multi_agent/reporting.py`:

```python
def format_selectivity_des_report(outcome) -> str:
    """Render a two-section selectivity-DES pipeline report."""
    n_compatible = sum(1 for r in outcome.ligand_des_results if r.des_compatible)
    header = [
        f"=== Selectivity-DES Pipeline: {outcome.target_metal} over {outcome.competitor_metal} ===",
        f"Outer cycles run: {outcome.n_outer_cycles_run} | Converged: {'yes' if outcome.converged else 'no'}",
        f"Shortlisted ligands: {len(outcome.ligand_des_results)} | DES-compatible: {n_compatible}",
        "=" * 52,
    ]

    # Section 1: selectivity results with des_compatible flag
    sec1 = [
        "",
        "--- Section 1: Selectivity Results ---",
        "",
        "ligand | log_k_target | log_k_competitor | delta_log_k | score | des_compatible",
    ]
    sel_results = outcome.selectivity_outcome.results if outcome.selectivity_outcome else []
    compatible_smiles = {
        r.ligand.ligand_smiles for r in outcome.ligand_des_results if r.des_compatible
    }
    for r in sel_results:
        des_flag = "yes" if r.ligand_smiles in compatible_smiles else "no"
        sec1.append(
            f"{r.ligand_smiles} | {r.log_k_target:.2f} | {r.log_k_competitor:.2f} | "
            f"{r.delta_log_k:.2f} | {r.composite_score:.2f} | {des_flag}"
        )

    # Section 2: per-ligand DES partner blocks
    sec2 = ["", "--- Section 2: DES Partners ---"]
    for ldr in outcome.ligand_des_results:
        r = ldr.ligand
        compat_str = "YES" if ldr.des_compatible else "NO"
        sec2.append(
            f"\nLigand: {r.ligand_smiles}  "
            f"(score={r.composite_score:.2f}, ΔlogK={r.delta_log_k:.2f}) "
            f"— DES-compatible: {compat_str}"
        )
        if ldr.des_results:
            sec2.append("  partner | min_tm_k | eutectic_ratio | rationale")
            for dr in ldr.des_results:
                sec2.append(
                    f"  {dr.curve.smiles_b} | {dr.min_tm_k:.1f} | "
                    f"{dr.eutectic_ratio_b:.2f} | {dr.rationale}"
                )
        else:
            sec2.append("  No DES partners found.")

    warning_lines: list[str] = []
    if outcome.warnings:
        warning_lines.append("")
        warning_lines.append("Warnings:")
        for w in outcome.warnings:
            warning_lines.append(f"- {w}")

    return "\n".join(header + sec1 + sec2 + warning_lines)
```

- [ ] **Step 4: Run report tests to verify they pass**

```bash
pytest tests/test_selectivity_des_pipeline.py -k "report" -v
```

Expected: all 6 report tests pass.

- [ ] **Step 5: Run full suite**

```bash
pytest tests/ -x -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add des_multi_agent/reporting.py tests/test_selectivity_des_pipeline.py
git commit -m "feat(selectivity-des): add format_selectivity_des_report"
```

---

## Task 5: CLI Integration

**Files:**
- Modify: `des_multi_agent/cli.py`
- Modify: `tests/test_selectivity_des_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_selectivity_des_pipeline.py`:

```python
from unittest.mock import patch as _patch
from des_multi_agent.cli import build_parser, main as cli_main


def test_cli_selectivity_des_routes_to_pipeline(tmp_path):
    fake_ckpt = tmp_path / "ckpt.pt"
    fake_ckpt.write_text("x")
    fake_outcome = _make_pipeline_outcome()
    with _patch(
        "des_multi_agent.cli.run_selectivity_des_pipeline",
        return_value=fake_outcome,
    ) as mock_run, _patch("des_multi_agent.cli.format_selectivity_des_report", return_value="REPORT"):
        cli_main([
            "--workflow", "selectivity-des",
            "--target-metal-ion", "Cu2+",
            "--competitor-metal-ion", "Zn2+",
            "--checkpoint-path", str(fake_ckpt),
        ])
    mock_run.assert_called_once()
    kwargs = mock_run.call_args[1]
    assert kwargs["target_metal"] == "Cu2+"
    assert kwargs["competitor_metal"] == "Zn2+"


def test_cli_selectivity_des_requires_target_metal_ion():
    with pytest.raises(SystemExit):
        cli_main([
            "--workflow", "selectivity-des",
            "--competitor-metal-ion", "Zn2+",
            "--checkpoint-path", "/fake/ckpt.pt",
        ])


def test_cli_selectivity_des_requires_checkpoint_path():
    with pytest.raises(SystemExit):
        cli_main([
            "--workflow", "selectivity-des",
            "--target-metal-ion", "Cu2+",
            "--competitor-metal-ion", "Zn2+",
        ])


def test_cli_metal_selectivity_workflow_unchanged():
    """Existing metal-selectivity workflow must still parse without error."""
    parser = build_parser()
    args = parser.parse_args([
        "--workflow", "metal-selectivity",
        "--target-metal-ion", "Cu2+",
        "--competitor-metal-ion", "Zn2+",
    ])
    assert args.workflow == "metal-selectivity"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_selectivity_des_pipeline.py::test_cli_selectivity_des_routes_to_pipeline tests/test_selectivity_des_pipeline.py::test_cli_selectivity_des_requires_target_metal_ion -v
```

Expected: FAIL — `selectivity-des` not a valid workflow choice.

- [ ] **Step 3: Add `selectivity-des` to `--workflow` choices in `build_parser`**

In `des_multi_agent/cli.py`, find the line:
```python
parser.add_argument("--workflow", choices=["des", "metal-binding", "metal-selectivity"], default="des")
```

Replace it with:
```python
parser.add_argument("--workflow", choices=["des", "metal-binding", "metal-selectivity", "selectivity-des"], default="des")
```

- [ ] **Step 4: Add the 5 new CLI arguments**

After the `--selectivity-weight` argument block in `build_parser`, add:

```python
    parser.add_argument(
        "--n-des-candidates",
        type=int,
        default=20,
        dest="n_des_candidates",
        help="DES candidate search breadth per ligand per cycle (selectivity-des workflow)",
    )
    parser.add_argument(
        "--n-des-cycles",
        type=int,
        default=3,
        dest="n_des_cycles",
        help="DES iteration depth per ligand (selectivity-des workflow)",
    )
    parser.add_argument(
        "--n-outer-cycles",
        type=int,
        default=2,
        dest="n_outer_cycles",
        help="Outer loop iteration cap for selectivity-des workflow",
    )
    parser.add_argument(
        "--min-delta-log-k",
        type=float,
        default=0.0,
        dest="min_delta_log_k",
        help="Minimum delta log K threshold for Phase 1 → Phase 2 bridge filter",
    )
    parser.add_argument(
        "--top-ligands",
        type=int,
        default=3,
        dest="top_ligands",
        help="Maximum ligands passed from Phase 1 to Phase 2 (selectivity-des workflow)",
    )
```

- [ ] **Step 5: Update the import line for report functions in `cli.py`**

Find the line importing `format_metal_selectivity_report` (near the top of `cli.py`) and add `format_selectivity_des_report` and `run_selectivity_des_pipeline`:

```python
from .reporting import (
    format_metal_binding_report, format_metal_binding_screen_report,
    format_metal_selectivity_report, format_selectivity_des_report,
)
from .workflows.selectivity_des_pipeline import run_selectivity_des_pipeline
```

- [ ] **Step 6: Add the `selectivity-des` routing branch in `main()`**

In `main()`, immediately before the `if args.workflow == "metal-selectivity":` block, insert:

```python
    if args.workflow == "selectivity-des":
        if not args.target_metal_ion:
            parser.error("selectivity-des workflow requires --target-metal-ion")
        if not args.competitor_metal_ion:
            parser.error("selectivity-des workflow requires --competitor-metal-ion")
        if not args.checkpoint_path:
            parser.error("selectivity-des workflow requires --checkpoint-path")
        pipeline_outcome = run_selectivity_des_pipeline(
            target_metal=args.target_metal_ion,
            competitor_metal=args.competitor_metal_ion,
            checkpoint_path=args.checkpoint_path,
            config_path=args.config_path,
            n_ligands=args.n,
            n_des_candidates=args.n_des_candidates,
            n_selectivity_cycles=args.n_cycles,
            n_des_cycles=args.n_des_cycles,
            n_outer_cycles=args.n_outer_cycles,
            min_delta_log_k=args.min_delta_log_k,
            top_ligands=args.top_ligands,
            w_affinity=args.affinity_weight,
            w_selectivity=args.selectivity_weight,
            stability_model_path=args.stability_constant_model_path,
            llm_cfg=llm_cfg,
        )
        print(format_selectivity_des_report(pipeline_outcome))
        _print_summary("selectivity-des", pipeline_outcome)
        return
```

- [ ] **Step 7: Run CLI tests to verify they pass**

```bash
pytest tests/test_selectivity_des_pipeline.py -k "cli" -v
```

Expected: all 4 CLI tests pass.

- [ ] **Step 8: Run full test suite**

```bash
pytest tests/ -x -q
```

Expected: all pass. Note the total test count — it should be the prior count plus the new tests in this file.

- [ ] **Step 9: Commit**

```bash
git add des_multi_agent/cli.py tests/test_selectivity_des_pipeline.py
git commit -m "feat(selectivity-des): add selectivity-des CLI workflow with 5 new args"
```
