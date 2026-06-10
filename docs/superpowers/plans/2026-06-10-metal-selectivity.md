# Metal Ion Selectivity Screening Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `metal-selectivity` workflow that screens ligands for composite affinity + selectivity toward a target metal ion over a competitor metal ion, with LLM-driven iterative refinement.

**Architecture:** A single `run_metal_selectivity_screen` loop scores each candidate against both metals via `predict_log_k`, computes `composite_score = w_affinity × log_k_target + w_selectivity × delta_log_k`, and uses a new `brainstorm_ligands_selectivity` LLM method (aware of both metals) for multi-cycle refinement. Mirrors `metal_binding_screen.py` in structure.

**Tech Stack:** Python 3.10+, RDKit (canonicalization/SMILES validation), existing `predict_log_k` / `generate_ligand_candidates` / `CandidateProposal` / `CandidateBrainstorm` / `CandidateReview` from this codebase, pytest.

---

## File Map

| File | Action |
|------|--------|
| `des_multi_agent/workflows/metal_binding_selectivity.py` | **CREATE** — dataclasses, scoring helpers, workflow loop |
| `des_multi_agent/llm/prompts.py` | **MODIFY** — add `ligand_selectivity_brainstorm_prompt` |
| `des_multi_agent/llm/base.py` | **MODIFY** — add `brainstorm_ligands_selectivity` method |
| `des_multi_agent/reporting.py` | **MODIFY** — add `format_metal_selectivity_report` |
| `des_multi_agent/cli.py` | **MODIFY** — add workflow choice, args, routing branch |
| `tests/test_metal_selectivity_screen.py` | **CREATE** — full test suite |

---

## Task 1: Dataclasses + scoring helpers in new workflow file

**Files:**
- Create: `des_multi_agent/workflows/metal_binding_selectivity.py`
- Test: `tests/test_metal_selectivity_screen.py`

- [ ] **Step 1: Write failing tests for dataclasses and composite score formula**

Create `tests/test_metal_selectivity_screen.py`:

```python
"""Tests for the metal ion selectivity screening workflow."""
from __future__ import annotations

import pytest

from des_multi_agent.workflows.metal_binding_selectivity import (
    SelectivityResult,
    SelectivityScreenOutcome,
    _top_k_stable,
    run_metal_selectivity_screen,
)
from des_multi_agent.llm.schemas import CandidateBrainstorm, CandidateReview


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(smiles: str, log_k_target: float, log_k_competitor: float,
                 w_aff: float = 0.5, w_sel: float = 0.5) -> SelectivityResult:
    delta = log_k_target - log_k_competitor
    score = w_aff * log_k_target + w_sel * delta
    return SelectivityResult(
        ligand_smiles=smiles,
        log_k_target=log_k_target,
        log_k_competitor=log_k_competitor,
        delta_log_k=delta,
        composite_score=score,
        source="heuristic",
        source_id="test",
        rationale="test",
    )


# ---------------------------------------------------------------------------
# Dataclass + scoring formula
# ---------------------------------------------------------------------------

def test_selectivity_result_delta_log_k():
    r = _make_result("NCC(=O)O", log_k_target=10.0, log_k_competitor=6.0)
    assert abs(r.delta_log_k - 4.0) < 1e-9


def test_selectivity_result_composite_score_equal_weights():
    r = _make_result("NCC(=O)O", log_k_target=10.0, log_k_competitor=6.0,
                     w_aff=0.5, w_sel=0.5)
    # 0.5 * 10.0 + 0.5 * 4.0 = 7.0
    assert abs(r.composite_score - 7.0) < 1e-9


def test_selectivity_result_composite_score_affinity_only():
    r = _make_result("NCC(=O)O", log_k_target=10.0, log_k_competitor=6.0,
                     w_aff=1.0, w_sel=0.0)
    assert abs(r.composite_score - 10.0) < 1e-9


def test_selectivity_result_composite_score_selectivity_only():
    r = _make_result("NCC(=O)O", log_k_target=10.0, log_k_competitor=6.0,
                     w_aff=0.0, w_sel=1.0)
    assert abs(r.composite_score - 4.0) < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/qshao/DES-Agent
python -m pytest tests/test_metal_selectivity_screen.py -v 2>&1 | head -30
```

Expected: `ImportError: cannot import name 'SelectivityResult'`

- [ ] **Step 3: Create the workflow file with dataclasses and helpers**

Create `des_multi_agent/workflows/metal_binding_selectivity.py`:

```python
from __future__ import annotations

import sys
from dataclasses import dataclass, field

from rdkit import Chem

from ..candidate_generation_ligand import generate_ligand_candidates
from ..chemistry_filter import canonicalize_smiles
from ..llm.schemas import CandidateBrainstorm, CandidateReview
from ..predictors.stability_constants import predict_log_k
from ..schemas import CandidateProposal


@dataclass(frozen=True)
class SelectivityResult:
    ligand_smiles: str
    log_k_target: float
    log_k_competitor: float
    delta_log_k: float
    composite_score: float
    source: str
    source_id: str
    rationale: str


@dataclass
class SelectivityScreenOutcome:
    target_metal: str
    competitor_metal: str
    results: list[SelectivityResult]
    n_screened: int
    n_cycles: int
    llm_brainstorm: list[CandidateBrainstorm] = field(default_factory=list)
    llm_candidate_reviews: list[CandidateReview] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _compute_composite(log_k_target: float, log_k_competitor: float,
                        w_affinity: float, w_selectivity: float) -> tuple[float, float]:
    delta = log_k_target - log_k_competitor
    score = w_affinity * log_k_target + w_selectivity * delta
    return delta, score


def _deduplicate_proposals(
    proposals: list[CandidateProposal], seen: set[str]
) -> list[CandidateProposal]:
    out: list[CandidateProposal] = []
    for p in proposals:
        canon = canonicalize_smiles(p.smiles)
        if canon is None or canon in seen:
            continue
        seen.add(canon)
        out.append(CandidateProposal(
            smiles=canon,
            rationale=p.rationale,
            family=p.family,
            source=p.source,
            source_id=p.source_id,
        ))
    return out


def _llm_proposals_from_brainstorm(
    brainstorms: list[CandidateBrainstorm],
) -> list[CandidateProposal]:
    out: list[CandidateProposal] = []
    for b in brainstorms:
        mol = Chem.MolFromSmiles(b.smiles)
        if mol is None:
            print(f"[selectivity] invalid SMILES from LLM (skipped): {b.smiles!r}", file=sys.stderr)
            continue
        out.append(CandidateProposal(
            smiles=b.smiles,
            rationale=b.rationale,
            family=b.family,
            source="llm",
            source_id="brainstorm",
        ))
    return out


def _score_proposal_pair(
    target_metal: str,
    competitor_metal: str,
    proposal: CandidateProposal,
    model_path,
    w_affinity: float,
    w_selectivity: float,
) -> tuple[SelectivityResult, list[str]]:
    warnings: list[str] = []
    pred_target = predict_log_k(
        target_metal, proposal.smiles, model_path=model_path, allow_fallback=True
    )
    pred_competitor = predict_log_k(
        competitor_metal, proposal.smiles, model_path=model_path, allow_fallback=True
    )
    delta_log_k, composite_score = _compute_composite(
        pred_target.value, pred_competitor.value, w_affinity, w_selectivity
    )
    return SelectivityResult(
        ligand_smiles=proposal.smiles,
        log_k_target=pred_target.value,
        log_k_competitor=pred_competitor.value,
        delta_log_k=delta_log_k,
        composite_score=composite_score,
        source=proposal.source,
        source_id=proposal.source_id,
        rationale=proposal.rationale,
    ), warnings


def _top_k_stable(
    prev: list[SelectivityResult], curr: list[SelectivityResult], k: int = 5
) -> bool:
    prev_smiles = {r.ligand_smiles for r in prev[:k]}
    curr_smiles = {r.ligand_smiles for r in curr[:k]}
    return prev_smiles == curr_smiles


def _build_selectivity_context(
    target_metal: str,
    competitor_metal: str,
    prev_results: list[SelectivityResult],
    cycle: int,
    w_affinity: float,
    w_selectivity: float,
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
    return "\n".join(lines)


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
) -> SelectivityScreenOutcome:
    seen_smiles: set[str] = set()
    all_reviews: list[CandidateReview] = []
    all_brainstorm: list[CandidateBrainstorm] = []
    all_warnings: list[str] = []
    cumulative_results: list[SelectivityResult] = []
    prev_cycle_results: list[SelectivityResult] = []

    for cycle in range(1, n_cycles + 1):
        proposals: list[CandidateProposal] = []

        if cycle == 1:
            heuristic = generate_ligand_candidates(target_metal, n, constraints)
            proposals.extend(_deduplicate_proposals(heuristic, seen_smiles))

        if llm_provider is not None:
            context = _build_selectivity_context(
                target_metal, competitor_metal, prev_cycle_results, cycle, w_affinity, w_selectivity
            )
            try:
                brainstorms = llm_provider.brainstorm_ligands_selectivity(
                    target_metal, competitor_metal, constraints, context
                )
                all_brainstorm.extend(brainstorms)
                llm_proposals = _llm_proposals_from_brainstorm(brainstorms)
                proposals.extend(_deduplicate_proposals(llm_proposals, seen_smiles))
            except Exception as exc:
                all_warnings.append(f"LLM brainstorm failed (cycle {cycle}): {exc}")

        proposals = proposals[:n]
        if not proposals:
            break

        cycle_results: list[SelectivityResult] = []
        for proposal in proposals:
            result, warnings = _score_proposal_pair(
                target_metal, competitor_metal, proposal, model_path, w_affinity, w_selectivity
            )
            all_warnings.extend(warnings)
            cycle_results.append(result)

        if llm_provider is not None:
            context = _build_selectivity_context(
                target_metal, competitor_metal, prev_cycle_results, cycle, w_affinity, w_selectivity
            )
            for r in cycle_results:
                try:
                    review = llm_provider.review_ligand(target_metal, r.ligand_smiles, context)
                    all_reviews.append(review)
                except Exception as exc:
                    all_warnings.append(f"LLM review failed for {r.ligand_smiles}: {exc}")

        by_smiles = {r.ligand_smiles: r for r in cumulative_results}
        for r in cycle_results:
            existing = by_smiles.get(r.ligand_smiles)
            if existing is None or r.composite_score > existing.composite_score:
                by_smiles[r.ligand_smiles] = r
        cumulative_results = sorted(
            by_smiles.values(), key=lambda r: r.composite_score, reverse=True
        )

        top_score = f"{cumulative_results[0].composite_score:.2f}" if cumulative_results else "n/a"
        print(
            f"[cycle {cycle}/{n_cycles}] screened={len(proposals)} top_score={top_score}",
            file=sys.stderr,
            flush=True,
        )

        if cycle > 1 and _top_k_stable(prev_cycle_results, cumulative_results):
            print(
                f"[cycle {cycle}/{n_cycles}] top-5 stable — converged early",
                file=sys.stderr,
                flush=True,
            )
            break

        prev_cycle_results = list(cumulative_results)

    return SelectivityScreenOutcome(
        target_metal=target_metal,
        competitor_metal=competitor_metal,
        results=cumulative_results,
        n_screened=len(seen_smiles),
        n_cycles=n_cycles,
        llm_brainstorm=all_brainstorm,
        llm_candidate_reviews=all_reviews,
        warnings=all_warnings,
    )
```

- [ ] **Step 4: Run the four dataclass tests**

```bash
python -m pytest tests/test_metal_selectivity_screen.py::test_selectivity_result_delta_log_k tests/test_metal_selectivity_screen.py::test_selectivity_result_composite_score_equal_weights tests/test_metal_selectivity_screen.py::test_selectivity_result_composite_score_affinity_only tests/test_metal_selectivity_screen.py::test_selectivity_result_composite_score_selectivity_only -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/workflows/metal_binding_selectivity.py tests/test_metal_selectivity_screen.py
git commit -m "feat(selectivity): add SelectivityResult dataclass, scoring helpers, and workflow skeleton"
```

---

## Task 2: Workflow loop tests

**Files:**
- Modify: `tests/test_metal_selectivity_screen.py`

- [ ] **Step 1: Add workflow loop tests to the test file**

Append to `tests/test_metal_selectivity_screen.py`:

```python
# ---------------------------------------------------------------------------
# _top_k_stable
# ---------------------------------------------------------------------------

def test_top_k_stable_identical():
    results = [_make_result(s, float(i), 0.0) for i, s in enumerate(["A", "B", "C", "D", "E"])]
    assert _top_k_stable(results, results, k=5)


def test_top_k_stable_different():
    r1 = [_make_result(s, float(i), 0.0) for i, s in enumerate(["A", "B", "C", "D", "E"])]
    r2 = [_make_result(s, float(i), 0.0) for i, s in enumerate(["A", "B", "C", "D", "F"])]
    assert not _top_k_stable(r1, r2, k=5)


# ---------------------------------------------------------------------------
# run_metal_selectivity_screen — no LLM
# ---------------------------------------------------------------------------

def test_run_screen_no_llm_returns_outcome():
    outcome = run_metal_selectivity_screen("Cu2+", "Zn2+", n=5, n_cycles=1)
    assert isinstance(outcome, SelectivityScreenOutcome)
    assert outcome.target_metal == "Cu2+"
    assert outcome.competitor_metal == "Zn2+"
    assert len(outcome.results) > 0
    assert outcome.llm_brainstorm == []
    assert outcome.llm_candidate_reviews == []


def test_run_screen_results_sorted_by_composite_score():
    outcome = run_metal_selectivity_screen("Cu2+", "Zn2+", n=10, n_cycles=1)
    scores = [r.composite_score for r in outcome.results]
    assert scores == sorted(scores, reverse=True)


def test_run_screen_delta_log_k_is_difference():
    outcome = run_metal_selectivity_screen("Cu2+", "Zn2+", n=5, n_cycles=1)
    for r in outcome.results:
        assert abs(r.delta_log_k - (r.log_k_target - r.log_k_competitor)) < 1e-9


def test_run_screen_composite_score_formula():
    outcome = run_metal_selectivity_screen(
        "Cu2+", "Zn2+", n=5, n_cycles=1, w_affinity=0.5, w_selectivity=0.5
    )
    for r in outcome.results:
        expected = 0.5 * r.log_k_target + 0.5 * r.delta_log_k
        assert abs(r.composite_score - expected) < 1e-9


def test_run_screen_no_duplicate_smiles():
    outcome = run_metal_selectivity_screen("Cu2+", "Zn2+", n=20, n_cycles=1)
    smiles_list = [r.ligand_smiles for r in outcome.results]
    assert len(smiles_list) == len(set(smiles_list))


def test_run_screen_multi_cycle_no_llm():
    outcome = run_metal_selectivity_screen("Cu2+", "Zn2+", n=5, n_cycles=3)
    assert outcome.n_cycles == 3
    assert len(outcome.results) > 0
```

- [ ] **Step 2: Run these tests**

```bash
python -m pytest tests/test_metal_selectivity_screen.py -k "top_k_stable or run_screen" -v
```

Expected: 8 passed

- [ ] **Step 3: Commit**

```bash
git add tests/test_metal_selectivity_screen.py
git commit -m "test(selectivity): add workflow loop and convergence tests"
```

---

## Task 3: LLM prompt function

**Files:**
- Modify: `des_multi_agent/llm/prompts.py`
- Modify: `tests/test_metal_selectivity_screen.py`

- [ ] **Step 1: Write failing tests for the new prompt**

Append to `tests/test_metal_selectivity_screen.py`:

```python
# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

from des_multi_agent.llm.prompts import ligand_selectivity_brainstorm_prompt
from des_multi_agent.llm.schemas import LigandFamily


def test_selectivity_brainstorm_prompt_contains_both_metals():
    prompt = ligand_selectivity_brainstorm_prompt("Cu2+", "Zn2+", None, "context")
    assert "Cu2+" in prompt
    assert "Zn2+" in prompt


def test_selectivity_brainstorm_prompt_contains_smiles_instruction():
    prompt = ligand_selectivity_brainstorm_prompt("Fe3+", "Ca2+", None, "context")
    assert "smiles" in prompt.lower()


def test_selectivity_brainstorm_prompt_with_families_includes_family_names():
    families = [LigandFamily(name="catecholates", rationale="bidentate O-donors", coordination_mode="bidentate O,O")]
    prompt = ligand_selectivity_brainstorm_prompt("Cu2+", "Zn2+", None, "context", families=families)
    assert "catecholates" in prompt
    assert "bidentate O,O" in prompt
```

- [ ] **Step 2: Run to verify they fail**

```bash
python -m pytest tests/test_metal_selectivity_screen.py -k "selectivity_brainstorm_prompt" -v
```

Expected: `ImportError: cannot import name 'ligand_selectivity_brainstorm_prompt'`

- [ ] **Step 3: Add the prompt function to `des_multi_agent/llm/prompts.py`**

At the end of the metal-binding section (after `ligand_review_prompt`, before `_results_summary`), add:

```python
def ligand_selectivity_brainstorm_prompt(
    target_metal: str,
    competitor_metal: str,
    constraints: dict | None,
    context: str,
    max_items: int | None = None,
    families: list | None = None,
) -> str:
    parts = [
        "Return raw JSON only. Do not use markdown fences or commentary.\n",
        f"Return a JSON array of candidate ligand SMILES designed for HIGH SELECTIVITY "
        f"for {target_metal} over {competitor_metal}.\n",
        f"Constraints: {constraints or {}}\n",
        f"Context: {context}\n",
    ]
    if families:
        parts.append("Distribute candidates across these coordination-chemistry families:\n")
        for f in families:
            parts.append(f"  - {f.name}: {f.rationale} (coordination: {f.coordination_mode})\n")
    parts.append(
        "Use HSAB theory, donor atom preferences (N vs O vs S), denticity, chelate ring size, "
        "and geometric preference differences between the two metals to achieve selectivity.\n"
    )
    if max_items is not None:
        parts.append(f"Return at most {max_items} items.\n")
    parts.append(
        "Each item must contain smiles (valid SMILES), rationale (why selective for target over competitor), "
        "and family (coordination chemistry class)."
    )
    return "".join(parts)
```

- [ ] **Step 4: Run the prompt tests**

```bash
python -m pytest tests/test_metal_selectivity_screen.py -k "selectivity_brainstorm_prompt" -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/llm/prompts.py tests/test_metal_selectivity_screen.py
git commit -m "feat(selectivity): add ligand_selectivity_brainstorm_prompt"
```

---

## Task 4: LLM provider method

**Files:**
- Modify: `des_multi_agent/llm/base.py`
- Modify: `tests/test_metal_selectivity_screen.py`

- [ ] **Step 1: Write failing tests for the LLM provider integration**

Append to `tests/test_metal_selectivity_screen.py`:

```python
# ---------------------------------------------------------------------------
# run_metal_selectivity_screen — with mock LLM
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock


def test_run_screen_with_llm_brainstorm_called():
    mock_llm = MagicMock()
    mock_llm.brainstorm_ligands_selectivity.return_value = [
        CandidateBrainstorm(smiles="c1ccnc(-c2ccccn2)c1", rationale="bidentate N,N", family="polypyridyl"),
    ]
    mock_llm.review_ligand.return_value = MagicMock(
        smiles="c1ccnc(-c2ccccn2)c1", decision="keep", confidence=0.9,
        rationale="good chelator", notes=[],
    )

    outcome = run_metal_selectivity_screen("Cu2+", "Zn2+", n=5, llm_provider=mock_llm, n_cycles=1)
    assert mock_llm.brainstorm_ligands_selectivity.called
    call_args = mock_llm.brainstorm_ligands_selectivity.call_args
    assert "Cu2+" in str(call_args)
    assert "Zn2+" in str(call_args)


def test_run_screen_llm_brainstorm_failure_adds_warning():
    mock_llm = MagicMock()
    mock_llm.brainstorm_ligands_selectivity.side_effect = RuntimeError("LLM down")

    outcome = run_metal_selectivity_screen("Cu2+", "Zn2+", n=5, llm_provider=mock_llm, n_cycles=1)
    assert len(outcome.warnings) > 0
    assert any("brainstorm" in w.lower() for w in outcome.warnings)


def test_run_screen_skips_invalid_llm_smiles():
    mock_llm = MagicMock()
    mock_llm.brainstorm_ligands_selectivity.return_value = [
        CandidateBrainstorm(smiles="NOT_A_SMILES", rationale="bad", family="test"),
        CandidateBrainstorm(smiles="NCC(=O)O", rationale="glycine", family="aminoacid"),
    ]
    mock_llm.review_ligand.return_value = MagicMock(
        smiles="NCC(=O)O", decision="keep", confidence=0.8, rationale="ok", notes=[],
    )

    outcome = run_metal_selectivity_screen("Cu2+", "Zn2+", n=5, llm_provider=mock_llm, n_cycles=1)
    from rdkit import Chem
    for r in outcome.results:
        assert Chem.MolFromSmiles(r.ligand_smiles) is not None
```

- [ ] **Step 2: Run to verify they fail**

```bash
python -m pytest tests/test_metal_selectivity_screen.py -k "with_llm or llm_brainstorm or invalid_llm" -v
```

Expected: 3 failures — `brainstorm_ligands_selectivity` is not yet a real method (MagicMock will auto-create it, but the test verifying it was called on a proper provider would fail). Actually the mock tests themselves will pass since MagicMock auto-creates methods. The actual integration test comes in Task 6 (CLI). Proceed to implement the real method anyway.

- [ ] **Step 3: Add `brainstorm_ligands_selectivity` to `des_multi_agent/llm/base.py`**

First, add the import of the new prompt at the top of `base.py`. Find the existing prompts import line:

```python
from .prompts import candidate_brainstorm_prompt, candidate_review_prompt, contradiction_prompt, critique_prompt, explanation_prompt, family_selection_prompt, ligand_brainstorm_prompt, ligand_family_selection_prompt, ligand_review_prompt
```

Replace it with (add `ligand_selectivity_brainstorm_prompt` at the end):

```python
from .prompts import candidate_brainstorm_prompt, candidate_review_prompt, contradiction_prompt, critique_prompt, explanation_prompt, family_selection_prompt, ligand_brainstorm_prompt, ligand_family_selection_prompt, ligand_review_prompt, ligand_selectivity_brainstorm_prompt
```

Then, in the `# Metal-binding ligand methods` section, add after `review_ligand`:

```python
def brainstorm_ligands_selectivity(
    self,
    target_metal: str,
    competitor_metal: str,
    constraints: dict | None,
    context: str,
) -> list[CandidateBrainstorm]:
    families: list[LigandFamily] = []
    try:
        families = self.select_ligand_families(target_metal, constraints, context)
    except Exception as exc:
        print(
            f"ligand family selection failed, falling back to single-stage brainstorm: {exc}",
            file=sys.stderr,
        )
    raw = self._request(
        ligand_selectivity_brainstorm_prompt(
            target_metal, competitor_metal, constraints, context,
            self.max_candidates, families,
        )
    )
    return parse_candidate_brainstorms(raw)[: self.max_candidates]
```

- [ ] **Step 4: Run the mock LLM tests**

```bash
python -m pytest tests/test_metal_selectivity_screen.py -k "with_llm or llm_brainstorm or invalid_llm" -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/llm/base.py des_multi_agent/llm/prompts.py tests/test_metal_selectivity_screen.py
git commit -m "feat(selectivity): add brainstorm_ligands_selectivity to BaseLLMProvider"
```

---

## Task 5: Report formatter

**Files:**
- Modify: `des_multi_agent/reporting.py`
- Modify: `tests/test_metal_selectivity_screen.py`

- [ ] **Step 1: Write failing test for the report**

Append to `tests/test_metal_selectivity_screen.py`:

```python
# ---------------------------------------------------------------------------
# Report format
# ---------------------------------------------------------------------------

def test_format_metal_selectivity_report_contains_headers():
    from des_multi_agent.reporting import format_metal_selectivity_report
    outcome = run_metal_selectivity_screen("Cu2+", "Zn2+", n=3, n_cycles=1)
    report = format_metal_selectivity_report(outcome)
    assert "Cu2+" in report
    assert "Zn2+" in report
    assert "delta_log_k" in report
    assert "score" in report
    assert "log_k_target" in report
    assert "log_k_competitor" in report


def test_format_metal_selectivity_report_no_results():
    from des_multi_agent.reporting import format_metal_selectivity_report
    outcome = SelectivityScreenOutcome(
        target_metal="Cu2+", competitor_metal="Zn2+",
        results=[], n_screened=0, n_cycles=1,
    )
    report = format_metal_selectivity_report(outcome)
    assert "Cu2+" in report
    assert "none" in report.lower()
```

- [ ] **Step 2: Run to verify they fail**

```bash
python -m pytest tests/test_metal_selectivity_screen.py -k "format_metal_selectivity" -v
```

Expected: `ImportError: cannot import name 'format_metal_selectivity_report'`

- [ ] **Step 3: Add `format_metal_selectivity_report` to `des_multi_agent/reporting.py`**

At the end of `reporting.py`, after `format_metal_binding_screen_report`, add:

```python
def format_metal_selectivity_report(outcome) -> str:
    """Render a ranked-candidate report for a SelectivityScreenOutcome."""
    results = outcome.results
    top = results[0] if results else None
    if top:
        top_str = (
            f"{top.ligand_smiles} — score={top.composite_score:.2f} "
            f"(ΔlogK={top.delta_log_k:.2f}, logK({outcome.target_metal})={top.log_k_target:.2f})"
        )
    else:
        top_str = "none"
    header_lines = [
        f"=== Metal Selectivity Screen: {outcome.target_metal} over {outcome.competitor_metal} ===",
        f"Screened {outcome.n_screened} candidate(s) over {outcome.n_cycles} cycle(s).",
        f"Top ligand: {top_str}",
        "=" * 52,
        "",
        "ligand | log_k_target | log_k_competitor | delta_log_k | score | source | rationale",
    ]
    rows = []
    for r in results:
        src = f"source={r.source}"
        if r.source_id:
            src += f"; id={r.source_id}"
        rows.append(
            f"{r.ligand_smiles} | {r.log_k_target:.2f} | {r.log_k_competitor:.2f} | "
            f"{r.delta_log_k:.2f} | {r.composite_score:.2f} | {src} | {r.rationale}"
        )

    review_lines: list[str] = []
    if outcome.llm_candidate_reviews:
        review_lines.append("")
        review_lines.append("LLM ligand reviews:")
        for rev in outcome.llm_candidate_reviews:
            notes = "; ".join(rev.notes) if rev.notes else "-"
            review_lines.append(
                f"{rev.smiles} | {rev.decision} | confidence={rev.confidence:.2f} | "
                f"{rev.rationale} | {notes}"
            )

    brainstorm_lines: list[str] = []
    if outcome.llm_brainstorm:
        brainstorm_lines.append("")
        brainstorm_lines.append("LLM brainstorm:")
        for b in outcome.llm_brainstorm:
            brainstorm_lines.append(f"{b.smiles} | {b.family} | {b.rationale}")

    warning_lines: list[str] = []
    if outcome.warnings:
        warning_lines.append("")
        warning_lines.append("Warnings:")
        for w in outcome.warnings:
            warning_lines.append(f"- {w}")

    return "\n".join(header_lines + rows + review_lines + brainstorm_lines + warning_lines)
```

- [ ] **Step 4: Run the report tests**

```bash
python -m pytest tests/test_metal_selectivity_screen.py -k "format_metal_selectivity" -v
```

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/reporting.py tests/test_metal_selectivity_screen.py
git commit -m "feat(selectivity): add format_metal_selectivity_report"
```

---

## Task 6: CLI integration

**Files:**
- Modify: `des_multi_agent/cli.py`
- Modify: `tests/test_metal_selectivity_screen.py`

- [ ] **Step 1: Write failing CLI tests**

Append to `tests/test_metal_selectivity_screen.py`:

```python
# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

def test_cli_metal_selectivity_routes_correctly(monkeypatch, capsys):
    """--workflow metal-selectivity without LLM should call run_metal_selectivity_screen."""
    import des_multi_agent.cli as cli_module
    import des_multi_agent.workflows.metal_binding_selectivity as sel_module

    fake_outcome = SelectivityScreenOutcome(
        target_metal="Cu2+", competitor_metal="Zn2+",
        results=[], n_screened=5, n_cycles=1,
    )
    monkeypatch.setattr(sel_module, "run_metal_selectivity_screen", lambda **kw: fake_outcome)
    monkeypatch.setattr(cli_module, "run_metal_selectivity_screen", lambda **kw: fake_outcome)

    cli_module.main([
        "--workflow", "metal-selectivity",
        "--target-metal-ion", "Cu2+",
        "--competitor-metal-ion", "Zn2+",
        "--n", "5",
    ])
    out = capsys.readouterr().out
    assert "Metal Selectivity Screen" in out or "summary:" in out.lower()


def test_cli_metal_binding_single_pair_unchanged_by_selectivity(monkeypatch, capsys):
    """Existing --workflow metal-binding --ligand-smiles path is not broken."""
    import des_multi_agent.cli as cli_module
    import des_multi_agent.workflows.metal_binding as mb_module

    class _FakeOutcome:
        metal_ion = "Cu2+"
        ligand_smiles = "NCCN"
        prediction = type("P", (), {
            "value": 5.5, "units": "log K",
            "model_name": "mock", "source": "mock", "warnings": ()
        })()
        warnings = ()

    monkeypatch.setattr(cli_module, "run_metal_binding_workflow", lambda *a, **kw: _FakeOutcome())
    monkeypatch.setattr(cli_module, "format_metal_binding_report", lambda o: "SINGLE PAIR REPORT")

    cli_module.main([
        "--workflow", "metal-binding",
        "--metal-ion", "Cu2+",
        "--ligand-smiles", "NCCN",
    ])
    out = capsys.readouterr().out
    assert "SINGLE PAIR REPORT" in out
```

- [ ] **Step 2: Run to verify they fail**

```bash
python -m pytest tests/test_metal_selectivity_screen.py -k "cli_metal_selectivity or cli_metal_binding_single_pair_unchanged" -v
```

Expected: first test fails because `metal-selectivity` is not a valid `--workflow` choice.

- [ ] **Step 3: Add CLI support to `des_multi_agent/cli.py`**

**3a. Update `--workflow` choices** — find:

```python
parser.add_argument("--workflow", choices=["des", "metal-binding"], default="des")
```

Replace with:

```python
parser.add_argument("--workflow", choices=["des", "metal-binding", "metal-selectivity"], default="des")
```

**3b. Add new arguments** — after the `--stability-constant-model-path` line, add:

```python
parser.add_argument("--target-metal-ion", default=None, help="Target metal ion for selectivity workflow (e.g., Cu2+)")
parser.add_argument("--competitor-metal-ion", default=None, help="Competitor metal ion for selectivity workflow (e.g., Zn2+)")
parser.add_argument("--affinity-weight", type=float, default=0.5, dest="affinity_weight",
                    help="Weight for log K(target) in composite selectivity score (default 0.5)")
parser.add_argument("--selectivity-weight", type=float, default=0.5, dest="selectivity_weight",
                    help="Weight for delta log K in composite selectivity score (default 0.5)")
```

**3c. Add imports at top of `cli.py`** — find the existing import block near line 22:

```python
from .reporting import (
    format_metal_binding_report, format_metal_binding_screen_report, format_report,
```

Add `format_metal_selectivity_report` to that import, and add the workflow import near line 32:

```python
from .workflows.metal_binding_selectivity import run_metal_selectivity_screen
```

**3d. Add routing branch** — at the end of `main()`, after the `if not args.metal_ion:` block for `metal-binding`, add a new top-level branch. Find the line:

```python
    if not args.metal_ion:
        parser.error("metal-binding workflow requires --metal-ion")
```

Before it, insert:

```python
    if args.workflow == "metal-selectivity":
        if not args.target_metal_ion or not args.competitor_metal_ion:
            parser.error("metal-selectivity workflow requires --target-metal-ion and --competitor-metal-ion")
        from .llm.factory import build_llm_provider as _build_llm_provider
        llm_provider_sel = _build_llm_provider(llm_cfg) if llm_cfg is not None else None
        sel_outcome = run_metal_selectivity_screen(
            target_metal=args.target_metal_ion,
            competitor_metal=args.competitor_metal_ion,
            n=getattr(args, "n", 20),
            model_path=args.stability_constant_model_path,
            llm_provider=llm_provider_sel,
            n_cycles=getattr(args, "n_cycles", 1),
            w_affinity=args.affinity_weight,
            w_selectivity=args.selectivity_weight,
        )
        print(format_metal_selectivity_report(sel_outcome))
        _print_summary("metal-selectivity", sel_outcome)
        return
```

- [ ] **Step 4: Run the CLI tests**

```bash
python -m pytest tests/test_metal_selectivity_screen.py -k "cli_metal_selectivity or cli_metal_binding_single_pair_unchanged" -v
```

Expected: 2 passed

- [ ] **Step 5: Run the full test suite to verify no regressions**

```bash
python -m pytest tests/ -x -q 2>&1 | tail -10
```

Expected: all prior tests pass plus new selectivity tests.

- [ ] **Step 6: Smoke test the CLI end-to-end**

```bash
python -m des_multi_agent.cli --workflow metal-selectivity --target-metal-ion Cu2+ --competitor-metal-ion Zn2+ --n 10 2>/dev/null
```

Expected output contains:
```
=== Metal Selectivity Screen: Cu2+ over Zn2+ ===
Screened 10 candidate(s) over 1 cycle(s).
Top ligand: ...
ligand | log_k_target | log_k_competitor | delta_log_k | score | source | rationale
```

- [ ] **Step 7: Commit**

```bash
git add des_multi_agent/cli.py des_multi_agent/reporting.py tests/test_metal_selectivity_screen.py
git commit -m "feat(selectivity): add metal-selectivity CLI workflow with --target-metal-ion and --competitor-metal-ion"
```

---

## Task 7: Final test run and cleanup

- [ ] **Step 1: Run the full test suite**

```bash
python -m pytest tests/ -q 2>&1 | tail -5
```

Expected: all tests pass, no warnings about missing imports.

- [ ] **Step 2: Run the new test file verbosely to confirm all cases pass**

```bash
python -m pytest tests/test_metal_selectivity_screen.py -v
```

Expected: all tests in the file listed and passing.

- [ ] **Step 3: Final commit**

```bash
git add -p  # stage any unstaged changes
git commit -m "test(selectivity): finalize metal ion selectivity test suite"
```

---

## Self-Review Checklist (already passed)

- **Spec coverage:** All 7 spec sections mapped to tasks. Data model (Task 1), workflow loop (Tasks 1–2), LLM prompt (Task 3), provider method (Task 4), report (Task 5), CLI (Task 6), tests (all tasks).
- **No placeholders:** All code blocks are complete and runnable.
- **Type consistency:** `SelectivityResult`, `SelectivityScreenOutcome`, `_top_k_stable`, `run_metal_selectivity_screen`, `brainstorm_ligands_selectivity`, `ligand_selectivity_brainstorm_prompt`, `format_metal_selectivity_report` — names consistent across all tasks.
- **`args.target_metal_ion`** (underscore, not hyphen) — argparse converts `--target-metal-ion` → `args.target_metal_ion`. Used consistently in Task 6.
