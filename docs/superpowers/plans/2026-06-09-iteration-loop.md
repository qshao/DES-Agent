# Iteration Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the gap between the current single-shot DES screener and the ultimate vision of a self-iterating, learning multi-agent system that proposes, predicts, validates, and refines candidates over multiple cycles.

**Architecture:** Five tightly sequenced improvements — H3 (chemical contradiction detection) provides a new LLM validation signal; H2 (viscosity-aware composite ranking) upgrades the ranking function so viscosity actually influences which candidates survive; H4 (cycle-aware context) enriches the brainstorm prompt with prior-cycle top hits; H1 (multi-cycle runner) loops `run_search_report` N times and accumulates per-cycle deltas; H5 (convergence detection) short-circuits the loop when the top-K set stabilises. The multi-cycle logic lives in a new `multi_cycle.py` to keep `orchestrator.py` focused.

**Tech Stack:** Python stdlib, RDKit (already a dep), existing `LLMProvider` / `BaseLLMProvider` pattern, `dataclasses`, `frozenset` for convergence comparison.

---

## File map

| Action | Path | Responsibility |
|---|---|---|
| Modify | `des_multi_agent/llm/schemas.py` | Add `ContradictionNote` dataclass; add `CandidateFamily` dataclass (H6) |
| Modify | `des_multi_agent/llm/prompts.py` | Add `contradiction_prompt`; add `family_selection_prompt`; update `candidate_brainstorm_prompt` to accept families (H6) |
| Modify | `des_multi_agent/llm/parser.py` | Add `parse_contradiction_notes`; add `parse_candidate_families` (H6) |
| Modify | `des_multi_agent/llm/provider.py` | Add `detect_contradictions` abstract method; add `select_candidate_families` abstract method (H6) |
| Modify | `des_multi_agent/llm/base.py` | Implement `detect_contradictions`; implement two-stage `brainstorm_candidates` + `select_candidate_families` (H6) |
| Modify | `des_multi_agent/ranking.py` | Add `rank_results_composite` |
| Modify | `des_multi_agent/orchestrator.py` | Move viscosity prediction before ranking; wire contradiction detection; add `prior_cycle_top_results` param; add `contradiction_notes` to `SearchOutcome`; pass family ledger to `_build_iterative_context` (H6) |
| Modify | `des_multi_agent/reporting.py` | Render `contradiction_notes` in `format_report` |
| Create | `des_multi_agent/multi_cycle.py` | `CycleDelta` (+ `family_ledger` H6), `MultiCycleOutcome`, `run_multi_cycle_search` (H1 + H5 + H6) |
| Modify | `des_multi_agent/cli.py` | `--n-cycles`, `--viscosity-threshold`, `--viscosity-weight` |
| Create | `tests/test_h1_h2_h3_h4_h5.py` | Tests for H1–H5 |
| Create | `tests/test_h6.py` | Tests for H6 (family-level brainstorm + ledger) |

---

## Task 1 — H3 schema, prompt, and parser

**Files:**
- Modify: `des_multi_agent/llm/schemas.py`
- Modify: `des_multi_agent/llm/prompts.py`
- Modify: `des_multi_agent/llm/parser.py`
- Test: `tests/test_h1_h2_h3_h4_h5.py`

- [ ] **Step 1.1 — Write the failing tests**

Create `tests/test_h1_h2_h3_h4_h5.py`:

```python
from __future__ import annotations
import pytest


# ── H3: ContradictionNote schema ─────────────────────────────────────────────

def test_contradiction_note_fields():
    from des_multi_agent.llm.schemas import ContradictionNote
    note = ContradictionNote(smiles="CCO", agreement="conflict", explanation="High polarity mismatch.")
    assert note.smiles == "CCO"
    assert note.agreement == "conflict"
    assert note.explanation == "High polarity mismatch."


def test_parse_contradiction_notes_agree():
    from des_multi_agent.llm.parser import parse_contradiction_notes
    raw = '[{"smiles": "CCO", "agreement": "agree", "explanation": "Fits HBD/HBA profile."}]'
    notes = parse_contradiction_notes(raw)
    assert len(notes) == 1
    assert notes[0].smiles == "CCO"
    assert notes[0].agreement == "agree"


def test_parse_contradiction_notes_conflict():
    from des_multi_agent.llm.parser import parse_contradiction_notes
    raw = '[{"smiles": "c1ccccc1", "agreement": "conflict", "explanation": "Aromatic ring unlikely to form HBD."}]'
    notes = parse_contradiction_notes(raw)
    assert notes[0].agreement == "conflict"


def test_parse_contradiction_notes_skips_invalid():
    """Items missing smiles or agreement are silently skipped."""
    from des_multi_agent.llm.parser import parse_contradiction_notes
    raw = '[{"smiles": "CCO"}, {"agreement": "agree"}, {"smiles": "OCC", "agreement": "uncertain", "explanation": "OK"}]'
    notes = parse_contradiction_notes(raw)
    assert len(notes) == 1
    assert notes[0].smiles == "OCC"


def test_parse_contradiction_notes_empty_list():
    from des_multi_agent.llm.parser import parse_contradiction_notes
    assert parse_contradiction_notes("[]") == []


def test_parse_contradiction_notes_wrapped():
    """Handles LLM wrapping the array under a key."""
    from des_multi_agent.llm.parser import parse_contradiction_notes
    raw = '{"items": [{"smiles": "CC(=O)O", "agreement": "agree", "explanation": "Acid fits."}]}'
    notes = parse_contradiction_notes(raw)
    assert len(notes) == 1


def test_contradiction_prompt_contains_smiles(fake_des_result):
    from des_multi_agent.llm.prompts import contradiction_prompt
    text = contradiction_prompt([fake_des_result], "context")
    assert fake_des_result.curve.smiles_b in text


def test_contradiction_prompt_instructs_json(fake_des_result):
    from des_multi_agent.llm.prompts import contradiction_prompt
    text = contradiction_prompt([fake_des_result], "context")
    assert "JSON" in text
    assert "agreement" in text
```

- [ ] **Step 1.2 — Add the `fake_des_result` fixture** (append to the same file):

```python
# ── shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def fake_des_result():
    from dataclasses import dataclass
    from des_multi_agent.evaluation import DesResult

    @dataclass(frozen=True)
    class _Curve:
        smiles_a: str = "Cc1ccc(O)cc1"
        smiles_b: str = "CCO"
        tm_pred_k: list = None
        ratios: list = None
        t1_k: float = 330.0
        t2_k: float = 289.0
        def __post_init__(self):
            if self.tm_pred_k is None:
                object.__setattr__(self, "tm_pred_k", [300.0, 240.0, 250.0])
            if self.ratios is None:
                object.__setattr__(self, "ratios", [0.1, 0.5, 0.9])

    return DesResult(
        curve=_Curve(),
        absolute_pass=True,
        relative_pass=True,
        is_des=True,
        rationale="test",
        min_tm_k=240.0,
    )
```

- [ ] **Step 1.3 — Run to verify failures**

```
python -m pytest tests/test_h1_h2_h3_h4_h5.py::test_contradiction_note_fields \
  tests/test_h1_h2_h3_h4_h5.py::test_parse_contradiction_notes_agree \
  --tb=short -q
```
Expected: `ImportError` or `AttributeError` — `ContradictionNote` not yet defined.

- [ ] **Step 1.4 — Add `ContradictionNote` to `des_multi_agent/llm/schemas.py`**

Append after `CritiqueNote`:

```python
@dataclass(frozen=True)
class ContradictionNote:
    smiles: str
    agreement: str   # "agree" | "conflict" | "uncertain"
    explanation: str
```

- [ ] **Step 1.5 — Add `contradiction_prompt` to `des_multi_agent/llm/prompts.py`**

Append after `critique_prompt`:

```python
def contradiction_prompt(results: Sequence[DesResult], context: str, max_items: int | None = None) -> str:
    parts = [
        "Return raw JSON only. Do not use markdown fences or commentary.\n",
        "Return a JSON array examining whether each ML DES prediction is chemically plausible.\n",
        f"Context: {context}\n",
        "Results:\n",
        f"{_results_summary(results)}\n",
    ]
    if max_items is not None:
        parts.append(f"Return at most {max_items} items.\n")
    parts.append(
        'Each item must contain smiles, agreement ("agree", "conflict", or "uncertain"), and explanation.'
    )
    return "".join(parts)
```

- [ ] **Step 1.6 — Add `parse_contradiction_notes` to `des_multi_agent/llm/parser.py`**

Add `ContradictionNote` to the import at the top of `parser.py`:

```python
from .schemas import CandidateBrainstorm, CandidateReview, CritiqueNote, ExplanationNote, ContradictionNote
```

Append after `parse_critique_notes`:

```python
def parse_contradiction_notes(raw: str) -> list[ContradictionNote]:
    data = _coerce_json(raw)
    if not isinstance(data, list):
        return []
    out: list[ContradictionNote] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        smiles = str(item.get("smiles", "")).strip()
        agreement = str(item.get("agreement", "")).strip().lower()
        explanation = str(item.get("explanation", "")).strip()
        if not smiles or not agreement or not explanation:
            continue
        out.append(ContradictionNote(smiles=smiles, agreement=agreement, explanation=explanation))
    return out
```

- [ ] **Step 1.7 — Run tests to verify pass**

```
python -m pytest tests/test_h1_h2_h3_h4_h5.py -k "contradiction" --tb=short -q
```
Expected: all 8 contradiction tests PASS.

- [ ] **Step 1.8 — Commit**

```bash
git add des_multi_agent/llm/schemas.py des_multi_agent/llm/prompts.py \
        des_multi_agent/llm/parser.py tests/test_h1_h2_h3_h4_h5.py
git commit -m "$(cat <<'EOF'
feat(H3): add ContradictionNote schema, prompt, and parser

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2 — H3 provider wiring

**Files:**
- Modify: `des_multi_agent/llm/provider.py`
- Modify: `des_multi_agent/llm/base.py`
- Test: `tests/test_h1_h2_h3_h4_h5.py`

- [ ] **Step 2.1 — Write the failing tests** (append to `tests/test_h1_h2_h3_h4_h5.py`):

```python
# ── H3: provider detect_contradictions ───────────────────────────────────────

def test_detect_contradictions_returns_notes(monkeypatch, fake_des_result):
    """BaseLLMProvider.detect_contradictions calls the LLM and parses results."""
    from des_multi_agent.llm.base import BaseLLMProvider

    class _StubProvider(BaseLLMProvider):
        request_profile = None
        def extract_text(self, raw):
            return raw

    provider = _StubProvider.__new__(_StubProvider)
    monkeypatch.setattr(
        provider, "_request",
        lambda prompt: '[{"smiles": "CCO", "agreement": "agree", "explanation": "Good HBD donor."}]'
    )
    notes = provider.detect_contradictions([fake_des_result], "ctx")
    assert len(notes) == 1
    assert notes[0].agreement == "agree"


def test_detect_contradictions_provider_is_abstract():
    """LLMProvider declares detect_contradictions as abstract."""
    import inspect
    from des_multi_agent.llm.provider import LLMProvider
    assert "detect_contradictions" in {
        name for name, _ in inspect.getmembers(LLMProvider, predicate=inspect.isfunction)
    }
```

- [ ] **Step 2.2 — Run to verify failures**

```
python -m pytest tests/test_h1_h2_h3_h4_h5.py -k "detect_contradictions" --tb=short -q
```
Expected: `AttributeError` — method not on provider.

- [ ] **Step 2.3 — Add abstract method to `des_multi_agent/llm/provider.py`**

Add import at top:

```python
from .schemas import CandidateBrainstorm, CandidateReview, CritiqueNote, ContradictionNote, ExplanationNote
```

Add the abstract method after `critique_results`:

```python
@abstractmethod
def detect_contradictions(self, results: list[DesResult], context: str) -> list[ContradictionNote]:
    raise NotImplementedError
```

- [ ] **Step 2.4 — Implement in `des_multi_agent/llm/base.py`**

Add imports at top of `base.py`:

```python
from .parser import (
    parse_candidate_brainstorms, parse_candidate_review,
    parse_contradiction_notes, parse_critique_notes, parse_explanation_notes,
)
from .prompts import (
    candidate_brainstorm_prompt, candidate_review_prompt,
    contradiction_prompt, critique_prompt, explanation_prompt,
)
from .schemas import (
    CandidateBrainstorm, CandidateReview, ContradictionNote, CritiqueNote, ExplanationNote,
)
```

Add method after `critique_results` in `BaseLLMProvider`:

```python
def detect_contradictions(self, results: list[DesResult], context: str) -> list[ContradictionNote]:
    raw = self._request(contradiction_prompt(results, context, len(results) or None))
    return parse_contradiction_notes(raw)
```

- [ ] **Step 2.5 — Run tests**

```
python -m pytest tests/test_h1_h2_h3_h4_h5.py -k "detect_contradictions" --tb=short -q
```
Expected: 2 PASS.

- [ ] **Step 2.6 — Commit**

```bash
git add des_multi_agent/llm/provider.py des_multi_agent/llm/base.py \
        tests/test_h1_h2_h3_h4_h5.py
git commit -m "$(cat <<'EOF'
feat(H3): wire detect_contradictions into LLMProvider and BaseLLMProvider

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3 — H2 viscosity-aware composite ranking

**Files:**
- Modify: `des_multi_agent/ranking.py`
- Test: `tests/test_h1_h2_h3_h4_h5.py`

- [ ] **Step 3.1 — Write the failing tests** (append to `tests/test_h1_h2_h3_h4_h5.py`):

```python
# ── H2: viscosity-aware composite ranking ────────────────────────────────────

def _make_result(smiles_b, min_tm_k, is_des=True, t1=330.0, t2=289.0):
    from dataclasses import dataclass
    from des_multi_agent.evaluation import DesResult

    @dataclass(frozen=True)
    class _Curve:
        smiles_a: str = "Cc1ccc(O)cc1"
        smiles_b: str = ""
        tm_pred_k: list = None
        ratios: list = None
        t1_k: float = 330.0
        t2_k: float = 289.0
        def __post_init__(self):
            if self.tm_pred_k is None:
                object.__setattr__(self, "tm_pred_k", [min_tm_k + 10, min_tm_k, min_tm_k + 5])
            if self.ratios is None:
                object.__setattr__(self, "ratios", [0.1, 0.5, 0.9])

    return DesResult(
        curve=_Curve(smiles_b=smiles_b, t1_k=t1, t2_k=t2),
        absolute_pass=is_des,
        relative_pass=is_des,
        is_des=is_des,
        rationale="test",
        min_tm_k=min_tm_k,
    )


def test_composite_ranking_prefers_low_viscosity():
    """A DES-former with lower viscosity ranks above one with higher viscosity
    even when both have the same Tm."""
    from des_multi_agent.ranking import rank_results_composite
    high_tm = _make_result("CCO", 240.0)      # good Tm, low viscosity
    low_visc = _make_result("OCCO", 240.0)    # same Tm, but lower viscosity
    visc = {"CCO": 500.0, "OCCO": 50.0}
    ranked = rank_results_composite([high_tm, low_visc], visc, viscosity_weight=0.5)
    assert ranked[0].curve.smiles_b == "OCCO"


def test_composite_ranking_threshold_moves_high_visc_below_passing():
    """With a viscosity threshold, high-viscosity DES-formers appear after
    low-viscosity ones regardless of Tm."""
    from des_multi_agent.ranking import rank_results_composite
    good = _make_result("CCO", 230.0)     # better Tm, low viscosity
    sticky = _make_result("OCCO", 235.0)  # worse Tm, high viscosity
    visc = {"CCO": 80.0, "OCCO": 600.0}
    ranked = rank_results_composite([good, sticky], visc, viscosity_threshold_cp=500.0)
    assert ranked[0].curve.smiles_b == "CCO"


def test_composite_ranking_no_visc_falls_back_to_min_tm():
    """When no viscosity data is available, sort by min_tm_k ascending."""
    from des_multi_agent.ranking import rank_results_composite
    r1 = _make_result("CCO", 250.0)
    r2 = _make_result("OCCO", 230.0)
    ranked = rank_results_composite([r1, r2], {})
    assert ranked[0].curve.smiles_b == "OCCO"


def test_composite_ranking_non_des_always_last():
    """Non-DES candidates always sort after DES-formers."""
    from des_multi_agent.ranking import rank_results_composite
    des = _make_result("CCO", 240.0, is_des=True)
    non_des = _make_result("OCCO", 200.0, is_des=False)  # lower Tm but not DES
    ranked = rank_results_composite([non_des, des], {})
    assert ranked[0].is_des is True
```

- [ ] **Step 3.2 — Run to verify failures**

```
python -m pytest tests/test_h1_h2_h3_h4_h5.py -k "composite_ranking" --tb=short -q
```
Expected: `ImportError` — `rank_results_composite` not yet defined.

- [ ] **Step 3.3 — Add `rank_results_composite` to `des_multi_agent/ranking.py`**

Full replacement of `ranking.py`:

```python
from __future__ import annotations

from .evaluation import DesResult


def rank_results(results: list[DesResult]) -> list[DesResult]:
    return sorted(
        results,
        key=lambda r: (
            not r.is_des,
            r.min_tm_k,
            -sum(r.curve.tm_pred_k) / len(r.curve.tm_pred_k),
        ),
    )


def rank_results_composite(
    results: list[DesResult],
    visc_by_smiles_b: dict[str, float],
    *,
    viscosity_weight: float = 0.3,
    viscosity_threshold_cp: float | None = None,
) -> list[DesResult]:
    """Rank by a composite of Tm-drop and viscosity.

    When visc_by_smiles_b is empty, falls back to rank_results ordering.
    When viscosity_threshold_cp is set, candidates above the threshold are
    sorted after those that pass it (regardless of Tm).
    """
    if not visc_by_smiles_b:
        return rank_results(results)

    des = [r for r in results if r.is_des]
    non_des = sorted([r for r in results if not r.is_des], key=lambda r: r.min_tm_k)

    def _composite_score(r: DesResult) -> float:
        baseline = min(r.curve.t1_k, r.curve.t2_k)
        tm_score = (baseline - r.min_tm_k) / baseline if baseline else 0.0
        visc_cp = visc_by_smiles_b.get(r.curve.smiles_b)
        visc_score = 1.0 / (1.0 + visc_cp / 100.0) if visc_cp is not None else 0.5
        return (1.0 - viscosity_weight) * tm_score + viscosity_weight * visc_score

    if viscosity_threshold_cp is not None:
        passing = [r for r in des if visc_by_smiles_b.get(r.curve.smiles_b, 0.0) <= viscosity_threshold_cp]
        failing = [r for r in des if visc_by_smiles_b.get(r.curve.smiles_b, 0.0) > viscosity_threshold_cp]
        ranked_des = (
            sorted(passing, key=_composite_score, reverse=True)
            + sorted(failing, key=_composite_score, reverse=True)
        )
    else:
        ranked_des = sorted(des, key=_composite_score, reverse=True)

    return ranked_des + non_des
```

- [ ] **Step 3.4 — Run tests**

```
python -m pytest tests/test_h1_h2_h3_h4_h5.py -k "composite_ranking" --tb=short -q
```
Expected: 4 PASS.

- [ ] **Step 3.5 — Commit**

```bash
git add des_multi_agent/ranking.py tests/test_h1_h2_h3_h4_h5.py
git commit -m "$(cat <<'EOF'
feat(H2): add rank_results_composite with viscosity threshold gate

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4 — H2 + H3 + H4 orchestrator wiring

**Files:**
- Modify: `des_multi_agent/orchestrator.py`
- Modify: `des_multi_agent/reporting.py`
- Test: `tests/test_h1_h2_h3_h4_h5.py`

- [ ] **Step 4.1 — Write the failing tests** (append to `tests/test_h1_h2_h3_h4_h5.py`):

```python
# ── H3 + H4: orchestrator wiring ────────────────────────────────────────────

def test_search_outcome_has_contradiction_notes():
    """SearchOutcome exposes contradiction_notes field."""
    from des_multi_agent.orchestrator import SearchOutcome
    outcome = SearchOutcome(
        results=[], annotated_results=[], candidate_proposals=[],
        candidate_reviews=[], brainstorm_candidates=[],
        explanation_notes=[], critique_notes=[], llm_warnings=[],
        contradiction_notes=[],
    )
    assert outcome.contradiction_notes == []


def test_run_search_report_accepts_prior_cycle_top_results(monkeypatch, tmp_path):
    """run_search_report accepts prior_cycle_top_results without error."""
    import yaml
    from des_multi_agent.orchestrator import run_search_report

    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_bytes(b"")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("device: cpu\nembedding:\n  method: morgan\n  morgan:\n    radius: 2\n    n_bits: 2048\n    use_chirality: false\n")

    monkeypatch.setattr("des_multi_agent.orchestrator.generate_candidates", lambda *a, **kw: [])
    monkeypatch.setattr("des_multi_agent.orchestrator.filter_candidates", lambda *a, **kw: [])
    monkeypatch.setattr("des_multi_agent.orchestrator.rank_results", lambda r: r)
    monkeypatch.setattr("des_multi_agent.orchestrator.rank_results_composite", lambda *a, **kw: [])

    outcome = run_search_report(
        "CCO", 1, str(ckpt), str(cfg),
        prior_cycle_top_results=[],
    )
    assert outcome.contradiction_notes == []


def test_contradiction_notes_appear_in_format_report(fake_des_result):
    """format_report includes a contradiction notes section when notes are present."""
    from des_multi_agent.llm.schemas import ContradictionNote
    from des_multi_agent.reporting import format_report

    notes = [ContradictionNote(smiles="CCO", agreement="conflict", explanation="Polarity mismatch.")]
    text = format_report([fake_des_result], contradiction_notes=notes)
    assert "conflict" in text
    assert "Polarity mismatch" in text


def test_iterative_context_includes_prior_results(fake_des_result):
    """When prior_cycle_top_results is set, the brainstorm context mentions prior top hits."""
    from des_multi_agent.orchestrator import _build_iterative_context
    ctx = _build_iterative_context("base context", [fake_des_result])
    assert "Prior cycle" in ctx
    assert fake_des_result.curve.smiles_b in ctx
```

- [ ] **Step 4.2 — Run to verify failures**

```
python -m pytest tests/test_h1_h2_h3_h4_h5.py \
  -k "search_outcome_has_contradiction or run_search_report_accepts or contradiction_notes_appear or iterative_context" \
  --tb=short -q
```
Expected: `TypeError` / `ImportError` — fields / functions not yet present.

- [ ] **Step 4.3 — Add `contradiction_notes` to `SearchOutcome` and `_build_iterative_context` to `des_multi_agent/orchestrator.py`**

At the top of `orchestrator.py`, extend the import from `llm/schemas`:
```python
from .llm.schemas import CandidateBrainstorm, CandidateReview, ContradictionNote, CritiqueNote, ExplanationNote
```

Extend the `SearchOutcome` dataclass (add after `critique_notes`):
```python
contradiction_notes: list[ContradictionNote] = field(default_factory=list)
```

Add `_build_iterative_context` helper after `_search_context`:
```python
def _build_iterative_context(base_context: str, prior_top: list) -> str:
    if not prior_top:
        return base_context
    lines = "\n".join(
        f"  - {r.curve.smiles_b}: min_tm_k={r.min_tm_k:.1f} K, is_des={r.is_des}"
        for r in prior_top[:5]
    )
    return base_context + f"\nPrior cycle top results (bias generation toward these chemical families):\n{lines}"
```

Add `prior_cycle_top_results: list | None = None` parameter to `run_search_report` signature (append at end of param list).

In `run_search_report`, replace the block where `brainstorm_candidates` is called to use the iterative context when prior results are available:

```python
    if provider is not None and candidates_file is None:
        try:
            brainstorm_context = review_context
            if prior_cycle_top_results:
                brainstorm_context = _build_iterative_context(review_context, prior_cycle_top_results)
            llm_candidates = provider.brainstorm_candidates(
                component_a,
                None,
                brainstorm_context,
            )
        except Exception as exc:
            llm_warnings.append(f"LLM brainstorming failed: {exc}")
            llm_candidates = []
```

- [ ] **Step 4.4 — Move viscosity prediction before ranking and wire H2 + H3**

In `run_search_report`, relocate the viscosity prediction call from after `final_results =` (line ~454) to immediately after the ML prediction loop closes (after `results` list is populated, before `rank_results`). Add the import for `rank_results_composite`:

```python
from .ranking import rank_results, rank_results_composite
```

Replace the `ranked = rank_results(results)` line with:

```python
    # H2 — predict viscosity early so it can influence ranking
    viscosity_predictions = _predict_viscosity_predictions(component_a, filtered, viscosity_model_path, llm_warnings)
    visc_by_smiles_b = {p.component_b: p.value for p in viscosity_predictions}
    if visc_by_smiles_b:
        ranked = rank_results_composite(
            results, visc_by_smiles_b,
            viscosity_weight=viscosity_weight,
            viscosity_threshold_cp=viscosity_threshold_cp,
        )
    else:
        ranked = rank_results(results)
```

Add `viscosity_weight: float = 0.3` and `viscosity_threshold_cp: float | None = None` to the `run_search_report` signature.

Remove the old `viscosity_predictions = _predict_viscosity_predictions(...)` line from its original location (~line 454) since it has been moved.

Wire H3 — add contradiction detection after `critique_notes` block:

```python
    contradiction_notes: list[ContradictionNote] = []
    if provider is not None:
        try:
            contradiction_notes = provider.detect_contradictions(final_results, review_context)
        except Exception as exc:
            llm_warnings.append(f"LLM contradiction detection failed: {exc}")
```

Update `export_outcome` construction to include `contradiction_notes=contradiction_notes`.

Update the `export_des_run_bundle` payload call — add to `_build_des_export_payload` or the call site:

In `_build_des_export_payload`, add to the returned dict:
```python
"contradiction_notes": [asdict(n) for n in outcome.contradiction_notes],
```

- [ ] **Step 4.5 — Add contradiction notes to `format_report` in `des_multi_agent/reporting.py`**

Add `contradiction_notes` parameter:

```python
def format_report(
    results,
    annotated_results=None,
    candidate_proposals=None,
    candidate_reviews=None,
    explanation_notes=None,
    critique_notes=None,
    brainstorm_candidates=None,
    llm_warnings=None,
    memory_notes=None,
    viscosity_predictions=None,
    resolve_names: bool = True,
    show_curves: bool = False,
    contradiction_notes=None,   # ← new
) -> str:
```

Add rendering block inside `format_report` after the `critique_notes` block:

```python
    if contradiction_notes:
        lines.append("")
        lines.append("LLM contradiction analysis:")
        for note in contradiction_notes:
            lines.append(f"{note.smiles} | {note.agreement} | {note.explanation}")
```

- [ ] **Step 4.6 — Run tests**

```
python -m pytest tests/test_h1_h2_h3_h4_h5.py \
  -k "search_outcome_has_contradiction or run_search_report_accepts or contradiction_notes_appear or iterative_context" \
  --tb=short -q
```
Expected: 4 PASS.

- [ ] **Step 4.7 — Commit**

```bash
git add des_multi_agent/orchestrator.py des_multi_agent/ranking.py \
        des_multi_agent/reporting.py tests/test_h1_h2_h3_h4_h5.py
git commit -m "$(cat <<'EOF'
feat(H2+H3+H4): wire composite ranking, contradiction detection, cycle-aware context

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5 — H1 + H5 multi-cycle runner

**Files:**
- Create: `des_multi_agent/multi_cycle.py`
- Test: `tests/test_h1_h2_h3_h4_h5.py`

- [ ] **Step 5.1 — Write the failing tests** (append to `tests/test_h1_h2_h3_h4_h5.py`):

```python
# ── H1 + H5: multi-cycle runner ──────────────────────────────────────────────

def test_multi_cycle_outcome_fields():
    """MultiCycleOutcome and CycleDelta are importable dataclasses."""
    from des_multi_agent.multi_cycle import CycleDelta, MultiCycleOutcome
    delta = CycleDelta(
        cycle=1, n_screened=5, n_des=2,
        top_smiles=frozenset(["CCO"]),
        new_entrants=["CCO"], dropouts=[], converged=False,
    )
    assert delta.cycle == 1
    outcome = MultiCycleOutcome(
        final_outcome=None,
        cycle_deltas=[delta],
        total_cycles=1,
        converged=False,
    )
    assert outcome.total_cycles == 1


def test_run_multi_cycle_search_runs_n_cycles(monkeypatch, tmp_path):
    """run_multi_cycle_search calls run_search_report exactly n_cycles times."""
    call_count = {"n": 0}

    def _fake_search(*args, **kwargs):
        call_count["n"] += 1
        from unittest.mock import MagicMock
        outcome = MagicMock()
        outcome.results = []
        return outcome

    monkeypatch.setattr("des_multi_agent.multi_cycle.run_search_report", _fake_search)

    from des_multi_agent.multi_cycle import run_multi_cycle_search
    import pathlib
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_bytes(b"")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("")

    result = run_multi_cycle_search("CCO", 2, str(ckpt), str(cfg), n_cycles=3)
    assert call_count["n"] == 3
    assert result.total_cycles == 3


def test_run_multi_cycle_search_stops_on_convergence(monkeypatch, tmp_path):
    """Convergence is detected when top-K set is unchanged across two cycles."""
    from dataclasses import dataclass
    from des_multi_agent.evaluation import DesResult

    @dataclass(frozen=True)
    class _Curve:
        smiles_b: str
        smiles_a: str = "Cc1ccc(O)cc1"
        tm_pred_k: list = None
        ratios: list = None
        t1_k: float = 330.0
        t2_k: float = 289.0
        def __post_init__(self):
            if self.tm_pred_k is None:
                object.__setattr__(self, "tm_pred_k", [240.0, 230.0])
            if self.ratios is None:
                object.__setattr__(self, "ratios", [0.1, 0.9])

    fixed_result = DesResult(
        curve=_Curve(smiles_b="CCO"), absolute_pass=True,
        relative_pass=True, is_des=True, rationale="t", min_tm_k=230.0,
    )

    def _stable_search(*args, **kwargs):
        from unittest.mock import MagicMock
        outcome = MagicMock()
        outcome.results = [fixed_result]
        return outcome

    monkeypatch.setattr("des_multi_agent.multi_cycle.run_search_report", _stable_search)

    from des_multi_agent.multi_cycle import run_multi_cycle_search
    ckpt = tmp_path / "ckpt.pt"; ckpt.write_bytes(b"")
    cfg = tmp_path / "config.yaml"; cfg.write_text("")

    result = run_multi_cycle_search("CCO", 2, str(ckpt), str(cfg), n_cycles=5, top_k_convergence=1)
    # should stop after cycle 2 (first convergence)
    assert result.converged is True
    assert result.total_cycles == 2


def test_cycle_delta_tracks_new_entrants_and_dropouts(monkeypatch, tmp_path):
    """CycleDelta records which SMILES entered and left the top-K between cycles."""
    from dataclasses import dataclass
    from des_multi_agent.evaluation import DesResult

    @dataclass(frozen=True)
    class _Curve:
        smiles_b: str
        smiles_a: str = "Cc1ccc(O)cc1"
        tm_pred_k: list = None
        ratios: list = None
        t1_k: float = 330.0
        t2_k: float = 289.0
        def __post_init__(self):
            if self.tm_pred_k is None:
                object.__setattr__(self, "tm_pred_k", [230.0])
            if self.ratios is None:
                object.__setattr__(self, "ratios", [0.5])

    def _r(smiles):
        return DesResult(
            curve=_Curve(smiles_b=smiles), absolute_pass=True,
            relative_pass=True, is_des=True, rationale="t", min_tm_k=230.0,
        )

    cycle_results = [
        [_r("CCO"), _r("OCCO")],   # cycle 1 top-2: CCO, OCCO
        [_r("OCCO"), _r("CC(=O)O")],  # cycle 2 top-2: OCCO (stays), CC(=O)O (new), CCO drops
    ]
    call_idx = {"n": 0}

    def _rotating_search(*args, **kwargs):
        from unittest.mock import MagicMock
        outcome = MagicMock()
        outcome.results = cycle_results[call_idx["n"]]
        call_idx["n"] += 1
        return outcome

    monkeypatch.setattr("des_multi_agent.multi_cycle.run_search_report", _rotating_search)

    from des_multi_agent.multi_cycle import run_multi_cycle_search
    ckpt = tmp_path / "ckpt.pt"; ckpt.write_bytes(b"")
    cfg = tmp_path / "config.yaml"; cfg.write_text("")

    result = run_multi_cycle_search("CCO", 2, str(ckpt), str(cfg), n_cycles=2, top_k_convergence=2)
    delta2 = result.cycle_deltas[1]
    assert "CC(=O)O" in delta2.new_entrants
    assert "CCO" in delta2.dropouts
```

- [ ] **Step 5.2 — Run to verify failures**

```
python -m pytest tests/test_h1_h2_h3_h4_h5.py -k "multi_cycle or cycle_delta" --tb=short -q
```
Expected: `ModuleNotFoundError` — `multi_cycle` not yet created.

- [ ] **Step 5.3 — Create `des_multi_agent/multi_cycle.py`**

```python
"""H1 + H5 — multi-cycle DES screening with convergence detection."""
from __future__ import annotations

from dataclasses import dataclass, field

from .orchestrator import run_search_report
from .schemas import DesThresholds
from .uncertainty import UncertaintyPolicy


@dataclass
class CycleDelta:
    cycle: int
    n_screened: int
    n_des: int
    top_smiles: frozenset
    new_entrants: list[str]
    dropouts: list[str]
    converged: bool


@dataclass
class MultiCycleOutcome:
    final_outcome: object   # SearchOutcome — avoid circular import
    cycle_deltas: list[CycleDelta]
    total_cycles: int
    converged: bool


def run_multi_cycle_search(
    component_a: str,
    n: int,
    checkpoint_path: str,
    config_path: str = "ml_des_mp/config.yaml",
    *,
    n_cycles: int = 3,
    top_k_convergence: int = 5,
    thresholds: DesThresholds | None = None,
    uncertainty_policy: UncertaintyPolicy | None = None,
    llm_cfg=None,
    llm_request_fn=None,
    discovery_path: str | None = None,
    viscosity_model_path: str | None = None,
    viscosity_weight: float = 0.3,
    viscosity_threshold_cp: float | None = None,
    output_dir: str | None = None,
    ensemble_checkpoints: list[str] | None = None,
    candidates_file: str | None = None,
) -> MultiCycleOutcome:
    """Run up to n_cycles iterations, passing top hits forward as brainstorm context.

    Stops early (H5) when the top-K canonical SMILES set is identical across
    two consecutive cycles.
    """
    cycle_deltas: list[CycleDelta] = []
    prev_top: frozenset = frozenset()
    last_outcome = None

    for cycle in range(1, n_cycles + 1):
        prior_results = last_outcome.results[:top_k_convergence] if last_outcome else None

        per_cycle_dir: str | None = None
        if output_dir is not None:
            import pathlib
            per_cycle_dir = str(pathlib.Path(output_dir) / f"cycle_{cycle:02d}")

        outcome = run_search_report(
            component_a=component_a,
            n=n,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            thresholds=thresholds,
            uncertainty_policy=uncertainty_policy,
            llm_cfg=llm_cfg,
            llm_request_fn=llm_request_fn,
            discovery_path=discovery_path,
            viscosity_model_path=viscosity_model_path,
            viscosity_weight=viscosity_weight,
            viscosity_threshold_cp=viscosity_threshold_cp,
            output_dir=per_cycle_dir,
            ensemble_checkpoints=ensemble_checkpoints,
            candidates_file=candidates_file,
            prior_cycle_top_results=prior_results,
        )

        top_k = frozenset(
            r.curve.smiles_b for r in outcome.results[:top_k_convergence] if r.is_des
        )
        new_entrants = sorted(top_k - prev_top)
        dropouts = sorted(prev_top - top_k)
        converged = (top_k == prev_top) and cycle > 1

        cycle_deltas.append(CycleDelta(
            cycle=cycle,
            n_screened=len(outcome.results),
            n_des=sum(1 for r in outcome.results if r.is_des),
            top_smiles=top_k,
            new_entrants=new_entrants,
            dropouts=dropouts,
            converged=converged,
        ))

        last_outcome = outcome
        prev_top = top_k

        if converged:
            break

    return MultiCycleOutcome(
        final_outcome=last_outcome,
        cycle_deltas=cycle_deltas,
        total_cycles=len(cycle_deltas),
        converged=cycle_deltas[-1].converged if cycle_deltas else False,
    )
```

- [ ] **Step 5.4 — Run tests**

```
python -m pytest tests/test_h1_h2_h3_h4_h5.py -k "multi_cycle or cycle_delta" --tb=short -q
```
Expected: 4 PASS.

- [ ] **Step 5.5 — Commit**

```bash
git add des_multi_agent/multi_cycle.py tests/test_h1_h2_h3_h4_h5.py
git commit -m "$(cat <<'EOF'
feat(H1+H5): add run_multi_cycle_search with CycleDelta and convergence detection

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6 — CLI wiring (H1, H2)

**Files:**
- Modify: `des_multi_agent/cli.py`
- Test: `tests/test_h1_h2_h3_h4_h5.py`

- [ ] **Step 6.1 — Write the failing tests** (append to `tests/test_h1_h2_h3_h4_h5.py`):

```python
# ── CLI: H1 + H2 flags ───────────────────────────────────────────────────────

def test_cli_n_cycles_parsed():
    from des_multi_agent.cli import build_parser
    args = build_parser().parse_args(["--n-cycles", "4"])
    assert args.n_cycles == 4


def test_cli_n_cycles_default_is_one():
    from des_multi_agent.cli import build_parser
    args = build_parser().parse_args([])
    assert args.n_cycles == 1


def test_cli_viscosity_threshold_parsed():
    from des_multi_agent.cli import build_parser
    args = build_parser().parse_args(["--viscosity-threshold", "300"])
    assert args.viscosity_threshold == 300.0


def test_cli_viscosity_threshold_default_is_none():
    from des_multi_agent.cli import build_parser
    args = build_parser().parse_args([])
    assert args.viscosity_threshold is None


def test_cli_viscosity_weight_parsed():
    from des_multi_agent.cli import build_parser
    args = build_parser().parse_args(["--viscosity-weight", "0.5"])
    assert args.viscosity_weight == pytest.approx(0.5)


def test_cli_viscosity_weight_default():
    from des_multi_agent.cli import build_parser
    args = build_parser().parse_args([])
    assert args.viscosity_weight == pytest.approx(0.3)
```

- [ ] **Step 6.2 — Run to verify failures**

```
python -m pytest tests/test_h1_h2_h3_h4_h5.py -k "cli_n_cycles or cli_viscosity" --tb=short -q
```
Expected: `SystemExit` / `AttributeError` — flags not yet defined.

- [ ] **Step 6.3 — Add flags to `des_multi_agent/cli.py`**

In `build_parser()`, add these three arguments after the `--dry-run` argument:

```python
    parser.add_argument(
        "--n-cycles",
        type=int,
        default=1,
        dest="n_cycles",
        help="Number of screening iterations; the top-K hits from each cycle seed the next (default: 1 = single shot)",
    )
    parser.add_argument(
        "--viscosity-threshold",
        type=float,
        default=None,
        dest="viscosity_threshold",
        help="Maximum acceptable viscosity (cP); DES-formers above this threshold sort below passing candidates",
    )
    parser.add_argument(
        "--viscosity-weight",
        type=float,
        default=0.3,
        dest="viscosity_weight",
        help="Weight [0,1] of the viscosity component in composite ranking (default: 0.3)",
    )
```

- [ ] **Step 6.4 — Wire multi-cycle dispatch in `main()`**

Add import at the top of `cli.py`:

```python
from .multi_cycle import run_multi_cycle_search
```

In the DES workflow section of `main()`, replace the single `run_search_report` call with a branch:

```python
        if getattr(args, "n_cycles", 1) > 1:
            multi_outcome = run_multi_cycle_search(
                component_a=args.component_a,
                n=args.n,
                checkpoint_path=checkpoint_path,
                config_path=args.config_path,
                thresholds=thresholds,
                uncertainty_policy=uncertainty_policy,
                llm_cfg=llm_cfg,
                discovery_path=args.discovery_path,
                viscosity_model_path=args.viscosity_model_path,
                viscosity_weight=args.viscosity_weight,
                viscosity_threshold_cp=args.viscosity_threshold,
                output_dir=args.output_dir,
                ensemble_checkpoints=ensemble_checkpoints,
                candidates_file=getattr(args, "candidates_file", None),
                n_cycles=args.n_cycles,
            )
            outcome = multi_outcome.final_outcome
            # Print cycle delta summary to stderr
            for delta in multi_outcome.cycle_deltas:
                status = "converged" if delta.converged else f"cycle {delta.cycle}"
                new = f"+{len(delta.new_entrants)}" if delta.new_entrants else ""
                out = f"-{len(delta.dropouts)}" if delta.dropouts else ""
                print(
                    f"[cycle {delta.cycle}/{multi_outcome.total_cycles}] "
                    f"screened={delta.n_screened} des={delta.n_des} "
                    f"top-K changes: {new or '0'} new, {out or '0'} dropped"
                    + (" — CONVERGED" if delta.converged else ""),
                    file=sys.stderr,
                )
        else:
            outcome = run_search_report(
                component_a=args.component_a,
                n=args.n,
                checkpoint_path=checkpoint_path,
                config_path=args.config_path,
                thresholds=thresholds,
                uncertainty_policy=uncertainty_policy,
                llm_cfg=llm_cfg,
                discovery_path=args.discovery_path,
                viscosity_model_path=args.viscosity_model_path,
                viscosity_weight=args.viscosity_weight,
                viscosity_threshold_cp=args.viscosity_threshold,
                output_dir=args.output_dir,
                ensemble_checkpoints=ensemble_checkpoints,
                candidates_file=getattr(args, "candidates_file", None),
                save_run_memory_path=getattr(args, "save_run_memory", None),
                reuse_run_path=getattr(args, "reuse_run", None),
            )
```

Note: the existing call to `run_search_report` is the `else` branch above; the rest of the `main()` reporting code (`format_report` / print) continues to operate on `outcome` as before.

- [ ] **Step 6.5 — Run tests**

```
python -m pytest tests/test_h1_h2_h3_h4_h5.py -k "cli_n_cycles or cli_viscosity" --tb=short -q
```
Expected: 6 PASS.

- [ ] **Step 6.6 — Commit**

```bash
git add des_multi_agent/cli.py tests/test_h1_h2_h3_h4_h5.py
git commit -m "$(cat <<'EOF'
feat(H1+H2): add --n-cycles, --viscosity-threshold, --viscosity-weight CLI flags

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7 — Full suite verification

- [ ] **Step 7.1 — Run all new tests**

```
python -m pytest tests/test_h1_h2_h3_h4_h5.py --tb=short -q
```
Expected: all tests PASS, zero failures.

- [ ] **Step 7.2 — Run full suite**

```
python -m pytest --tb=short -q
```
Expected: all tests PASS (no regressions from prior tranches).

- [ ] **Step 7.3 — Fix any failures** before proceeding.

- [ ] **Step 7.4 — Final commit**

```bash
git add -p   # review any uncommitted changes
git commit -m "$(cat <<'EOF'
test: full suite green after H1–H5 iteration loop

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

---

## Task 8 — H6 schema, prompt, parser, and provider abstract

**Goal:** Introduce `CandidateFamily` as a first-class schema type so the LLM can reason over chemical families (polyols, amides, imidazolium salts, etc.) before proposing specific SMILES. This is the foundation for two-stage brainstorming and the family ledger.

**Files:**
- Modify: `des_multi_agent/llm/schemas.py`
- Modify: `des_multi_agent/llm/prompts.py`
- Modify: `des_multi_agent/llm/parser.py`
- Modify: `des_multi_agent/llm/provider.py`
- Create: `tests/test_h6.py`

- [ ] **Step 8.1 — Write the failing tests**

Create `tests/test_h6.py`:

```python
from __future__ import annotations
import pytest


# ── H6: CandidateFamily schema ───────────────────────────────────────────────

def test_candidate_family_fields():
    from des_multi_agent.llm.schemas import CandidateFamily
    f = CandidateFamily(name="polyols", rationale="Multiple OH groups donate H-bonds.", hbd_hba_role="HBD")
    assert f.name == "polyols"
    assert f.hbd_hba_role == "HBD"


def test_parse_candidate_families_basic():
    from des_multi_agent.llm.parser import parse_candidate_families
    raw = '[{"name": "polyols", "rationale": "Strong HBD.", "hbd_hba_role": "HBD"}]'
    families = parse_candidate_families(raw)
    assert len(families) == 1
    assert families[0].name == "polyols"


def test_parse_candidate_families_skips_incomplete():
    from des_multi_agent.llm.parser import parse_candidate_families
    raw = '[{"name": "polyols"}, {"name": "amides", "rationale": "ok", "hbd_hba_role": "HBA"}]'
    families = parse_candidate_families(raw)
    assert len(families) == 1
    assert families[0].name == "amides"


def test_parse_candidate_families_empty():
    from des_multi_agent.llm.parser import parse_candidate_families
    assert parse_candidate_families("[]") == []


def test_parse_candidate_families_wrapped():
    from des_multi_agent.llm.parser import parse_candidate_families
    raw = '{"items": [{"name": "amines", "rationale": "HBA.", "hbd_hba_role": "HBA"}]}'
    families = parse_candidate_families(raw)
    assert len(families) == 1


def test_family_selection_prompt_contains_component_a():
    from des_multi_agent.llm.prompts import family_selection_prompt
    text = family_selection_prompt("CC(=O)O", None, "ctx")
    assert "CC(=O)O" in text
    assert "JSON" in text
    assert "hbd_hba_role" in text


def test_candidate_brainstorm_prompt_includes_families():
    from des_multi_agent.llm.schemas import CandidateFamily
    from des_multi_agent.llm.prompts import candidate_brainstorm_prompt
    families = [CandidateFamily(name="polyols", rationale="HBD.", hbd_hba_role="HBD")]
    text = candidate_brainstorm_prompt("CC(=O)O", None, "ctx", families=families)
    assert "polyols" in text


def test_select_candidate_families_is_abstract():
    import inspect
    from des_multi_agent.llm.provider import LLMProvider
    assert "select_candidate_families" in {
        name for name, _ in inspect.getmembers(LLMProvider, predicate=inspect.isfunction)
    }
```

- [ ] **Step 8.2 — Run to verify failures**

```
python -m pytest tests/test_h6.py --tb=short -q
```
Expected: `ImportError` / `AttributeError` — `CandidateFamily` and related symbols not yet defined.

- [ ] **Step 8.3 — Add `CandidateFamily` to `des_multi_agent/llm/schemas.py`**

Append after `ContradictionNote`:

```python
@dataclass(frozen=True)
class CandidateFamily:
    name: str           # e.g., "polyols", "amides", "imidazolium salts"
    rationale: str      # why this family suits DES formation with component A
    hbd_hba_role: str   # "HBD", "HBA", or "both"
```

- [ ] **Step 8.4 — Add `family_selection_prompt` to `des_multi_agent/llm/prompts.py`**

Add after `candidate_brainstorm_prompt`:

```python
def family_selection_prompt(
    component_a: str,
    constraints: dict | None,
    context: str,
    max_families: int = 6,
) -> str:
    return "".join([
        "Return raw JSON only. Do not use markdown fences or commentary.\n",
        "Return a JSON array of chemical families to explore as DES partner candidates.\n",
        f"Component A: {component_a}\n",
        f"Constraints: {constraints or {}}\n",
        f"Context: {context}\n",
        f"Return at most {max_families} families.\n",
        'Each item must contain name, rationale, and hbd_hba_role ("HBD", "HBA", or "both").',
    ])
```

Update `candidate_brainstorm_prompt` to accept an optional `families` list and inject it into the prompt:

```python
def candidate_brainstorm_prompt(
    component_a: str,
    constraints: dict | None,
    context: str,
    max_items: int | None = None,
    families=None,
) -> str:
    parts = [
        "Return raw JSON only. Do not use markdown fences or commentary.\n",
        "Return a JSON array of candidate partner molecules for DES screening.\n",
        f"Component A: {component_a}\n",
        f"Constraints: {constraints or {}}\n",
        f"Context: {context}\n",
    ]
    if families:
        parts.append("Distribute candidates across these chemical families:\n")
        for f in families:
            parts.append(f"  - {f.name}: {f.rationale} (role: {f.hbd_hba_role})\n")
    if max_items is not None:
        parts.append(f"Return at most {max_items} items.\n")
    parts.append("Each item must contain smiles, rationale, and family.")
    return "".join(parts)
```

- [ ] **Step 8.5 — Add `parse_candidate_families` to `des_multi_agent/llm/parser.py`**

Add `CandidateFamily` to the import:

```python
from .schemas import CandidateBrainstorm, CandidateFamily, CandidateReview, ContradictionNote, CritiqueNote, ExplanationNote
```

Append after `parse_contradiction_notes`:

```python
def parse_candidate_families(raw: str) -> list[CandidateFamily]:
    data = _coerce_json(raw)
    if not isinstance(data, list):
        return []
    out: list[CandidateFamily] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        rationale = str(item.get("rationale", "")).strip()
        hbd_hba_role = str(item.get("hbd_hba_role", "")).strip()
        if not name or not rationale or not hbd_hba_role:
            continue
        out.append(CandidateFamily(name=name, rationale=rationale, hbd_hba_role=hbd_hba_role))
    return out
```

- [ ] **Step 8.6 — Add `select_candidate_families` abstract method to `des_multi_agent/llm/provider.py`**

Add `CandidateFamily` to the import:

```python
from .schemas import CandidateBrainstorm, CandidateFamily, CandidateReview, ContradictionNote, CritiqueNote, ExplanationNote
```

Add abstract method after `brainstorm_candidates`:

```python
@abstractmethod
def select_candidate_families(
    self, component_a: str, constraints: dict | None, context: str
) -> list[CandidateFamily]:
    raise NotImplementedError
```

- [ ] **Step 8.7 — Run tests**

```
python -m pytest tests/test_h6.py --tb=short -q
```
Expected: all 8 tests PASS.

- [ ] **Step 8.8 — Commit**

```bash
git add des_multi_agent/llm/schemas.py des_multi_agent/llm/prompts.py \
        des_multi_agent/llm/parser.py des_multi_agent/llm/provider.py \
        tests/test_h6.py
git commit -m "$(cat <<'EOF'
feat(H6): add CandidateFamily schema, family_selection_prompt, parser, and provider abstract

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9 — H6 two-stage brainstorm + family ledger + enriched context

**Goal:** Wire the two-stage brainstorm into `BaseLLMProvider` (stage 1: select families; stage 2: generate SMILES distributed across those families). Build a `family_ledger` in `CycleDelta` that tracks DES-positive hit counts per family. Enrich `_build_iterative_context` so each cycle learns which families were productive.

**Files:**
- Modify: `des_multi_agent/llm/base.py`
- Modify: `des_multi_agent/multi_cycle.py`
- Modify: `des_multi_agent/orchestrator.py`
- Test: `tests/test_h6.py`

- [ ] **Step 9.1 — Write the failing tests** (append to `tests/test_h6.py`):

```python
# ── H6: two-stage brainstorm ─────────────────────────────────────────────────

def test_brainstorm_calls_family_selection_first(monkeypatch):
    """BaseLLMProvider.brainstorm_candidates calls select_candidate_families before the brainstorm."""
    from des_multi_agent.llm.base import BaseLLMProvider
    from des_multi_agent.llm.schemas import CandidateFamily

    call_order = []

    class _StubProvider(BaseLLMProvider):
        request_profile = None
        def extract_text(self, raw): return raw

    provider = _StubProvider.__new__(_StubProvider)

    def _fake_families(component_a, constraints, context):
        call_order.append("families")
        return [CandidateFamily(name="polyols", rationale="HBD.", hbd_hba_role="HBD")]

    def _fake_request(prompt):
        call_order.append("brainstorm")
        # families name should appear in the prompt
        assert "polyols" in prompt
        return '[{"smiles": "OCCO", "rationale": "diol", "family": "polyols"}]'

    monkeypatch.setattr(provider, "select_candidate_families", _fake_families)
    monkeypatch.setattr(provider, "_request", _fake_request)
    object.__setattr__(provider, "max_candidates", 10)

    results = provider.brainstorm_candidates("CC(=O)O", None, "ctx")
    assert call_order == ["families", "brainstorm"]
    assert results[0].smiles == "OCCO"


def test_brainstorm_falls_back_when_family_selection_raises(monkeypatch):
    """If select_candidate_families raises, brainstorm_candidates continues without families."""
    from des_multi_agent.llm.base import BaseLLMProvider

    class _StubProvider(BaseLLMProvider):
        request_profile = None
        def extract_text(self, raw): return raw

    provider = _StubProvider.__new__(_StubProvider)

    def _failing_families(*a, **kw):
        raise RuntimeError("LLM timeout")

    def _fake_request(prompt):
        return '[{"smiles": "OCCO", "rationale": "ok", "family": "polyols"}]'

    monkeypatch.setattr(provider, "select_candidate_families", _failing_families)
    monkeypatch.setattr(provider, "_request", _fake_request)
    object.__setattr__(provider, "max_candidates", 10)

    # should not raise
    results = provider.brainstorm_candidates("CC(=O)O", None, "ctx")
    assert len(results) == 1


def test_select_candidate_families_implemented_in_base(monkeypatch):
    """BaseLLMProvider.select_candidate_families calls the LLM and parses families."""
    from des_multi_agent.llm.base import BaseLLMProvider

    class _StubProvider(BaseLLMProvider):
        request_profile = None
        def extract_text(self, raw): return raw

    provider = _StubProvider.__new__(_StubProvider)
    monkeypatch.setattr(
        provider, "_request",
        lambda prompt: '[{"name": "amides", "rationale": "HBA.", "hbd_hba_role": "HBA"}]'
    )
    families = provider.select_candidate_families("CC(=O)O", None, "ctx")
    assert len(families) == 1
    assert families[0].name == "amides"


# ── H6: family ledger in CycleDelta ─────────────────────────────────────────

def test_cycle_delta_has_family_ledger():
    from des_multi_agent.multi_cycle import CycleDelta
    delta = CycleDelta(
        cycle=1, n_screened=5, n_des=2,
        top_smiles=frozenset(["CCO"]),
        new_entrants=["CCO"], dropouts=[], converged=False,
        family_ledger={"polyols": 2},
    )
    assert delta.family_ledger["polyols"] == 2


def test_multi_cycle_builds_family_ledger(monkeypatch, tmp_path):
    """run_multi_cycle_search builds a family_ledger from brainstorm_candidates + DES results."""
    from dataclasses import dataclass
    from des_multi_agent.evaluation import DesResult
    from des_multi_agent.llm.schemas import CandidateBrainstorm

    @dataclass(frozen=True)
    class _Curve:
        smiles_b: str
        smiles_a: str = "Cc1ccc(O)cc1"
        tm_pred_k: list = None
        ratios: list = None
        t1_k: float = 330.0
        t2_k: float = 289.0
        def __post_init__(self):
            if self.tm_pred_k is None:
                object.__setattr__(self, "tm_pred_k", [230.0])
            if self.ratios is None:
                object.__setattr__(self, "ratios", [0.5])

    des_result = DesResult(
        curve=_Curve(smiles_b="OCCO"), absolute_pass=True,
        relative_pass=True, is_des=True, rationale="t", min_tm_k=230.0,
    )
    brainstorm = CandidateBrainstorm(smiles="OCCO", rationale="diol", family="polyols")

    from unittest.mock import MagicMock
    outcome = MagicMock()
    outcome.results = [des_result]
    outcome.brainstorm_candidates = [brainstorm]

    monkeypatch.setattr("des_multi_agent.multi_cycle.run_search_report", lambda *a, **kw: outcome)

    from des_multi_agent.multi_cycle import run_multi_cycle_search
    ckpt = tmp_path / "ckpt.pt"; ckpt.write_bytes(b"")
    cfg = tmp_path / "config.yaml"; cfg.write_text("")

    result = run_multi_cycle_search("CCO", 2, str(ckpt), str(cfg), n_cycles=1)
    assert result.cycle_deltas[0].family_ledger.get("polyols", 0) >= 1


# ── H6: enriched iterative context ──────────────────────────────────────────

def test_build_iterative_context_includes_family_ledger(monkeypatch):
    from des_multi_agent.orchestrator import _build_iterative_context
    from dataclasses import dataclass
    from des_multi_agent.evaluation import DesResult

    @dataclass(frozen=True)
    class _Curve:
        smiles_b: str = "OCCO"
        smiles_a: str = "Cc1ccc(O)cc1"
        tm_pred_k: list = None
        ratios: list = None
        t1_k: float = 330.0
        t2_k: float = 289.0
        def __post_init__(self):
            if self.tm_pred_k is None:
                object.__setattr__(self, "tm_pred_k", [230.0])
            if self.ratios is None:
                object.__setattr__(self, "ratios", [0.5])

    result = DesResult(
        curve=_Curve(), absolute_pass=True, relative_pass=True,
        is_des=True, rationale="t", min_tm_k=230.0,
    )
    ledger = {"polyols": 3, "amides": 1}
    ctx = _build_iterative_context("base", [result], family_ledger=ledger)
    assert "polyols" in ctx
    assert "3" in ctx
```

- [ ] **Step 9.2 — Run to verify failures**

```
python -m pytest tests/test_h6.py -k "two_stage or brainstorm_calls or brainstorm_falls or select_candidate or family_ledger or iterative_context_includes" --tb=short -q
```
Expected: failures — methods not yet updated.

- [ ] **Step 9.3 — Implement `select_candidate_families` and two-stage `brainstorm_candidates` in `des_multi_agent/llm/base.py`**

Add `CandidateFamily` and `parse_candidate_families` / `family_selection_prompt` to imports:

```python
from .parser import (
    parse_candidate_brainstorms, parse_candidate_families, parse_candidate_review,
    parse_contradiction_notes, parse_critique_notes, parse_explanation_notes,
)
from .prompts import (
    candidate_brainstorm_prompt, candidate_review_prompt, contradiction_prompt,
    critique_prompt, explanation_prompt, family_selection_prompt,
)
from .schemas import (
    CandidateBrainstorm, CandidateFamily, CandidateReview,
    ContradictionNote, CritiqueNote, ExplanationNote,
)
```

Replace `brainstorm_candidates` and add `select_candidate_families`:

```python
def select_candidate_families(
    self, component_a: str, constraints: dict | None, context: str
) -> list[CandidateFamily]:
    raw = self._request(family_selection_prompt(component_a, constraints, context))
    return parse_candidate_families(raw)

def brainstorm_candidates(
    self, component_a: str, constraints: dict | None, context: str
) -> list[CandidateBrainstorm]:
    families: list[CandidateFamily] = []
    try:
        families = self.select_candidate_families(component_a, constraints, context)
    except Exception:
        pass  # fall back to single-stage if family selection fails
    raw = self._request(
        candidate_brainstorm_prompt(component_a, constraints, context, self.max_candidates, families)
    )
    return parse_candidate_brainstorms(raw)[: self.max_candidates]
```

- [ ] **Step 9.4 — Add `family_ledger` to `CycleDelta` in `des_multi_agent/multi_cycle.py`**

Update `CycleDelta` dataclass:

```python
@dataclass
class CycleDelta:
    cycle: int
    n_screened: int
    n_des: int
    top_smiles: frozenset
    new_entrants: list[str]
    dropouts: list[str]
    converged: bool
    family_ledger: dict[str, int] = field(default_factory=dict)
```

In `run_multi_cycle_search`, build the ledger immediately after `outcome` is received (before `CycleDelta` is constructed):

```python
        # H6 — build family ledger: DES-positive hit count per chemical family
        smiles_to_family = {
            bc.smiles: bc.family for bc in getattr(outcome, "brainstorm_candidates", [])
        }
        family_ledger: dict[str, int] = {}
        for r in outcome.results:
            if r.is_des:
                fam = smiles_to_family.get(r.curve.smiles_b, "unknown")
                family_ledger[fam] = family_ledger.get(fam, 0) + 1
```

Pass `family_ledger=family_ledger` to `CycleDelta(...)`.

Update the `prior_results` assignment to forward the accumulated ledger to the next cycle context:

```python
        prior_results = last_outcome.results[:top_k_convergence] if last_outcome else None
        # Accumulate ledger across cycles for richer context
        accumulated_ledger: dict[str, int] = {}
        for d in cycle_deltas:
            for fam, n in d.family_ledger.items():
                accumulated_ledger[fam] = accumulated_ledger.get(fam, 0) + n
```

Pass `family_ledger=accumulated_ledger` when calling `run_search_report` via the `prior_cycle_top_results` mechanism. Since the ledger is consumed by `_build_iterative_context`, add a `prior_family_ledger` parameter to `run_search_report` (see next step).

- [ ] **Step 9.5 — Enrich `_build_iterative_context` and `run_search_report` in `des_multi_agent/orchestrator.py`**

Update `_build_iterative_context` signature and body:

```python
def _build_iterative_context(
    base_context: str,
    prior_top: list,
    family_ledger: dict[str, int] | None = None,
) -> str:
    if not prior_top:
        return base_context
    lines = "\n".join(
        f"  - {r.curve.smiles_b}: min_tm_k={r.min_tm_k:.1f} K, is_des={r.is_des}"
        for r in prior_top[:5]
    )
    ctx = base_context + f"\nPrior cycle top results (bias generation toward these chemical families):\n{lines}"
    if family_ledger:
        top_families = sorted(family_ledger.items(), key=lambda x: -x[1])[:3]
        fam_lines = "\n".join(f"  - {fam}: {n} DES-positive hits" for fam, n in top_families)
        ctx += f"\nTop productive chemical families:\n{fam_lines}"
    return ctx
```

Add `prior_family_ledger: dict[str, int] | None = None` to `run_search_report` signature.

In the brainstorm block, pass `family_ledger` when building iterative context:

```python
            if prior_cycle_top_results:
                brainstorm_context = _build_iterative_context(
                    review_context, prior_cycle_top_results, family_ledger=prior_family_ledger
                )
```

In `des_multi_agent/multi_cycle.py`, pass `prior_family_ledger=accumulated_ledger` in the `run_search_report` call for cycles 2+:

```python
        outcome = run_search_report(
            ...
            prior_cycle_top_results=prior_results,
            prior_family_ledger=accumulated_ledger if cycle > 1 else None,
        )
```

- [ ] **Step 9.6 — Run tests**

```
python -m pytest tests/test_h6.py --tb=short -q
```
Expected: all tests PASS.

- [ ] **Step 9.7 — Run full suite to catch regressions**

```
python -m pytest --tb=short -q
```
Expected: all tests PASS.

- [ ] **Step 9.8 — Commit**

```bash
git add des_multi_agent/llm/base.py des_multi_agent/multi_cycle.py \
        des_multi_agent/orchestrator.py tests/test_h6.py
git commit -m "$(cat <<'EOF'
feat(H6): two-stage brainstorm with family ledger and enriched cycle context

LLM first selects chemical families (CandidateFamily), then distributes
candidates across them. Each CycleDelta records a family_ledger mapping
family → DES-positive hits. The ledger is fed forward to _build_iterative_context
so subsequent cycles know which chemical families were productive.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 10 — Full suite verification (H6)

- [ ] **Step 10.1 — Run H6 tests**

```
python -m pytest tests/test_h6.py --tb=short -q
```
Expected: all H6 tests PASS.

- [ ] **Step 10.2 — Run full suite**

```
python -m pytest --tb=short -q
```
Expected: all tests PASS, zero regressions from H1–H5.

- [ ] **Step 10.3 — Fix any failures** before proceeding.

---

## Self-review checklist

**Spec coverage:**
- H1 (`--n-cycles` multi-cycle loop): Tasks 5 + 6 ✅
- H2 (viscosity-aware composite ranking): Tasks 3 + 4 + 6 ✅
- H3 (chemical contradiction detection): Tasks 1 + 2 + 4 ✅
- H4 (cycle-aware candidate generation): Task 4 (`_build_iterative_context` + `prior_cycle_top_results`) ✅
- H5 (convergence detection): Task 5 (`converged` flag in cycle loop) ✅
- H6 (chemical-group-level brainstorm + family ledger): Tasks 8 + 9 + 10

**Placeholder scan:** No TBDs, no "add appropriate error handling" phrases, every code step shows the actual code.

**Type consistency:**
- `ContradictionNote` defined in Task 1, imported in Tasks 2 + 4 consistently.
- `CandidateFamily` defined in Task 8, imported in Tasks 8 (parser/provider) + 9 (base) consistently.
- `rank_results_composite` defined in Task 3, called in Task 4 orchestrator wiring with identical signature `(results, visc_by_smiles_b, viscosity_weight=..., viscosity_threshold_cp=...)`.
- `CycleDelta` / `MultiCycleOutcome` defined in Task 5 (extended in Task 9 with `family_ledger`), referenced in Task 6 without signature break.
- `prior_cycle_top_results` added to `run_search_report` in Task 4, passed from `run_multi_cycle_search` in Task 5.
- `prior_family_ledger` added to `run_search_report` in Task 9, passed from cycle loop in `multi_cycle.py`.
- `viscosity_weight` / `viscosity_threshold_cp` added to `run_search_report` in Task 4, forwarded from both CLI paths in Task 6.
- `_build_iterative_context` extended in Task 9 with optional `family_ledger` param — existing callers pass no ledger and behaviour is unchanged (backward compatible).
