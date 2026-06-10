# Metal Ion Selectivity Screening Agent — Design Spec

**Date:** 2026-06-10
**Status:** Approved

## Overview

A new agent workflow that screens ligands for **selectivity** toward a target metal ion over a competitor metal ion. For each candidate ligand, the ML model predicts log K for both metals; a composite score balancing selectivity (Δlog K) and affinity (log K target) ranks candidates. An LLM brainstorm loop proposes increasingly selective ligands over n cycles.

---

## 1. Scoring and Data Model

### Composite Score

```
delta_log_k    = log_k_target − log_k_competitor
composite_score = w_affinity × log_k_target + w_selectivity × delta_log_k
```

Defaults: `w_affinity = 0.5`, `w_selectivity = 0.5`. Both are user-configurable via CLI.

### New Dataclasses (`workflows/metal_binding_selectivity.py`)

```python
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
    results: list[SelectivityResult]   # sorted by composite_score descending
    n_screened: int
    n_cycles: int
    llm_brainstorm: list[CandidateBrainstorm]
    llm_candidate_reviews: list[CandidateReview]
    warnings: list[str]
```

---

## 2. Workflow Loop

**Entry point:** `run_metal_selectivity_screen` in `workflows/metal_binding_selectivity.py`

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
) -> SelectivityScreenOutcome
```

**Per-cycle steps:**

1. **Cycle 1:** generate heuristic candidates via `generate_ligand_candidates(target_metal, n)`
2. **All cycles (if LLM available):** call `llm_provider.brainstorm_ligands_selectivity(target_metal, competitor_metal, constraints, context)`
3. Deduplicate against `seen_smiles` using `canonicalize_smiles`
4. For each candidate: call `predict_log_k(target_metal, smiles)` and `predict_log_k(competitor_metal, smiles)`; compute `delta_log_k` and `composite_score`
5. Optionally call `llm_provider.review_ligand(target_metal, smiles, context)` for each result
6. Merge with cumulative results keeping best `composite_score` per SMILES
7. Convergence: `_top_k_stable()` on composite-score-sorted list; stop early if top-5 set unchanged

**Context string passed to LLM each cycle:**

```
Target metal: Cu2+
Competitor metal: Zn2+
Selectivity weight: 0.5 | Affinity weight: 0.5
Cycle: 2
Top ligands from previous cycle:
  - OC(=O)CN(CC(=O)O)CC(=O)O: log_K(Cu2+)=14.06, log_K(Zn2+)=9.10, ΔlogK=4.96, score=11.58
  - ...
```

---

## 3. LLM Prompt and Provider Method

### New prompt function: `ligand_selectivity_brainstorm_prompt` (`llm/prompts.py`)

```
You are an expert coordination chemist designing ligands with HIGH SELECTIVITY
for {target_metal} over {competitor_metal}.

Use HSAB theory, donor atom preferences, denticity, and chelate ring geometry
to propose ligands that bind {target_metal} strongly while discriminating
against {competitor_metal}.

Current best results:
{context}

Focus on: donor atom type (N vs O vs S), chelate ring size, charge match,
and geometric preference differences between the two metals.

Propose {max_items} ligands as JSON:
[{"smiles": "...", "rationale": "...", "family": "..."}]
```

### New method: `brainstorm_ligands_selectivity` (`llm/base.py`)

- Calls `select_ligand_families(target_metal, constraints, context)` first (reuses existing family-selection step)
- Then calls `ligand_selectivity_brainstorm_prompt` with both metals and families
- Falls back to single-stage if family selection fails
- Returns `list[CandidateBrainstorm]` (same schema as existing brainstorm)

---

## 4. CLI

**New workflow name:** `metal-selectivity`

**New CLI arguments:**
- `--target-metal-ion` — target metal (e.g., `Cu2+`)
- `--competitor-metal-ion` — competitor metal (e.g., `Zn2+`)
- `--affinity-weight` — weight for log K(target) in composite score (default: 0.5)
- `--selectivity-weight` — weight for Δlog K in composite score (default: 0.5)

**Example invocation:**
```bash
python -m des_multi_agent.cli \
  --workflow metal-selectivity \
  --target-metal-ion Cu2+ \
  --competitor-metal-ion Zn2+ \
  --n 20 \
  --n-cycles 3 \
  --affinity-weight 0.5 \
  --selectivity-weight 0.5
```

The `--workflow metal-selectivity` branch in `main()` calls `run_metal_selectivity_screen` and prints `format_metal_selectivity_report`.

Existing `--workflow metal-binding` behaviour is unchanged.

---

## 5. Report Format (`reporting.py`)

New function: `format_metal_selectivity_report(outcome) -> str`

```
=== Metal Selectivity Screen: Cu2+ over Zn2+ ===
Screened 20 candidate(s) over 3 cycle(s).
Top ligand: OC(=O)CN(CC(=O)O)CC(=O)O — score=11.58 (ΔlogK=4.96, logK(Cu2+)=14.06)
====================================================

ligand | log_k_target | log_k_competitor | delta_log_k | score | source | rationale

LLM brainstorm:
<smiles> | <family> | <rationale>

LLM ligand reviews:
<smiles> | <decision> | confidence=<x> | <rationale>

Warnings:
- ...
```

---

## 6. Tests (`tests/test_metal_selectivity_screen.py`)

- Composite score formula correctness
- Results sorted by `composite_score` descending
- `delta_log_k` = `log_k_target − log_k_competitor`
- Deduplication via canonicalization
- Invalid LLM SMILES skipped, valid ones scored
- LLM brainstorm called with both metal names
- LLM brainstorm failure degrades gracefully (warning added, no crash)
- Convergence: `_top_k_stable` stops loop early
- CLI routes `--workflow metal-selectivity` to `run_metal_selectivity_screen`
- Existing `--workflow metal-binding` single-pair mode unchanged
- Report contains target metal, competitor metal, `delta_log_k`, `score` headers

---

## 7. Files Changed

| File | Change |
|------|--------|
| `des_multi_agent/workflows/metal_binding_selectivity.py` | **NEW** — `SelectivityResult`, `SelectivityScreenOutcome`, `run_metal_selectivity_screen`, `_build_selectivity_context` |
| `des_multi_agent/llm/prompts.py` | Add `ligand_selectivity_brainstorm_prompt` |
| `des_multi_agent/llm/base.py` | Add `brainstorm_ligands_selectivity` method |
| `des_multi_agent/reporting.py` | Add `format_metal_selectivity_report` |
| `des_multi_agent/cli.py` | Add `metal-selectivity` workflow branch + new args |
| `tests/test_metal_selectivity_screen.py` | **NEW** — full test suite |
