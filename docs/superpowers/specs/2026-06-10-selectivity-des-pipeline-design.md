# Selectivity-DES Pipeline — Design Spec

**Date:** 2026-06-10
**Status:** Approved

## Overview

A two-phase pipeline that first screens ligands for metal-ion selectivity, then searches for chemical components that form deep eutectic solvents (DES) with the most selective ligands. The two phases run in separated iteration cycles, connected by an outer feedback loop that steers Phase 1 toward ligands that are both selective *and* DES-compatible.

---

## 1. Architecture and Flow

Three nested loops, each with its own convergence guard:

```
OUTER LOOP (max n_outer_cycles, stops when DES-compatible set stable)
│
├── PHASE 1 — Selectivity screening (n_selectivity_cycles)
│   LLM context includes: top ligands from previous cycle +
│   which prior ligands formed DES (outer cycle ≥ 2)
│
│   ↓  Bridge: threshold filter (delta_log_k >= min_delta_log_k) + cap (top_ligands)
│   ↓  Fallback: if filter yields zero, take top-N unconditionally + warning
│
└── PHASE 2 — DES partner search, one multi-cycle run per shortlisted ligand
    Each ligand runs independently (n_des_cycles)
    Results collected → DES-compatible set
    If set == previous outer pass → CONVERGED
```

**New entry point:** `run_selectivity_des_pipeline` in `des_multi_agent/workflows/selectivity_des_pipeline.py`

**Minimal surgery to existing code:** one additive parameter added to `run_metal_selectivity_screen` — `des_compatible_hints: list[str] | None = None` — folded into the LLM context string. No existing callers break.

**New CLI workflow:** `--workflow selectivity-des`

---

## 2. Data Model

New dataclasses in `des_multi_agent/workflows/selectivity_des_pipeline.py`:

```python
@dataclass(frozen=True)
class LigandDesResult:
    ligand: SelectivityResult       # selectivity scores for this ligand
    des_results: list[DesResult]    # DES partners found (may be empty)
    n_des_screened: int             # candidates evaluated in Phase 2
    des_compatible: bool            # True if any DES partner found

@dataclass
class SelectivityDesPipelineOutcome:
    target_metal: str
    competitor_metal: str
    selectivity_outcome: SelectivityScreenOutcome   # final Phase 1 state
    ligand_des_results: list[LigandDesResult]       # one entry per shortlisted ligand
    n_outer_cycles_run: int                         # actual passes completed
    converged: bool
    warnings: list[str]
```

`SelectivityScreenOutcome` and `DesResult` are reused directly — no new schemas required for either phase.

`converged` is `True` when the outer loop exited because the DES-compatible SMILES set stabilised between two consecutive passes; `False` when the cycle cap was reached.

---

## 3. Workflow Loop

### Entry point signature

```python
def run_selectivity_des_pipeline(
    target_metal: str,
    competitor_metal: str,
    checkpoint_path: str,
    config_path: str = "ml_des_mp/config.yaml",
    n_ligands: int = 20,             # selectivity candidates per cycle
    n_des_candidates: int = 20,      # DES candidates per ligand per cycle
    n_selectivity_cycles: int = 3,
    n_des_cycles: int = 3,
    n_outer_cycles: int = 2,
    min_delta_log_k: float = 0.0,    # selectivity threshold for bridge
    top_ligands: int = 3,            # max ligands passing Phase 1 → Phase 2
    w_affinity: float = 0.5,
    w_selectivity: float = 0.5,
    stability_model_path=None,
    llm_cfg=None,
    constraints: dict | None = None,
) -> SelectivityDesPipelineOutcome
```

### Per outer cycle

**Step 1 — Phase 1 (selectivity screening)**

Call `run_metal_selectivity_screen` with `des_compatible_hints=des_compatible_smiles` and `des_incompatible_hints=des_incompatible_smiles`. Both sets are derived from the previous outer cycle's `ligand_des_results`: compatible = those where `des_compatible=True`, incompatible = those shortlisted but where `des_compatible=False`. On outer cycle ≥ 2, `_build_selectivity_context` appends two new lines to the LLM context string:

```
Ligands that formed DES in previous pass (prefer similar scaffolds):
  - <smiles> ...
Ligands that did NOT form DES (avoid similar scaffolds):
  - <smiles> ...
```

Both lines are omitted on outer cycle 1 (sets are empty). The hints guide LLM brainstorming without affecting scoring or the heuristic path.

**Step 2 — Bridge (threshold + cap)**

Filter `selectivity_outcome.results` to entries where `delta_log_k >= min_delta_log_k`, then take `[:top_ligands]`. If the filter yields zero candidates, take `results[:top_ligands]` unconditionally and append a warning to `warnings`.

**Step 3 — Phase 2 (DES partner search)**

For each shortlisted ligand call `run_multi_cycle_search(component_a=ligand.ligand_smiles, n=n_des_candidates, n_cycles=n_des_cycles, ...)`. Runs are independent: an exception on one ligand appends a warning and continues to the next.

**Step 4 — Convergence check**

Collect `des_compatible_smiles = {r.ligand.ligand_smiles for r in ligand_des_results if r.des_compatible}`. If this set equals the previous outer cycle's set and outer cycle > 1, set `converged=True` and break.

---

## 4. CLI

New `--workflow selectivity-des` branch in `des_multi_agent/cli.py`.

### Reused existing arguments

| Arg | Maps to |
|-----|---------|
| `--target-metal-ion` | `target_metal` |
| `--competitor-metal-ion` | `competitor_metal` |
| `--affinity-weight` | `w_affinity` |
| `--selectivity-weight` | `w_selectivity` |
| `--n-cycles` | `n_selectivity_cycles` |
| `--n` | `n_ligands` |
| `--llm-config` | `llm_cfg` (loaded via `load_llm_config`) |
| `--stability-constant-model-path` | `stability_model_path` |
| `--checkpoint-path` | `checkpoint_path` |
| `--config-path` | `config_path` |

### New arguments (selectivity-des only)

| Arg | Type | Default | Description |
|-----|------|---------|-------------|
| `--n-des-candidates` | int | 20 | DES search breadth per ligand per cycle |
| `--n-des-cycles` | int | 3 | DES iteration depth per ligand |
| `--n-outer-cycles` | int | 2 | Outer loop cap |
| `--min-delta-log-k` | float | 0.0 | Selectivity threshold for bridge filter |
| `--top-ligands` | int | 3 | Max ligands bridging Phase 1 → Phase 2 |

### Required arguments for selectivity-des

`--target-metal-ion`, `--competitor-metal-ion`, `--checkpoint-path`. Missing any raises `parser.error`.

### Example invocation

```bash
python -m des_multi_agent.cli \
  --workflow selectivity-des \
  --target-metal-ion Cu2+ \
  --competitor-metal-ion Zn2+ \
  --n 20 --n-cycles 3 \
  --n-des-candidates 20 --n-des-cycles 3 \
  --n-outer-cycles 2 --top-ligands 3 --min-delta-log-k 0.5 \
  --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt \
  --config-path ml_des_mp/config.yaml \
  --stability-constant-model-path artifacts/stability_constants/model.json
```

---

## 5. Report Format

New `format_selectivity_des_report(outcome: SelectivityDesPipelineOutcome) -> str` in `des_multi_agent/reporting.py`.

```
=== Selectivity-DES Pipeline: Cu2+ over Zn2+ ===
Outer cycles run: 2 | Converged: yes
Shortlisted ligands: 3 | DES-compatible: 2
====================================================

--- Section 1: Selectivity Results ---

ligand | log_k_target | log_k_competitor | delta_log_k | score | des_compatible
OC(=O)CNCC(=O)O      | 14.06 | 9.10  | 4.96 | 11.58 | yes
NCC(=O)O             | 11.23 | 9.10  | 2.13 |  8.18 | yes
c1cnc2ccc3ncccc3c2c1 | 10.87 | 10.79 | 0.08 |  5.47 | no

--- Section 2: DES Partners ---

Ligand: OC(=O)CNCC(=O)O  (score=11.58, ΔlogK=4.96) — DES-compatible: YES
  partner    | min_tm_k | eutectic_ratio | rationale
  CC(=O)NCCO | 287.1    | 0.33           | min Tm=287.1 K ...
  OCC(O)CO   | 291.4    | 0.50           | min Tm=291.4 K ...

Ligand: NCC(=O)O  (score=8.18, ΔlogK=2.13) — DES-compatible: YES
  partner | min_tm_k | eutectic_ratio | rationale
  ...

Ligand: c1cnc2ccc3ncccc3c2c1  (score=5.47, ΔlogK=0.08) — DES-compatible: NO
  No DES partners found.

Warnings:
- Bridge filter found 0 ligands above min_delta_log_k=0.5; using top-3 unconditionally.
```

The `des_compatible` column in Section 1 links visually to Section 2 — readers can scan Section 1 for rankings and jump to Section 2 for partner detail on the winners.

---

## 6. Tests

New `tests/test_selectivity_des_pipeline.py`.

### Data model
- `LigandDesResult.des_compatible` is `True` iff `des_results` contains at least one `is_des=True` entry
- `SelectivityDesPipelineOutcome` fields populated correctly from mock sub-outcomes

### Bridge logic
- Threshold filter keeps only ligands with `delta_log_k >= min_delta_log_k`
- Cap respected: never more than `top_ligands` passed to Phase 2
- Fallback: when all ligands are below threshold, top-N taken unconditionally and a warning is added

### Outer loop
- Converges early when DES-compatible SMILES set is identical across two consecutive passes
- Runs full `n_outer_cycles` when set keeps changing
- `converged=True` only on early exit; `False` when cap reached

### Feedback
- `des_compatible_hints` passed to `run_metal_selectivity_screen` on outer cycle ≥ 2
- LLM context string contains both DES-compatible and DES-incompatible ligand lists

### Phase 2 resilience
- One ligand's DES failure (exception) adds a warning and does not abort the others
- Zero DES candidates found produces `des_compatible=False` with no crash

### Report
- Both section headers present in output
- `des_compatible` column present in Section 1
- Per-ligand block present in Section 2 with DES partner table
- "No DES partners found" rendered for DES-incompatible ligands

### CLI
- `--workflow selectivity-des` routes to `run_selectivity_des_pipeline`
- Missing `--target-metal-ion`, `--competitor-metal-ion`, or `--checkpoint-path` raises error
- Existing `--workflow metal-selectivity` behaviour unchanged

---

## 7. Files Changed

| File | Change |
|------|--------|
| `des_multi_agent/workflows/selectivity_des_pipeline.py` | **NEW** — `LigandDesResult`, `SelectivityDesPipelineOutcome`, `run_selectivity_des_pipeline` |
| `des_multi_agent/workflows/metal_binding_selectivity.py` | Add `des_compatible_hints` param to `run_metal_selectivity_screen` and `_build_selectivity_context` |
| `des_multi_agent/reporting.py` | Add `format_selectivity_des_report` |
| `des_multi_agent/cli.py` | Add `selectivity-des` workflow branch + 5 new args |
| `tests/test_selectivity_des_pipeline.py` | **NEW** — full test suite |
