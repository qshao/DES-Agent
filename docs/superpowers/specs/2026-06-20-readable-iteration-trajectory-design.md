# Readable Iteration Trajectory — Design Spec

**Date:** 2026-06-20
**Status:** Approved design (Approach A), pending implementation-plan handoff
**Author:** brainstormed with the user

## Goal

Make the *trajectory* of every iterative DES-Agent search readable — not only the
final report, but the per-cycle progression: what entered and left the shortlist
each cycle, which chemical families were reinforced, and why the search converged
(or didn't). Deliver this as a durable, reviewable Markdown artifact plus a tidy
live console trace, for the DES multi-cycle search and the two metal workflows.

## Background / Problem

Iterative searches compute rich per-cycle state and then throw most of it away:

- **DES multi-cycle** (`run_multi_cycle_search`) keeps a `CycleDelta` per cycle
  (entrants, dropouts, family ledger, convergence flag) but only a terse
  one-line-per-cycle count reaches the user — on **stderr**
  (`[cycle 2/3] screened=5 des=5 top-K changes: +1 new, -0 dropped`). The
  rendered report shows only the **last** cycle. The `CycleDelta` also stores top
  hits as an unordered SMILES *set* with **no metric values**, so even what is
  retained can't show how the top candidates ranked.
- **Metal selectivity** (`run_metal_selectivity_screen`) and the
  **selectivity-DES pipeline** (`run_selectivity_des_pipeline`) discard per-cycle
  state entirely: their loops print `[cycle N/M] …` / `[outer N/M] …` to stderr
  and keep only cumulative/final results. There is no history object to render.

So "make any outcome during iteration readable" requires two layers: **capture**
a per-cycle, render-oriented snapshot inside each workflow's loop (where the
cycle's outcome is still in hand), then **render** those snapshots through one
shared, well-bounded renderer.

## Scope

In scope (the three iterative workflows the user selected):

1. DES multi-cycle search — `des_multi_agent/multi_cycle.py`
2. Metal selectivity screen — `des_multi_agent/workflows/metal_binding_selectivity.py`
3. Selectivity-DES pipeline — `des_multi_agent/workflows/selectivity_des_pipeline.py`

Out of scope (see "Out of Scope" below): restructuring the existing dense
final-report pipe tables in `reporting.py`; single-cycle (`n_cycles=1`) runs,
which have no trajectory to show.

## Architecture

A new module `des_multi_agent/trajectory.py` owns the **capture model** and the
**renderer**; it has no dependency on any workflow (workflows depend on it, not
the reverse). Each workflow builds snapshots in its loop and attaches a
`SearchTrajectory` to its existing outcome object. The CLI renders the trajectory
to stdout and, when an output directory is set, writes `trajectory.md`.

```
workflow loop  ──builds──>  CycleSnapshot (one per cycle)
                                  │
                          SearchTrajectory  ──attached to──> {Multi,Selectivity,Pipeline}Outcome
                                  │
            ┌─────────────────────┴─────────────────────┐
   format_trajectory_console(traj)            format_trajectory_report(traj)
        → stdout live trace                        → trajectory.md (Markdown)
```

### Data model (`trajectory.py`)

```python
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TopEntry:
    """One ranked candidate as it stood in a given cycle."""
    label: str            # display: "acetamide (CC(=O)N)"
    metric_name: str      # "min_tm_k" | "delta_log_k" | "composite_score"
    metric_value: float
    secondary: str        # short context, e.g. "Δ11.9%, high confidence" or "logK(Ni)=8.2"


@dataclass(frozen=True)
class CycleSnapshot:
    """Render-oriented record of one iteration. Built where the cycle outcome exists."""
    cycle: int                                  # 1-based
    n_screened: int
    n_hits: int                                 # DES-positive (DES) / target-selective (metal-selectivity) / DES-compatible (pipeline)
    top_entries: list[TopEntry]                 # best-first, the cycle's top-K
    new_entrants: list[str]                     # labels that entered top-K vs prior cycle
    dropouts: list[str]                         # labels that left top-K vs prior cycle
    family_ledger: dict[str, int] = field(default_factory=dict)  # family→hit count (DES; {} for metal)
    converged: bool = False
    convergence_reason: str = ""                # "" unless converged
    notable_warnings: list[str] = field(default_factory=list)    # cycle-scoped, capped at 3


@dataclass(frozen=True)
class SearchTrajectory:
    """Full readable record of an iterative run."""
    workflow: str                               # "des" | "metal-selectivity" | "selectivity-des"
    headline: str                               # "DES partners for ethanol (CCO)"
    metric_label: str                           # human axis name, e.g. "min Tm (K)" | "ΔlogK"
    snapshots: list[CycleSnapshot]
    total_cycles: int
    converged: bool
    convergence_reason: str                     # overall; "" if ran to cycle budget
    final_summary: list[TopEntry]               # final-cycle top entries (the readable "final outcome")
```

`metric_value`/`secondary` are computed at capture time, so the renderer is pure
formatting with no chemistry knowledge.

### Capture wiring

**DES — `multi_cycle.py`.** Inside the existing `for cycle in …` loop, after
`outcome` is computed and `top_k`/`new_entrants`/`dropouts`/`family_ledger`
already exist, also build a `CycleSnapshot`:

- `top_entries`: from `outcome.results[:top_k_convergence]`, best-first, using
  `display_name(r.curve.smiles_b)` for `label`, `metric_name="min_tm_k"`,
  `metric_value=r.min_tm_k`, and `secondary` = relative-drop % + confidence label
  (reuse `_confidence_label` via the annotated result when available).
- `new_entrants`/`dropouts`: convert the existing SMILES deltas to display labels.
- `family_ledger`: the cycle's `family_ledger` Counter (already computed).
- `converged`/`convergence_reason`: `converged` already computed; reason =
  `"top-{K} shortlist identical to previous cycle"`.
- `notable_warnings`: first 3 of `outcome.llm_warnings` that start with
  `[GROUNDING]` or `[REALITY]` (the chemistry-relevant ones).

Append each snapshot to a local list. After the loop, build a `SearchTrajectory`
(`workflow="des"`, `headline=f"DES partners for {display_name(component_a)}"`,
`metric_label="min Tm (K)"`, `final_summary` = last snapshot's `top_entries`,
overall `converged`/reason from the final delta). Add a field
`trajectory: SearchTrajectory | None = None` to `MultiCycleOutcome` and populate
it. `CycleDelta` and `cycle_deltas` are unchanged (existing tests and the
convergence logic keep using them).

**Metal selectivity — `metal_binding_selectivity.py`.** Inside the cycle loop,
after `cumulative_results` is ranked, build a `CycleSnapshot`:

- `top_entries`: top-5 of `cumulative_results` by `composite_score`,
  `metric_name="composite_score"`, `metric_value=r.composite_score`, `secondary`
  = `f"ΔlogK={r.delta_log_k:.2f}, logK({target})={r.log_k_target:.2f}"`.
- `n_hits`: count of results with positive ΔlogK (selective toward target).
- `new_entrants`/`dropouts`: top-5 ligand-SMILES set vs previous cycle's.
- `family_ledger`: `{}` (metal workflow has no DES family ledger).
- `converged`/reason: reuse the existing `_top_k_stable` early-stop signal;
  reason = `"top-5 ligand set stable vs previous cycle"`.

Add `trajectory: SearchTrajectory | None = None` to `SelectivityScreenOutcome`
(`workflow="metal-selectivity"`, `metric_label="composite score"`,
`headline=f"{target} over {competitor} selectivity"`).

**Selectivity-DES pipeline — `selectivity_des_pipeline.py`.** One snapshot per
**outer** cycle:

- `n_hits`: number of DES-compatible ligands this outer cycle.
- `top_entries`: the DES-compatible ligands, best-first by composite score,
  `metric_name="composite_score"`, `secondary` = `f"ΔlogK={…}, DES-compatible"`.
- `new_entrants`/`dropouts`: DES-compatible ligand-SMILES set vs previous outer cycle.
- `converged`/reason: reuse the existing `new_compatible == prev_compatible`
  early-stop; reason = `"DES-compatible ligand set stable vs previous outer cycle"`.

Add `trajectory: SearchTrajectory | None = None` to
`SelectivityDesPipelineOutcome` (`workflow="selectivity-des"`,
`metric_label="composite score"`).

All three additive fields default to `None`, so every existing construction site
and test is unaffected.

### Renderer (`trajectory.py`)

Two pure functions, no I/O:

```python
def format_trajectory_report(traj: SearchTrajectory) -> str:   # Markdown artifact
def format_trajectory_console(traj: SearchTrajectory) -> str:  # compact plain text
```

**`format_trajectory_report` (Markdown)** — example for DES, 3 cycles:

```markdown
# Search Trajectory — DES partners for ethanol (CCO)

Workflow: des  ·  Cycles run: 3  ·  Converged: yes (top-5 shortlist identical to previous cycle)

## Cycle 1 — 5 screened, 5 hits
Top by min Tm (K):
1. ethylene glycol (OCCO) — 201.8  (Δ22.6%, high confidence)
2. glycerol (OCC(O)CO) — 221.2  (Δ18.4%, moderate-high confidence)
3. acetamide (CC(=O)N) — 238.9  (Δ11.9%, high confidence)
Families reinforced: amide (1), diol (1), carboxylic acid (1)

## Cycle 2 — 5 screened, 5 hits
Shortlist change vs cycle 1: +1 entered (1,2-propanediol (CC(O)CO)), -1 left (water (O))
Top by min Tm (K):
1. ethylene glycol (OCCO) — 201.8  (Δ22.6%, high confidence)
…
Families reinforced: diol (2), amide (1)

## Cycle 3 — 5 screened, 5 hits  ✓ converged
Shortlist change vs cycle 2: none — top-5 identical
Converged: top-5 shortlist identical to previous cycle

## Final shortlist
1. ethylene glycol (OCCO) — min Tm 201.8 K
2. glycerol (OCC(O)CO) — min Tm 221.2 K
3. acetamide (CC(=O)N) — min Tm 238.9 K
```

Rules: omit the "Shortlist change" line on cycle 1; render "none — … identical"
when both entrant/dropout lists are empty; render `Families reinforced:` only when
`family_ledger` is non-empty (so metal workflows skip it); append `✓ converged`
to the heading of the converging cycle; list `notable_warnings` under a
`> warnings:` block when present.

**`format_trajectory_console`** — one block, ≤ ~2 lines per cycle, for stdout:

```
Trajectory — DES partners for ethanol (CCO)  (3 cycles, converged)
  cycle 1: 5 screened, 5 hits · top: ethylene glycol 201.8 K · families: amide, diol, carboxylic acid
  cycle 2: +1/-1 shortlist · top: ethylene glycol 201.8 K · families: diol, amide
  cycle 3: stable ✓ converged
```

### CLI + artifact wiring

A small writer in `trajectory.py`:

```python
def write_trajectory_artifact(output_dir: str | Path, traj: SearchTrajectory) -> Path:
    """Atomically write trajectory.md into output_dir; return its path."""
```

It writes via a temp file + `os.replace` (mirroring `exporting.py`'s atomic
pattern) and returns the path. It does **not** touch `export_des_run_bundle`
(which is DES-payload specific and shared with single-cycle runs).

In `cli.py`, for each of the three workflows when iterative (`n_cycles > 1` /
pipeline always iterates):

1. After the workflow returns, if its outcome carries a non-`None` `trajectory`,
   `print(format_trajectory_console(traj))` to **stdout** (replaces the current
   terse per-cycle stderr loop for DES; adds one for the metal workflows).
2. If `args.output_dir` is set, call `write_trajectory_artifact(args.output_dir, traj)`.

The existing final `format_report*` output is unchanged and still printed.

## Error handling

- Capture is best-effort and must never break a run: each workflow wraps its
  snapshot construction in a local `try/except` that, on failure, appends a
  warning (`"[TRAJECTORY] snapshot capture failed (cycle N): …"`) and continues —
  the search result is never affected.
- The renderer assumes a well-formed `SearchTrajectory` (built only by our
  capture code) and does no defensive parsing; it handles empty `snapshots`
  (returns a single "no cycles recorded" line) and empty `top_entries`
  (renders "no hits this cycle").
- `write_trajectory_artifact` lets `OSError` propagate to the CLI, which already
  surfaces export failures; a failed artifact write does not suppress the printed
  console trajectory or the final report.

## Testing (TDD)

Unit (pure, deterministic — no LLM, no ML):

1. `tests/test_trajectory_model.py` — `CycleSnapshot`/`SearchTrajectory`
   construction and field defaults; `TopEntry` formatting inputs.
2. `tests/test_trajectory_render.py` — `format_trajectory_report` and
   `format_trajectory_console` on hand-built trajectories: cycle-1 omits the
   change line; converged cycle gets `✓ converged`; `Families reinforced:` shown
   only with a non-empty ledger; "none — … identical" when no entrants/dropouts;
   warnings block rendered when present.

Capture (monkeypatched orchestrator/predictors, mirroring
`tests/test_orchestrator_partner_reality_report.py` stubs):

3. `tests/test_trajectory_capture_des.py` — `run_multi_cycle_search` over 2–3
   stubbed cycles yields `outcome.trajectory` with correct per-cycle
   `new_entrants`/`dropouts`/`family_ledger` and a populated `final_summary`.
4. `tests/test_trajectory_capture_metal.py` — `run_metal_selectivity_screen` and
   `run_selectivity_des_pipeline` attach a `SearchTrajectory` with one snapshot
   per (outer) cycle and correct top-entry metrics.

Integration:

5. `tests/test_cli_trajectory_artifact.py` — an iterative DES run with
   `--output-dir` writes a `trajectory.md` containing the headline and a
   `## Cycle 1` heading; stdout contains the console trajectory block.

Regression: full suite green; additive `None`-default fields keep existing
construction sites and golden example outputs unchanged. The committed example
outputs that exercise `n_cycles>1` (e.g. `examples/multi_cycle_des`) gain a
`trajectory.md` and a stdout trajectory block — those example fixtures are
re-captured as part of the work.

## Out of Scope

- Restructuring the dense final-report pipe tables in `reporting.py` (the
  trajectory artifact already gives a clean final shortlist; reworking the legacy
  tables is a separate, higher-churn effort). Noted as a follow-up.
- Single-cycle runs (`n_cycles=1`): no trajectory is produced or written.
- Any change to convergence logic, ranking, or chemistry — this work only
  captures and renders state that the search already produces.
- Machine-readable trajectory export (JSON): the Markdown artifact is the
  deliverable; a JSON sibling can be a follow-up if a dashboard needs it.
