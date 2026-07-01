---
name: des-run-debug
description: "Diagnosing and interpreting DES-Agent runs: reading trajectory artifacts, understanding convergence signals, grounding warnings, ranking score factors, and common failure patterns. Use when a run finishes poorly, cycling without improvement, or when you need to explain what happened in a run."
---

# DES-Agent Run Debugging Reference

This skill covers how to read a completed run, diagnose poor results, and
understand what each part of the output means.

---

## Output Artifacts

A run with `--output-dir <dir>` produces:

| File | Contents |
|------|----------|
| `trajectory.md` | Human-readable per-cycle narrative |
| `trajectory.json` | Machine-readable equivalent (same data) |
| `run.json` | Full structured results (candidates, scores, metadata) |
| `run.csv` | Flat CSV of all screened candidates |
| `run.manifest.json` | Provenance: CLI args, timestamps, file checksums |

If `--output-dir` is not set, the per-cycle trace still prints to stderr but
no files are written.

---

## Reading the Console Trace (stderr)

During a run, each cycle prints a one-line summary:

```
Trajectory — choline_cl vs ? (3 cycles, no)
  Cycle 1 | 18 screened, 4 hits
  Cycle 2 | 22 screened, 6 hits  · +2/-1 shortlist
  Cycle 3 | 20 screened, 6 hits  · shortlist unchanged
```

- `+2/-1 shortlist` — 2 new candidates entered the top-N; 1 fell out.
- `shortlist unchanged` — identical top-N to the previous cycle.
- `(3 cycles, no)` — ran 3 cycles, did **not** converge early.
- `(2 cycles, yes)` — converged at cycle 2; reason appears in `trajectory.md`.

---

## Reading `trajectory.md`

Structure of each cycle block:

```
### Cycle 2

Screened: 22  |  Hits: 6

Top by min_tm_k:
1. urea (CN(C)=O) — 260.4  (Δ15.2%, high confidence)
2. glycerol (OCC(O)CO) — 272.1  (Δ8.7%, medium confidence)
...

Shortlist change vs previous cycle: +2 entered (urea, betaine), -1 left (oxalic acid)

Family ledger: alcohol: 3 hits, amide: 2 hits, carboxylic acid: 1 hit

[GROUNDING] DES plausibility contradicted for CCOC(=O)OCC: weak HBD/HBA complementarity
```

**`min_tm_k`** — lowest predicted eutectic temperature (K). Lower = better DES.  
**`composite_score`** — used by metal-binding workflows; higher = better binding.  
**`Δ%`** — how far the predicted eutectic is below the pure-component minimum
(a proxy for how strong the eutectic effect is).  
**Family ledger** — hit count per chemical family across the cycle; used for UCB1 exploration.

---

## Convergence Reasons

| Reason text | Meaning |
|-------------|---------|
| `top-N shortlist identical to previous cycle` | Top-N candidates unchanged between two consecutive cycles; DES workflow |
| `top-5 stable` | Metal-binding: top-5 composite scores unchanged |
| *(empty, converged=False)* | Hit max cycles without converging |

Early convergence is usually good (stable shortlist found). If the shortlist
stabilised on poor candidates (e.g. all `min_tm_k > 290 K`), the run
converged to a local optimum — see "Run cycling without improvement" below.

---

## `[GROUNDING]` Warnings

All grounding events carry the `[GROUNDING]` prefix. They appear in
`notable_warnings` in `trajectory.json` (per-cycle) and in the stderr trace.

| Pattern | What happened |
|---------|---------------|
| `[GROUNDING] DES plausibility contradicted for <smiles>: …` | H-bond complementarity with component A is `none`; candidate took −0.25 ranking penalty |
| `[GROUNDING] Family contradicted for <smiles>: …` | LLM family label doesn't match SMARTS check; candidate took −0.25 penalty |
| `[GROUNDING] Coordination contradicted for <smiles>: …` | LLM claimed a coordination mode inconsistent with computed donor atoms/denticity |
| `[GROUNDING] Ligand dropped (reality): <smiles>: …` | Metal-binding: ligand has no donor atoms, invalid SMILES, or structural sanity failure; removed from candidates |

A high rate of `[GROUNDING]` warnings (>3/cycle) means the LLM is generating
chemically implausible proposals; check the context quality or tighten constraints.

---

## Ranking Score Components

`ranking_score` is a composite that starts from the ML prediction and is
modified by several deterministic layers (applied in this order):

| Layer | Effect | Source |
|-------|--------|--------|
| ML prediction | Base score: `min_tm_k` (lower = higher score) or `composite_score` | `orchestrator.py` / metal workflows |
| LLM candidate review | `deprioritize` → −0.25 | `_apply_candidate_reviews` |
| Grounding penalty | Contradicted claim → −0.25 per violation | `_apply_grounding_penalties` |
| H-bond complementarity bias | Strong pair ↑, mismatched pair ↓ (max ±0.10) | `_apply_hbond_bias` |
| Tanimoto diversity penalty | Max similarity ≥ 0.70 to a known-bad candidate → up to −0.10 | `_apply_tanimoto_diversity_penalty` |
| Run-memory bias | Prior `good`/`bad` labels shift score ±0.05 per label | `_apply_run_memory_bias` |

Score is always clamped to `[0.0, ∞)`. A candidate scoring 0.0 is effectively
last; it was not dropped only because dropping requires a harder gate.

---

## `trajectory.json` Schema (key fields)

```json
{
  "workflow": "des_multi_cycle",
  "headline": "choline_cl (3 cycles)",
  "total_cycles": 3,
  "converged": false,
  "convergence_reason": "",
  "snapshots": [
    {
      "cycle": 1,
      "n_screened": 18,
      "n_hits": 4,
      "top_entries": [
        {"label": "urea (CN(C)=O)", "metric_name": "min_tm_k",
         "metric_value": 260.4, "secondary": "Δ15.2%, high confidence"}
      ],
      "new_entrants": [],
      "dropouts": [],
      "family_ledger": {"amide": 2, "alcohol": 1},
      "converged": false,
      "convergence_reason": "",
      "notable_warnings": []
    }
  ],
  "final_summary": [...]
}
```

Parse with standard JSON; all fields are present on every snapshot (defaults:
empty lists/dicts, `converged=false`, `convergence_reason=""`).

---

## Common Failure Patterns

### Run converged immediately (cycle 1 = cycle 2 shortlist)

**Cause:** The initial heuristic candidates already dominate; the LLM adds
nothing new. Common when component A has very few known partners.

**Fix:** Add `--n-cycles 4` or more; reduce `--n-candidates` (forces more
exploration per cycle); or seed with a broader family list via `--constraints`.

---

### Run cycling without improvement (min_tm_k stays above 290 K)

**Cause:** LLM keeps proposing the same structural classes; UCB1 has not yet
discovered a productive family.

**Diagnosis:** Check `family_ledger` across cycles — if one family dominates
hits and its UCB score saturates, exploration stalls.

**Fix options:**
1. `--diversity-mode explore` — instructs the LLM to prefer novel families
2. Pass `--run-memory <prior-run.json>` from a different component A to seed
   cross-molecule family knowledge
3. Manually specify `--constraints '{"avoid_families": ["alcohol"]}'` to force
   exploration away from the dominant family

---

### Many `[GROUNDING] DES plausibility contradicted` warnings

**Cause:** LLM is proposing molecules with the wrong H-bond role for component A
(e.g. proposing HBD partners for a component A that is already a strong HBD).

**Diagnosis:** Check `run.json → component_a → hbond_role`. If `role="HBD"`,
the LLM should be proposing HBA partners.

**Fix:** The system already passes computed H-bond facts to the LLM; if the
problem persists, add `"require_role": "HBA"` to `--constraints`.

---

### Many `[GROUNDING] Ligand dropped (reality)` warnings (metal workflows)

**Cause:** LLM is generating molecules with no donor atoms (no N/O/S available
for coordination) — e.g. purely carbocyclic structures or hydrocarbons.

**Diagnosis:** Look at the dropped SMILES in the warning — if they're aromatic
rings without heteroatoms, the LLM is hallucinating coordination without
chemistry backing.

**Fix:** The known-ligand menu injection should reduce this. If it persists,
check that `metal_ion` is in standard format (e.g. `"Cu2+"` not `"Cu(II)"`) —
a malformed key returns an empty menu.

---

### `total_cycles` lower than `--n-cycles`

**Cause:** Either early convergence (shortlist stable) or a mid-run exception
that halted iteration. Check convergence_reason in `trajectory.json`.

If `converged=false` and `total_cycles < n_cycles`, look for an `ERROR` or
traceback in stderr — a failed ML batch call or file-write error can abort a
cycle.

---

## Run Memory: Using Prior Results

Pass a prior run's output to bias the next run:

```bash
des run-search choline_cl --output-dir run2/ \
    --run-memory run1/run.json \
    --run-memory-dir history/
```

- `--run-memory <file>` — loads `good`/`bad` labels from a single run
- `--run-memory-dir <dir>` — aggregates all `run.json` files in the directory
- Labels apply a ±0.05 ranking bias per occurrence (not a hard filter)
- Cross-run family scores, scaffold counts, and FG SAR are also loaded when
  the `accumulated_*` fields are present in the JSON

To label a result: edit `run.json` and set `"feedback": "good"` or
`"feedback": "bad"` on individual candidate entries, then pass the file as
`--run-memory` in the next run.

---

## Quick Diagnostic Checklist

1. Open `trajectory.json` → check `total_cycles` and `converged`
2. Scan `notable_warnings` across all snapshots for `[GROUNDING]` density
3. Compare `family_ledger` across cycles — is one family monopolising hits?
4. Check `final_summary[0].metric_value` — is the best result actually good?
5. If metal workflow: confirm `metal_ion` format (e.g. `"Cu2+"`, `"Zn2+"`,
   `"Fe3+"`) — the Irving-Williams table uses that exact format; a mismatch
   returns a near-zero stability score and an empty ligand menu
6. If DES workflow: confirm component A resolves (`des list-molecules | grep <name>`)
