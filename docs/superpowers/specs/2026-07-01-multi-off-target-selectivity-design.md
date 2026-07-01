# Multi-Off-Target Metal Selectivity Design

## Goal

Extend the metal-selectivity workflow from "one target metal vs one competitor metal" to "one target metal vs N off-target metals." A ligand only ranks as selective if it beats *every* off-target, not just one — the practical question a wet-lab chemist actually asks ("will this ligand grab my target metal cleanly out of a mixture with several other metals present?").

Both `metal-selectivity` and `selectivity-des` gain this. The single-off-target case (today's exact usage) is byte-identical in every layer — report text, CLI behavior, DFT tiebreak, LLM context — so nothing that already works changes.

## Scope

`des_multi_agent/workflows/metal_binding_selectivity.py`, `des_multi_agent/workflows/selectivity_des_pipeline.py`, `des_multi_agent/llm/{prompts,base}.py`, `des_multi_agent/cli.py`, `des_multi_agent/reporting.py`. `des_multi_agent/chemistry/dft_selectivity.py` needs no signature change (see DFT section below). DES-partner search (Phase 2 of `selectivity-des`) is unaffected — it consumes a flat selective-ligand shortlist regardless of how many off-targets produced it.

---

## Data Model

**`run_metal_selectivity_screen`'s `competitor_metal` parameter** widens from `str` to `str | list[str]`. A bare string still means "one off-target" and is normalized to a single-element list on entry — every existing call site that passes `competitor_metal="Zn2+"` keeps working unchanged. New multi-off-target callers pass `competitor_metal=["Zn2+", "Fe3+", "Ni2+"]`.

**`SelectivityResult`** (frozen dataclass) gains two fields:
- `log_k_competitors: dict[str, float]` — every off-target's predicted log K, full breakdown.
- `worst_competitor_metal: str` — which off-target is the bottleneck (highest predicted log K among off-targets — the one keeping the ligand from being clean).

`log_k_competitor: float` and `delta_log_k: float` keep their existing field names and meaning, but now always hold the **worst-case** off-target's value: `log_k_competitor == log_k_competitors[worst_competitor_metal]`, and `delta_log_k = log_k_target − log_k_competitor`. For N=1 this is identical to today's behavior (the one off-target IS the worst-case by definition).

**`SelectivityScreenOutcome.competitor_metal: str`** renames to **`competitor_metals: list[str]`**. This field is constructed directly in only a handful of places (`reporting.py`, `cli.py`, a few tests) — small, mechanical blast radius.

---

## Ranking

`delta_log_k = log_k_target − max(log_k_off-target for each off-target)`. This feeds `_compute_composite` exactly as today (no change to that function's formula) — it now just receives the worst-case number instead of the sole competitor's number. A candidate only scores well if it beats every off-target simultaneously.

---

## DFT Tiebreak Stage

No signature change to `dft_selectivity_adjustment(dft_result, target_metal, competitor_metal)` — it already compares one target against one competitor's HSAB softness. The workflow now passes each candidate's own `worst_competitor_metal` as the `competitor_metal` argument, so DFT tiebreaks against whichever off-target is actually limiting that specific candidate (different candidates may have different limiting off-targets).

---

## LLM Prompts

`_build_selectivity_context`, `dft_nomination_prompt`, and `ligand_selectivity_brainstorm_prompt` change their competitor-facing text:
- Header line: `"Competitor metal: {x}"` → `"Off-target metals: {x}, {y}, {z}"` (comma-joined, order preserved from the CLI/API input).
- Per-candidate context lines keep showing one log K number for the off-target side (the worst-case value) to stay concise regardless of N — e.g. `log_K(Cu2+)=8.20, log_K(worst off-target)=6.80, ΔlogK=1.40`.

No change to nomination/brainstorm *logic* — only the rendered text.

---

## CLI

`--competitor-metal-ion` accepts a comma-separated string: `--competitor-metal-ion "Zn2+,Fe3+,Ni2+"`. The CLI splits on comma and strips whitespace before calling `run_metal_selectivity_screen`, passing a list. A single value with no comma (today's usage) behaves exactly as before — no argparse type change, no new flag.

---

## Report

**N=1:** byte-identical to today — same header (`"{target} over {competitor}"`), same columns, same formatting. No new column appears.

**N>1:**
- Header: `"=== Metal Selectivity Screen: {target} over {off1}, {off2}, {off3} ==="`.
- The `log_k_competitor` / `delta_log_k` columns keep showing the worst-case numbers (consistent with what drives ranking).
- One additional column, `off_target_breakdown`, shows the full per-metal picture compactly: `Zn2+=10.20, Fe3+=11.80*` — the asterisk marks the limiting (worst-case) off-target for that row. This avoids a variable-width table for arbitrary N while still surfacing which specific metal is the problem.

---

## Error Handling

- If any individual off-target's `predict_log_k` call fails, that candidate is dropped with a warning (matching today's existing failure behavior for the single-competitor case — `_score_proposal_pair` already returns `None` + a warning on any prediction exception; this now applies per-off-target, and a candidate is dropped if *any* off-target prediction fails, since a partial breakdown can't support a trustworthy worst-case).
- `competitor_metal=[]` (empty list, if ever passed) is a caller error — `run_metal_selectivity_screen` raises `ValueError("competitor_metal must contain at least one metal ion")` immediately, matching the existing pattern of raising early on missing required CLI arguments (`cli.py` already does `parser.error(...)` when `--competitor-metal-ion` is omitted entirely).
- Duplicate off-targets in the input (e.g. `"Zn2+,Zn2+"`) are de-duplicated silently (order-preserving) before scoring — no error, since a chemist supplying the same metal twice isn't a meaningful mistake worth failing on.

---

## Testing Plan

`tests/test_metal_binding_selectivity.py` (existing file, additions):
- `competitor_metal` as a bare string still produces `SelectivityResult.log_k_competitors` with exactly one entry and unchanged `log_k_competitor`/`delta_log_k` values (regression).
- `competitor_metal=["Zn2+", "Fe3+"]` produces `log_k_competitors` with both entries; `worst_competitor_metal` correctly identifies the higher-log-K one; `delta_log_k` matches `log_k_target - max(...)`.
- A candidate whose prediction fails for one (but not all) off-targets is dropped entirely, with a warning.
- `competitor_metal=[]` raises `ValueError`.
- Duplicate off-targets in the input are de-duplicated.

`tests/test_dft_integration.py` (additions):
- The DFT stage's `_dft_adj` call receives each candidate's own `worst_competitor_metal`, not a single outcome-wide competitor — verify via a two-candidate fixture where the two candidates have different limiting off-targets.

`tests/test_reporting.py` or equivalent (additions):
- N=1 report output is byte-identical to a captured baseline (regression against the exact existing format).
- N>1 report includes the off-target list in the header and the `off_target_breakdown` column with the asterisk on the correct (worst-case) metal.

`tests/test_cli.py` or equivalent (additions):
- `--competitor-metal-ion "Zn2+,Fe3+"` is parsed into a list and forwarded correctly.
- `--competitor-metal-ion "Zn2+"` (no comma) is forwarded as a single-element list, and the resulting report is unchanged from today's single-competitor format.

**Regression:** full existing suite (927 tests as of commit `6f84e81`) continues to pass; the N=1 code path is exercised by nearly every existing metal-selectivity test today, so this is the primary regression net.

---

## Global Constraints

- N=1 (today's usage) produces byte-identical report output, CLI behavior, DFT tiebreak behavior, and LLM prompt structure — the multi-off-target machinery is additive, not a rewrite of the existing single-competitor path.
- `run_metal_selectivity_screen`'s `competitor_metal` parameter name is NOT renamed — only its accepted type widens (`str | list[str]`) — to minimize churn across existing call sites and tests.
- Ranking is always worst-case (`delta_log_k = log_k_target − max(off-target log Ks)`) — no per-run aggregation mode flag (YAGNI; can be added later if a real need for "average" aggregation emerges).
- `dft_selectivity_adjustment`'s signature is unchanged; the workflow supplies the per-candidate worst-case off-target as its `competitor_metal` argument.
- A candidate is dropped (not partially scored) if any single off-target's prediction fails.
- `selectivity-des` gets the same `competitor_metal: str | list[str]` widening; its Phase 2 (DES partner search) is unmodified.
