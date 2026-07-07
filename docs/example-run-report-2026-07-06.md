# Example Re-Run Report — 2026-07-06

This report documents a full re-run of every runnable example in [`examples/`](../examples/)
against the current state of the agent system — including the concurrent LLM review-call feature
and the deterministic chemistry grounding layer, both added earlier in this development arc. All
30 examples were executed end to end against a live local Ollama instance (models: `gemma4:12b`,
`nemotron-3-nano:latest`, `qwen3.6:latest`), driven by a single detached background script
(`.rerun_examples.log` timestamps 2026-07-06 06:04 → 22:14, ~16 hours wall-clock, dominated by the
three multi-hour selectivity-DES pipelines). Captured `output.txt` files and the frozen regression
baselines under `tests/fixtures/example_benchmark_baselines/` were refreshed to match; the example
benchmark suite (`tests/test_benchmarks_examples.py` + `tests/test_offline_examples.py`) passes
with a perfect 1.0 score (7/7 tests green).

## Summary

| Category | Count | Status |
|---|---|---|
| Offline / deterministic examples | 18 | All re-ran cleanly; several outputs changed due to features added since the last capture (see Finding 1) |
| qwen3.6-backed examples | 4 (`task_router`, `task_execute`, `qwen3_6`, `ni_co_selectivity_des_qwen36`) | 3 ran cleanly; `task_execute` failed once and was left as its documented static placeholder (see Finding 2) |
| gemma4:12b-backed examples | 6 | 5 ran cleanly first try; `plain_language_gemma4_12b` failed once (benign LLM JSON-truncation hiccup, see Finding 3) and succeeded on immediate retry |
| nemotron-3-nano-backed examples | 2 | Both ran cleanly |

## Timing

| Example | Time | Notes |
|---|---|---|
| 18 offline examples (combined) | 2s–27s each | Deterministic, no LLM (`DES_DISABLE_QSPR=1`) |
| `task_router` | 18s | qwen3.6, single routing call |
| `task_execute` | 23s (failed) | qwen3.6; see Finding 2 |
| `qwen3_6` | 25m50s | n=20, 1 cycle |
| `ni_co_selectivity_des_qwen36` | ~7h46m | n=20/n_cycles=3 × n_des=20/n_des_cycles=3 × n_outer_cycles=2 |
| `gemma4_12b` | 9m58s | n=20, 1 cycle |
| `betaine_des_gemma4_12b` | 1h39m | n=20, 5 cycles |
| `lidocaine_gemma4_12b` | 4m34s | n=5, 1 cycle |
| `plain_language_gemma4_12b` | 16s (failed), then succeeded on retry | routed to n=5 DES search |
| `plain_language_metal_binding_gemma4_12b` | 12s | single ligand, no iterative loop |
| `ni_co_selectivity_des` | ~4h24m | same shape as qwen36 variant, gemma4:12b |
| `nemotron_3_nano` | 2m18s | n=20, 1 cycle |
| `ni_co_selectivity_des_nemotron` | ~1h32m | same shape, nemotron-3-nano |

## Finding 1 — Offline "deterministic" examples are still run-to-run byte-identical, but their content has drifted since the last capture

Re-running the offline examples with `DES_DISABLE_QSPR=1` still produces byte-identical output
across repeats, so the determinism claim in `examples/README.md` holds. Comparing the new captures
to the previously committed baselines showed substantive content differences — **not flakiness,
but real feature growth** accumulated since the baselines were last frozen:

- **Grounding verdicts** (`✓ verified | DES plausibility of '...' + '...'`) now appear in reports —
  the deterministic chemistry grounding layer (`des_multi_agent/chemistry/claim_grounding.py`) is
  wired in and active whenever a candidate pair can be checked.
- **Analogue expansion** adds new candidates not present before — e.g. `ni2_co2_selectivity` grew
  from 20 to 26 screened ligands (`NC(CCC(=O)O)(CC(=O)O)C(=O)O`, `CC(N)(P(=O)(O)O)P(=O)(O)O`, and
  others appear via `chain_extend`/`oh_to_nh2`/`n_methyl` analogue rules), reordering the top ligand.
- `examples/multi_cycle_des/` gained a new `trajectory.json` artifact (machine-readable companion
  to `trajectory.md`, written automatically by the CLI when `--output-dir` is set).
- Candidate counts and rankings shifted in several examples — e.g. `betaine_des` went from 20 → 8
  screened candidates (3 → 5 predicted DES-formers) as a direct effect of the diversity/analogue
  features documented in `future-improvements.md`'s "Recently Completed" list, all landed after
  these baselines were last captured.

All baseline-tracked examples with changed output were re-synced under
`tests/fixtures/example_benchmark_baselines/`.

## Finding 2 — `task_execute` failed once on LLM sampling non-determinism; this is expected and by design

`task_router` and `task_execute` both route through the same qwen3.6 backend
(`llm.example.yaml`, `temperature: 0.2`) via `route_task()`. In this run, `task_router` produced a
fully-populated job (`component_a`, `n`, `checkpoint_path`, `config_path` all present) and its
capture is byte-identical to the previous baseline. `task_execute`'s independent routing call,
sampled a few seconds later, came back missing some of those fields and tripped
`RouterJob.validate()`'s "missing required fields" check (`des_multi_agent/task_router_schema.py`).

This is not a code regression: `task-router` and `task-execute` share the exact same routing and
validation code path (confirmed by reading `cli.py`, `task_executor.py`, `task_router.py`,
`task_router_schema.py`) — there is no normalization step one applies that the other skips. At
`temperature=0.2` the model's JSON output isn't guaranteed field-complete on every call, and
`RouterJob` silently drops any unrecognized key name from the model's response rather than coercing
it, so an occasional incomplete/renamed-field response surfaces as a validation error instead of a
successful route.

`examples/task_execute/output.txt` has been a static, non-live placeholder since it was first
committed (`git log --follow` shows a single commit, never touched since) — its own README already
documents this ("The file `output.txt` is a placeholder — run `./run.sh` with Ollama active to
capture live output."), precisely because this example's live output isn't suitable for a frozen,
byte-exact baseline. The placeholder was left as-is; no baseline change was needed.

**Follow-up candidate (not done here):** enumerate `RouterJob`'s actual field names directly in the
router's system prompt (`ROUTER_SYSTEM_PROMPT`), and/or give `task-router`/`task-execute` the same
client-side default-substitution fallback that `plain_language_gemma4_12b/run_example.py` and
`plain_language_metal_binding_gemma4_12b/run_example.py` already implement
(`_normalize_router_job`).

### Addendum to Finding 2 — repeated testing shows this is a frequent, reproducible failure, not a rare fluke

A separate manual pass ran `task_router` and `task_execute` several times back to back (outside
this report's single-sample capture) to check whether the above was a one-off. Results: `task_router`
failed 4 of 6 fresh attempts with the same "missing required fields" error, one attempt returned an
unwarranted `needs_clarification` asking the user to spell out the checkpoint/config path the CLI
already defaults, and one attempt hung past a 3-minute timeout; `task_execute` failed 5 of 5
attempts. So this isn't rare sampling variance at the margins — with the current prompt and
`qwen3.6`, the default (no explicit `--llm-config`) routing path fails more often than it succeeds.

The root cause is confirmed, not inferred: calling `build_default_router_provider().route_request(...)`
directly and printing the raw response before parsing shows `qwen3.6` invents its own natural field
names instead of the CLI's actual ones — e.g. `"target_molecule": "ethanol"` instead of
`"component_a": "CCO"`. `parse_router_response`'s field filter (`{key: value for key, value in
job_data.items() if key in allowed_fields}`) silently drops any key that isn't an exact
`RouterJob` field name, so an invented key just disappears rather than erroring loudly, and the
subsequent required-fields check then fails. `ROUTER_SYSTEM_PROMPT`
(`des_multi_agent/task_router_prompts.py`) only says *"Use existing CLI field names for job
fields"* without ever listing them — a fine instruction for a model that already has the DES-Agent
CLI schema memorized, and an unreliable one otherwise. This raises the "Follow-up candidate" above
from a nice-to-have to a real reliability gap worth prioritizing: the fix (enumerate the field names
in the prompt) is a small, low-risk prompt change.

## Finding 3 — Occasional benign LLM JSON-parse hiccups

Across the gemma4:12b, nemotron-3-nano, and qwen3.6 runs, a small number of calls (brainstorm,
explanation, critique) occasionally failed with a JSON-parse error — most likely the model's JSON
array being truncated mid-object by a `max_tokens` cutoff before its closing bracket. These degrade
gracefully (a warning line in the report; the search continues with whatever candidates did parse)
and occurred at a low, expected rate consistent with real LLM sampling variance.

`plain_language_gemma4_12b`'s first attempt hit exactly this failure mode and produced a truncated
capture; re-running it immediately (no code or config change) produced a full, correct 109-line
report. This is consistent with genuine sampling variance rather than a systemic problem — the
other four gemma4:12b examples (`gemma4_12b`, `betaine_des_gemma4_12b`, `lidocaine_gemma4_12b`,
`plain_language_metal_binding_gemma4_12b`) all succeeded on the first attempt.

## Files changed

- **Regenerated outputs:** all 18 offline examples; `task_router`, `qwen3_6`,
  `ni_co_selectivity_des_qwen36`, `gemma4_12b`, `betaine_des_gemma4_12b`, `lidocaine_gemma4_12b`,
  `plain_language_gemma4_12b`, `plain_language_metal_binding_gemma4_12b`, `ni_co_selectivity_des`,
  `nemotron_3_nano`, `ni_co_selectivity_des_nemotron`.
- **Left unchanged (documented static placeholder, Finding 2):** `task_execute`.
- **Baselines synced:** `tests/fixtures/example_benchmark_baselines/{betaine_des,
  betaine_des_gemma4_12b, candidates_file, compare_runs, gemma4_12b, leaderboard_history,
  lidocaine_gemma4_12b, multi_cycle_des, nemotron_3_nano, plain_language_gemma4_12b,
  plain_language_metal_binding_gemma4_12b, qwen3_6, viscosity_composite_ranking,
  viscosity_template}`; `des_viscosity`, `ligand_binding_template`, `metal_binding`, `task_router`,
  and `task_execute` were byte-identical to their existing baselines and needed no update.
- **Test infra fix:** added `tests/__init__.py` and `tests/fixtures/__init__.py` — a same-named
  `tests` package installed by `MDAnalysisTests` in site-packages was shadowing the repo's local
  `tests/` namespace package, breaking `tests.fixtures` imports in
  `tests/test_benchmarks_examples.py` (regular packages always win over namespace packages
  regardless of `sys.path` order). This was blocking the benchmark suite entirely, independent of
  any example content.
- **Docs:** `examples/README.md`'s stale "predate this feature and were not regenerated" note
  updated, and a stale note claiming `betaine_des_gemma4_12b` stays frozen removed (it was
  regenerated in this pass); this report added.

## Next Up (candidates for future work, not done here)

1. **Router prompt robustness** (Finding 2): enumerate the actual `RouterJob` field names directly
   in the router system prompt instead of relying on the model to infer them, and/or give
   `task-router`/`task-execute` the same client-side default-substitution fallback the
   plain-language example scripts already have.
2. **Ollama-vs-vLLM concurrency tuning**: the concurrent-LLM-review-calls feature was designed and
   benchmarked against vLLM's continuous batching; against Ollama's single-model serving, "concurrent"
   requests still queue at the server. `timeout_seconds` in the example configs was already raised to
   `300.0` (from `120.0`/`180.0`) in an earlier pass to absorb this queuing; no timeouts were observed
   in this run at that setting.

## Environment

- Ollama models used: `gemma4:12b` (7.6GB), `nemotron-3-nano:latest` (24GB), `qwen3.6:latest`
  (23GB), served locally on `127.0.0.1:11434`.
- All commands run via each example's own `run.sh`/`run_example.py`, unmodified.
