# vLLM Example Run Report — 2026-07-07

This report documents a new set of examples: vLLM-backed twins of every Ollama-backed example in
[`examples/`](../examples/) for which a local, vLLM-compatible Hugging Face checkpoint is
available. Each twin lives in its own new `examples/<name>_vllm/` folder alongside the original —
nothing in the existing Ollama-backed folders was modified. Every twin folder documents its Input,
Process, and Output in its own `README.md`; this report covers the run as a whole: exact server
commands, GB10-specific workarounds, timings against the Ollama baseline, and two real
cross-backend findings surfaced along the way.

## Scope

| Group | vLLM model | New folders |
|---|---|---|
| gemma4:12b twins | `google/gemma-4-12B-it` (bf16) | `gemma4_12b_vllm`, `betaine_des_gemma4_12b_vllm`, `lidocaine_gemma4_12b_vllm`, `plain_language_gemma4_12b_vllm`, `plain_language_metal_binding_gemma4_12b_vllm`, `ni_co_selectivity_des_vllm` |
| qwen3.6 twins | `Qwen/Qwen3.6-35B-A3B-FP8` | `qwen3_6_vllm`, `task_router_vllm`, `task_execute_vllm`, `ni_co_selectivity_des_qwen36_vllm` |

**Excluded:** `nemotron_3_nano` / `ni_co_selectivity_des_nemotron` have no vLLM twin — no local
Hugging Face checkpoint for `nemotron-3-nano` is available offline in this environment, and
downloading one speculatively was out of scope for this pass.

Both checkpoints were already present locally from an earlier vLLM-vs-Ollama benchmark
(`docs/future-improvements.md` item 21) — `google/gemma-4-12B-it` (23G) and
`Qwen/Qwen3.6-35B-A3B-FP8` (35G), the closest open-weights matches to Ollama's `gemma4:12b` and
`qwen3.6:latest` Q4_K_M pulls.

## Server commands

Both models needed a fix beyond what item 21's benchmark documented, because both are natively
multimodal models and vLLM's dummy-input profiling step (used to estimate memory usage before
serving) exercises the image/video encoder path by default, which fails for both checkpoints under
vLLM 0.24.0 on this hardware (`ValueError: Failed to apply Gemma4UnifiedProcessor...` /
`Qwen3VLProcessor...` on synthetic multimodal dummy tokens). `--language-model-only` disables all
multimodal inputs and profiling, which is correct here since this workload is text-only.

Available free memory was also lower than expected — this GB10 devkit's 128GB is unified CPU/GPU
memory, and ~20GB was already in use by other processes on the box, so the default
`--gpu-memory-utilization 0.9` request exceeded what was actually free.

```bash
# Gemma group
vllm serve google/gemma-4-12B-it --port 8000 \
  --language-model-only --gpu-memory-utilization 0.6

# Qwen group — plus the FP8-MoE-on-Blackwell workaround from item 21
VLLM_DEEP_GEMM_WARMUP=skip vllm serve Qwen/Qwen3.6-35B-A3B-FP8 --port 8000 \
  --moe-backend triton --linear-backend triton \
  --gpu-memory-utilization 0.6 --language-model-only
```

Gemma took ~2 minutes to load weights + compile; Qwen (35B, FP8) took ~5 minutes. Both were
confirmed healthy via `curl -s http://localhost:8000/v1/models` and a real chat completion before
any example was run.

## Timing vs. the Ollama baseline

All Ollama timings are from [`docs/example-run-report-2026-07-06.md`](example-run-report-2026-07-06.md)'s
2026-07-06 re-run, same parameters, same hardware.

| Example | vLLM | Ollama | Delta |
|---|---|---|---|
| `gemma4_12b_vllm` | 647s | 598s | +8% slower |
| `betaine_des_gemma4_12b_vllm` | 4162s (~69m) | 5945s (~99m) | 30% faster |
| `lidocaine_gemma4_12b_vllm` | 270s | 274s | ~even |
| `plain_language_gemma4_12b_vllm` | failed (see Finding 1) | 16s failed, then succeeded on retry | n/a |
| `plain_language_metal_binding_gemma4_12b_vllm` | 17s | 12s | +42% slower (both trivially fast) |
| `ni_co_selectivity_des_vllm` | 17843s (~4h57m) | 15864s (~4h24m) | 12% slower |
| `qwen3_6_vllm` | 329s | 1550s | **4.7x faster** |
| `task_router_vllm` | failed (see Finding 2) | 18s succeeded | n/a |
| `task_execute_vllm` | failed (see Finding 2) | 23s failed (placeholder, non-comparable) | n/a |
| `ni_co_selectivity_des_qwen36_vllm` | 19926s (~5h32m) | 27972s (~7h46m) | **29% faster** |

**Takeaway:** results are genuinely mixed, consistent with item 21's original finding — there is no
clean "vLLM is faster" or "Ollama is faster" story here. The Gemma group is roughly at parity to
modestly slower under vLLM; the Qwen group is substantially faster under vLLM (both the simple DES
demo and the heaviest multi-hour pipeline). This is a different result from item 21's original
Ollama-favors-Qwen finding — the gap there was attributed to forced Triton-kernel fallbacks being
slower than DeepGEMM/CUTLASS on more mainstream hardware; this run shows vLLM comfortably ahead for
Qwen despite the same fallback still being in effect, and no `max_workers=8` concurrency advantage
this workload can exploit in either single-request routing calls or the sequential per-ligand
selectivity loop. The most likely explanation is `--language-model-only`, absent from item 21's
original run, meaningfully speeding up scheduling/profiling for both models — but this wasn't
isolated as a controlled variable, so treat it as an observation, not a proven cause.

## Finding 1 — vLLM-served Gemma checkpoint consistently mis-fills the `config` job field

`plain_language_gemma4_12b_vllm` failed all 8 attempts (the driver's automated attempt plus 7
manual retries) with the identical failure shape:
`FileNotFoundError: Path does not exist: /home/qshao/DES-Agent/shipped...`. The router's JSON
response reliably paraphrases the request's "shipped ml_des_mp checkpoint and config" phrase into a
`config` field value like `"shipped"`, `"shipped_default"`, or `"shipped_config"` — never the
literal `"default"` that `run_example.py`'s `_normalize_router_job` checks for before substituting
the real config path. The same request against the Ollama-quantized Gemma checkpoint
(`plain_language_gemma4_12b`) does not exhibit this; its router reliably omits or nulls the
`config` field, which the normalizer does catch.

8/8 identical-shape failures across sampling variation in the exact wording (`"shipped"` x2,
`"shipped_default"` x5, `"shipped_config"` x1) indicates this is a stable property of this
checkpoint's phrasing tendency for this exact prompt under vLLM (bf16), not one-off noise. The
example's `output.txt` captures one such failing run in full (request, raw router JSON, normalized
job, traceback) rather than a synthetic success. No production code was changed —
`_normalize_router_job`'s fallback set is example-local convenience logic, not something this
documentation pass is in scope to fix.

## Finding 2 — `Qwen/Qwen3.6-35B-A3B-FP8` is a "thinking" model that breaks the router's JSON extractor

`task_router_vllm` and `task_execute_vllm` fail every attempt against this vLLM checkpoint. Root
cause, confirmed by calling the provider directly and inspecting the raw response: this checkpoint
reasons inside `<think>...</think>` tags before producing its final answer, and — critically — that
reasoning text itself contains draft/example JSON snippets while the model works out field names
("Wait, should I use exact CLI field names?... Actually, to be strictly compliant, I'll output:
```json ... ```"). `des_multi_agent/llm/parser.py`'s `_extract_json_block` extracts JSON with a
greedy regex (`\{[\s\S]*\}`, matching from the first `{` to the *last* `}` in the whole response) —
a reasonable heuristic against a model that answers directly, but against a thinking model's output
it spans across multiple JSON-like blocks in one match, producing invalid or wrong-shaped JSON.

The exact symptom varies by which draft block the regex's span happens to land on for a given
sample — three different argparse-level errors were observed across repeated attempts:
`Failed to parse router response: Expecting value: line 1 column 2 (char 1)`,
`router response job is missing required fields for des: component_a, n, checkpoint_path,
config_path`, and `router response must be a JSON object`. All three are the same root cause.

Raising `max_tokens` from 1024 to 4096 (tested manually, not applied to the example configs) lets
the model reach its actual `</think>` and final JSON — but does not fix the extraction: the greedy
regex still spans from the first draft block to the final one, yielding `Extra data: line 13 column
4` instead. There is no example-local fix available; `_extract_json_block` is shared core parsing
code used across the whole LLM layer, not something in scope for this documentation pass to modify.

The Ollama-served `qwen3.6` tag used by `../task_router/`/`../task_execute/` does not exhibit this
for this prompt — it answers directly without visible `<think>` reasoning, so extraction only ever
sees one JSON block.

**Why the DES workflow examples (`qwen3_6_vllm`, `ni_co_selectivity_des_qwen36_vllm`) succeeded
despite using the same checkpoint:** `des_multi_agent/llm/base.py`'s brainstorm/review/explanation
calls go through the same `_extract_json_block`/JSON-parsing path and are just as exposed to this
failure mode — both successful captures contain ~20 `[WARNING]`/`not valid JSON` lines each,
confirming the same extraction problem does occur there too. The difference is architectural: those
call sites already have a graceful-degradation contract (a parse failure becomes a warning line and
the search continues with whatever candidates did parse — see
[`docs/example-run-report-2026-07-06.md`](example-run-report-2026-07-06.md) Finding 4 for the
established pattern), whereas `task-router`'s single routing call has no equivalent fallback — a
parse failure is immediately fatal via `parser.error()`. Both `task_router_vllm`'s `output.txt` and
`task_execute_vllm`'s `output.txt` capture one such failing run in full (stderr merged in, unlike
the Ollama originals, so the argparse error is visible in the artifact).

## Finding 3 — the `llm.example.yaml` swap/restore mechanism held up under real failures

`task_router_vllm`/`task_execute_vllm` needed a workaround since `task-router`/`task-execute`
hardcode the shared `llm.example.yaml` with no override flag (see each folder's `README.md`): their
`run.sh` backs up that file, overwrites it with a vLLM config for the call, and restores the
original via `trap ... EXIT`. Both examples failed on every attempt (Finding 2) — a real test of
whether the trap fires correctly on a non-zero exit under `set -euo pipefail`, not just the happy
path. It did, every time, confirmed via `diff <(git show HEAD:llm.example.yaml) llm.example.yaml`
after each run.

## Behavioral notes (not failures)

- `ni_co_selectivity_des_vllm` (gemma) reports `Converged: no` after 2 outer cycles, `3/3`
  DES-compatible ligands — identical convergence/compatibility counts to the Ollama capture
  (`ni_co_selectivity_des/output.txt`), differing only in which specific ligands were shortlisted.
- `ni_co_selectivity_des_qwen36_vllm` reports `Converged: yes`, `4/5` DES-compatible ligands, vs.
  the Ollama capture's `Converged: yes`, `5/5` — a minor difference attributable to ordinary LLM
  sampling variance in the brainstorm/review stages, not a functional regression.
- `betaine_des_gemma4_12b_vllm` screened 21 candidates (6 DES-formers) vs. the Ollama capture's
  count — again ordinary brainstorm variance, not a functional difference.

## Files changed

- **New:** 10 `examples/*_vllm/` folders (see Scope table), each with `llm.*_vllm.yaml`,
  `run.sh`/`run_example.py`, `input.txt`, `output.txt`, `README.md`.
- **New (repo root):** `llm.ni_co_selectivity_vllm.yaml`, `llm.ni_co_qwen36_vllm.yaml` — mirrors the
  existing `llm.ni_co_selectivity.yaml`/`llm.ni_co_qwen36.yaml` convention for the selectivity-DES
  pipeline's LLM config.
- **Unchanged:** every existing Ollama-backed `examples/` folder, `llm.example.yaml` (restored to
  its original committed content after every `task_router_vllm`/`task_execute_vllm` run),
  `des_multi_agent/` production code (no fixes applied for Findings 1–2, both documented as
  out-of-scope follow-ups).
- **Docs:** `examples/README.md` gained a "vLLM backend twins" section; this report added.

## Follow-ups (not done here)

1. **Extraction robustness for thinking models** (Finding 2): `_extract_json_block` needs to
   isolate content after a `</think>` tag (when present) before applying its brace-matching regex,
   or switch to non-greedy/balanced-brace matching that stops at the first complete top-level JSON
   object rather than spanning to the last `}` in the response.
2. **Router job field normalization** (Finding 1, echoing the same follow-up in
   `docs/example-run-report-2026-07-06.md`'s Finding 2): the plain-language example scripts'
   `_normalize_router_job` fallback-value set is too narrow (`{None, "", "default"}`) to catch the
   range of plausible paraphrases a model can produce for "use the default/shipped X."
3. **Nemotron vLLM twin**: would need sourcing a Hugging Face checkpoint for `nemotron-3-nano`
   compatible with vLLM, not attempted here.

## Environment

- Hardware: NVIDIA GB10 devkit (Grace-Blackwell, aarch64, sm_121, ~128GB unified memory, ~97GB free
  at run time).
- vLLM 0.24.0. Models: `google/gemma-4-12B-it` (bf16, 23G), `Qwen/Qwen3.6-35B-A3B-FP8` (35G).
- Driver: a detached background script (`setsid nohup ... & disown`, survives session close) started
  the Gemma server, ran the Gemma-group examples, stopped it, started the Qwen server, ran the
  Qwen-group examples, stopped it. Total wall-clock ~12.5 hours (2026-07-07 06:24–18:36),
  dominated by the two multi-hour selectivity-DES pipelines.
