# Task Router Example (vLLM)

vLLM twin of [`../task_router/`](../task_router) — translates the same plain-language request into
a JSON job, using a local vLLM server serving `Qwen/Qwen3.6-35B-A3B-FP8` instead of Ollama.

## Input

- Plain-language request: see [`input.txt`](./input.txt)

## Process

`task-router` (`des_multi_agent/task_router.py`) hardcodes its LLM config path to the shared
`llm.example.yaml` at the repo root (`DEFAULT_ROUTER_LLM_CONFIG`) — there is no `--llm-config`
override flag for the routing step itself (only the downstream workflow in `task-execute` accepts
one). To point the router at vLLM without permanently changing the shared config used by the
Ollama-backed examples, [`run.sh`](./run.sh):

1. Starts (outside this script) a vLLM server:
   ```bash
   VLLM_DEEP_GEMM_WARMUP=skip vllm serve Qwen/Qwen3.6-35B-A3B-FP8 --port 8000 \
     --moe-backend triton --linear-backend triton --gpu-memory-utilization 0.6 --language-model-only
   ```
2. Backs up the repo root's `llm.example.yaml`.
3. Temporarily overwrites it with a `provider: vllm` / `model_name: Qwen/Qwen3.6-35B-A3B-FP8`
   config for the duration of the call.
4. Runs `task-router` against the request in `input.txt`.
5. Restores the original `llm.example.yaml` from the backup via a `trap ... EXIT`, which fires on
   success, failure, or interruption alike — the shared file is never left mutated.

```bash
./run.sh
```

## Output

**Known limitation, captured as-is:** every attempt against this vLLM-served checkpoint fails.
`Qwen/Qwen3.6-35B-A3B-FP8` is a "thinking" model — it reasons inside `<think>...</think>` tags
before its final answer, and that reasoning text itself contains draft/example JSON snippets while
the model works out field names. `des_multi_agent/llm/parser.py`'s `_extract_json_block` extracts
JSON with a greedy regex (`\{[\s\S]*\}`, first `{` to last `}` in the whole response) that was
written for models answering directly — against a thinking model's output it spans across multiple
JSON-like blocks at once, yielding invalid or wrong-shaped JSON. Symptom varies by exact sampling:
observed failures include `Failed to parse router response: Expecting value: ...`, `router response
job is missing required fields for des: ...`, and `router response must be a JSON object` — three
different surface errors, same root cause. The Ollama-served `qwen3.6` tag used by
`../task_router/` does not exhibit this (its output doesn't show `<think>` reasoning for this
prompt), so this is a cross-backend difference specific to this HF checkpoint's default behavior
under vLLM.

[`output.txt`](./output.txt) captures one such failing run in full (unlike `../task_router/`, this
wrapper merges stderr into `output.txt` so the argparse error is visible rather than discarded).
This is out of scope to fix here — `_extract_json_block` is shared core parsing code, not
example-local logic. See
[`docs/vllm-example-run-report-2026-07-07.md`](/home/qshao/DES-Agent/docs/vllm-example-run-report-2026-07-07.md)
for the full write-up, including a test that confirms raising `max_tokens` so the model reaches its
final JSON doesn't help — the extraction still grabs from the first draft block.
