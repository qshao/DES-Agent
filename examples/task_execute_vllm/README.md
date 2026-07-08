# Task-Execute Example (vLLM)

vLLM twin of [`../task_execute/`](../task_execute) — routes the same plain-language request and
runs the matching workflow in one step, using a local vLLM server serving
`Qwen/Qwen3.6-35B-A3B-FP8` instead of Ollama.

## Input

- Plain-language request: see [`input.txt`](./input.txt)

## Process

`task-execute` routes through the exact same hardcoded `llm.example.yaml` path as `task-router`
(see [`../task_router_vllm/README.md`](../task_router_vllm/README.md) for why). `run.sh` uses the
same backup/overwrite/restore-via-trap mechanism around the shared config file, pointed at a vLLM
server:

```bash
VLLM_DEEP_GEMM_WARMUP=skip vllm serve Qwen/Qwen3.6-35B-A3B-FP8 --port 8000 \
  --moe-backend triton --linear-backend triton --gpu-memory-utilization 0.6 --language-model-only
./run.sh
```

## Output

**Known limitation, captured as-is:** `task-execute` routes through the exact same code path as
`task-router` (see [`../task_router_vllm/README.md`](../task_router_vllm/README.md) for the full
explanation) — `Qwen/Qwen3.6-35B-A3B-FP8` is a "thinking" model whose `<think>...</think>`
reasoning embeds draft JSON snippets that the shared `_extract_json_block` greedy-regex extractor
grabs alongside the real final answer, producing invalid or wrong-shaped JSON. Every attempt against
this vLLM checkpoint fails this way; the Ollama-served `qwen3.6` tag used by `../task_execute/` does
not exhibit it for this prompt.

Unlike `../task_execute/run.sh` (which discards stderr), this wrapper merges stderr into
`output.txt` so the failure is captured rather than silently discarded. This is out of scope to fix
here — `_extract_json_block` is shared core parsing code. See
[`docs/vllm-example-run-report-2026-07-07.md`](/home/qshao/DES-Agent/docs/vllm-example-run-report-2026-07-07.md)
for the full write-up.

## How to Adapt

Same as [`../task_execute/README.md`](../task_execute/README.md#how-to-adapt) — change the request
string in `run.sh`, or point `LLM_CONFIG_FILE`-equivalent block inside `run.sh` at a different
vLLM-served model.
