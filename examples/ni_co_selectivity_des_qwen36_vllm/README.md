# Ni²⁺/Co²⁺ Selectivity-DES Example — Qwen 3.6 (vLLM)

vLLM twin of [`../ni_co_selectivity_des_qwen36/`](../ni_co_selectivity_des_qwen36) — the identical
selectivity-DES pipeline (5 shortlisted ligands in Phase 1, DES search for the top 3 in Phase 2),
backed by a local vLLM server serving `Qwen/Qwen3.6-35B-A3B-FP8` instead of Ollama.

## Input

Identical parameters to `../ni_co_selectivity_des_qwen36/`: see [`input.txt`](./input.txt). The
only change is `llm_config=llm.ni_co_qwen36_vllm.yaml` (repo root) instead of
`llm.ni_co_qwen36.yaml`.

## Process

1. Start the vLLM server with the GB10 FP8-MoE workaround (see
   [`../qwen3_6_vllm/README.md`](../qwen3_6_vllm/README.md) for why these flags are needed on this
   hardware):
   ```bash
   VLLM_DEEP_GEMM_WARMUP=skip vllm serve Qwen/Qwen3.6-35B-A3B-FP8 --port 8000 \
     --moe-backend triton --linear-backend triton --gpu-memory-utilization 0.6 --language-model-only
   ```
2. Confirm with `curl -s http://localhost:8000/v1/models`.
3. Run:
   ```bash
   ./run.sh
   # or with a custom output directory:
   ./run.sh runs/my_ni_co_qwen36_vllm_run
   ```
   Same CLI invocation as `../ni_co_selectivity_des_qwen36/run.sh`, pointed at
   `llm.ni_co_qwen36_vllm.yaml`. Default output directory is
   `runs/ni_co_selectivity_des_qwen36_vllm_001/`.

## Output

[`output.txt`](./output.txt) has the same two-section structure as the Ollama capture (Phase 1
selectivity table for 5 ligands, Phase 2 DES partner blocks for the top 3). See
[`docs/vllm-example-run-report-2026-07-07.md`](/home/qshao/DES-Agent/docs/vllm-example-run-report-2026-07-07.md)
for the timing comparison against `../ni_co_selectivity_des_qwen36/`.

## LLM config

[`llm.ni_co_qwen36_vllm.yaml`](../../llm.ni_co_qwen36_vllm.yaml) points to
`Qwen/Qwen3.6-35B-A3B-FP8` on a local vLLM server at `localhost:8000/v1`.
