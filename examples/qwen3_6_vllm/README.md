# Qwen 3.6 Example (vLLM)

vLLM twin of [`../qwen3_6/`](../qwen3_6) — the identical DES demo run, swapping the Ollama backend
for a local vLLM OpenAI-compatible server serving the FP8 Hugging Face checkpoint
`Qwen/Qwen3.6-35B-A3B-FP8` (the closest open-weights match to Ollama's `qwen3.6:latest` Q4_K_M
pull, a 35B mixture-of-experts model).

## Input

- Component A: `CCO`
- Candidate search count: `20`
- Checkpoint: `ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt`
- LLM config: [`llm.qwen3_6_vllm.yaml`](./llm.qwen3_6_vllm.yaml) (`provider: vllm`, `model_name: Qwen/Qwen3.6-35B-A3B-FP8`)
- Captured input: [`input.txt`](./input.txt)

## Process

1. Start the vLLM server. This checkpoint's FP8 MoE kernels are not yet mature on this hardware
   (NVIDIA GB10, Blackwell, compute capability sm_121) under vLLM 0.24.0 — the default
   DeepGEMM/CUTLASS FP8 kernels raise `Assertion error ... Unknown SF transformation` at load time.
   Falling back to Triton kernels for both MoE and quantized-linear layers works around it, along
   with skipping the DeepGEMM kernel-warmup step (which otherwise ignores the chosen backend):
   ```bash
   VLLM_DEEP_GEMM_WARMUP=skip vllm serve Qwen/Qwen3.6-35B-A3B-FP8 --port 8000 \
     --moe-backend triton --linear-backend triton --gpu-memory-utilization 0.6 --language-model-only
   ```
2. Confirm with `curl -s http://localhost:8000/v1/models`.
3. Run `./run.sh` — identical invocation to `../qwen3_6/run.sh`, pointed at the vLLM config.

## Output

[`output.txt`](./output.txt) has the same structure as `../qwen3_6/output.txt`: ranked DES results,
uncertainty annotations, LLM brainstorm candidates, explanation and critique notes. See
[`docs/vllm-example-run-report-2026-07-07.md`](/home/qshao/DES-Agent/docs/vllm-example-run-report-2026-07-07.md)
for the vLLM vs. Ollama timing comparison — the earlier item-21 benchmark in
[`docs/future-improvements.md`](/home/qshao/DES-Agent/docs/future-improvements.md) found Ollama
~1.8x faster than vLLM for this exact model pair on this hardware, due to the forced Triton kernel
fallback above.
