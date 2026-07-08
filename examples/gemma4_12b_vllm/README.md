# Gemma 4-12B Example (vLLM)

vLLM twin of [`../gemma4_12b/`](../gemma4_12b) — the identical DES demo run, swapping the Ollama
backend for a local vLLM OpenAI-compatible server serving the bf16 Hugging Face checkpoint
`google/gemma-4-12B-it` (the closest open-weights match to Ollama's `gemma4:12b` Q4_K_M pull).

## Input

- Component A: `CCO`
- Candidate search count: `20`
- Checkpoint: `ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt`
- LLM config: [`llm.gemma4_12b_vllm.yaml`](./llm.gemma4_12b_vllm.yaml) (`provider: vllm`, `model_name: google/gemma-4-12B-it`, `api_base_url: http://localhost:8000/v1`)
- Captured input: [`input.txt`](./input.txt)

## Process

1. Start the vLLM server (text-only — `--language-model-only` disables the model's multimodal
   image/video encoder path, which is unused here and otherwise fails vLLM's dummy-input profiling
   step on this checkpoint):
   ```bash
   vllm serve google/gemma-4-12B-it --port 8000 --language-model-only
   ```
2. Wait for `curl -s http://localhost:8000/v1/models` to return the model, then run:
   ```bash
   ./run.sh
   ```
   The wrapper saves stdout to `output.txt` and suppresses stderr so the captured artifact starts
   with the report table — identical invocation to `../gemma4_12b/run.sh` except for the LLM config
   file.

## Output

[`output.txt`](./output.txt) contains the same report shape as `../gemma4_12b/output.txt`: ranked
DES results, uncertainty annotations, two-stage LLM brainstorm, explanation/critique notes, and
grounding/contradiction analysis where applicable. Compare the two directly to see how backend
choice (vLLM bf16 vs. Ollama Q4_K_M) affects candidate brainstorming and timing for the same prompt
and parameters.

See [`docs/vllm-example-run-report-2026-07-07.md`](/home/qshao/DES-Agent/docs/vllm-example-run-report-2026-07-07.md)
for timings and cross-backend findings.
