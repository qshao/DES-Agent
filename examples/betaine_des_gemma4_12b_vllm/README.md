# Betaine DES Screening with Gemma 4-12B (vLLM)

vLLM twin of [`../betaine_des_gemma4_12b/`](../betaine_des_gemma4_12b) — identical betaine
DES-screening pipeline (multi-cycle iterative search, viscosity-aware ranking, chemistry-advisor
reasoning), swapping the Ollama backend for a local vLLM server serving
`google/gemma-4-12B-it` (bf16).

## Input

- Component A: `C[N+](C)(C)CC(=O)[O-]` (betaine, zwitterionic form)
- Candidate search count per cycle: `20`, max cycles: `5`
- Checkpoint: `ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt`
- Tm ceiling: `340 K`, minimum relative Tm drop: `5%`
- Viscosity model: `artifacts/designsolvents/viscosity/model.json`, weight `0.7`
- LLM config: [`llm.gemma4_12b_vllm.yaml`](./llm.gemma4_12b_vllm.yaml)
- Captured input: [`input.txt`](./input.txt)

## Process

1. Start the vLLM server: `vllm serve google/gemma-4-12B-it --port 8000 --language-model-only`
   (`--language-model-only` disables the unused multimodal image/video encoder path — required for
   this checkpoint to profile correctly under vLLM).
2. Confirm the server is serving with `curl -s http://localhost:8000/v1/models`.
3. Run `./run.sh` — same CLI invocation as `../betaine_des_gemma4_12b/run.sh`, pointed at the vLLM
   config instead of the Ollama one.

## Output

[`output.txt`](./output.txt) has the same structure as `../betaine_des_gemma4_12b/output.txt`: the
DES screening table, LLM candidate reviews, two-stage brainstorm, contradiction/grounding analysis,
chemistry-advisor notes, and viscosity predictions. See
[`docs/vllm-example-run-report-2026-07-07.md`](/home/qshao/DES-Agent/docs/vllm-example-run-report-2026-07-07.md)
for the vLLM vs. Ollama timing comparison for this multi-cycle run.

## How to Adapt

Same knobs as the Ollama version apply — see [`../betaine_des_gemma4_12b/README.md`](../betaine_des_gemma4_12b/README.md#how-to-adapt).
