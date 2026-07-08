# Lidocaine + Gemma 4-12B Example (vLLM)

vLLM twin of [`../lidocaine_gemma4_12b/`](../lidocaine_gemma4_12b) — same lidocaine DES screening
run, using a local vLLM server serving `google/gemma-4-12B-it` (bf16) instead of Ollama.

## Input

- Component A: `lidocaine` free base
- SMILES: `CCN(CC)CC(=O)Nc1c(C)cccc1C`
- Candidate search count: `5`
- Checkpoint: `ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt`
- LLM config: [`llm.gemma4_12b_vllm.yaml`](./llm.gemma4_12b_vllm.yaml)
- Captured input: [`input.txt`](./input.txt)

## Process

1. Start the vLLM server: `vllm serve google/gemma-4-12B-it --port 8000 --language-model-only`.
2. Confirm with `curl -s http://localhost:8000/v1/models`.
3. Run `./run.sh` — identical invocation to `../lidocaine_gemma4_12b/run.sh`, pointed at the vLLM
   config.

## Output

[`output.txt`](./output.txt) has the same structure as the Ollama capture: ranked DES results,
uncertainty annotations, Gemma's two-stage brainstorm, proposal-diversity controls (`explore` mode
here), explanation/critique notes, and contradiction analysis when available. See
[`docs/vllm-example-run-report-2026-07-07.md`](/home/qshao/DES-Agent/docs/vllm-example-run-report-2026-07-07.md)
for timing notes.
