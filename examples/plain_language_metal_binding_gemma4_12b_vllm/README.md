# Plain-Language Gemma 4-12B Metal-Binding Example (vLLM)

vLLM twin of [`../plain_language_metal_binding_gemma4_12b/`](../plain_language_metal_binding_gemma4_12b)
— the same plain-language-request → router → metal-binding-workflow pipeline, backed by a local
vLLM server serving `google/gemma-4-12B-it` (bf16) instead of Ollama.

## Input

- Plain-language request: see [`input.txt`](./input.txt)
- Stability model: `artifacts/stability_constants/model.json`
- LLM config: [`llm.gemma4_12b_vllm.yaml`](./llm.gemma4_12b_vllm.yaml)

## Process

1. Start the vLLM server: `vllm serve google/gemma-4-12B-it --port 8000 --language-model-only`.
2. Confirm with `curl -s http://localhost:8000/v1/models`.
3. Run `./run.sh`, which runs [`run_example.py`](./run_example.py) — identical to
   `../plain_language_metal_binding_gemma4_12b/run_example.py` except `LLM_CONFIG_FILE` points at
   `llm.gemma4_12b_vllm.yaml`.

## Output

[`output.txt`](./output.txt) contains the plain-language request, the raw router JSON output, the
normalized metal-binding job, and the metal-binding prediction report. See
[`docs/vllm-example-run-report-2026-07-07.md`](/home/qshao/DES-Agent/docs/vllm-example-run-report-2026-07-07.md)
for timing notes.
