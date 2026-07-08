# Plain-Language Gemma 4-12B DES Example (vLLM)

vLLM twin of [`../plain_language_gemma4_12b/`](../plain_language_gemma4_12b) — the same
plain-language-request → router → DES-workflow pipeline, backed by a local vLLM server serving
`google/gemma-4-12B-it` (bf16) instead of Ollama.

## Input

- Plain-language request: see [`input.txt`](./input.txt) (identical to the Ollama version, with
  "Ollama" replaced by "the local Gemma 4-12B model (served via vLLM)" — the router only reasons
  over the chemistry content, not the backend name)
- LLM config: [`llm.gemma4_12b_vllm.yaml`](./llm.gemma4_12b_vllm.yaml)

## Process

1. Start the vLLM server: `vllm serve google/gemma-4-12B-it --port 8000 --language-model-only`.
2. Confirm with `curl -s http://localhost:8000/v1/models`.
3. Run `./run.sh`, which runs [`run_example.py`](./run_example.py) — identical to
   `../plain_language_gemma4_12b/run_example.py` except `LLM_CONFIG_FILE` points at
   `llm.gemma4_12b_vllm.yaml`. The script calls the router directly (`provider.route_request`),
   normalizes the JSON job client-side (`_normalize_router_job`), then runs the DES workflow with
   `run_search_report`.

## Output

**Known limitation, captured as-is:** this vLLM-served checkpoint (bf16, `google/gemma-4-12B-it`)
reproducibly paraphrases the request's "shipped ml_des_mp checkpoint and config" phrase into a
`config` job field like `"shipped"`, `"shipped_default"`, or `"shipped_config"` — 8/8 attempts
returned one of these three variants, never the literal `"default"` that
[`run_example.py`](./run_example.py)'s `_normalize_router_job` checks for before substituting the
real config path. Every attempt fails identically with
`FileNotFoundError: Path does not exist: /home/qshao/DES-Agent/shipped...`. The same request against
the Ollama-quantized Gemma checkpoint (`../plain_language_gemma4_12b/`) does not exhibit this —
its router reliably omits or nulls the `config` field, which the normalizer does catch.

[`output.txt`](./output.txt) captures this actual failing run in full (request, raw router JSON,
normalized job, and the traceback) rather than a synthetic success, since 8 consecutive attempts
converged on the same failure mode — this reads as a stable property of this checkpoint's phrasing
under vLLM, not one-off sampling noise. See
[`docs/vllm-example-run-report-2026-07-07.md`](/home/qshao/DES-Agent/docs/vllm-example-run-report-2026-07-07.md)
for the full write-up. No production code was changed — `_normalize_router_job`'s fallback set is
example-local convenience logic (mirrored across the plain-language example scripts), not something
this documentation pass is in scope to fix.
