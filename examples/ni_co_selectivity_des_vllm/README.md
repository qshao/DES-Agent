# Ni²⁺/Co²⁺ Selectivity-DES Example (vLLM)

vLLM twin of [`../ni_co_selectivity_des/`](../ni_co_selectivity_des) — the identical two-phase
selectivity-DES pipeline (Ni²⁺-selective ligand screening, then DES partner search for the top
shortlisted ligands, with an outer feedback loop), backed by a local vLLM server serving
`google/gemma-4-12B-it` (bf16) instead of Ollama. See the original README for the full pipeline
architecture diagram and scientific goal — this file only documents what differs.

## Input

Identical parameters to `../ni_co_selectivity_des/`: target `Ni2+` vs. competitor `Co2+`, Phase 1
`n=20`/`n_cycles=3`/`top_ligands=3`, Phase 2 `n_des_candidates=20`/`n_des_cycles=3`, 2 outer cycles.
See [`input.txt`](./input.txt). The only change is `llm_config=llm.ni_co_selectivity_vllm.yaml`
(repo root) instead of `llm.ni_co_selectivity.yaml`.

## Process

1. Start the vLLM server:
   ```bash
   vllm serve google/gemma-4-12B-it --port 8000 --language-model-only
   ```
2. Confirm with `curl -s http://localhost:8000/v1/models`.
3. Run:
   ```bash
   ./run.sh
   # or with a custom output directory:
   ./run.sh runs/my_ni_co_run
   ```
   Same CLI invocation as `../ni_co_selectivity_des/run.sh`, pointed at
   `llm.ni_co_selectivity_vllm.yaml`. Default output directory is
   `runs/ni_co_selectivity_des_vllm_001/`.

## Output

[`output.txt`](./output.txt) has the same two-section structure as the Ollama capture: the Phase 1
selectivity table (ligand, log K(Ni²⁺), log K(Co²⁺), ΔlogK, score, DES-compatibility) and Phase 2
DES partner blocks per shortlisted ligand. See
[`docs/vllm-example-run-report-2026-07-07.md`](/home/qshao/DES-Agent/docs/vllm-example-run-report-2026-07-07.md)
for the wall-clock comparison against the Ollama run — this is the longest-running example in the
suite (multi-hour, dominated by LLM calls across 2 outer cycles × 3 ligands × 3 DES cycles).

## LLM config

[`llm.ni_co_selectivity_vllm.yaml`](../../llm.ni_co_selectivity_vllm.yaml) points to
`google/gemma-4-12B-it` on a local vLLM server at `localhost:8000/v1`.
