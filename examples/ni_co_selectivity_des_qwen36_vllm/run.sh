#!/usr/bin/env bash
# Ni2+/Co2+ selectivity-DES pipeline with Qwen 3.6 (vLLM)
#
# vLLM twin of ../ni_co_selectivity_des_qwen36/ — identical pipeline
# parameters, swapping the Ollama backend for a local vLLM OpenAI-compatible
# server.
#
# Prerequisites:
#   - vLLM server running locally with Qwen/Qwen3.6-35B-A3B-FP8 loaded:
#       VLLM_DEEP_GEMM_WARMUP=skip vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
#         --port 8000 --moe-backend triton --linear-backend triton
#     (check: curl -s http://localhost:8000/v1/models)
#   - ChemBERTa checkpoint in ml_des_mp/runs/
#   - Viscosity and stability-constant artifacts in artifacts/
#
# Usage:
#   ./run.sh               # default output dir (runs/ni_co_selectivity_des_qwen36_vllm_001)
#   ./run.sh runs/myrun    # custom output directory

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

OUTPUT_DIR="${1:-runs/ni_co_selectivity_des_qwen36_vllm_001}"
mkdir -p "${OUTPUT_DIR}"

echo "Proposal diversity settings: mode=balanced, max_similarity=0.82, per_family_budget=2"

python -m des_multi_agent.cli \
  --workflow selectivity-des \
  --target-metal-ion  "Ni2+" \
  --competitor-metal-ion "Co2+" \
  \
  `# --- Phase 1: selectivity screening (shortlist top 5 HBD/HBA ligands) ---` \
  --n 20 \
  --n-cycles 3 \
  --affinity-weight 0.4 \
  --selectivity-weight 0.6 \
  --min-delta-log-k 0.3 \
  --top-ligands 5 \
  --proposal-diversity-mode balanced \
  --proposal-max-similarity 0.82 \
  --proposal-per-family-budget 2 \
  --stability-constant-model-path artifacts/stability_constants/model.json \
  \
  `# --- Phase 2: DES partner search for the top 3 shortlisted ligands ---` \
  --n-des-candidates 20 \
  --n-des-cycles 3 \
  --abs-tm-threshold 350 \
  --viscosity-model-path artifacts/designsolvents/viscosity/model.json \
  --viscosity-threshold 200 \
  --viscosity-weight 0.4 \
  \
  `# --- Outer feedback loop ---` \
  --n-outer-cycles 2 \
  \
  `# --- ML checkpoint ---` \
  --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt \
  --config-path ml_des_mp/config.yaml \
  \
  `# --- LLM (Qwen 3.6 via vLLM) ---` \
  --llm-config llm.ni_co_qwen36_vllm.yaml \
  \
  `# --- Save outputs ---` \
  --output-dir "${OUTPUT_DIR}" \
  > "${SCRIPT_DIR}/output.txt" 2>/dev/null

echo "Results written to ${SCRIPT_DIR}/output.txt"
echo "Run artifacts saved in ${OUTPUT_DIR}/"
