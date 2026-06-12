#!/usr/bin/env bash
# Ni2+/Co2+ selectivity-DES pipeline with Qwen 3.6 (Ollama)
#
# Goal: find five ligands that bind Ni2+ selectively over Co2+, where each
# ligand is a hydrogen-bond donor or acceptor, then find deep eutectic
# solvents containing the top three ligands with Tm < 350 K, low viscosity,
# and hydrophobic character where possible.
#
# Prerequisites:
#   - Ollama running locally with qwen3.6 loaded
#     (check: curl -s http://localhost:11434/api/tags)
#     (pull:  ollama pull qwen3.6)
#   - ChemBERTa checkpoint in ml_des_mp/runs/
#   - Viscosity and stability-constant artifacts in artifacts/
#
# Usage:
#   ./run.sh               # default output dir (runs/ni_co_selectivity_des_qwen36_001)
#   ./run.sh runs/myrun    # custom output directory

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

OUTPUT_DIR="${1:-runs/ni_co_selectivity_des_qwen36_001}"
mkdir -p "${OUTPUT_DIR}"

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
  `# --- LLM (Qwen 3.6 via Ollama) ---` \
  --llm-config llm.ni_co_qwen36.yaml \
  \
  `# --- Save outputs ---` \
  --output-dir "${OUTPUT_DIR}" \
  > "${SCRIPT_DIR}/output.txt" 2>/dev/null

echo "Results written to ${SCRIPT_DIR}/output.txt"
echo "Run artifacts saved in ${OUTPUT_DIR}/"
