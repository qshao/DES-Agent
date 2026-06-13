#!/usr/bin/env bash
# Ni2+/Co2+ selectivity-DES pipeline with Gemma4-12B (Ollama)
#
# Goal: find ligands that bind Ni2+ selectively over Co2+, where the ligand
# also forms a deep eutectic solvent with a low-viscosity, low-melting partner.
#
# Prerequisites:
#   - Ollama running locally with gemma4:12b loaded
#     (check: curl -s http://localhost:11434/api/tags)
#   - ChemBERTa checkpoint in ml_des_mp/runs/
#   - Viscosity and stability-constant artifacts in artifacts/
#
# Usage:
#   ./run.sh               # uses default output dir (runs/ni_co_selectivity_des_001)
#   ./run.sh runs/myrun    # saves results into a custom directory

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

OUTPUT_DIR="${1:-runs/ni_co_selectivity_des_001}"
mkdir -p "${OUTPUT_DIR}"

echo "Proposal diversity settings: mode=balanced, max_similarity=0.82, per_family_budget=2"

python -m des_multi_agent.cli \
  --workflow selectivity-des \
  --target-metal-ion  "Ni2+" \
  --competitor-metal-ion "Co2+" \
  \
  `# --- Phase 1: selectivity screening ---` \
  --n 20 \
  --n-cycles 3 \
  --affinity-weight 0.4 \
  --selectivity-weight 0.6 \
  --min-delta-log-k 0.3 \
  --top-ligands 3 \
  --proposal-diversity-mode balanced \
  --proposal-max-similarity 0.82 \
  --proposal-per-family-budget 2 \
  --stability-constant-model-path artifacts/stability_constants/model.json \
  \
  `# --- Phase 2: DES partner search ---` \
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
  `# --- LLM (Gemma4-12B via Ollama) ---` \
  --llm-config llm.ni_co_selectivity.yaml \
  \
  `# --- Save outputs ---` \
  --output-dir "${OUTPUT_DIR}" \
  > "${SCRIPT_DIR}/output.txt" 2>/dev/null

echo "Results written to ${SCRIPT_DIR}/output.txt"
echo "Run artifacts saved in ${OUTPUT_DIR}/"
