#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
CHECKPOINT_PATH="${DES_CHECKPOINT_PATH:-ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt}"

cd "${REPO_ROOT}"

python -m des_multi_agent.cli \
  --workflow des \
  --component-a "betaine" \
  --n 20 \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --config-path ml_des_mp/config.yaml \
  --abs-tm-threshold 340 \
  --rel-drop-min 0.05 \
  --viscosity-model-path artifacts/designsolvents/viscosity/model.json \
  --viscosity-weight 0.7 \
  --n-cycles 5 \
  --llm-config "${SCRIPT_DIR}/llm.gemma4_12b_vllm.yaml"   --proposal-diversity-mode balanced   --proposal-max-similarity 0.84   --proposal-per-family-budget 2 \
  > "${SCRIPT_DIR}/output.txt" 2>/dev/null
