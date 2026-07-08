#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

CHECKPOINT_PATH="${DES_CHECKPOINT_PATH:-ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt}"
OUTPUT_FILE="${SCRIPT_DIR}/output.txt"

python -m examples.demo_des_search \
  --component-a "CCN(CC)CC(=O)Nc1c(C)cccc1C" \
  --n 5 \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --llm-config "${SCRIPT_DIR}/llm.gemma4_12b_vllm.yaml"   --proposal-diversity-mode explore   --proposal-max-similarity 0.78   --proposal-per-family-budget 1 \
  > "${OUTPUT_FILE}" 2>/dev/null
