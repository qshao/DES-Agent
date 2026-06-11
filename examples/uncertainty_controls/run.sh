#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

CHECKPOINT_PATH="ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt"
CONFIG_PATH="ml_des_mp/config.yaml"

echo "=== report_only: annotate uncertainty without changing ranking ===" > "${SCRIPT_DIR}/output.txt"
python -m des_multi_agent.cli \
  --workflow des \
  --component-a "CCO" \
  --n 5 \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --config-path "${CONFIG_PATH}" \
  --uncertainty-mode report_only \
  >> "${SCRIPT_DIR}/output.txt" 2>/dev/null

echo "" >> "${SCRIPT_DIR}/output.txt"
echo "=== penalize: soft down-rank candidates below min-trust-score ===" >> "${SCRIPT_DIR}/output.txt"
python -m des_multi_agent.cli \
  --workflow des \
  --component-a "CCO" \
  --n 5 \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --config-path "${CONFIG_PATH}" \
  --uncertainty-mode penalize \
  --min-trust-score 0.85 \
  --soft-penalty-weight 0.3 \
  >> "${SCRIPT_DIR}/output.txt" 2>/dev/null

echo "" >> "${SCRIPT_DIR}/output.txt"
echo "=== filter: hard-remove candidates below min-trust-score 0.9 ===" >> "${SCRIPT_DIR}/output.txt"
python -m des_multi_agent.cli \
  --workflow des \
  --component-a "CCO" \
  --n 5 \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --config-path "${CONFIG_PATH}" \
  --uncertainty-mode filter \
  --min-trust-score 0.9 \
  >> "${SCRIPT_DIR}/output.txt" 2>/dev/null
