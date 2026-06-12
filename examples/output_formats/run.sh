#!/usr/bin/env bash
set -euo pipefail

# Deterministic capture: use experimental lookup + heuristic only (QSPR weights
# are not committed and training is GPU-stochastic). See examples/README.md.
export DES_DISABLE_QSPR=1

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

CHECKPOINT_PATH="ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt"
CONFIG_PATH="ml_des_mp/config.yaml"

echo "=== table (default) ===" > "${SCRIPT_DIR}/output.txt"
python -m des_multi_agent.cli \
  --workflow des \
  --component-a "CCO" \
  --n 5 \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --config-path "${CONFIG_PATH}" \
  --format table \
  >> "${SCRIPT_DIR}/output.txt" 2>/dev/null

echo "" >> "${SCRIPT_DIR}/output.txt"
echo "=== json ===" >> "${SCRIPT_DIR}/output.txt"
python -m des_multi_agent.cli \
  --workflow des \
  --component-a "CCO" \
  --n 5 \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --config-path "${CONFIG_PATH}" \
  --format json \
  >> "${SCRIPT_DIR}/output.txt" 2>/dev/null

echo "" >> "${SCRIPT_DIR}/output.txt"
echo "=== csv ===" >> "${SCRIPT_DIR}/output.txt"
python -m des_multi_agent.cli \
  --workflow des \
  --component-a "CCO" \
  --n 5 \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --config-path "${CONFIG_PATH}" \
  --format csv \
  >> "${SCRIPT_DIR}/output.txt" 2>/dev/null

echo "" >> "${SCRIPT_DIR}/output.txt"
echo "=== prose ===" >> "${SCRIPT_DIR}/output.txt"
python -m des_multi_agent.cli \
  --workflow des \
  --component-a "CCO" \
  --n 5 \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --config-path "${CONFIG_PATH}" \
  --format prose \
  >> "${SCRIPT_DIR}/output.txt" 2>/dev/null
