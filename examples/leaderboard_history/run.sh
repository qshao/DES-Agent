#!/usr/bin/env bash
set -euo pipefail

# Deterministic capture: use experimental lookup + heuristic only (QSPR weights
# are not committed and training is GPU-stochastic). See examples/README.md.
export DES_DISABLE_QSPR=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CHECKPOINT_PATH="${DES_CHECKPOINT_PATH:-ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt}"
CONFIG_PATH="${DES_CONFIG_PATH:-ml_des_mp/config.yaml}"
HISTORY_DIR="/tmp/des_history"

cd "${REPO_ROOT}"

# Clean up any prior history directory so the example is reproducible
rm -rf "${HISTORY_DIR}"

{
  echo "=== RUN 1 ==="
  python -m des_multi_agent.cli --workflow des --component-a "ethanol" --n 5 \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --config-path "${CONFIG_PATH}" \
    --output-dir "${HISTORY_DIR}/run_01" 2>/dev/null
  echo

  echo "=== RUN 2 ==="
  python -m des_multi_agent.cli --workflow des --component-a "ethanol" --n 5 \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --config-path "${CONFIG_PATH}" \
    --output-dir "${HISTORY_DIR}/run_02" 2>/dev/null
  echo

  echo "=== LEADERBOARD ==="
  python -m des_multi_agent.cli leaderboard "${HISTORY_DIR}"
  echo

  echo "=== HISTORY ==="
  python -m des_multi_agent.cli history "${HISTORY_DIR}"
} > "${SCRIPT_DIR}/output.txt" 2>&1
