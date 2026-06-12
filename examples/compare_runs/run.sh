#!/usr/bin/env bash
set -euo pipefail

# Deterministic capture: use experimental lookup + heuristic only (QSPR weights
# are not committed and training is GPU-stochastic). See examples/README.md.
export DES_DISABLE_QSPR=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CHECKPOINT_PATH="${DES_CHECKPOINT_PATH:-ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt}"
CONFIG_PATH="${DES_CONFIG_PATH:-ml_des_mp/config.yaml}"

cd "${REPO_ROOT}"

{
  echo "=== RUN A (n=3) ==="
  python -m des_multi_agent.cli --workflow des --component-a "CCO" --n 3 \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --config-path "${CONFIG_PATH}" \
    --save-run-memory /tmp/des_compare_run_a.json 2>/dev/null
  echo

  echo "=== RUN B (n=5) ==="
  python -m des_multi_agent.cli --workflow des --component-a "CCO" --n 5 \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --config-path "${CONFIG_PATH}" \
    --save-run-memory /tmp/des_compare_run_b.json 2>/dev/null
  echo

  echo "=== COMPARE ==="
  python -m des_multi_agent.cli compare-runs \
    /tmp/des_compare_run_a.json \
    /tmp/des_compare_run_b.json
} > "${SCRIPT_DIR}/output.txt" 2>&1
