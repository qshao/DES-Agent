#!/usr/bin/env bash
set -euo pipefail

# Deterministic capture: use experimental lookup + heuristic only (QSPR weights
# are not committed and training is GPU-stochastic). See examples/README.md.
export DES_DISABLE_QSPR=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_FILE="${SCRIPT_DIR}/output.txt"
MEMORY_FILE="${SCRIPT_DIR}/run.memory.json"
CHECKPOINT_PATH="${DES_CHECKPOINT_PATH:-ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt}"
CONFIG_PATH="${DES_CONFIG_PATH:-ml_des_mp/config.yaml}"

cd "${REPO_ROOT}"

{
  echo "INPUT"
  cat "${SCRIPT_DIR}/input.txt"
  echo
  echo "SAVE MEMORY"
  python -m des_multi_agent.cli --workflow des --component-a "CCO" --n 5 --checkpoint-path "${CHECKPOINT_PATH}" --config-path "${CONFIG_PATH}" --proposal-diversity-mode balanced --proposal-max-similarity 0.82 --proposal-per-family-budget 2 --save-run-memory "${MEMORY_FILE}"
  echo
  echo "LABEL MEMORY"
  python -m des_multi_agent.cli label-run --run "${MEMORY_FILE}" --label "O=good" --label "CC(=O)O=bad"
  echo
  echo "REUSE MEMORY"
  python -m des_multi_agent.cli --workflow des --component-a "CCO" --n 5 --checkpoint-path "${CHECKPOINT_PATH}" --config-path "${CONFIG_PATH}" --proposal-diversity-mode balanced --proposal-max-similarity 0.82 --proposal-per-family-budget 2 --reuse-run "${MEMORY_FILE}"
} > "${OUTPUT_FILE}"
