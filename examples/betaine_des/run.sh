#!/usr/bin/env bash
set -euo pipefail

# Deterministic capture: use experimental lookup + heuristic only (QSPR weights
# are not committed and training is GPU-stochastic). See examples/README.md.
export DES_DISABLE_QSPR=1

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

python -m des_multi_agent.cli \
  --workflow des \
  --component-a "C[N+](C)(C)CC(=O)[O-]" \
  --n 20 \
  --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt \
  --config-path ml_des_mp/config.yaml \
  --abs-tm-threshold 340 \
  --rel-drop-min 0.05 \
  --viscosity-model-path artifacts/designsolvents/viscosity/model.json \
  --viscosity-weight 0.7 \
  --n-cycles 5 \
  > "${SCRIPT_DIR}/output.txt" 2>/dev/null
