#!/usr/bin/env bash
set -euo pipefail

# Deterministic capture: use experimental lookup + heuristic only (QSPR weights
# are not committed and training is GPU-stochastic). See examples/README.md.
export DES_DISABLE_QSPR=1

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

python -m examples.demo_des_search   --component-a "ethanol"   --n 5   --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt   --viscosity-model-path artifacts/designsolvents/viscosity/model.json   > "${SCRIPT_DIR}/output.txt" 2>/dev/null
