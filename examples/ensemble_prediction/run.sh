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
  --component-a "CCO" \
  --n 5 \
  --ensemble \
  --config-path ml_des_mp/config.yaml \
  > "${SCRIPT_DIR}/output.txt" 2>/dev/null
