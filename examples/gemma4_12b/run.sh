#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

OUTPUT_FILE="${SCRIPT_DIR}/output.txt"

python -m examples.demo_des_search \
  --component-a "CCO" \
  --n 20 \
  --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt \
  --llm-config "${SCRIPT_DIR}/llm.gemma4_12b.yaml" \
  > "${OUTPUT_FILE}" 2>/dev/null
