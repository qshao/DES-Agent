#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

# Optional: pass an LLM config file as the first argument to enable three full
# brainstorm cycles.  Without it, only the heuristic cycle (cycle 1) runs.
LLM_CONFIG="${1:-}"

if [ -n "${LLM_CONFIG}" ]; then
    python "${SCRIPT_DIR}/run.py" "${LLM_CONFIG}" \
        > "${SCRIPT_DIR}/output.txt" 2>/dev/null
else
    python "${SCRIPT_DIR}/run.py" \
        > "${SCRIPT_DIR}/output.txt" 2>/dev/null
fi
