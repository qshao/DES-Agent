#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"
python -m des_multi_agent.cli task-execute \
  "find DES partners for ethanol (CCO) using 20 candidates" \
  > "${SCRIPT_DIR}/output.txt" 2>/dev/null
