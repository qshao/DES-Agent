#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REQUEST="$(tr '\n' ' ' < "${SCRIPT_DIR}/input.txt")"

cd "${REPO_ROOT}"
python -m des_multi_agent.cli task-router "${REQUEST}" > "${SCRIPT_DIR}/output.txt"
