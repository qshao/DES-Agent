#!/usr/bin/env bash
# vLLM twin of ../task_execute/ — see task_router_vllm/run.sh for why this
# wrapper temporarily swaps the shared llm.example.yaml to point at the vLLM
# server and restores it afterward via a trap.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ROUTER_CFG="${REPO_ROOT}/llm.example.yaml"
BACKUP_CFG="${SCRIPT_DIR}/.llm.example.yaml.bak"

cd "${REPO_ROOT}"
cp "${ROUTER_CFG}" "${BACKUP_CFG}"
trap 'cp "${BACKUP_CFG}" "${ROUTER_CFG}"; rm -f "${BACKUP_CFG}"' EXIT

cat > "${ROUTER_CFG}" <<'YAML'
llm:
  enabled: true
  provider: vllm
  model_name: Qwen/Qwen3.6-35B-A3B-FP8
  api_base_url: http://localhost:8000/v1
  max_candidates: 20
  max_tokens: 1024
  temperature: 0.2
  timeout_seconds: 300.0
YAML

# Unlike ../task_execute/run.sh (stderr discarded), stderr is merged here so
# a routing failure is captured in output.txt — see README.md for why this
# currently fails against this vLLM checkpoint.
python -m des_multi_agent.cli task-execute \
  "find DES partners for ethanol (CCO) using 20 candidates" \
  > "${SCRIPT_DIR}/output.txt" 2>&1
