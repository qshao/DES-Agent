#!/usr/bin/env bash
# vLLM twin of ../task_router/ — task-router hardcodes the shared
# llm.example.yaml as its LLM config (des_multi_agent/task_router.py's
# DEFAULT_ROUTER_LLM_CONFIG) and has no --llm-config override flag, so this
# wrapper temporarily points that shared file at the vLLM server for the
# duration of the call and restores the original content afterward via a
# trap (runs on success, failure, or interrupt alike).
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

REQUEST="$(tr '\n' ' ' < "${SCRIPT_DIR}/input.txt")"
# Unlike ../task_router/run.sh (stdout only), stderr is merged here so a
# routing failure is captured in output.txt rather than silently discarded —
# see README.md for why this currently fails against this vLLM checkpoint.
python -m des_multi_agent.cli task-router "${REQUEST}" > "${SCRIPT_DIR}/output.txt" 2>&1
