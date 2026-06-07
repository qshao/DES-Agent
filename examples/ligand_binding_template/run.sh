#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

python -m des_multi_agent.cli   --workflow metal-binding   --metal-ion "Cu2+"   --ligand-smiles "NCCN"   --stability-constant-model-path artifacts/stability_constants/model.json   > "${SCRIPT_DIR}/output.txt" 2>/dev/null
