#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

CHECKPOINT_PATH="${DES_CHECKPOINT_PATH:-ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt}"
if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "Checkpoint not found: $CHECKPOINT_PATH"
  echo "Set DES_CHECKPOINT_PATH to a local trained checkpoint and rerun."
  exit 1
fi

ARGS=(python -m examples.demo_des_search --component-a "CCO" --n 5 --checkpoint-path "$CHECKPOINT_PATH")
if [[ -n "${DES_DISCOVERY_PATH:-}" ]]; then
  if [[ ! -d "$DES_DISCOVERY_PATH" ]]; then
    echo "Discovery directory not found: $DES_DISCOVERY_PATH"
    echo "Set DES_DISCOVERY_PATH to a local directory containing literature.yaml and library.yaml."
    exit 1
  fi
  ARGS+=(--discovery-path "$DES_DISCOVERY_PATH")
fi

"${ARGS[@]}"
