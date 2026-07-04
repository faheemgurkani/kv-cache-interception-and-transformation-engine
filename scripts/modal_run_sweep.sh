#!/usr/bin/env bash
# Launch detached parallel eval sweep on Modal A10G GPUs.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

CONTEXT_LENGTHS="${CONTEXT_LENGTHS:-128,512,4096,8192,16384,32768}"
LABELS="${LABELS:-}"
OUTPUT="${OUTPUT:-phase5_modal_sweep}"

ARGS=(--context-lengths "$CONTEXT_LENGTHS" --output "$OUTPUT")
if [[ -n "$LABELS" ]]; then
  ARGS+=(--labels "$LABELS")
fi

modal run --detach modal_app/sweep.py::main "${ARGS[@]}"
