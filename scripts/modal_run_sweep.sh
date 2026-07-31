#!/usr/bin/env bash
# Launch detached parallel eval sweep on Modal A10G GPUs.
set -euo pipefail
cd "$(dirname "$0")/.."

PRESET="${PRESET:-turboquant}"
CONTEXT_LENGTHS="${CONTEXT_LENGTHS:-128,256,512}"
LABELS="${LABELS:-}"
OUTPUT="${OUTPUT:-olmo2_phase5_turboquant}"
NO_RESUME="${NO_RESUME:-}"

resolve_modal() {
  if command -v modal >/dev/null 2>&1 && modal --version >/dev/null 2>&1; then
    echo "modal"
    return
  fi
  for py in python3.11 /usr/local/bin/python3.11 /Library/Frameworks/Python.framework/Versions/3.11/bin/python3; do
    if command -v "$py" >/dev/null 2>&1 && "$py" -c "import modal" >/dev/null 2>&1; then
      echo "$py -m modal"
      return
    fi
  done
  echo "modal not found" >&2
  exit 1
}

MODAL_CMD="$(resolve_modal)"

ARGS=(--preset "$PRESET" --context-lengths "$CONTEXT_LENGTHS" --output "$OUTPUT")
if [[ -n "$LABELS" ]]; then
  ARGS+=(--labels "$LABELS")
fi
if [[ -n "$NO_RESUME" ]]; then
  ARGS+=(--no-resume)
fi

# shellcheck disable=SC2086
$MODAL_CMD run --detach modal_app/sweep.py::main "${ARGS[@]}"
