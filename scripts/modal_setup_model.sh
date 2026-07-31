#!/usr/bin/env bash
# Download configured model (configs/model.yaml) into Modal Volume (run once).
set -euo pipefail
cd "$(dirname "$0")/.."

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
# shellcheck disable=SC2086
$MODAL_CMD run modal_app/worker.py::ensure_model
