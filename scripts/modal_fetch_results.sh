#!/usr/bin/env bash
# Pull evaluation JSON payloads from Modal results volume to results/modal_volume_olmo2/
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-results/modal_volume_olmo2}"
mkdir -p "$OUT"

resolve_python() {
  for py in python3.11 /usr/local/bin/python3.11 /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 python3; do
    if command -v "$py" >/dev/null 2>&1; then
      echo "$py"
      return
    fi
  done
  echo "python3" 
}

PY="$(resolve_python)"
VOLUME="$($PY - <<'PY'
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path("configs/modal.yaml").read_text())
print(cfg.get("volumes", {}).get("results", "kv-engine-results-olmo2"))
PY
)"

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
$MODAL_CMD volume get "$VOLUME" / "$OUT" --force
echo "Downloaded Modal results from $VOLUME to $OUT"
echo "Merge: $PY scripts/restructure_olmo2_modal_results.py"
