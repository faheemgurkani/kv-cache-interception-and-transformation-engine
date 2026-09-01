#!/usr/bin/env bash
# Upload a shortlist checkpoint into its Modal model volume (idempotent).
# Usage:
#   KV_MODEL_CONFIG=configs/model_gemma3_270m.yaml \
#   KV_MODAL_CONFIG=configs/modal_gemma3.yaml \
#   bash scripts/modal_setup_shortlist_model.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

export KV_MODEL_CONFIG="${KV_MODEL_CONFIG:-configs/model_gemma3_270m.yaml}"
export KV_MODAL_CONFIG="${KV_MODAL_CONFIG:-configs/modal_gemma3.yaml}"

resolve_modal() {
  if command -v modal >/dev/null 2>&1 && modal --version >/dev/null 2>&1; then
    echo "modal"
    return
  fi
  if python -c "import modal" >/dev/null 2>&1; then
    echo "python -m modal"
    return
  fi
  echo "modal not found" >&2
  exit 1
}

MODAL_CMD="$(resolve_modal)"
MODEL_VOL="$(python - <<'PY'
import os, yaml
from pathlib import Path
print(yaml.safe_load(Path(os.environ["KV_MODAL_CONFIG"]).read_text())["volumes"]["model"])
PY
)"
LOCAL_MODEL="$(python - <<'PY'
import os, yaml
from pathlib import Path
print(yaml.safe_load(Path(os.environ["KV_MODEL_CONFIG"]).read_text())["local_path"])
PY
)"
REMOTE_NAME="$(basename "$LOCAL_MODEL")"

if [[ ! -f "$LOCAL_MODEL/config.json" ]]; then
  echo "Local checkpoint missing: $LOCAL_MODEL" >&2
  exit 1
fi

echo "Putting $LOCAL_MODEL → volume $MODEL_VOL:/$REMOTE_NAME"
# shellcheck disable=SC2086
$MODAL_CMD volume put "$MODEL_VOL" "$LOCAL_MODEL" "/$REMOTE_NAME" --force
# shellcheck disable=SC2086
$MODAL_CMD run modal_app/worker.py::ensure_model
echo "Done. Volume listing:"
# shellcheck disable=SC2086
$MODAL_CMD volume ls "$MODEL_VOL" "/$REMOTE_NAME"
