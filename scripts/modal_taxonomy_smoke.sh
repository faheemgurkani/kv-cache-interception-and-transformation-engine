#!/usr/bin/env bash
# Dummy-local taxonomy smoke, then Gemma3-270M Modal coverage of every active compressor.
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

echo "==> Dummy local taxonomy smoke"
python scripts/run_taxonomy_smoke.py --dummy

if [[ "${DUMMY_ONLY:-}" == "1" ]]; then
  exit 0
fi

MODAL_CMD="$(resolve_modal)"
MODEL_VOL="$(python - <<'PY'
import os, yaml
from pathlib import Path
cfg = yaml.safe_load(Path(os.environ["KV_MODAL_CONFIG"]).read_text())
print(cfg["volumes"]["model"])
PY
)"
LOCAL_MODEL="$(python - <<'PY'
import os, yaml
from pathlib import Path
cfg = yaml.safe_load(Path(os.environ["KV_MODEL_CONFIG"]).read_text())
print(cfg["local_path"])
PY
)"
REMOTE_NAME="$(basename "$LOCAL_MODEL")"

echo "==> Ensure Modal model volume: $MODEL_VOL /$REMOTE_NAME"
# shellcheck disable=SC2086
$MODAL_CMD volume ls "$MODEL_VOL" "/$REMOTE_NAME" >/dev/null 2>&1 || true
if ! $MODAL_CMD volume ls "$MODEL_VOL" "/$REMOTE_NAME" 2>/dev/null | grep -q "config.json"; then
  echo "Uploading $LOCAL_MODEL → $MODEL_VOL /$REMOTE_NAME"
  # shellcheck disable=SC2086
  $MODAL_CMD volume put "$MODEL_VOL" "$LOCAL_MODEL" "/$REMOTE_NAME" --force
fi

echo "==> Modal taxonomy smoke (Gemma3-270M, ctx=128, 6 compressors)"
python scripts/run_taxonomy_smoke.py --modal --skip-dummy --sync --no-resume \
  --output "${OUTPUT:-taxonomy_smoke_gemma3}"
