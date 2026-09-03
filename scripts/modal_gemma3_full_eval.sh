#!/usr/bin/env bash
# Gemma3-270M full redesigned eval: 6 compressors × ctx {128,256,512} on Modal A10G only.
#
# Usage:
#   bash scripts/modal_gemma3_full_eval.sh scope   # one identity job @ ctx=128
#   bash scripts/modal_gemma3_full_eval.sh full     # 18-job grid + KPI audit
#   bash scripts/modal_gemma3_full_eval.sh setup    # volume + model upload only
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

export KV_MODEL_CONFIG="${KV_MODEL_CONFIG:-configs/model_gemma3_270m.yaml}"
export KV_MODAL_CONFIG="${KV_MODAL_CONFIG:-configs/modal_gemma3.yaml}"
CONTEXT_LENGTHS="${CONTEXT_LENGTHS:-128,256,512}"
OUTPUT="${OUTPUT:-gemma3_full_eval}"
PRESET="${PRESET:-taxonomy_full}"
MODE="${1:-full}"

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
RESULTS_VOL="$(python - <<'PY'
import os, yaml
from pathlib import Path
print(yaml.safe_load(Path(os.environ["KV_MODAL_CONFIG"]).read_text())["volumes"]["results"])
PY
)"
GPU="$(python - <<'PY'
import os, yaml
from pathlib import Path
cfg = yaml.safe_load(Path(os.environ["KV_MODAL_CONFIG"]).read_text())
print(",".join(cfg.get("gpu_fallbacks") or [cfg.get("gpu", "a10g")]))
PY
)"
LOCAL_MODEL="$(python - <<'PY'
import os, yaml
from pathlib import Path
print(yaml.safe_load(Path(os.environ["KV_MODEL_CONFIG"]).read_text())["local_path"])
PY
)"
REMOTE_NAME="$(basename "$LOCAL_MODEL")"

echo "Model config:  $KV_MODEL_CONFIG"
echo "Modal config:  $KV_MODAL_CONFIG"
echo "GPU policy:    $GPU (must be single type)"
echo "Model volume:  $MODEL_VOL"
echo "Results volume: $RESULTS_VOL"

if [[ "$GPU" == *","* ]]; then
  echo "ERROR: multiple GPU fallbacks configured — full eval requires one GPU type only." >&2
  exit 1
fi

if [[ ! -f "$LOCAL_MODEL/config.json" ]]; then
  echo "Local checkpoint missing: $LOCAL_MODEL" >&2
  exit 1
fi

setup_volumes() {
  echo "==> Upload checkpoint if needed"
  if ! $MODAL_CMD volume ls "$MODEL_VOL" "/$REMOTE_NAME" 2>/dev/null | grep -q "config.json"; then
    $MODAL_CMD volume put "$MODEL_VOL" "$LOCAL_MODEL" "/$REMOTE_NAME" --force
  else
    echo "Model already on volume $MODEL_VOL/$REMOTE_NAME"
  fi
  echo "==> ensure_model (idempotent)"
  $MODAL_CMD run modal_app/worker.py::ensure_model
  echo "==> Volume listings"
  $MODAL_CMD volume ls "$MODEL_VOL" "/$REMOTE_NAME" | head -20
  $MODAL_CMD volume ls "$RESULTS_VOL" "/" 2>/dev/null | tail -5 || true
}

case "$MODE" in
  setup)
    setup_volumes
    ;;
  scope)
    setup_volumes
    echo "==> Scope test: identity_baseline @ ctx=128 ($GPU only)"
    python scripts/run_taxonomy_smoke.py --modal --skip-dummy --sync --no-resume --skip-validate \
      --preset "$PRESET" --context-lengths 128 --labels identity_baseline \
      --output "${OUTPUT}_scope"
    ;;
  full)
    setup_volumes
    echo "==> Full eval: preset=$PRESET ctx=$CONTEXT_LENGTHS (6×3=18 jobs, $GPU only)"
    python scripts/run_taxonomy_smoke.py --modal --skip-dummy --sync --no-resume \
      --preset "$PRESET" --context-lengths "$CONTEXT_LENGTHS" \
      --output "$OUTPUT"
    ;;
  retry)
    setup_volumes
    echo "==> Retry OOM jobs: identity/turboquant/qjl @ ctx=256,512 ($GPU only)"
    python scripts/run_taxonomy_smoke.py --modal --skip-dummy --sync --no-resume \
      --preset "$PRESET" --context-lengths 256,512 \
      --labels identity_baseline,tq_full_b4,qjl_default \
      --output "${OUTPUT}_retry"
    ;;
  *)
    echo "Usage: $0 {setup|scope|full|retry}" >&2
    exit 1
    ;;
esac

if [[ "$MODE" == "full" || "$MODE" == "retry" ]]; then
  python - <<'PY'
import json
import os
from pathlib import Path
from eval.paper_collection import audit_paper_collection, write_paper_collection_report
from eval.kpi_schema import validate_bundle
from modal_app.merge import write_merged_reports

root = Path("results")
stem = os.environ.get("OUTPUT", "gemma3_full_eval")
paths = [p for p in sorted(root.glob(f"{stem}_*.json")) if "_scope" not in p.stem and "_retry" not in p.stem]
paths += sorted(root.glob(f"{stem}_retry_*.json"))
payloads = []
for path in paths:
    data = json.loads(path.read_text())
    payloads.extend(data.get("results", data))
seen = set()
merged = []
for item in payloads:
    key = (item.get("label"), item.get("context_length"), item.get("compressor"))
    if key in seen:
        continue
    seen.add(key)
    merged.append(item)
out = root / f"{stem}_merged_latest.json"
out.write_text(json.dumps({"results": merged}, indent=2))
write_merged_reports(merged, root, f"{stem}_merged")
audit = audit_paper_collection(merged, modal=True)
report_dir = root / f"{stem}_paper"
report_dir.mkdir(parents=True, exist_ok=True)
write_paper_collection_report(audit, report_dir / "PAPER_COLLECTION_REPORT.md")
errors = validate_bundle(merged, require_smoke_extras=True, execution_platform="modal", require_full_taxonomy=True)
print(f"Merged {len(merged)} jobs -> {out}")
print(f"Collection OK: {not errors and audit['collection_ok']}")
if errors:
    for err in errors[:10]:
        print(" ", err)
if audit.get("quality_anomalies"):
    for err in audit["quality_anomalies"]:
        print(" anomaly:", err)
PY
fi
