#!/usr/bin/env bash
# Re-run ONLY the jobs invalidated by the QJL ProdQJL estimator fix
# (float S@q; sign-quantize keys only).
#
# What this covers (6 jobs total if --model both):
#   Qwen3-1.7B : qjl_default × ctx {128,256,512}  → 3 jobs
#   OLMo 2 1B  : qjl_default × ctx {128,256,512}  → 3 jobs
#
# What this does NOT re-run (still valid):
#   Identity / TurboQuant / RocketKV on either model
#   (TurboQuant uses QJL only for residual *encoding* of values, not the
#    attention estimator that was wrong.)
#
# Usage:
#   bash scripts/rerun_qjl_prodqjl_fix.sh                 # both models, full 3 contexts
#   bash scripts/rerun_qjl_prodqjl_fix.sh --model qwen3   # Qwen3 only
#   bash scripts/rerun_qjl_prodqjl_fix.sh --model olmo2   # OLMo2 only
#   bash scripts/rerun_qjl_prodqjl_fix.sh --smoke         # 1 job/model @ ctx=128 first
#   bash scripts/rerun_qjl_prodqjl_fix.sh --sync          # wait + merge locally (no detach)
#   bash scripts/rerun_qjl_prodqjl_fix.sh --fetch-only    # fetch+merge existing Modal jobs
#
# Outputs land in *new* result dirs (old Phase-5 QJL bundles are left untouched):
#   results/phase5_modal_qjl_prodqjl/
#   results/olmo2_phase5_qjl_prodqjl/
#
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="both"          # qwen3 | olmo2 | both
SMOKE=0
SYNC=0
FETCH_ONLY=0
CONTEXT_LENGTHS="128,256,512"
LABELS="qjl_default"
DETACH=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    --sync) SYNC=1; DETACH=0; shift ;;
    --fetch-only) FETCH_ONLY=1; shift ;;
    --contexts) CONTEXT_LENGTHS="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,35p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1 (try --help)" >&2
      exit 1
      ;;
  esac
done

if [[ "$SMOKE" == "1" ]]; then
  CONTEXT_LENGTHS="128"
fi

case "$MODEL" in
  qwen3|olmo2|both) ;;
  *)
    echo "--model must be qwen3|olmo2|both (got: $MODEL)" >&2
    exit 1
    ;;
esac

resolve_modal() {
  if command -v modal >/dev/null 2>&1 && modal --version >/dev/null 2>&1; then
    echo "modal"
    return
  fi
  if [[ -x .venv/bin/python ]] && .venv/bin/python -c "import modal" >/dev/null 2>&1; then
    echo ".venv/bin/python -m modal"
    return
  fi
  for py in python3.11 /usr/local/bin/python3.11 /Library/Frameworks/Python.framework/Versions/3.11/bin/python3; do
    if command -v "$py" >/dev/null 2>&1 && "$py" -c "import modal" >/dev/null 2>&1; then
      echo "$py -m modal"
      return
    fi
  done
  echo "modal not found / not importable" >&2
  exit 1
}

MODAL_CMD="$(resolve_modal)"
PY="${PY:-.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="python3"
fi

# --- Config swap helpers (model is selected via configs/model.yaml + configs/modal.yaml) ---
MODEL_CFG="configs/model.yaml"
MODAL_CFG="configs/modal.yaml"
BACKUP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/kvbench_qjl_rerun.XXXXXX")"
cleanup() {
  if [[ -f "$BACKUP_DIR/model.yaml" ]]; then
    cp "$BACKUP_DIR/model.yaml" "$MODEL_CFG"
  fi
  if [[ -f "$BACKUP_DIR/modal.yaml" ]]; then
    cp "$BACKUP_DIR/modal.yaml" "$MODAL_CFG"
  fi
  rm -rf "$BACKUP_DIR"
}
trap cleanup EXIT
cp "$MODEL_CFG" "$BACKUP_DIR/model.yaml"
cp "$MODAL_CFG" "$BACKUP_DIR/modal.yaml"

activate_model() {
  local which="$1"
  case "$which" in
    qwen3)
      cp configs/model_qwen3.yaml configs/model.yaml
      cp configs/modal_qwen3.yaml configs/modal.yaml
      echo "[config] active model = Qwen3-1.7B (volumes from configs/modal_qwen3.yaml)"
      ;;
    olmo2)
      # Restore the repo's OLMo2 configs from the backup taken at start
      # (current committed defaults), then re-assert OLMo2 pair if backups
      # were already swapped. Prefer explicit OLMo2 content via git HEAD if needed.
      if [[ -f configs/model.yaml ]] && grep -q "OLMo-2" "$BACKUP_DIR/model.yaml" 2>/dev/null; then
        cp "$BACKUP_DIR/model.yaml" configs/model.yaml
        cp "$BACKUP_DIR/modal.yaml" configs/modal.yaml
      else
        # Fallback: write OLMo2 pair inline from known backup naming
        cat > configs/model.yaml <<'EOF'
model_name: allenai/OLMo-2-0425-1B
local_path: models/olmo2_1b

context_lengths:
  - 128
  - 256
  - 512

bitwidths:
  - 2
  - 3
  - 4

turboquant:
  default_bitwidth: 4
  default_stage: full
  attn_implementation: eager
  torch_dtype: float16
EOF
        cat > configs/modal.yaml <<'EOF'
gpu: A10G
gpu_fallbacks:
  - A10G
  - L4
  - any
timeout_hours: 4
volumes:
  model: kv-engine-olmo2
  results: kv-engine-results-olmo2
secrets:
  huggingface: huggingface-secret
EOF
      fi
      echo "[config] active model = OLMo 2 1B (volumes kv-engine-olmo2 / kv-engine-results-olmo2)"
      ;;
  esac
}

fetch_and_merge() {
  local which="$1"
  local out_dir="$2"
  local fetch_dir="$3"
  local restructure="$4"

  mkdir -p "$fetch_dir"
  local volume
  volume="$($PY - <<PY
import yaml
from pathlib import Path
print(yaml.safe_load(Path("configs/modal.yaml").read_text())["volumes"]["results"])
PY
)"
  echo "[fetch] modal volume get $volume → $fetch_dir"
  # shellcheck disable=SC2086
  $MODAL_CMD volume get "$volume" / "$fetch_dir" --force

  mkdir -p "results/${out_dir}"
  # Prefer dedicated restructure when available; else merge_local for qjl_default_* only
  if [[ -n "$restructure" && -f "$restructure" ]]; then
    echo "[merge] $PY $restructure  (filter to QJL inside script / or use merge_local)"
  fi
  # shellcheck disable=SC2086
  $MODAL_CMD run modal_app/sweep.py::merge_local \
    --input-dir "$fetch_dir" \
    --output "$out_dir" \
    --label-prefixes qjl_default
  echo "[merge] wrote results/${out_dir}/ (qjl_default only)"
}

launch_qjl() {
  local which="$1"
  local output="$2"
  activate_model "$which"

  local args=(--preset qjl --context-lengths "$CONTEXT_LENGTHS" --labels "$LABELS" --no-resume --output "$output")
  if [[ "$SYNC" == "1" ]]; then
    args+=(--sync)
  fi

  echo
  echo "============================================================"
  echo " Launching QJL ProdQJL re-run: model=$which"
  echo " contexts=$CONTEXT_LENGTHS  labels=$LABELS  output=$output"
  echo " mode=$([ "$SYNC" == "1" ] && echo sync || echo detach)"
  echo "============================================================"

  if [[ "$DETACH" == "1" && "$SYNC" != "1" ]]; then
    # shellcheck disable=SC2086
    $MODAL_CMD run --detach modal_app/sweep.py::main "${args[@]}"
  else
    # shellcheck disable=SC2086
    $MODAL_CMD run modal_app/sweep.py::main "${args[@]}"
  fi
}

echo "KVBench QJL ProdQJL re-run"
echo "  Affected: QJL attention estimator only (1 config × 3 ctx × models)."
echo "  Skipped: identity, TurboQuant, RocketKV (unaffected)."
echo

if [[ "$FETCH_ONLY" == "1" ]]; then
  if [[ "$MODEL" == "qwen3" || "$MODEL" == "both" ]]; then
    activate_model qwen3
    fetch_and_merge qwen3 "phase5_modal_qjl_prodqjl" "results/modal_volume_qjl_prodqjl_qwen3" ""
  fi
  if [[ "$MODEL" == "olmo2" || "$MODEL" == "both" ]]; then
    activate_model olmo2
    fetch_and_merge olmo2 "olmo2_phase5_qjl_prodqjl" "results/modal_volume_qjl_prodqjl_olmo2" ""
  fi
  echo "Done (fetch-only)."
  exit 0
fi

# Launch
if [[ "$MODEL" == "qwen3" || "$MODEL" == "both" ]]; then
  launch_qjl qwen3 "phase5_modal_qjl_prodqjl"
fi
if [[ "$MODEL" == "olmo2" || "$MODEL" == "both" ]]; then
  launch_qjl olmo2 "olmo2_phase5_qjl_prodqjl"
fi

echo
if [[ "$SYNC" == "1" ]]; then
  echo "Sync runs finished. Merged CSVs/JSON should be under results/*_qjl_prodqjl*/"
else
  echo "Detached jobs spawned. When Modal finishes:"
  echo "  bash scripts/rerun_qjl_prodqjl_fix.sh --fetch-only --model $MODEL"
  echo
  echo "Optional smoke before full grid:"
  echo "  bash scripts/rerun_qjl_prodqjl_fix.sh --smoke --model qwen3 --sync"
fi

echo
echo "After merge, regenerate paper plots that include QJL:"
echo "  MPLBACKEND=Agg .venv/bin/python docs/research_paper_writeup/scripts/generate_result_plots.py"
echo "(Point the plot loader at the new *_qjl_prodqjl bundles, or copy job JSONs into the"
echo " canonical phase5_modal_qjl / olmo2_phase5_qjl trees once you accept the numbers.)"
