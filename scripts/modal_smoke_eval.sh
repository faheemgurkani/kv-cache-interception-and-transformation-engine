#!/usr/bin/env bash
# Modal smoke eval: one job @ ctx=128 to verify preset wiring before full sweeps.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

PRESET="${1:-qjl}"
LABEL="${2:-}"
OUTPUT="${OUTPUT:-phase5_modal_${PRESET}_smoke}"

case "$PRESET" in
  qjl)
    LABEL="${LABEL:-qjl_default}"
    ;;
  rocketkv)
    LABEL="${LABEL:-rocketkv_r512}"
    ;;
  baseline)
    LABEL="${LABEL:-identity_baseline}"
    ;;
  turboquant)
    LABEL="${LABEL:-tq_full_b4}"
    ;;
  *)
    echo "Unknown preset: $PRESET (use qjl|rocketkv|baseline|turboquant)" >&2
    exit 1
    ;;
esac

modal run modal_app/sweep.py::main \
  --preset "$PRESET" \
  --context-lengths 128 \
  --labels "$LABEL" \
  --no-resume \
  --sync \
  --output "$OUTPUT"
