#!/usr/bin/env bash
# Launch RocketKV Modal sweep (3 configs × 3 context lengths = 9 jobs).
set -euo pipefail
cd "$(dirname "$0")/.."

PRESET=rocketkv \
OUTPUT="${OUTPUT:-phase5_modal_rocketkv}" \
CONTEXT_LENGTHS="${CONTEXT_LENGTHS:-128,256,512}" \
LABELS="${LABELS:-}" \
bash scripts/modal_run_sweep.sh
