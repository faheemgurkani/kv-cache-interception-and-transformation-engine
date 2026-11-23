#!/usr/bin/env bash
# Launch QJL Modal sweep (1 config × 3 context lengths = 3 jobs).
set -euo pipefail
"$(dirname "$0")/../.venv/bin/python" "$(dirname "$0")/modal_assert_profile.py" || exit 1
cd "$(dirname "$0")/.."

PRESET=qjl \
OUTPUT="${OUTPUT:-olmo2_phase5_qjl}" \
CONTEXT_LENGTHS="${CONTEXT_LENGTHS:-128,256,512}" \
LABELS="${LABELS:-}" \
NO_RESUME="${NO_RESUME:-1}" \
bash scripts/modal_run_sweep.sh
