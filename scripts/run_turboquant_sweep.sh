#!/usr/bin/env bash
# Full TurboQuant evaluation sweep per docs/EVALUATION_PLAN.md
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

LOG_DIR="$ROOT/results/sweep_logs"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
MASTER_LOG="$LOG_DIR/sweep_${STAMP}.log"

run_one() {
  local label="$1"
  shift
  echo "=== [$label] $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$MASTER_LOG"
  python scripts/run_eval.py "$@" 2>&1 | tee -a "$LOG_DIR/${label}.log" | tee -a "$MASTER_LOG"
  echo "=== [$label] done ===" | tee -a "$MASTER_LOG"
}

COMMON=(--all-context-lengths --include-baselines)

run_one "identity_baseline" --compressor identity --output "sweep_identity_${STAMP}" "${COMMON[@]}"
run_one "tq_full_b2" --compressor turboquant --stage full --bitwidth 2 --output "sweep_tq_full_b2_${STAMP}" "${COMMON[@]}"
run_one "tq_full_b3" --compressor turboquant --stage full --bitwidth 3 --output "sweep_tq_full_b3_${STAMP}" "${COMMON[@]}"
run_one "tq_full_b4" --compressor turboquant --stage full --bitwidth 4 --output "sweep_tq_full_b4_${STAMP}" "${COMMON[@]}"
run_one "tq_mse_b4" --compressor turboquant --stage wht_quant --bitwidth 4 --output "sweep_tq_mse_b4_${STAMP}" "${COMMON[@]}"

echo "SWEEP COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$MASTER_LOG"
