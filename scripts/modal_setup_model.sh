#!/usr/bin/env bash
# Download Qwen3-1.7B into Modal Volume (run once).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true
modal run modal_app/worker.py::ensure_model
