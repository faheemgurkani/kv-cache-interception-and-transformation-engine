#!/usr/bin/env bash
# Pull evaluation JSON payloads from Modal results volume to results/modal_volume/
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-results/modal_volume}"
mkdir -p "$OUT"
modal volume get kv-engine-results / "$OUT" --force
echo "Downloaded Modal results to $OUT"
echo "Merge locally: modal run modal_app/sweep.py::merge_local --input-dir $OUT"
