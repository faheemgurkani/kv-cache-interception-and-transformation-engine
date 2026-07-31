#!/usr/bin/env bash
# Pull evaluation JSON payloads from Modal results volume to results/modal_volume_olmo2/
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-results/modal_volume_olmo2}"
mkdir -p "$OUT"

# Resolve results volume name from configs/modal.yaml (default: kv-engine-results-olmo2)
VOLUME="$(python - <<'PY'
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path("configs/modal.yaml").read_text())
print(cfg.get("volumes", {}).get("results", "kv-engine-results-olmo2"))
PY
)"

modal volume get "$VOLUME" / "$OUT" --force
echo "Downloaded Modal results from $VOLUME to $OUT"
echo "Merge locally: modal run modal_app/sweep.py::merge_local --input-dir $OUT"
