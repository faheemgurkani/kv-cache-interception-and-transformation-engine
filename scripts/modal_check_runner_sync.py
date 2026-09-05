"""Confirm Modal sees the runner peak-memory override removal."""

from __future__ import annotations

from pathlib import Path

import modal

from modal_app.image import cuda_image
from modal_app.worker import app


@app.function(image=cuda_image, timeout=120)
def check_runner() -> dict:
    text = Path("/root/kv-cache-engine/eval/runner.py").read_text()
    return {
        "has_do_not_force": "Do NOT force run_peak_memory" in text,
        "has_old_override": "run_peak_memory = True\n            run_gpu_utilization = True" in text,
        "reload_present": "clean VRAM slate" in text,
    }


@app.local_entrypoint()
def main() -> None:
    print(check_runner.remote())
