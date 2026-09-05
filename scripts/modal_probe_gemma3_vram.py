"""One-shot Modal probe: code sync + model VRAM for Gemma3 full-eval."""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import modal
import torch

from modal_app.image import MODEL_MOUNT, RESULTS_MOUNT, cuda_image, model_volume, results_volume
from modal_app.worker import app


@app.function(
    image=cuda_image,
    gpu="A10G",
    timeout=600,
    volumes={MODEL_MOUNT: model_volume, RESULTS_MOUNT: results_volume},
)
def probe_gemma3() -> dict:
    import framework.kv_engine as kve
    from framework.model import ModelLayer

    src = inspect.getsource(kve.KVCacheEngine._refresh_decode_prefix)
    model_root = Path(MODEL_MOUNT)
    candidates = [
        model_root / "gemma3_270m",
        model_root / "models" / "gemma3_270m",
        Path("/root/kv-cache-engine/models/gemma3_270m"),
    ]
    model_path = next((p for p in candidates if p.exists()), candidates[0])
    ml = ModelLayer(model_path=str(model_path))
    nparams = sum(p.numel() for p in ml.model.parameters())
    return {
        "has_noop_refresh": "Always rebuild from compressed" in src,
        "has_reload_comment": "clean VRAM slate" in Path("/root/kv-cache-engine/eval/runner.py").read_text(),
        "refresh_head": src.splitlines()[0:6],
        "model_path": str(ml.model_path),
        "dtype": str(ml.torch_dtype),
        "nparams": nparams,
        "alloc_gb": torch.cuda.memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "mount_entries": sorted(p.name for p in model_root.iterdir())[:30] if model_root.exists() else [],
        "KV_MODEL_CONFIG": os.environ.get("KV_MODEL_CONFIG"),
        "KV_MODAL_CONFIG": os.environ.get("KV_MODAL_CONFIG"),
    }


@app.local_entrypoint()
def main() -> None:
    print(probe_gemma3.remote())
