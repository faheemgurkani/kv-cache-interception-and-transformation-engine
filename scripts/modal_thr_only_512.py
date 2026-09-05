"""Throughput-only identity@512 gen=64 — isolate SYSTEM leak."""

from __future__ import annotations

from pathlib import Path

import modal
import torch

from modal_app.image import MODEL_MOUNT, RESULTS_MOUNT, cuda_image, model_volume, results_volume
from modal_app.worker import app


@app.function(
    image=cuda_image,
    gpu="A10G",
    timeout=1800,
    volumes={MODEL_MOUNT: model_volume, RESULTS_MOUNT: results_volume},
)
def thr_only() -> dict:
    import traceback

    from compressors.identity import IdentityCompressor
    from eval.system.latency_throughput import evaluate_throughput
    from framework.model import ModelLayer
    from data.loader import build_long_context_ids, load_wikitext2

    ml = ModelLayer(model_path=str(Path(MODEL_MOUNT) / "gemma3_270m"))
    ids = build_long_context_ids(ml.tokenizer, load_wikitext2(), 512).to(ml.device)
    torch.cuda.reset_peak_memory_stats()
    try:
        m = evaluate_throughput(ml, ids, IdentityCompressor(), num_new_tokens=64)
        return {
            "status": "ok",
            "tps": m.tokens_per_second,
            "alloc_gb": torch.cuda.memory_allocated() / 1e9,
            "max_alloc_gb": torch.cuda.max_memory_allocated() / 1e9,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "error": str(exc),
            "traceback": traceback.format_exc()[-2000:],
            "alloc_gb": torch.cuda.memory_allocated() / 1e9,
            "max_alloc_gb": torch.cuda.max_memory_allocated() / 1e9,
        }


@app.local_entrypoint()
def main() -> None:
    print(thr_only.remote())
