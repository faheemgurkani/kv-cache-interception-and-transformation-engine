"""Full EvaluationRunner identity@512 on Modal A10G (mirrors worker)."""

from __future__ import annotations

from pathlib import Path

import modal
import torch

from modal_app.image import MODEL_MOUNT, RESULTS_MOUNT, cuda_image, model_volume, results_volume
from modal_app.worker import app


@app.function(
    image=cuda_image,
    gpu="A10G",
    timeout=3600,
    volumes={MODEL_MOUNT: model_volume, RESULTS_MOUNT: results_volume},
)
def run_identity_full() -> dict:
    import gc
    import traceback

    from compressors.registry import get_compressor
    from eval.runner import EvaluationRunner
    from framework.config import load_eval_config, load_model_config
    from framework.model import ModelLayer

    model_path = str(Path(MODEL_MOUNT) / "gemma3_270m")
    compressor = get_compressor("identity")
    model_layer = ModelLayer(model_path=model_path)
    runner = EvaluationRunner(
        model_layer=model_layer,
        compressor=compressor,
        eval_config=load_eval_config(),
        model_config=load_model_config(),
    )
    try:
        result = runner.run(
            512,
            run_perplexity=True,
            run_throughput=True,
            run_retrieval=True,
            run_instruction_following=True,
            run_reasoning=False,
            run_peak_memory=False,
            run_gpu_utilization=False,
            run_kernel_cost=False,
            run_memory_bandwidth=False,
            collect_hardware_metrics=True,
            include_baselines=True,
            generated_tokens=64,
        )
        payload = result.to_dict()
        payload["status"] = "ok"
        payload["mem"] = {
            "alloc_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "max_alloc_gb": torch.cuda.max_memory_allocated() / 1e9,
        }
        return payload
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "mem": {
                "alloc_gb": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else None,
                "reserved_gb": torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else None,
                "max_alloc_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None,
            },
        }
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@app.local_entrypoint()
def main() -> None:
    out = run_identity_full.remote()
    print("status", out.get("status"))
    print("mem", out.get("mem"))
    if out.get("status") != "ok":
        print("error", out.get("error"))
        print(out.get("traceback", "")[-2000:])
    else:
        print("keys", sorted(out.keys())[:30])
        print("compressor", out.get("compressor"), "ctx", out.get("context_length"))
