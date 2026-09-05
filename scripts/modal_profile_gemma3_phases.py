"""Profile VRAM after fidelity/behavior and after model reload on Modal A10G."""

from __future__ import annotations

import gc
from pathlib import Path

import modal
import torch

from modal_app.image import MODEL_MOUNT, RESULTS_MOUNT, cuda_image, model_volume, results_volume
from modal_app.worker import app


def _mem() -> dict:
    if not torch.cuda.is_available():
        return {"alloc_gb": 0.0, "reserved_gb": 0.0}
    torch.cuda.synchronize()
    return {
        "alloc_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
        "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 3),
        "max_alloc_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
    }


@app.function(
    image=cuda_image,
    gpu="A10G",
    timeout=1800,
    volumes={MODEL_MOUNT: model_volume, RESULTS_MOUNT: results_volume},
)
def profile_phases() -> dict:
    from compressors.identity import IdentityCompressor
    from eval.behavior import evaluate_behavior
    from eval.fidelity import evaluate_fidelity
    from eval.system.latency_throughput import evaluate_throughput
    from framework.model import ModelLayer
    from data.loader import build_long_context_ids, load_wikitext2

    out: dict = {"steps": []}
    model_path = str(Path(MODEL_MOUNT) / "gemma3_270m")
    ml = ModelLayer(model_path=model_path)
    out["steps"].append({"after_load": _mem(), "dtype": str(ml.torch_dtype)})

    text = load_wikitext2()
    input_ids = build_long_context_ids(ml.tokenizer, text, 512).to(ml.device)
    comp = IdentityCompressor()

    torch.cuda.reset_peak_memory_stats()
    fidelity = evaluate_fidelity(ml, input_ids, comp)
    out["steps"].append({"after_fidelity": _mem(), "rep": fidelity.representation.rmse if hasattr(fidelity.representation, "rmse") else None})

    gc.collect()
    torch.cuda.empty_cache()
    out["steps"].append({"after_fidelity_free": _mem()})

    behavior = evaluate_behavior(
        ml,
        input_ids,
        comp,
        run_task_quality=True,
        run_retrieval=True,
        run_reasoning=False,
        run_instruction_following=True,
        include_baseline=True,
        perplexity_stride=512,
        context_length=512,
    )
    out["steps"].append(
        {
            "after_behavior": _mem(),
            "ppl": behavior.perplexity,
        }
    )

    # Mimic runner reload
    path = ml.model_path
    device = ml.device
    dtype = ml.torch_dtype
    try:
        ml.model.to("cpu")
    except Exception as exc:  # noqa: BLE001
        out["to_cpu_error"] = str(exc)
    del ml
    gc.collect()
    torch.cuda.empty_cache()
    out["steps"].append({"after_del_model": _mem()})

    ml2 = ModelLayer(model_path=path, device=device, torch_dtype=dtype)
    out["steps"].append({"after_reload": _mem()})

    try:
        thr = evaluate_throughput(ml2, input_ids, IdentityCompressor(), num_new_tokens=8)
        out["steps"].append({"after_throughput": _mem(), "tps": thr.tokens_per_second})
        out["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        out["status"] = "error"
        out["error"] = str(exc)
        out["steps"].append({"after_throughput_error": _mem()})
    return out


@app.local_entrypoint()
def main() -> None:
    print(profile_phases.remote())
