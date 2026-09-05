"""Step-through EvaluationRunner with VRAM logs (identity@512)."""

from __future__ import annotations

import gc
from pathlib import Path

import modal
import torch

from modal_app.image import MODEL_MOUNT, RESULTS_MOUNT, cuda_image, model_volume, results_volume
from modal_app.worker import app


def _mem(tag: str) -> dict:
    torch.cuda.synchronize()
    return {
        "tag": tag,
        "alloc_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
        "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 3),
        "max_alloc_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
    }


@app.function(
    image=cuda_image,
    gpu="A10G",
    timeout=3600,
    volumes={MODEL_MOUNT: model_volume, RESULTS_MOUNT: results_volume},
)
def profile_runner() -> dict:
    from compressors.registry import get_compressor
    from eval.behavior import evaluate_behavior
    from eval.fidelity import evaluate_fidelity
    from eval.system import evaluate_system
    from framework.attention_patches import ensure_vanilla_attention
    from framework.compatibility_probe import run_compatibility_probe
    from framework.config import load_eval_config, load_model_config
    from framework.model import ModelLayer
    from framework.model_capabilities import load_compatibility_manifest
    from data.loader import build_long_context_ids, load_wikitext2

    logs = []
    model_path = str(Path(MODEL_MOUNT) / "gemma3_270m")
    model_config = load_model_config()
    eval_config = load_eval_config()
    ml = ModelLayer(model_path=model_path)
    comp = get_compressor("identity")
    logs.append(_mem("load"))

    try:
        ensure_vanilla_attention(ml.model)
    except AttributeError:
        pass
    logs.append(_mem("vanilla_attn"))

    dataset = load_wikitext2()
    input_ids = build_long_context_ids(ml.tokenizer, dataset, 512).to(ml.device)
    logs.append(_mem("context"))

    probe = run_compatibility_probe(
        ml,
        manifest=load_compatibility_manifest(ml.config, yaml_config=model_config),
    )
    logs.append({**_mem("probe"), "gates": getattr(probe, "gates", None)})

    torch.cuda.reset_peak_memory_stats()
    fidelity = evaluate_fidelity(ml, input_ids, comp)
    logs.append(_mem("fidelity"))
    gc.collect()
    torch.cuda.empty_cache()
    logs.append(_mem("fidelity_free"))

    behavior = evaluate_behavior(
        ml,
        input_ids,
        comp,
        run_task_quality=True,
        run_retrieval=True,
        run_reasoning=False,
        run_instruction_following=True,
        include_baseline=True,
        perplexity_stride=int(eval_config.get("perplexity_stride", 512)),
        context_length=512,
    )
    logs.append({**_mem("behavior"), "ppl": behavior.perplexity})

    # reload
    path, device, dtype = ml.model_path, ml.device, ml.torch_dtype
    try:
        ml.model.to("cpu")
    except Exception as exc:  # noqa: BLE001
        logs.append({"to_cpu_error": str(exc)})
    del ml
    gc.collect()
    torch.cuda.empty_cache()
    logs.append(_mem("after_del"))

    ml2 = ModelLayer(model_path=path, device=device, torch_dtype=dtype)
    logs.append(_mem("reload"))

    try:
        system = evaluate_system(
            ml2,
            input_ids,
            get_compressor("identity"),
            run_throughput=True,
            run_peak_memory=False,
            run_gpu_utilization=False,
            include_baseline=True,
            num_new_tokens=64,
            actual_kv_memory_bytes=fidelity.memory.compressed_bytes,
            uncompressed_kv_memory_bytes=fidelity.memory.uncompressed_bytes,
        )
        logs.append({**_mem("system_ok"), "tps": system.throughput.tokens_per_second if system.throughput else None})
        return {"status": "ok", "logs": logs}
    except Exception as exc:  # noqa: BLE001
        import traceback

        logs.append({**_mem("system_err"), "error": str(exc)})
        return {"status": "error", "logs": logs, "traceback": traceback.format_exc()[-2500:]}


@app.local_entrypoint()
def main() -> None:
    out = profile_runner.remote()
    print("status", out["status"])
    for row in out["logs"]:
        print(row)
    if out["status"] != "ok":
        print(out.get("traceback", ""))
