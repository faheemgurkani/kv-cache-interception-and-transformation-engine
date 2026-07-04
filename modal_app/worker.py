"""Modal GPU worker — runs one evaluation job per container."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import modal
from modal_app.image import (
    DEFAULT_GPU,
    DEFAULT_TIMEOUT_SEC,
    MODEL_MOUNT,
    RESULTS_MOUNT,
    cuda_image,
    model_volume,
    results_volume,
)
from modal_app.job_spec import EvalJobSpec

app = modal.App("kv-cache-engine-eval", image=cuda_image)


@app.function(
    gpu=DEFAULT_GPU,
    timeout=DEFAULT_TIMEOUT_SEC,
    volumes={
        MODEL_MOUNT: model_volume,
        RESULTS_MOUNT: results_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def eval_worker(job: dict) -> dict:
    """Run a single (compressor, context_length) evaluation on CUDA."""
    from compressors.registry import get_compressor
    from eval.runner import EvaluationRunner
    from framework.config import load_eval_config, load_model_config
    from framework.model import ModelLayer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = EvalJobSpec(**job)
    model_config = load_model_config()
    eval_config = load_eval_config()

    model_path = Path(MODEL_MOUNT) / "qwen3_1.7b"
    if not model_path.exists():
        model_path.mkdir(parents=True, exist_ok=True)
        model_name = model_config.get("model_name", "Qwen/Qwen3-1.7B")
        AutoTokenizer.from_pretrained(model_name).save_pretrained(model_path)
        AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").save_pretrained(model_path)
        model_volume.commit()

    kwargs: dict = {}
    if spec.bitwidth is not None:
        kwargs["bitwidth"] = spec.bitwidth
    if spec.stage is not None:
        kwargs["stage"] = spec.stage
    compressor = get_compressor(spec.compressor, **kwargs)

    runner = EvaluationRunner(
        model_layer=ModelLayer(model_path=model_path),
        compressor=compressor,
        eval_config=eval_config,
        model_config=model_config,
    )
    result = runner.run(
        spec.context_length,
        include_baselines=True,
    )
    payload = result.to_dict()
    payload["label"] = spec.label
    payload["finished_at"] = datetime.now(UTC).isoformat()

    out_dir = Path(RESULTS_MOUNT)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{spec.label}_ctx{spec.context_length}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    results_volume.commit()
    return payload
