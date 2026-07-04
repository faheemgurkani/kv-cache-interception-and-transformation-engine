"""Modal GPU workers — one A10G container per evaluation job."""

from __future__ import annotations

import json
import traceback
from datetime import UTC, datetime
from pathlib import Path

import modal
from transformers import AutoModelForCausalLM, AutoTokenizer

from modal_app.image import MODEL_MOUNT, RESULTS_MOUNT, cuda_image, model_volume, results_volume
from modal_app.job_spec import EvalJobSpec
from modal_app.settings import gpu_spec, secret_name, timeout_seconds

app = modal.App("kv-cache-engine-eval", image=cuda_image)

_HF_SECRET = modal.Secret.from_name(secret_name(), required_keys=["HF_TOKEN"])
_VOLUMES = {MODEL_MOUNT: model_volume, RESULTS_MOUNT: results_volume}


def _model_dir() -> Path:
    return Path(MODEL_MOUNT) / "qwen3_1.7b"


def _ensure_model_weights() -> Path:
    """Download Qwen3 once into the persistent model volume."""
    from framework.config import load_model_config

    model_volume.reload()
    model_path = _model_dir()
    marker = model_path / "config.json"
    if marker.exists():
        return model_path

    model_path.mkdir(parents=True, exist_ok=True)
    model_name = load_model_config().get("model_name", "Qwen/Qwen3-1.7B")
    AutoTokenizer.from_pretrained(model_name).save_pretrained(model_path)
    AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").save_pretrained(model_path)
    model_volume.commit()
    return model_path


@app.function(
    timeout=60 * 60,
    volumes=_VOLUMES,
    secrets=[_HF_SECRET],
)
def ensure_model() -> str:
    """One-time (or idempotent) model download into Modal Volume."""
    path = _ensure_model_weights()
    return str(path)


@app.function(
    gpu=gpu_spec(),
    timeout=timeout_seconds(),
    volumes=_VOLUMES,
    secrets=[_HF_SECRET],
)
def eval_worker(job: dict) -> dict:
    """Run one (compressor × context_length) evaluation on CUDA."""
    spec = EvalJobSpec(**job)
    started_at = datetime.now(UTC).isoformat()

    try:
        model_volume.reload()
        results_volume.reload()

        from compressors.registry import get_compressor
        from eval.runner import EvaluationRunner
        from framework.config import load_eval_config, load_model_config
        from framework.model import ModelLayer

        model_path = _ensure_model_weights()

        kwargs: dict = {}
        if spec.bitwidth is not None:
            kwargs["bitwidth"] = spec.bitwidth
        if spec.stage is not None:
            kwargs["stage"] = spec.stage
        compressor = get_compressor(spec.compressor, **kwargs)

        runner = EvaluationRunner(
            model_layer=ModelLayer(model_path=model_path),
            compressor=compressor,
            eval_config=load_eval_config(),
            model_config=load_model_config(),
        )
        result = runner.run(
            spec.context_length,
            run_perplexity=not spec.skip_perplexity,
            run_throughput=not spec.skip_throughput,
            include_baselines=True,
        )

        payload = result.to_dict()
        payload["label"] = spec.label
        payload["job"] = spec.to_dict()
        payload["started_at"] = started_at
        payload["finished_at"] = datetime.now(UTC).isoformat()
        payload["status"] = "ok"

        out_path = Path(RESULTS_MOUNT) / f"{spec.result_stem}.json"
        out_path.write_text(json.dumps(payload, indent=2))
        results_volume.commit()
        return payload

    except Exception as exc:  # noqa: BLE001 — persist failure for sweep resume
        error_payload = {
            "label": spec.label,
            "job": spec.to_dict(),
            "started_at": started_at,
            "finished_at": datetime.now(UTC).isoformat(),
            "status": "error",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        out_path = Path(RESULTS_MOUNT) / f"{spec.result_stem}.error.json"
        out_path.write_text(json.dumps(error_payload, indent=2))
        results_volume.commit()
        return error_payload


@app.function(volumes={RESULTS_MOUNT: results_volume})
def list_completed_jobs() -> list[str]:
    """Return result stems already stored on the results volume."""
    results_volume.reload()
    stems: list[str] = []
    for path in Path(RESULTS_MOUNT).glob("*.json"):
        if path.name.endswith(".error.json"):
            continue
        stems.append(path.stem)
    return sorted(stems)
