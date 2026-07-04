"""Modal container image and shared paths."""

from __future__ import annotations

from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_MOUNT = "/models"
RESULTS_MOUNT = "/results"
HF_CACHE = MODEL_MOUNT
CODE_MOUNT = "/root/kv-cache-engine"

cuda_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install_from_requirements(str(PROJECT_ROOT / "requirements.txt"))
    .pip_install("fast-hadamard-transform", "modal")
    .add_local_dir(
        str(PROJECT_ROOT),
        remote_path=CODE_MOUNT,
        ignore=[
            ".venv",
            ".git",
            "results",
            "plots",
            ".cache",
            "models",
            "__pycache__",
            ".pytest_cache",
        ],
    )
    .env(
        {
            "KV_EVAL_DEVICE": "cuda",
            "HF_HUB_CACHE": HF_CACHE,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
            "PYTHONPATH": CODE_MOUNT,
        }
    )
)

model_volume = modal.Volume.from_name("kv-engine-qwen3", create_if_missing=True)
results_volume = modal.Volume.from_name("kv-engine-results", create_if_missing=True)

DEFAULT_GPU = "A10G"
DEFAULT_TIMEOUT_SEC = 4 * 60 * 60
