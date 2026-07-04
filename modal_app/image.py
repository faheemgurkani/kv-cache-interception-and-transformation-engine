"""Modal container image and persistent storage mounts."""

from __future__ import annotations

from pathlib import Path

import modal

from modal_app.settings import load_modal_config, volume_names

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_MOUNT = "/models"
RESULTS_MOUNT = "/results"
HF_CACHE = MODEL_MOUNT
CODE_MOUNT = "/root/kv-cache-engine"

_modal_cfg = load_modal_config()
_model_vol_name, _results_vol_name = volume_names(_modal_cfg)

cuda_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install_from_requirements(str(PROJECT_ROOT / "requirements-modal.txt"))
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
            ".env",
        ],
    )
    .env(
        {
            "KV_EVAL_DEVICE": "cuda",
            "HF_HUB_CACHE": HF_CACHE,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
            "PYTHONPATH": CODE_MOUNT,
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

model_volume = modal.Volume.from_name(_model_vol_name, create_if_missing=True)
results_volume = modal.Volume.from_name(_results_vol_name, create_if_missing=True)
