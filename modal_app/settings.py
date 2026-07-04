"""Load Modal runtime settings from configs/modal.yaml."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

DEFAULT_MODAL_CONFIG_NAME = Path("configs") / "modal.yaml"


def project_root() -> Path:
    """Resolve repo root in local dev and Modal container mounts."""
    for key in ("KV_PROJECT_ROOT", "PYTHONPATH"):
        raw = os.environ.get(key, "").strip()
        if not raw:
            continue
        for part in raw.split(":"):
            candidate = Path(part)
            if (candidate / DEFAULT_MODAL_CONFIG_NAME).exists():
                return candidate

    candidate = Path(__file__).resolve().parent.parent
    if (candidate / DEFAULT_MODAL_CONFIG_NAME).exists():
        return candidate

    code_mount = Path("/root/kv-cache-engine")
    if (code_mount / DEFAULT_MODAL_CONFIG_NAME).exists():
        return code_mount

    return candidate


PROJECT_ROOT = project_root()
DEFAULT_MODAL_CONFIG = PROJECT_ROOT / DEFAULT_MODAL_CONFIG_NAME


@lru_cache(maxsize=1)
def load_modal_config(path: Path | str | None = None) -> dict:
    config_path = Path(path) if path else DEFAULT_MODAL_CONFIG
    with config_path.open() as handle:
        return yaml.safe_load(handle)


def gpu_spec(config: dict | None = None) -> str | list[str]:
    """Return Modal gpu= argument with fallbacks."""
    cfg = config or load_modal_config()
    fallbacks = cfg.get("gpu_fallbacks") or [cfg.get("gpu", "A10G")]
    if len(fallbacks) == 1:
        return fallbacks[0]
    return fallbacks


def timeout_seconds(config: dict | None = None) -> int:
    cfg = config or load_modal_config()
    hours = int(cfg.get("timeout_hours", 4))
    return hours * 60 * 60


def secret_name(config: dict | None = None) -> str:
    cfg = config or load_modal_config()
    return cfg.get("secrets", {}).get("huggingface", "huggingface-secret")


def volume_names(config: dict | None = None) -> tuple[str, str]:
    cfg = config or load_modal_config()
    volumes = cfg.get("volumes", {})
    return (
        volumes.get("model", "kv-engine-qwen3"),
        volumes.get("results", "kv-engine-results"),
    )
