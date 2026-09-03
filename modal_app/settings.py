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


@lru_cache(maxsize=8)
def _load_modal_config_cached(config_path: str) -> dict:
    with Path(config_path).open() as handle:
        return yaml.safe_load(handle)


def load_modal_config(path: Path | str | None = None) -> dict:
    if path is None:
        raw = os.environ.get("KV_MODAL_CONFIG", "").strip()
        path = raw or DEFAULT_MODAL_CONFIG
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = project_root() / config_path
    return _load_modal_config_cached(str(config_path))


def gpu_spec(config: dict | None = None) -> str | list[str]:
    """Return Modal ``gpu=`` argument with fallbacks (see Modal GPU guide)."""
    cfg = config or load_modal_config()
    fallbacks = cfg.get("gpu_fallbacks") or [cfg.get("gpu", "a10g")]
    if len(fallbacks) == 1:
        return fallbacks[0]
    return fallbacks


def hardware_config(config: dict | None = None) -> dict:
    """Phase 10 hardware collection policy from ``configs/modal.yaml``."""
    cfg = config or load_modal_config()
    return cfg.get("hardware") or {}


def reference_gpu_label(config: dict | None = None) -> str:
    return hardware_config(config).get("reference_gpu", "NVIDIA A10G")


def collect_hardware_metrics(config: dict | None = None) -> bool:
    hw = hardware_config(config)
    return bool(hw.get("collect_peak_memory", True) or hw.get("collect_gpu_utilization", True))


def modal_runtime_env(config: dict | None = None) -> dict[str, str]:
    """Environment stamped into the Modal CUDA image for hardware-aware eval."""
    cfg = config or load_modal_config()
    primary_gpu = cfg.get("gpu_fallbacks", [cfg.get("gpu", "a10g")])[0]
    env = {
        "KV_EXECUTION_PLATFORM": "modal",
        "KV_HARDWARE_PROFILE": reference_gpu_label(cfg),
        "MODAL_GPU_REQUEST": str(primary_gpu),
    }
    if collect_hardware_metrics(cfg):
        env["KV_COLLECT_HARDWARE_METRICS"] = "1"
    model_cfg = os.environ.get("KV_MODEL_CONFIG", "").strip()
    if model_cfg:
        env["KV_MODEL_CONFIG"] = model_cfg
    modal_cfg = os.environ.get("KV_MODAL_CONFIG", "").strip()
    if modal_cfg:
        env["KV_MODAL_CONFIG"] = modal_cfg
    from eval.reproducibility.manifest import collect_git_sha

    git_sha = collect_git_sha(cwd=project_root())
    if git_sha:
        env["KV_GIT_SHA"] = git_sha
    return env


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
