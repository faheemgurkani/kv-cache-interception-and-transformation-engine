"""YAML configuration loaders."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_config_path(explicit: Path | str | None, env_key: str, default_relative: str) -> Path:
    if explicit is not None:
        path = Path(explicit)
        return path if path.is_absolute() else PROJECT_ROOT / path
    raw = os.environ.get(env_key, "").strip()
    if raw:
        path = Path(raw)
        return path if path.is_absolute() else PROJECT_ROOT / path
    return PROJECT_ROOT / default_relative


def load_model_config(config_path: Path | str | None = None) -> dict:
    path = _resolve_config_path(config_path, "KV_MODEL_CONFIG", "configs/model.yaml")
    with path.open() as f:
        return yaml.safe_load(f)


def load_eval_config(config_path: Path | str | None = None) -> dict:
    path = _resolve_config_path(config_path, "KV_EVAL_CONFIG", "configs/eval.yaml")
    with path.open() as f:
        return yaml.safe_load(f)
