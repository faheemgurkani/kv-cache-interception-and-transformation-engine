"""Tests for declarative compatibility manifests in model YAML (§30)."""

from pathlib import Path

import pytest
import yaml

from framework.config import PROJECT_ROOT
from framework.model import ModelLayer
from framework.model_capabilities import load_compatibility_manifest, resolve_model_capabilities, validate_manifest

SHORTLIST_CONFIGS = [
    "model_olmo2_1b.yaml",
    "model_qwen3_0.6b.yaml",
    "model_gemma3_270m.yaml",
    "model_tinydeepseek_0.5b.yaml",
    "model_falcon_h1_0.5b.yaml",
]


@pytest.mark.parametrize("config_name", SHORTLIST_CONFIGS)
def test_shortlist_yaml_manifest_has_required_sections(config_name: str):
    path = PROJECT_ROOT / "configs" / config_name
    with path.open() as handle:
        yaml_cfg = yaml.safe_load(handle)
    assert "compatibility" in yaml_cfg
    compat = yaml_cfg["compatibility"]
    assert "architecture" in compat
    assert "attention" in compat
    assert "evaluation" in compat


@pytest.mark.parametrize("config_name", SHORTLIST_CONFIGS)
def test_shortlist_manifest_matches_model_when_downloaded(config_name: str):
    path = PROJECT_ROOT / "configs" / config_name
    with path.open() as handle:
        yaml_cfg = yaml.safe_load(handle)
    model_path = PROJECT_ROOT / yaml_cfg["local_path"]
    if not model_path.exists():
        pytest.skip(f"{model_path.name} not downloaded")
    import torch

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_layer = ModelLayer(model_path=model_path, device=device)
    caps = resolve_model_capabilities(model_layer.config)
    manifest = load_compatibility_manifest(model_layer.config, yaml_config=yaml_cfg)
    validate_manifest(manifest, caps, model_layer.config)
