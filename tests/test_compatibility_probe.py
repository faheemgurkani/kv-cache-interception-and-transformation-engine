"""Tests for unified compatibility probe (§29)."""

from pathlib import Path

import pytest

from framework.compatibility_probe import run_compatibility_probe
from framework.config import load_model_config
from framework.model import ModelLayer
from framework.model_capabilities import load_compatibility_manifest, validate_manifest, resolve_model_capabilities

OLMO2 = Path(__file__).resolve().parent.parent / "models" / "olmo2_1b"


@pytest.mark.skipif(not OLMO2.exists(), reason="OLMo2 not downloaded")
def test_compatibility_probe_runs_all_gates():
    import torch

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_layer = ModelLayer(model_path=OLMO2, device=device)
    yaml_cfg = load_model_config(Path(__file__).resolve().parent.parent / "configs" / "model_olmo2_1b.yaml")
    manifest = load_compatibility_manifest(model_layer.config, yaml_config=yaml_cfg)
    probe = run_compatibility_probe(model_layer, manifest=manifest)
    assert probe.forward_ok is True
    assert set(probe.gates) == {"loader_state", "attention", "state_semantics"}
    assert probe.all_gates_passed is True
    assert probe.metadata["compatibility_gates"]["attention"]["passed"] is True


def test_manifest_validation_rejects_mismatch():
    config = type("Cfg", (), {"model_type": "olmo2", "num_attention_heads": 16, "num_key_value_heads": 16})()
    caps = resolve_model_capabilities(config)
    manifest = load_compatibility_manifest(config)
    validate_manifest(manifest, caps, config)
    bad = {**manifest, "architecture": {**manifest["architecture"], "q_heads": 99}}  # type: ignore[index]
    with pytest.raises(ValueError, match="q_heads"):
        validate_manifest(bad, caps, config)
