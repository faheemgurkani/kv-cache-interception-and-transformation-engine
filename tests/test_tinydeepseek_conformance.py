"""Partial conformance for TinyDeepSeek (Gate C fails by design on expanded KV)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from framework.compatibility import (
    check_attention_gate,
    check_loader_state_gate,
    check_state_semantics_gate,
)
from framework.model import ModelLayer
from framework.model_capabilities import resolve_model_capabilities

TINYDEEPSEEK = Path(__file__).resolve().parent.parent / "models" / "tinydeepseek_0.5b"


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.mark.skipif(not TINYDEEPSEEK.exists(), reason="TinyDeepSeek-0.5B not downloaded")
def test_tinydeepseek_partial_conformance_expanded_kv():
    model_layer = ModelLayer(model_path=TINYDEEPSEEK, device=_device())
    config = model_layer.config
    caps = resolve_model_capabilities(config)

    assert caps.adapter_registered is True
    assert caps.native_latent_cache is True

    input_ids = model_layer.tokenize("Expanded KV conformance probe")
    with torch.no_grad():
        outputs = model_layer.model(input_ids, use_cache=True, return_dict=True)

    assert check_loader_state_gate(
        model_loaded=True,
        forward_ok=True,
        past_key_values=outputs.past_key_values,
    ).passed
    assert check_attention_gate(config).passed is True

    semantics = check_state_semantics_gate(config, outputs.past_key_values)
    assert semantics.passed is False
    assert "expanded" in semantics.detail.lower() or "latent" in semantics.detail.lower()
