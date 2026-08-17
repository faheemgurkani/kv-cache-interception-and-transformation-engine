"""Conformance tests for supported model families (Phase 0 / WP1)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from compressors.identity import IdentityCompressor
from framework.compatibility import (
    check_attention_gate,
    check_loader_state_gate,
    check_state_semantics_gate,
)
from framework.kv_cache import get_cache_size_bytes, iter_layer_kv
from framework.model import ModelLayer
from framework.model_adapter import load_attention_ops, project_qkv, pre_attention_hidden
from framework.model_capabilities import resolve_model_capabilities
from framework.rope import build_rope_context
from framework.state_interface import iter_layer_states, total_state_bytes

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ConformanceSpec:
    name: str
    model_path: Path
    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int


CONFORMANCE_MODELS = [
    ConformanceSpec(
        name="olmo2_1b",
        model_path=PROJECT_ROOT / "models" / "olmo2_1b",
        num_layers=16,
        num_q_heads=16,
        num_kv_heads=16,
        head_dim=128,
    ),
    ConformanceSpec(
        name="qwen3_0.6b",
        model_path=PROJECT_ROOT / "models" / "qwen3_0.6b",
        num_layers=28,
        num_q_heads=16,
        num_kv_heads=8,
        head_dim=128,
    ),
    ConformanceSpec(
        name="qwen3_1.7b",
        model_path=PROJECT_ROOT / "models" / "legacy" / "qwen3_1.7b",
        num_layers=28,
        num_q_heads=16,
        num_kv_heads=8,
        head_dim=128,
    ),
]


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture(params=CONFORMANCE_MODELS, ids=lambda spec: spec.name)
def conformance_model(request):
    spec: ConformanceSpec = request.param
    if not spec.model_path.exists():
        pytest.skip(f"Model not downloaded: {spec.model_path}")
    return spec, ModelLayer(model_path=spec.model_path, device=_device())


def test_adapter_conformance(conformance_model):
    spec, model_layer = conformance_model
    config = model_layer.config
    caps = resolve_model_capabilities(config)

    assert model_layer.attn_implementation == "eager"
    assert config.use_cache is True
    assert caps.adapter_registered is True

    input_ids = model_layer.tokenize("The quick brown fox jumps over the lazy dog")
    with torch.no_grad():
        outputs = model_layer.model(
            input_ids,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

    loader_gate = check_loader_state_gate(
        model_loaded=True,
        forward_ok=True,
        past_key_values=outputs.past_key_values,
    )
    assert loader_gate.passed is True

    attention_gate = check_attention_gate(config)
    assert attention_gate.passed is True

    semantics_gate = check_state_semantics_gate(config, outputs.past_key_values)
    assert semantics_gate.passed is True

    layers = list(iter_layer_kv(outputs.past_key_values))
    assert len(layers) == spec.num_layers

    states = list(iter_layer_states(outputs.past_key_values, capabilities=caps))
    assert len(states) == spec.num_layers

    key, value = layers[0]
    assert key.ndim == 4 and value.ndim == 4
    assert key.shape[1] == spec.num_kv_heads
    assert key.shape[3] == spec.head_dim
    assert value.shape == key.shape
    assert key.shape[2] == input_ids.shape[1]

    ops = load_attention_ops(config)
    rope_ctx = build_rope_context(
        model_layer.model,
        outputs.hidden_states[0],
        torch.arange(input_ids.shape[1], device=model_layer.device).unsqueeze(0),
        config=config,
    )
    layer = model_layer.model.model.layers[0]
    attn = layer.self_attn
    hidden = pre_attention_hidden(layer, outputs.hidden_states[0], ops)
    query, key_proj, _value = project_qkv(attn, hidden, ops)
    cos, sin = rope_ctx.get_rope(0)
    query_rope, _ = ops.apply_rotary_pos_emb(query, key_proj, cos, sin)
    assert query_rope.shape[1] == spec.num_q_heads

    compressor = IdentityCompressor()
    compressed = compressor.compress(key, value, layer=0)
    key_hat, value_hat = compressor.decompress(compressed)
    assert torch.allclose(key, key_hat, atol=1e-5)
    assert torch.allclose(value, value_hat, atol=1e-5)

    cache_bytes = get_cache_size_bytes(outputs.past_key_values)
    state_bytes = total_state_bytes(outputs.past_key_values, capabilities=caps)
    assert cache_bytes == state_bytes
