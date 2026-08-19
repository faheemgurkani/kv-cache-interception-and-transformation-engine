"""Tests for state-aware compression dispatch (§28)."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from compressors.identity import IdentityCompressor
from eval.fidelity.memory import evaluate_memory_from_cache
from framework.kv_cache import apply_compressor
from framework.model import ModelLayer
from framework.model_capabilities import StateType
from framework.state_compression import compress_layer_state, compress_state, compressed_attention_layers
from framework.state_interface import AttentionKVState, LayerState, RecurrentState

OLMO2 = Path(__file__).resolve().parent.parent / "models" / "olmo2_1b"
FALCON = Path(__file__).resolve().parent.parent / "models" / "falcon_h1_0.5b"


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def test_compress_layer_state_preserves_recurrent():
    key = torch.randn(1, 2, 4, 64)
    value = torch.randn(1, 2, 4, 64)
    recurrent = torch.randn(1, 8, 64, 128)
    conv = torch.randn(1, 512, 4)
    state = LayerState(
        layer_idx=0,
        state_type=StateType.HYBRID,
        attention=AttentionKVState(key=key, value=value),
        recurrent=RecurrentState(recurrent_states=recurrent, conv_states=conv),
    )
    compressor = IdentityCompressor()
    compressed = compress_layer_state(state, compressor)
    assert compressed.recurrent is state.recurrent
    key_hat, value_hat = compressor.decompress(compressed.attention)  # type: ignore[arg-type]
    assert torch.allclose(key_hat, key)
    assert torch.allclose(value_hat, value)


def test_compress_state_matches_apply_compressor_on_conventional_kv():
    key = torch.randn(1, 8, 4, 128)
    value = torch.randn(1, 8, 4, 128)
    past = SimpleNamespace(
        layers=[SimpleNamespace(keys=key, values=value)],
    )
    compressor = IdentityCompressor()
    legacy = apply_compressor(past, compressor)
    stateful = compress_state(past, compressor)
    attn = compressed_attention_layers(stateful)
    assert len(attn) == len(legacy)
    assert attn[0].nbytes == legacy[0].nbytes


@pytest.mark.skipif(not OLMO2.exists(), reason="OLMo2-1B not downloaded")
def test_memory_eval_uses_compress_state_path():
    model_layer = ModelLayer(model_path=OLMO2, device=_device())
    input_ids = model_layer.tokenize("Memory path uses compress_state")
    with torch.no_grad():
        outputs = model_layer.forward_with_cache(input_ids, use_cache=True)
    compressor = IdentityCompressor()
    legacy_bytes = sum(
        item.nbytes for item in apply_compressor(outputs.past_key_values, compressor)
    )
    metrics = evaluate_memory_from_cache(
        model_layer,
        input_ids,
        compressor,
        outputs.past_key_values,
    )
    assert metrics.compressed_kv_bytes == legacy_bytes + compressor.shared_storage_bytes()
    assert metrics.compression_ratio == pytest.approx(1.0, rel=1e-3)


@pytest.mark.skipif(not FALCON.exists(), reason="Falcon-H1-0.5B not downloaded")
def test_memory_eval_hybrid_includes_recurrent_via_compress_state():
    model_layer = ModelLayer(model_path=FALCON, device=_device())
    input_ids = model_layer.tokenize("Hybrid memory path")
    with torch.no_grad():
        outputs = model_layer.forward_with_cache(input_ids, use_cache=True)
    compressor = IdentityCompressor()
    metrics = evaluate_memory_from_cache(
        model_layer,
        input_ids,
        compressor,
        outputs.past_key_values,
    )
    assert metrics.recurrent_state_bytes is not None and metrics.recurrent_state_bytes > 0
    assert metrics.kv_compression_ratio == pytest.approx(1.0, rel=1e-3)
    assert metrics.compression_ratio == pytest.approx(1.0, rel=1e-3)
    assert metrics.uncompressed_bytes == metrics.attention_kv_bytes + metrics.recurrent_state_bytes
