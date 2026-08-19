"""Tests for state-aware compression dispatch (§28)."""

from types import SimpleNamespace

import pytest
import torch

from compressors.identity import IdentityCompressor
from framework.kv_cache import apply_compressor
from framework.model_capabilities import StateType
from framework.state_compression import compress_layer_state, compress_state, compressed_attention_layers
from framework.state_interface import AttentionKVState, LayerState, RecurrentState


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
