"""Tests for typed inference-state iteration."""

from pathlib import Path

import pytest
import torch

from framework.kv_cache import get_cache_size_bytes, iter_layer_kv
from framework.model import ModelLayer
from framework.model_capabilities import resolve_model_capabilities
from framework.state_interface import hybrid_layer_detected, iter_layer_states, visible_state_bytes

OLMO2 = Path(__file__).resolve().parent.parent / "models" / "olmo2_1b"
FALCON = Path(__file__).resolve().parent.parent / "models" / "falcon_h1_0.5b"


@pytest.mark.skipif(not OLMO2.exists(), reason="OLMo2 not downloaded")
def test_iter_layer_states_matches_iter_layer_kv_for_olmo2():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_layer = ModelLayer(model_path=OLMO2, device=device)
    caps = resolve_model_capabilities(model_layer.config)
    input_ids = model_layer.tokenize("Hello")
    with torch.no_grad():
        outputs = model_layer.forward_with_cache(input_ids, use_cache=True)

    kv_pairs = list(iter_layer_kv(outputs.past_key_values))
    state_pairs = [
        (state.attention.key, state.attention.value)
        for state in iter_layer_states(outputs.past_key_values, capabilities=caps)
        if state.attention is not None
    ]
    assert len(kv_pairs) == len(state_pairs)
    for (k1, v1), (k2, v2) in zip(kv_pairs, state_pairs, strict=True):
        assert torch.equal(k1, k2)
        assert torch.equal(v1, v2)


@pytest.mark.skipif(not FALCON.exists(), reason="Falcon-H1 not downloaded")
def test_falcon_hybrid_state_is_detected():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_layer = ModelLayer(model_path=FALCON, device=device)
    caps = resolve_model_capabilities(model_layer.config)
    input_ids = model_layer.tokenize("Hello")
    with torch.no_grad():
        outputs = model_layer.forward_with_cache(input_ids, use_cache=True)

    assert hybrid_layer_detected(outputs.past_key_values) is True
    states = list(iter_layer_states(outputs.past_key_values, capabilities=caps))
    assert states[0].recurrent is not None
    assert states[0].recurrent.recurrent_states is not None

    attention_only = get_cache_size_bytes(outputs.past_key_values)
    total = visible_state_bytes(outputs.past_key_values)
    assert total > attention_only
    assert caps.state_semantics_complete is True
