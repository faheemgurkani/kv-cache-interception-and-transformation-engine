"""Tests for layer-aware RoPE context (backwards compatible global path)."""

from pathlib import Path

import pytest
import torch

from framework.model import ModelLayer
from framework.rope import build_rope_context

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "qwen3_0.6b"


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="Model not downloaded")
def test_global_rope_is_identical_for_all_layers():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_layer = ModelLayer(model_path=MODEL_DIR, device=device)
    input_ids = model_layer.tokenize("Hello world")
    with torch.no_grad():
        outputs = model_layer.model(
            input_ids,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    legacy = model_layer.model.model.rotary_emb(outputs.hidden_states[0], position_ids)
    ctx = build_rope_context(
        model_layer.model,
        outputs.hidden_states[0],
        position_ids,
        config=model_layer.config,
    )

    for layer_idx in range(model_layer.config.num_hidden_layers):
        cos, sin = ctx.get_rope(layer_idx)
        assert torch.equal(cos, legacy[0])
        assert torch.equal(sin, legacy[1])
