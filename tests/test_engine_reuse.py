"""Audit B7: reusing one KVCacheEngine for several prompts must not leak KV between them."""

from pathlib import Path

import pytest
import torch

from compressors.registry import get_compressor

GEMMA3 = Path(__file__).resolve().parent.parent / "models" / "gemma3_270m"


@pytest.mark.parametrize(
    "name,kwargs",
    [("identity", {}), ("rocketkv", dict(token_budget=24, hsa_budget=24, window_size=8))],
)
def test_second_generate_matches_a_fresh_engine(name, kwargs):
    pytest.importorskip("transformers")
    if not (GEMMA3 / "config.json").exists():
        pytest.skip("Gemma3 checkpoint not present")
    from framework.model import ModelLayer

    layer = ModelLayer(model_path=GEMMA3, device=torch.device("cpu"), torch_dtype=torch.float32)
    a = layer.tokenize("Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho.")[:, :40]
    b = layer.tokenize("The quick brown fox jumps over the lazy dog while the rain falls on the quiet old town.")[:, :40]

    reused = layer.make_kv_engine(get_compressor(name, **kwargs))
    reused.generate(a, max_new_tokens=4)  # leaves cache state behind
    second = reused.generate(b, max_new_tokens=6)

    fresh = layer.make_kv_engine(get_compressor(name, **kwargs)).generate(b, max_new_tokens=6)
    assert torch.equal(second, fresh)
