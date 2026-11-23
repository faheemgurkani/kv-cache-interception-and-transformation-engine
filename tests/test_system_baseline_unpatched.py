"""Audit S1: the SYSTEM baseline must run on unpatched attention."""

from pathlib import Path

import pytest
import torch

GEMMA3 = Path(__file__).resolve().parent.parent / "models" / "gemma3_270m"


def test_baseline_throughput_restores_vanilla_attention():
    pytest.importorskip("transformers")
    if not (GEMMA3 / "config.json").exists():
        pytest.skip("Gemma3 checkpoint not present")
    from compressors.registry import get_compressor
    from eval.system.latency_throughput import evaluate_throughput, evaluate_throughput_baseline
    from framework.model import ModelLayer

    layer = ModelLayer(model_path=GEMMA3, device=torch.device("cpu"), torch_dtype=torch.float32)
    ids = layer.tokenize("Baseline attention restore check for patched compressors.")[:, :16]
    attn0 = layer.model.model.layers[0].self_attn

    comp = get_compressor("qjl", bitwidth=1, seed=42)
    layer.make_kv_engine(comp)  # installs the QJL attention patch
    assert "forward" in vars(attn0) and getattr(layer.model, "_qjl_online_enabled", False)

    evaluate_throughput_baseline(layer, ids, num_new_tokens=2)
    assert "forward" not in vars(attn0) or getattr(layer.model, "_qjl_online_enabled", False) is False
    assert not getattr(layer.model, "_qjl_online_enabled", False)

    # The compressed pass after the baseline must be patched again.
    metrics = evaluate_throughput(layer, ids, comp, num_new_tokens=2)
    assert metrics.online_compressed_kv is True
    assert getattr(layer.model, "_qjl_online_enabled", False)
