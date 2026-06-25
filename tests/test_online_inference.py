"""Tests for online compressed-KV inference metrics."""

from pathlib import Path

import pytest

from compressors.identity import IdentityCompressor
from eval.perplexity import evaluate_perplexity, evaluate_perplexity_baseline
from eval.throughput import evaluate_throughput, evaluate_throughput_baseline
from framework.kv_engine import KVCacheEngine
from framework.model import ModelLayer

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "qwen3_1.7b"


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="Model not downloaded")
def test_online_perplexity_matches_baseline_for_identity():
    model_layer = ModelLayer()
    ids = model_layer.tokenize("Online compressed KV perplexity identity check sequence.")[:, :64]
    ppl_online = evaluate_perplexity(model_layer, ids, IdentityCompressor(), stride=32)
    ppl_baseline = evaluate_perplexity_baseline(model_layer, ids, stride=32)
    assert abs(ppl_online - ppl_baseline) / ppl_baseline < 0.05


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="Model not downloaded")
def test_online_throughput_uses_kv_engine():
    model_layer = ModelLayer()
    ids = model_layer.tokenize("Throughput path check.")[:, :16]
    metrics = evaluate_throughput(model_layer, ids, IdentityCompressor(), num_new_tokens=4)
    assert metrics.online_compressed_kv is True
    assert metrics.tokens_per_second > 0


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="Model not downloaded")
def test_baseline_throughput_does_not_use_compressed_path():
    model_layer = ModelLayer()
    ids = model_layer.tokenize("Baseline throughput path check.")[:, :16]
    metrics = evaluate_throughput_baseline(model_layer, ids, num_new_tokens=4)
    assert metrics.online_compressed_kv is False


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="Model not downloaded")
def test_kv_engine_generate_runs():
    model_layer = ModelLayer()
    engine = KVCacheEngine(model_layer.model, IdentityCompressor())
    ids = model_layer.tokenize("Generate loop check.")
    out = engine.generate(ids, max_new_tokens=3)
    assert out.shape[1] == ids.shape[1] + 3
