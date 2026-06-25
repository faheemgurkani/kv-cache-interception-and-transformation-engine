"""Tests for incremental online KV cache (no re-compression drift)."""

import math
from pathlib import Path

import pytest
import torch

from compressors.identity import IdentityCompressor
from compressors.turboquant import TurboQuantCompressor
from data.loader import build_long_context_ids, load_wikitext2
from eval.perplexity import evaluate_perplexity, evaluate_perplexity_baseline
from framework.kv_cache import decompress_to_legacy_cache, iter_layer_kv
from framework.kv_engine import KVCacheEngine
from framework.model import ModelLayer
from quantizers.turboquant_pipeline import TurboQuantStage

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "qwen3_1.7b"
CONTEXT_LENGTH = 128


def _eval_ids(model_layer: ModelLayer):
    dataset = load_wikitext2()
    return build_long_context_ids(model_layer.tokenizer, dataset, CONTEXT_LENGTH).to(model_layer.device)


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="Model not downloaded")
def test_incremental_cache_does_not_recompress_old_tokens():
    model_layer = ModelLayer()
    compressor = TurboQuantCompressor(bitwidth=4, stage=TurboQuantStage.FULL)
    engine = KVCacheEngine(model_layer.model, compressor)
    ids = model_layer.tokenize("Incremental cache stability check.")[:, :32]

    cache = None
    first_key_payload = None
    for t in range(ids.size(1)):
        _, cache = engine.step(ids[:, t : t + 1], compressed_cache=cache)
        payload = cache.layers[0].keys[0]  # type: ignore[index]
        if t == 0:
            first_key_payload = payload
        else:
            assert cache.layers[0].keys[0] is first_key_payload  # type: ignore[index]


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="Model not downloaded")
def test_online_kv_norms_stay_finite_for_full_turboquant():
    model_layer = ModelLayer()
    compressor = TurboQuantCompressor(bitwidth=4, stage=TurboQuantStage.FULL)
    engine = KVCacheEngine(model_layer.model, compressor)
    ids = model_layer.tokenize("KV norm finiteness over online steps.")[:, :128]

    cache = None
    for t in range(ids.size(1)):
        logits, cache = engine.step(ids[:, t : t + 1], compressed_cache=cache)
        assert torch.isfinite(logits).all()
        past = decompress_to_legacy_cache(cache.layers, compressor, model_layer.config, device=ids.device)
        for key, value in iter_layer_kv(past):
            assert torch.isfinite(key).all()
            assert torch.isfinite(value).all()


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="Model not downloaded")
def test_turboquant_full_online_perplexity_is_finite():
    model_layer = ModelLayer()
    ids = _eval_ids(model_layer)
    baseline = evaluate_perplexity(model_layer, ids, IdentityCompressor(), stride=512)
    compressor = TurboQuantCompressor(bitwidth=4, stage=TurboQuantStage.FULL)
    ppl = evaluate_perplexity(model_layer, ids, compressor, stride=512)
    assert math.isfinite(ppl)
    assert ppl < baseline * 2.0


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="Model not downloaded")
def test_wht_quant_online_perplexity_is_finite():
    model_layer = ModelLayer()
    ids = _eval_ids(model_layer)
    baseline = evaluate_perplexity(model_layer, ids, IdentityCompressor(), stride=512)
    compressor = TurboQuantCompressor(bitwidth=4, stage=TurboQuantStage.WHT_QUANT)
    ppl = evaluate_perplexity(model_layer, ids, compressor, stride=512)
    assert math.isfinite(ppl)
    assert ppl < baseline * 2.0


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="Model not downloaded")
def test_identity_online_perplexity_still_matches_baseline():
    model_layer = ModelLayer()
    ids = model_layer.tokenize("Identity baseline parity after incremental fix.")[:, :64]
    ppl_online = evaluate_perplexity(model_layer, ids, IdentityCompressor(), stride=32)
    ppl_baseline = evaluate_perplexity_baseline(model_layer, ids, stride=32)
    assert abs(ppl_online - ppl_baseline) / ppl_baseline < 0.05
