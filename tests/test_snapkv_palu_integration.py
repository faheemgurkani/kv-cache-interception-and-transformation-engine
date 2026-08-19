"""Model-level integration tests for SnapKV and Palu plug-ins."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from compressors.registry import get_compressor
from eval.runner import EvaluationRunner
from framework.kv_cache import decompress_to_legacy_cache, iter_layer_kv
from framework.kv_engine import KVCacheEngine
from framework.model import ModelLayer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_CANDIDATES = [
    PROJECT_ROOT / "models" / "qwen3_0.6b",
    PROJECT_ROOT / "models" / "legacy" / "qwen3_1.7b",
    PROJECT_ROOT / "models" / "olmo2_1b",
]


def _first_model_path() -> Path | None:
    for path in MODEL_CANDIDATES:
        if path.exists():
            return path
    return None


MODEL_PATH = _first_model_path()
pytestmark_model = pytest.mark.skipif(MODEL_PATH is None, reason="No SLM model downloaded")


@pytest.fixture
def model_layer() -> ModelLayer:
    return ModelLayer(model_path=MODEL_PATH)


@pytestmark_model
def test_snapkv_engine_step_produces_finite_logits(model_layer: ModelLayer):
    compressor = get_compressor("snapkv", max_capacity_prompt=64, window_size=8, kernel_size=5)
    engine = KVCacheEngine(model_layer.model, compressor)
    ids = model_layer.tokenize("SnapKV engine integration check for KVBench.")[:, :128]

    cache = None
    for t in range(ids.size(1)):
        step_in = ids if cache is None else ids[:, t : t + 1]
        logits, cache = engine.step(step_in, compressed_cache=cache)
        assert torch.isfinite(logits).all()
        assert cache.layers
        assert cache.seq_length > 0


@pytestmark_model
def test_snapkv_online_prefill_reduces_cache_when_over_budget(model_layer: ModelLayer):
    compressor = get_compressor("snapkv", max_capacity_prompt=64, window_size=8, kernel_size=5)
    engine = KVCacheEngine(model_layer.model, compressor)
    ids = model_layer.tokenize("word " * 300)[:, :128]
    seq_len = ids.shape[1]

    _, cache = engine.step(ids, compressed_cache=None)
    payload = cache.layers[0].keys
    assert payload.original_seq_len == seq_len
    if seq_len >= compressor.max_capacity_prompt:
        assert payload.keys.shape[2] == compressor.max_capacity_prompt
    else:
        assert payload.keys.shape[2] == seq_len


@pytestmark_model
def test_snapkv_decompress_to_legacy_finite(model_layer: ModelLayer):
    compressor = get_compressor("snapkv", max_capacity_prompt=64, window_size=8)
    engine = KVCacheEngine(model_layer.model, compressor)
    ids = model_layer.tokenize("Decompress round trip for SnapKV.")[:, :96]
    _, cache = engine.step(ids)
    past = decompress_to_legacy_cache(cache.layers, compressor, model_layer.config, device=ids.device)
    for key, value in iter_layer_kv(past):
        assert torch.isfinite(key).all()
        assert torch.isfinite(value).all()


@pytestmark_model
def test_palu_engine_step_produces_finite_logits(model_layer: ModelLayer):
    compressor = get_compressor("palu", compression_rate=0.5, group_size=4)
    engine = KVCacheEngine(model_layer.model, compressor)
    ids = model_layer.tokenize("Palu engine integration check for KVBench.")[:, :64]

    cache = None
    for t in range(min(16, ids.size(1))):
        step_in = ids if cache is None else ids[:, t : t + 1]
        logits, cache = engine.step(step_in, compressed_cache=cache)
        assert torch.isfinite(logits).all()


@pytestmark_model
def test_palu_bind_model_populates_shared_storage(model_layer: ModelLayer):
    compressor = get_compressor("palu", compression_rate=0.5, group_size=4)
    assert compressor.shared_storage_bytes() == 0
    compressor.bind_model(model_layer.model)
    assert compressor.shared_storage_bytes() > 0
    assert compressor.uses_weight_factors


@pytestmark_model
def test_snapkv_eval_runner_smoke(model_layer: ModelLayer):
    compressor = get_compressor("snapkv", max_capacity_prompt=128, window_size=16)
    runner = EvaluationRunner(model_layer=model_layer, compressor=compressor)
    result = runner.run(
        context_length=128,
        run_fidelity=True,
        run_behavior=True,
        run_perplexity=True,
        run_retrieval=False,
        run_instruction_following=False,
        run_system=True,
        run_throughput=True,
        generated_tokens=4,
        perplexity_stride=64,
    )
    assert result.taxonomy is not None
    assert result.taxonomy["primary"] == "eviction"
    assert result.fidelity is not None
    assert result.fidelity.memory.compression_ratio > 0
    assert result.behavior.perplexity is not None and math.isfinite(result.behavior.perplexity)
    assert result.system.throughput.tokens_per_second > 0
    assert result.cost is not None


@pytestmark_model
def test_palu_eval_runner_smoke(model_layer: ModelLayer):
    compressor = get_compressor("palu", compression_rate=0.5, group_size=4)
    runner = EvaluationRunner(model_layer=model_layer, compressor=compressor)
    result = runner.run(
        context_length=128,
        run_fidelity=True,
        run_behavior=True,
        run_perplexity=True,
        run_retrieval=False,
        run_instruction_following=False,
        run_system=True,
        run_throughput=True,
        generated_tokens=4,
        perplexity_stride=64,
    )
    assert result.taxonomy is not None
    assert result.taxonomy["primary"] == "projection"
    assert result.fidelity.representation.key_rmse >= 0.0
    assert math.isfinite(result.behavior.perplexity)
    assert result.cost.offline.calibration_required is True
