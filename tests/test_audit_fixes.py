"""Regression tests for plan-vs-harvest audit fixes."""

from __future__ import annotations

import math

import pytest
import torch

from compressors.palu import PaluCompressor
from compressors.snapkv import SnapKVCompressor
from eval.behavior.task_quality import evaluate_perplexity_result
from eval.cost.oaken_taxonomy import OakenCostLayer
from eval.cross_dim.correlation import analyze_correlations
from eval.cross_dim.points import CrossDimPoint
from eval.cost.accounting import evaluate_cost
from eval.fidelity.representation import evaluate_representation
from eval.paper_collection import paper_quality_anomalies
from eval.reproducibility.manifest import collect_git_sha
from eval.taxonomy_smoke import _fake_gemma_mqa_model


def test_collect_git_sha_returns_hex():
    sha = collect_git_sha()
    if sha is not None:
        assert len(sha) >= 7


def test_collect_git_sha_reads_env(monkeypatch):
    monkeypatch.setenv("KV_GIT_SHA", "abc123def456")
    assert collect_git_sha() == "abc123def456"


def test_palu_offline_cost_includes_bind_timing():
    compressor = PaluCompressor(compression_rate=0.5, group_size=1)
    compressor.bind_model(_fake_gemma_mqa_model())
    meta = compressor.offline_cost_metadata()
    assert meta.calibration_time_ms is not None
    assert meta.calibration_time_ms >= 0


def test_oaken_preprocessing_not_measured_without_timing():
    palu = PaluCompressor()
    cost = evaluate_cost(palu, context_length=128, fidelity=None, system=None)
    layers = {item.layer: item for item in cost.oaken_layers or []}
    assert layers[OakenCostLayer.OFFLINE_PREPROCESSING.value].measured is False


def test_oaken_preprocessing_measured_after_bind():
    palu = PaluCompressor(compression_rate=0.5, group_size=1)
    palu.bind_model(_fake_gemma_mqa_model())
    cost = evaluate_cost(palu, context_length=128, fidelity=None, system=None)
    layers = {item.layer: item for item in cost.oaken_layers or []}
    assert layers[OakenCostLayer.OFFLINE_PREPROCESSING.value].measured is True
    assert layers[OakenCostLayer.OFFLINE_PREPROCESSING.value].metrics["calibration_time_ms"] is not None


def test_palu_representation_degenerate_at_full_rank():
    palu = PaluCompressor(compression_rate=0.5, group_size=1)
    palu.bind_model(_fake_gemma_mqa_model())
    key = torch.randn(1, 1, 128, 256)
    value = torch.randn(1, 1, 128, 256)

    class _Past:
        def __iter__(self):
            yield (key, value)

    metrics = evaluate_representation(_Past(), palu)
    assert metrics.reconstruction_degenerate is True
    assert metrics.reconstruction_degenerate_reason is not None


def test_correlation_max_perplexity_ratio_excludes_qjl():
    points = [
        CrossDimPoint(
            point_id="identity",
            compressor="identity",
            context_length=128,
            compression_ratio=1.0,
            theoretical_compression_ratio=1.0,
            perplexity_ratio=1.05,
            log10_perplexity_ratio=math.log10(1.05),
            quality_score=0.9,
            attention_rmse=0.0,
            key_rmse=0.0,
            value_rmse=0.0,
            tokens_per_second=20.0,
            latency_ms_per_token=50.0,
            online_overhead_ms=50.0,
            retrieval_accuracy=1.0,
            instruction_compliance=1.0,
        ),
        CrossDimPoint(
            point_id="qjl",
            compressor="qjl",
            context_length=128,
            compression_ratio=1.5,
            theoretical_compression_ratio=1.5,
            perplexity_ratio=100.0,
            log10_perplexity_ratio=2.0,
            quality_score=0.1,
            attention_rmse=5.0,
            key_rmse=5.0,
            value_rmse=5.0,
            tokens_per_second=1.0,
            latency_ms_per_token=500.0,
            online_overhead_ms=500.0,
            retrieval_accuracy=0.0,
            instruction_compliance=0.0,
        ),
        CrossDimPoint(
            point_id="snapkv",
            compressor="snapkv",
            context_length=128,
            compression_ratio=2.0,
            theoretical_compression_ratio=2.0,
            perplexity_ratio=1.1,
            log10_perplexity_ratio=math.log10(1.1),
            quality_score=0.8,
            attention_rmse=1.0,
            key_rmse=0.5,
            value_rmse=0.5,
            tokens_per_second=30.0,
            latency_ms_per_token=30.0,
            online_overhead_ms=30.0,
            retrieval_accuracy=1.0,
            instruction_compliance=1.0,
        ),
    ]
    analysis = analyze_correlations(points, context_length=128, max_perplexity_ratio=5.0)
    assert analysis.point_count == 2


def test_paper_audit_flags_snapkv_identity_ppl():
    identity = {
        "label": "identity",
        "compressor": "identity",
        "behavior": {"task_quality": {"perplexity": 72.21885799589231}},
        "fidelity": {"memory": {"compression_ratio": 1.0}, "representation": {}, "attention": {}},
        "cost": {"oaken_layers": [{"layer": "offline_evaluation", "measured": True}]},
    }
    snapkv = {
        "label": "snapkv",
        "compressor": "snapkv",
        "behavior": {"task_quality": {"perplexity": 72.21885799589231}},
        "fidelity": {"memory": {"compression_ratio": 2.0}, "representation": {}, "attention": {}},
        "cost": {"oaken_layers": [{"layer": "offline_evaluation", "measured": True}]},
    }
    anomalies = paper_quality_anomalies([identity, snapkv])
    assert any("SnapKV PPL" in item for item in anomalies)


def test_multi_token_ppl_returns_n_tokens():
    pytest.importorskip("transformers")
    from pathlib import Path

    from framework.model import ModelLayer

    root = Path(__file__).resolve().parent.parent
    model_path = root / "models" / "gemma3_270m"
    if not (model_path / "config.json").exists():
        pytest.skip("Gemma3 checkpoint not present")
    model_layer = ModelLayer(model_path=model_path)
    ids = model_layer.tokenize("Audit fix PPL smoke.")[:, :32]
    result = evaluate_perplexity_result(
        model_layer,
        ids,
        SnapKVCompressor(max_capacity_prompt=16, window_size=8, kernel_size=5),
        max_length=32,
        stride=16,
    )
    assert result.n_tokens > 0
    assert result.perplexity > 1.0
