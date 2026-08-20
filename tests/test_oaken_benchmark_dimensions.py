"""Tests for Phase 26 Oaken taxonomy and Phase 27 benchmark dimensions."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from compressors.identity import IdentityCompressor
from compressors.qjl import QJLCompressor
from compressors.rocketkv import RocketKVCompressor
from compressors.turboquant import TurboQuantCompressor
from eval.cost.accounting import evaluate_cost
from eval.cost.benchmark_dimensions import derive_online_overhead_ms_per_token
from eval.cost.oaken_taxonomy import OakenCostLayer, build_oaken_layers, compressor_is_stateful
from eval.fidelity.memory import MemoryMetrics
from eval.system import SystemMetrics
from eval.system.latency_throughput import ThroughputMetrics
from quantizers.lloyd_max import build_centroids
from quantizers.turboquant_pipeline import TurboQuantStage


@dataclass
class _FakeFidelity:
    representation: object
    attention: object
    memory: MemoryMetrics

    def __init__(self, memory: MemoryMetrics):
        self.memory = memory
        self.representation = type("R", (), {"key_rmse": 0.5, "value_rmse": 1.0})()
        self.attention = type("A", (), {"rmse": 2.0, "cosine_similarity": 0.9})()


def _memory(**overrides) -> MemoryMetrics:
    defaults = dict(
        context_length=512,
        num_kv_elements=1000,
        uncompressed_bytes=2000,
        compressed_bytes=1000,
        shared_metadata_bytes=0,
        compression_ratio=2.0,
        effective_bits_per_kv_element=8.0,
        process_memory_mb=1.0,
    )
    defaults.update(overrides)
    return MemoryMetrics(**defaults)


def test_oaken_offline_evaluation_is_fidelity_not_dollar_cost():
    fidelity = _FakeFidelity(_memory())
    cost = evaluate_cost(IdentityCompressor(), context_length=512, fidelity=fidelity, system=None)  # type: ignore[arg-type]
    assert cost.oaken_layers is not None
    layers = {item.layer: item for item in cost.oaken_layers}
    assert layers[OakenCostLayer.OFFLINE_EVALUATION.value].measured is True
    assert layers[OakenCostLayer.OFFLINE_EVALUATION.value].metrics["compression_ratio"] == 2.0
    assert layers[OakenCostLayer.OFFLINE_PREPROCESSING.value].metrics["calibration_required"] is False


def test_oaken_online_layers_populated_with_system():
    throughput = ThroughputMetrics(
        context_length=512,
        generated_tokens=64,
        elapsed_seconds=1.0,
        tokens_per_second=64.0,
        latency_ms_per_token=15.625,
        end_to_end_latency_ms=1000.0,
        online_compressed_kv=True,
    )
    system = SystemMetrics(throughput=throughput)
    cost = evaluate_cost(IdentityCompressor(), context_length=512, fidelity=None, system=system)
    layers = {item.layer: item for item in cost.oaken_layers or []}
    assert layers[OakenCostLayer.END_TO_END_SERVING.value].measured is True
    assert layers[OakenCostLayer.END_TO_END_SERVING.value].metrics["latency_ms_per_token"] == pytest.approx(15.625)


def test_benchmark_dimensions_stateful_flags():
    assert compressor_is_stateful(QJLCompressor()) is True
    assert compressor_is_stateful(RocketKVCompressor()) is True
    assert compressor_is_stateful(IdentityCompressor()) is False


def test_benchmark_dimensions_turboquant_calibration():
    build_centroids.cache_clear()
    compressor = TurboQuantCompressor(bitwidth=4, stage=TurboQuantStage.WHT_QUANT)
    cost = evaluate_cost(compressor, context_length=512, fidelity=None, system=None)
    assert cost.benchmark_dimensions is not None
    assert cost.benchmark_dimensions.calibration_required is True
    assert cost.benchmark_dimensions.calibration_dataset == "gaussian_synthetic"
    assert cost.benchmark_dimensions.calibration_tokens == 1_000_000
    assert cost.benchmark_dimensions.stateful is False


def test_benchmark_dimensions_qjl_calibration_free_stateful():
    cost = evaluate_cost(QJLCompressor(), context_length=512, fidelity=None, system=None)
    dims = cost.benchmark_dimensions
    assert dims is not None
    assert dims.calibration_required is False
    assert dims.stateful is True


def test_online_overhead_ms_per_token_prefers_latency():
    from eval.cost.accounting import CompressionCostMetrics, CostMetrics, OfflineCostMetrics, OnlineCostMetrics

    cost = CostMetrics(
        compression=CompressionCostMetrics(None, None, None, None, None, None),
        offline=OfflineCostMetrics(calibration_required=False),
        online=OnlineCostMetrics(end_to_end_decode_cost_ms=640.0),
    )
    throughput = ThroughputMetrics(
        context_length=512,
        generated_tokens=64,
        elapsed_seconds=1.0,
        tokens_per_second=64.0,
        latency_ms_per_token=10.0,
        end_to_end_latency_ms=640.0,
        online_compressed_kv=True,
    )
    system = SystemMetrics(throughput=throughput)
    assert derive_online_overhead_ms_per_token(cost, system) == pytest.approx(10.0)


def test_online_overhead_fallback_divides_e2e_by_tokens():
    from eval.cost.accounting import CompressionCostMetrics, CostMetrics, OfflineCostMetrics, OnlineCostMetrics

    cost = CostMetrics(
        compression=CompressionCostMetrics(None, None, None, None, None, None),
        offline=OfflineCostMetrics(calibration_required=False),
        online=OnlineCostMetrics(end_to_end_decode_cost_ms=640.0),
    )
    throughput = ThroughputMetrics(
        context_length=512,
        generated_tokens=64,
        elapsed_seconds=1.0,
        tokens_per_second=64.0,
        latency_ms_per_token=None,
        end_to_end_latency_ms=640.0,
        online_compressed_kv=True,
    )
    system = SystemMetrics(throughput=throughput)
    assert derive_online_overhead_ms_per_token(cost, system) == pytest.approx(10.0)


def test_cost_export_includes_oaken_and_benchmark_blocks():
    cost = evaluate_cost(IdentityCompressor(), context_length=128, fidelity=None, system=None)
    payload = cost.to_dict()
    assert "oaken_layers" in payload
    assert len(payload["oaken_layers"]) == 5
    assert "benchmark_dimensions" in payload
    assert "stateful" in payload["benchmark_dimensions"]
