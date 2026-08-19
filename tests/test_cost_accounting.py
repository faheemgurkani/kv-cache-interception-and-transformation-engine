"""Unit tests for eval/cost/ (Phase 3 explicit cost accounting)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from compressors.identity import IdentityCompressor
from compressors.qjl import QJLCompressor
from compressors.rocketkv import RocketKVCompressor
from compressors.turboquant import TurboQuantCompressor
from eval.cost.accounting import evaluate_cost
from eval.fidelity.memory import MemoryMetrics
from eval.system import SystemMetrics
from eval.system.kernel_cost import KernelCostMetrics
from eval.system.latency_throughput import ThroughputMetrics
from quantizers.lloyd_max import build_centroids, last_centroid_calibration
from quantizers.turboquant_pipeline import TurboQuantStage


@dataclass
class _FakeFidelity:
    memory: MemoryMetrics


def _memory_metrics(**overrides) -> MemoryMetrics:
    defaults = dict(
        context_length=128,
        num_kv_elements=1000,
        uncompressed_bytes=2000,
        compressed_bytes=1000,
        shared_metadata_bytes=64,
        compression_ratio=2.0,
        effective_bits_per_kv_element=8.0,
        process_memory_mb=100.0,
    )
    defaults.update(overrides)
    return MemoryMetrics(**defaults)


def test_identity_offline_cost_no_calibration():
    compressor = IdentityCompressor()
    meta = compressor.offline_cost_metadata()
    assert meta.calibration_required is False
    assert compressor.theoretical_compression_ratio() == 1.0


def test_qjl_offline_cost_calibration_free():
    compressor = QJLCompressor()
    meta = compressor.offline_cost_metadata()
    assert meta.calibration_required is False
    assert meta.calibration_dataset == "fixed_seed_projection"
    assert compressor.theoretical_compression_ratio() == pytest.approx(1.882, rel=1e-2)


def test_rocketkv_theoretical_ratio_context_dependent():
    compressor = RocketKVCompressor(token_budget=256)
    assert compressor.theoretical_compression_ratio(context_length=512) == pytest.approx(2.0)
    assert compressor.theoretical_compression_ratio(context_length=128) == pytest.approx(1.0)
    assert compressor.theoretical_compression_ratio() is None


def test_turboquant_offline_cost_reports_calibration():
    build_centroids.cache_clear()
    compressor = TurboQuantCompressor(bitwidth=4, stage=TurboQuantStage.WHT_QUANT)
    stats = last_centroid_calibration()
    assert stats is not None
    meta = compressor.offline_cost_metadata()
    assert meta.calibration_required is True
    assert meta.calibration_dataset == "gaussian_synthetic"
    assert meta.calibration_tokens == 1_000_000
    assert meta.calibration_time_ms is not None and meta.calibration_time_ms >= 0
    assert compressor.theoretical_compression_ratio() == pytest.approx(4.0)


def test_evaluate_cost_aggregates_fidelity_and_system():
    compressor = IdentityCompressor()
    fidelity = _FakeFidelity(memory=_memory_metrics())
    throughput = ThroughputMetrics(
        context_length=128,
        generated_tokens=8,
        elapsed_seconds=0.4,
        ttft_ms=10.0,
        itl_ms_mean=5.0,
        itl_ms_p50=5.0,
        itl_ms_p99=6.0,
        decode_latency_ms=40.0,
        tokens_per_second=20.0,
        latency_ms_per_token=50.0,
        end_to_end_latency_ms=50.0,
        online_compressed_kv=True,
    )
    kernel = KernelCostMetrics(
        context_length=128,
        generated_tokens=8,
        total_step_time_ms=100.0,
        compress_time_ms=10.0,
        decompress_time_ms=5.0,
        compress_decompress_time_ms=15.0,
        attention_execution_time_ms=85.0,
        compress_decompress_overhead_frac=0.15,
    )
    system = SystemMetrics(throughput=throughput, kernel_cost=kernel)

    cost = evaluate_cost(compressor, context_length=128, fidelity=fidelity, system=system)  # type: ignore[arg-type]

    assert cost.compression.actual_compression_ratio == 2.0
    assert cost.compression.actual_memory_reduction_bytes == 1000
    assert cost.compression.metadata_overhead_bytes == 64
    assert cost.offline.calibration_required is False
    assert cost.online.compression_time_ms == 10.0
    assert cost.online.decompression_time_ms == 5.0
    assert cost.online.attention_cost_ms == 85.0
    assert cost.online.end_to_end_decode_cost_ms == 50.0
    assert cost.online.kernel_cost_measured is True

    payload = cost.to_dict()
    assert "compression" in payload
    assert "offline" in payload
    assert "online" in payload


def test_evaluate_cost_without_kernel_cost():
    compressor = IdentityCompressor()
    cost = evaluate_cost(compressor, context_length=128, fidelity=None, system=None)
    assert cost.compression.actual_compression_ratio is None
    assert cost.online.kernel_cost_measured is False
    assert cost.online.compression_time_ms is None
