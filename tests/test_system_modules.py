"""Unit and module-isolation tests for eval/system/."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from compressors.identity import IdentityCompressor
from eval.system import SystemMetrics, evaluate_system
from eval.system.device_metrics import PeakMemoryTracker
from eval.system.gpu_utilization import GPUUtilizationMetrics, evaluate_gpu_utilization
from eval.system.kernel_cost import KernelCostMetrics, _timed_methods, evaluate_kernel_cost
from eval.system.latency_throughput import ThroughputMetrics, _percentile, evaluate_throughput
from eval.system.memory_bandwidth import MemoryBandwidthMetrics, evaluate_memory_bandwidth
from eval.system.vram import PeakMemoryMetrics, evaluate_peak_vram
from framework.model import ModelLayer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_CANDIDATES = [
    PROJECT_ROOT / "models" / "legacy" / "qwen3_1.7b",
    PROJECT_ROOT / "models" / "olmo2_1b",
    PROJECT_ROOT / "models" / "qwen3_0.6b",
]


def _first_model_path() -> Path | None:
    for path in MODEL_CANDIDATES:
        if path.exists():
            return path
    return None


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --- Pure math / logic -------------------------------------------------------------


def test_percentile_interpolation():
    samples = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert _percentile(samples, 0.50) == 30.0
    assert _percentile(samples, 0.99) == 50.0
    assert _percentile([], 0.5) == 0.0


def test_throughput_derived_metrics():
    elapsed = 2.0
    num_new_tokens = 8
    tps = num_new_tokens / elapsed
    latency_ms = (elapsed / num_new_tokens) * 1000
    assert tps == 4.0
    assert latency_ms == 250.0


def test_memory_bandwidth_formula():
    total_bytes = 2 * (1000 + 2000 + 3000)  # three steps, 1000 nbytes each, read+write
    elapsed = 0.5
    gbps = total_bytes / elapsed / 1e9
    assert gbps == pytest.approx(0.000024, rel=1e-3)


def test_kernel_cost_overhead_math():
    total_ms = 200.0
    compress_ms = 30.0
    decompress_ms = 20.0
    cd_ms = compress_ms + decompress_ms
    attention_ms = max(total_ms - cd_ms, 0.0)
    overhead_frac = cd_ms / total_ms
    assert attention_ms == 150.0
    assert overhead_frac == pytest.approx(0.25)


def test_kernel_cost_metrics_fields():
    metrics = KernelCostMetrics(
        context_length=128,
        generated_tokens=4,
        total_step_time_ms=100.0,
        compress_time_ms=10.0,
        decompress_time_ms=5.0,
        compress_decompress_time_ms=15.0,
        attention_execution_time_ms=85.0,
        compress_decompress_overhead_frac=0.15,
    )
    payload = metrics.to_dict()
    assert payload["compress_decompress_overhead_frac"] == 0.15
    assert payload["attention_execution_time_ms"] == 85.0


def test_timed_methods_accumulates_compress_decompress():
    compressor = IdentityCompressor()

    class _Payload:
        nbytes = 16

    with _timed_methods(compressor) as totals:
        key = torch.randn(1, 2, 4, 8)
        value = torch.randn(1, 2, 4, 8)
        compressed = compressor.compress(key, value, layer=0)
        compressor.decompress(compressed)

    assert totals["compress"] >= 0.0
    assert totals["decompress"] >= 0.0


def test_evaluate_system_wiring_respects_flags(monkeypatch):
    model = MagicMock()
    compressor = MagicMock()
    input_ids = torch.tensor([[1, 2, 3, 4]])

    monkeypatch.setattr(
        "eval.system.evaluate_throughput",
        lambda *a, **k: ThroughputMetrics(4, 2, 1.0, 2.0, 500.0),
    )
    monkeypatch.setattr(
        "eval.system.evaluate_peak_vram",
        lambda *a, **k: PeakMemoryMetrics(4, 2, 100.0, 120.0, 130.0, "mps", False),
    )
    monkeypatch.setattr(
        "eval.system.evaluate_memory_bandwidth",
        lambda *a, **k: MemoryBandwidthMetrics(4, 2, 1.0, 8000, 0.008),
    )
    monkeypatch.setattr(
        "eval.system.evaluate_kernel_cost",
        lambda *a, **k: KernelCostMetrics(4, 2, 50.0, 5.0, 5.0, 10.0, 40.0, 0.2),
    )
    monkeypatch.setattr(
        "eval.system.evaluate_gpu_utilization",
        lambda *a, **k: GPUUtilizationMetrics(True, "process_cpu", 3, 42.0, 55.0),
    )

    full = evaluate_system(
        model,
        input_ids,
        compressor,
        run_throughput=True,
        run_peak_memory=True,
        run_memory_bandwidth=True,
        run_kernel_cost=True,
        run_gpu_utilization=True,
        actual_kv_memory_bytes=4096,
    )
    assert full.throughput is not None
    assert full.peak_memory is not None
    assert full.memory_bandwidth is not None
    assert full.kernel_cost is not None
    assert full.gpu_utilization is not None
    assert full.actual_kv_memory_bytes == 4096

    throughput_only = evaluate_system(
        model,
        input_ids,
        compressor,
        run_peak_memory=False,
        run_memory_bandwidth=False,
        run_kernel_cost=False,
        run_gpu_utilization=False,
    )
    assert throughput_only.throughput is not None
    assert throughput_only.peak_memory is None


def test_system_metrics_to_dict_shape():
    metrics = SystemMetrics(
        throughput=ThroughputMetrics(128, 4, 1.0, 4.0, 250.0, ttft_ms=300.0),
        actual_kv_memory_bytes=8192,
    )
    payload = metrics.to_dict()
    assert payload["latency_throughput"]["tokens_per_second"] == 4.0
    assert payload["actual_kv_memory_bytes"] == 8192


def test_memory_bandwidth_mocked_engine_accumulates_two_x_nbytes():
    """Verify read+write accounting: total_bytes_moved += 2 * cache.nbytes per step."""
    compressor = IdentityCompressor()
    model_layer = MagicMock()
    model_layer.device = torch.device("cpu")

    nbytes_per_step = 512
    steps = 3
    call_count = {"n": 0}

    class FakeCache:
        def __init__(self):
            self.nbytes = nbytes_per_step

    def fake_step(step_input, compressed_cache=None, **kwargs):
        call_count["n"] += 1
        vocab = 128
        logits = torch.zeros(step_input.shape[0], 1, vocab)
        return logits, FakeCache()

    fake_engine = MagicMock()
    fake_engine.step = fake_step
    model_layer.make_kv_engine.return_value = fake_engine

    input_ids = torch.tensor([[1, 2, 3, 4]])
    metrics = evaluate_memory_bandwidth(model_layer, input_ids, compressor, num_new_tokens=steps)

    assert call_count["n"] == steps
    assert metrics.total_kv_bytes_moved == 2 * nbytes_per_step * steps
    assert metrics.effective_bandwidth_gbps == pytest.approx(
        metrics.total_kv_bytes_moved / metrics.elapsed_seconds / 1e9,
        rel=1e-6,
    )


def test_timed_methods_wraps_compress_layer_from_kv():
    compressor = MagicMock()
    compressor.compress_kv = MagicMock(side_effect=lambda *a, **k: "k")
    compressor.decompress_kv = MagicMock(side_effect=lambda *a, **k: "d")

    def slow_layer(*args, **kwargs):
        time.sleep(0.001)
        return "layer"

    compressor.compress_layer_from_kv = slow_layer

    with _timed_methods(compressor) as totals:
        compressor.compress_layer_from_kv(1, 2, 3)
        compressor.compress_kv(torch.zeros(1))

    assert totals["compress"] >= 0.001


def test_peak_memory_tracker_process_rss():
    tracker = PeakMemoryTracker(torch.device("cpu"))
    tracker.reset()
    tracker.sample()
    snap = tracker.snapshot()
    assert snap.peak_process_rss_mb > 0
    assert snap.memory_backend in {"process_rss", "cuda", "mps"}


def test_gpu_utilization_non_cuda_uses_process_cpu():
    model_layer = MagicMock()
    model_layer.device = torch.device("cpu")
    fake_engine = MagicMock()
    fake_engine.generate.return_value = torch.tensor([[1, 2, 3]])
    model_layer.make_kv_engine.return_value = fake_engine

    with patch("torch.cuda.is_available", return_value=False):
        metrics = evaluate_gpu_utilization(model_layer, torch.tensor([[1, 2]]), IdentityCompressor(), num_new_tokens=1)

    assert metrics.available is True
    assert metrics.utilization_backend == "process_cpu"


# --- Module integration (real model) -----------------------------------------------


MODEL_PATH = _first_model_path()
pytestmark_model = pytest.mark.skipif(MODEL_PATH is None, reason="No shortlist/legacy model downloaded")


@pytest.fixture(scope="module")
def model_layer():
    return ModelLayer(model_path=MODEL_PATH, device=_device())


@pytestmark_model
def test_evaluate_throughput_ttft_and_itl(model_layer: ModelLayer):
    ids = model_layer.tokenize("System throughput module isolation.")[:, :12]
    metrics = evaluate_throughput(model_layer, ids, IdentityCompressor(), num_new_tokens=4)
    assert metrics.online_compressed_kv is True
    assert metrics.tokens_per_second > 0
    assert metrics.ttft_ms is not None and metrics.ttft_ms > 0
    assert metrics.itl_ms_mean is not None and metrics.itl_ms_mean > 0
    assert metrics.end_to_end_latency_ms == pytest.approx(metrics.elapsed_seconds * 1000, rel=1e-3)
    if metrics.itl_ms_mean is not None:
        assert metrics.decode_latency_ms == metrics.itl_ms_mean


@pytestmark_model
def test_evaluate_memory_bandwidth_positive(model_layer: ModelLayer):
    ids = model_layer.tokenize("Bandwidth check.")[:, :8]
    metrics = evaluate_memory_bandwidth(model_layer, ids, IdentityCompressor(), num_new_tokens=3)
    assert metrics.total_kv_bytes_moved > 0
    assert metrics.effective_bandwidth_gbps >= 0.0
    assert metrics.generated_tokens == 3


@pytestmark_model
def test_evaluate_kernel_cost_partition(model_layer: ModelLayer):
    ids = model_layer.tokenize("Kernel cost partition.")[:, :8]
    metrics = evaluate_kernel_cost(model_layer, ids, IdentityCompressor(), num_new_tokens=3)
    assert metrics.total_step_time_ms > 0
    assert metrics.compress_decompress_time_ms >= 0
    assert metrics.attention_execution_time_ms >= 0
    assert metrics.compress_decompress_overhead_frac == pytest.approx(
        metrics.compress_decompress_time_ms / metrics.total_step_time_ms,
        rel=1e-6,
    )
    assert (
        metrics.attention_execution_time_ms + metrics.compress_decompress_time_ms
        == pytest.approx(metrics.total_step_time_ms, rel=1e-4)
    )


@pytestmark_model
def test_evaluate_peak_vram_on_device(model_layer: ModelLayer):
    ids = model_layer.tokenize("VRAM probe.")[:, :8]
    metrics = evaluate_peak_vram(model_layer, ids, IdentityCompressor(), num_new_tokens=2)
    assert metrics.generated_tokens == 2
    assert metrics.memory_backend in {"cuda", "mps", "process_rss"}
    assert metrics.peak_process_rss_mb is not None and metrics.peak_process_rss_mb > 0
    assert metrics.peak_allocated_mb is not None and metrics.peak_allocated_mb > 0


@pytestmark_model
def test_rocketkv_kernel_cost_captures_compression_time(model_layer: ModelLayer):
    from compressors.rocketkv import RocketKVCompressor

    ids = model_layer.tokenize("RocketKV kernel cost.")[:, :8]
    compressor = RocketKVCompressor(token_budget=64, hsa_budget=64)
    metrics = evaluate_kernel_cost(model_layer, ids, compressor, num_new_tokens=2)
    assert metrics.total_step_time_ms > 0
    assert metrics.compress_time_ms > 0
    assert metrics.compress_decompress_overhead_frac > 0.0
