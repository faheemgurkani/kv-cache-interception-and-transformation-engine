"""SYSTEM: does the compression actually make inference better, end to end?

A method that reaches a high compression ratio but adds significant per-step
compute can lose to a method with a lower ratio and near-zero runtime overhead —
that tradeoff is exactly what SYSTEM is for; FIDELITY and BEHAVIOR alone can't see it.

Sub-metrics:
- latency_throughput — TTFT, inter-token latency (ITL), decode latency, tokens/sec,
  end-to-end latency (framework/kv_engine.py step loop)
- vram               — peak allocated/reserved CUDA memory
- memory_bandwidth   — effective KV bytes moved per second (analytical, see module docstring)
- kernel_cost        — compress/decompress time vs. the rest of the forward pass
- gpu_utilization    — best-effort NVML sampling (optional, CUDA + pynvml only)

`evaluate_system` also reports `actual_kv_memory` (== FIDELITY/memory.compressed_bytes)
so a SYSTEM-only view doesn't require cross-referencing FIDELITY output.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from compressors.base import KVCompressor
from eval.system.gpu_utilization import GPUUtilizationMetrics, evaluate_gpu_utilization
from eval.system.kernel_cost import KernelCostMetrics, evaluate_kernel_cost
from eval.system.latency_throughput import (
    ThroughputMetrics,
    evaluate_throughput,
    evaluate_throughput_baseline,
    measure_tokens_per_second,
)
from eval.system.memory_bandwidth import MemoryBandwidthMetrics, evaluate_memory_bandwidth
from eval.system.vram import PeakMemoryMetrics, evaluate_peak_vram
from framework.model import ModelLayer

__all__ = [
    "GPUUtilizationMetrics",
    "KernelCostMetrics",
    "MemoryBandwidthMetrics",
    "PeakMemoryMetrics",
    "SystemMetrics",
    "ThroughputMetrics",
    "evaluate_gpu_utilization",
    "evaluate_kernel_cost",
    "evaluate_memory_bandwidth",
    "evaluate_peak_vram",
    "evaluate_system",
    "evaluate_throughput",
    "evaluate_throughput_baseline",
    "measure_tokens_per_second",
]


@dataclass
class SystemMetrics:
    """Aggregate SYSTEM result. Every field is optional — callers opt into what they run."""

    throughput: ThroughputMetrics | None = None
    throughput_baseline: ThroughputMetrics | None = None
    peak_memory: PeakMemoryMetrics | None = None
    memory_bandwidth: MemoryBandwidthMetrics | None = None
    kernel_cost: KernelCostMetrics | None = None
    gpu_utilization: GPUUtilizationMetrics | None = None
    actual_kv_memory_bytes: int | None = None

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return {
            "latency_throughput": asdict(self.throughput) if self.throughput else None,
            "latency_throughput_baseline": asdict(self.throughput_baseline) if self.throughput_baseline else None,
            "peak_memory": self.peak_memory.to_dict() if self.peak_memory else None,
            "memory_bandwidth": self.memory_bandwidth.to_dict() if self.memory_bandwidth else None,
            "kernel_cost": self.kernel_cost.to_dict() if self.kernel_cost else None,
            "gpu_utilization": self.gpu_utilization.to_dict() if self.gpu_utilization else None,
            "actual_kv_memory_bytes": self.actual_kv_memory_bytes,
        }


def evaluate_system(
    model_layer: ModelLayer,
    input_ids: torch.Tensor,
    compressor: KVCompressor,
    *,
    run_throughput: bool = True,
    run_peak_memory: bool = False,
    run_memory_bandwidth: bool = False,
    run_kernel_cost: bool = False,
    run_gpu_utilization: bool = False,
    include_baseline: bool = False,
    num_new_tokens: int = 128,
    actual_kv_memory_bytes: int | None = None,
    uncompressed_kv_memory_bytes: int | None = None,
) -> SystemMetrics:
    """Run the requested SYSTEM sub-metrics. Only throughput (TTFT/ITL/tok-s) runs by
    default; the rest are opt-in since each adds its own generate() pass."""
    throughput_baseline = (
        evaluate_throughput_baseline(model_layer, input_ids, num_new_tokens=num_new_tokens)
        if include_baseline and run_throughput
        else None
    )
    throughput = (
        evaluate_throughput(model_layer, input_ids, compressor, num_new_tokens=num_new_tokens)
        if run_throughput
        else None
    )
    peak_memory = (
        evaluate_peak_vram(
            model_layer,
            input_ids,
            compressor,
            num_new_tokens=num_new_tokens,
            uncompressed_kv_bytes=uncompressed_kv_memory_bytes,
            compressed_kv_bytes=actual_kv_memory_bytes,
        )
        if run_peak_memory
        else None
    )
    memory_bandwidth = (
        evaluate_memory_bandwidth(model_layer, input_ids, compressor, num_new_tokens=num_new_tokens)
        if run_memory_bandwidth
        else None
    )
    kernel_cost = (
        evaluate_kernel_cost(model_layer, input_ids, compressor, num_new_tokens=num_new_tokens)
        if run_kernel_cost
        else None
    )
    gpu_utilization = (
        evaluate_gpu_utilization(model_layer, input_ids, compressor, num_new_tokens=num_new_tokens)
        if run_gpu_utilization
        else None
    )

    return SystemMetrics(
        throughput=throughput,
        throughput_baseline=throughput_baseline,
        peak_memory=peak_memory,
        memory_bandwidth=memory_bandwidth,
        kernel_cost=kernel_cost,
        gpu_utilization=gpu_utilization,
        actual_kv_memory_bytes=actual_kv_memory_bytes,
    )
