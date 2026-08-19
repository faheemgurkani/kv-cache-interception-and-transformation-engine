"""Unified cost accounting for every compression plug-in (Phase 3).

Schema (per RESEARCH_REDESIGN_PLAN.md Phase 3):

    METHOD
    ├── Compression — theoretical ratio, actual memory reduction, metadata overhead
    ├── Offline cost — calibration required?, dataset, tokens, time, memory
    └── Online cost  — compression/decompression/attention/end-to-end decode time
"""

from __future__ import annotations

from dataclasses import dataclass

from compressors.base import KVCompressor
from eval.fidelity import FidelityMetrics
from eval.system import SystemMetrics


@dataclass
class CompressionCostMetrics:
    theoretical_compression_ratio: float | None
    actual_compression_ratio: float | None
    actual_memory_reduction_bytes: int | None
    uncompressed_bytes: int | None
    compressed_bytes: int | None
    metadata_overhead_bytes: int | None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class OfflineCostMetrics:
    calibration_required: bool
    calibration_dataset: str | None = None
    calibration_tokens: int | None = None
    calibration_time_ms: float | None = None
    calibration_memory_bytes: int | None = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class OnlineCostMetrics:
    compression_time_ms: float | None = None
    decompression_time_ms: float | None = None
    attention_cost_ms: float | None = None
    end_to_end_decode_cost_ms: float | None = None
    compress_decompress_time_ms: float | None = None
    compress_decompress_overhead_frac: float | None = None
    kernel_cost_measured: bool = False

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class CostMetrics:
    compression: CompressionCostMetrics
    offline: OfflineCostMetrics
    online: OnlineCostMetrics

    def to_dict(self) -> dict:
        return {
            "compression": self.compression.to_dict(),
            "offline": self.offline.to_dict(),
            "online": self.online.to_dict(),
        }


def evaluate_cost(
    compressor: KVCompressor,
    *,
    context_length: int,
    fidelity: FidelityMetrics | None = None,
    system: SystemMetrics | None = None,
) -> CostMetrics:
    """Build the unified cost block from FIDELITY, SYSTEM, and compressor hooks."""
    memory = fidelity.memory if fidelity else None
    theoretical = compressor.theoretical_compression_ratio(context_length=context_length)
    offline_meta = compressor.offline_cost_metadata()

    compression = CompressionCostMetrics(
        theoretical_compression_ratio=theoretical,
        actual_compression_ratio=memory.compression_ratio if memory else None,
        actual_memory_reduction_bytes=(
            (memory.uncompressed_bytes - memory.compressed_bytes) if memory else None
        ),
        uncompressed_bytes=memory.uncompressed_bytes if memory else None,
        compressed_bytes=memory.compressed_bytes if memory else None,
        metadata_overhead_bytes=memory.shared_metadata_bytes if memory else None,
    )

    offline = OfflineCostMetrics(
        calibration_required=offline_meta.calibration_required,
        calibration_dataset=offline_meta.calibration_dataset,
        calibration_tokens=offline_meta.calibration_tokens,
        calibration_time_ms=offline_meta.calibration_time_ms,
        calibration_memory_bytes=offline_meta.calibration_memory_bytes,
    )

    kernel = system.kernel_cost if system else None
    throughput = system.throughput if system else None
    online = OnlineCostMetrics(
        compression_time_ms=kernel.compress_time_ms if kernel else None,
        decompression_time_ms=kernel.decompress_time_ms if kernel else None,
        attention_cost_ms=kernel.attention_execution_time_ms if kernel else None,
        end_to_end_decode_cost_ms=throughput.end_to_end_latency_ms if throughput else None,
        compress_decompress_time_ms=kernel.compress_decompress_time_ms if kernel else None,
        compress_decompress_overhead_frac=(
            kernel.compress_decompress_overhead_frac if kernel else None
        ),
        kernel_cost_measured=kernel is not None,
    )

    return CostMetrics(compression=compression, offline=offline, online=online)
