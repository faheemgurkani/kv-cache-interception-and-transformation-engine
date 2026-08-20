"""Per-method benchmark dimension table (Phase 27).

Standardizes calibration, statefulness, and online overhead columns for fair
cross-method comparison alongside FIDELITY / BEHAVIOR / SYSTEM metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from compressors.base import KVCompressor
from eval.system import SystemMetrics

if TYPE_CHECKING:
    from eval.cost.accounting import CostMetrics


@dataclass(frozen=True)
class BenchmarkDimensions:
    """Phase 27 comparison columns exported on every ``EvaluationResult.cost``."""

    calibration_required: bool
    calibration_dataset: str | None
    calibration_tokens: int | None
    calibration_time_ms: float | None
    calibration_memory_bytes: int | None
    stateful: bool
    online_overhead_ms_per_token: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration_required": self.calibration_required,
            "calibration_dataset": self.calibration_dataset,
            "calibration_tokens": self.calibration_tokens,
            "calibration_time_ms": self.calibration_time_ms,
            "calibration_memory_bytes": self.calibration_memory_bytes,
            "stateful": self.stateful,
            "online_overhead_ms_per_token": self.online_overhead_ms_per_token,
        }


def derive_online_overhead_ms_per_token(
    cost: CostMetrics | Any,
    system: SystemMetrics | None,
) -> float | None:
    """Best available per-token online overhead for Phase 27 reporting.

    Priority:
    1. ``SYSTEM.throughput.latency_ms_per_token`` (always collected in default runs)
    2. ``cost.online.end_to_end_decode_cost_ms / generated_tokens`` when only aggregate exists
    3. Sum of measured kernel compress/decompress + attention per step (upper-bound proxy)
    """
    throughput = system.throughput if system else None
    if throughput is not None and throughput.latency_ms_per_token is not None:
        return float(throughput.latency_ms_per_token)

    online = cost.online
    if (
        throughput is not None
        and online.end_to_end_decode_cost_ms is not None
        and throughput.generated_tokens > 0
    ):
        return float(online.end_to_end_decode_cost_ms) / float(throughput.generated_tokens)

    parts = [
        online.compression_time_ms,
        online.decompression_time_ms,
        online.attention_cost_ms,
    ]
    if any(p is not None for p in parts):
        return sum(p or 0.0 for p in parts)

    return None


def build_benchmark_dimensions(
    compressor: KVCompressor,
    cost: CostMetrics | Any,
    *,
    system: SystemMetrics | None = None,
) -> BenchmarkDimensions:
    from eval.cost.oaken_taxonomy import compressor_is_stateful

    offline = cost.offline
    return BenchmarkDimensions(
        calibration_required=offline.calibration_required,
        calibration_dataset=offline.calibration_dataset,
        calibration_tokens=offline.calibration_tokens,
        calibration_time_ms=offline.calibration_time_ms,
        calibration_memory_bytes=offline.calibration_memory_bytes,
        stateful=compressor_is_stateful(compressor),
        online_overhead_ms_per_token=derive_online_overhead_ms_per_token(cost, system),
    )


def benchmark_dimensions_from_dict(payload: dict[str, Any]) -> BenchmarkDimensions | None:
    """Load Phase 27 columns from nested ``cost.benchmark_dimensions`` in job JSON."""
    block = payload.get("benchmark_dimensions")
    if block is None:
        cost = payload.get("cost") or {}
        block = cost.get("benchmark_dimensions")
    if not isinstance(block, dict):
        return None
    return BenchmarkDimensions(
        calibration_required=bool(block.get("calibration_required")),
        calibration_dataset=block.get("calibration_dataset"),
        calibration_tokens=block.get("calibration_tokens"),
        calibration_time_ms=block.get("calibration_time_ms"),
        calibration_memory_bytes=block.get("calibration_memory_bytes"),
        stateful=bool(block.get("stateful")),
        online_overhead_ms_per_token=block.get("online_overhead_ms_per_token"),
    )
