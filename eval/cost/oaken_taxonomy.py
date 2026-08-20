"""Oaken-style cost layer taxonomy (Phase 26).

Separates *offline evaluation* (FIDELITY metrics) from *offline preprocessing cost*
(calibration / codebooks) and online transformation, attention, and end-to-end serving
costs. Prevents conflating ``offline`` with ``free`` when discussing KV compression.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from compressors.base import KVCompressor
from eval.fidelity import FidelityMetrics
from eval.system import SystemMetrics


class OakenCostLayer(str, Enum):
    """Five-way cost/evaluation distinction (Oaken-inspired, Phase 26)."""

    OFFLINE_EVALUATION = "offline_evaluation"
    OFFLINE_PREPROCESSING = "offline_preprocessing"
    ONLINE_TRANSFORMATION = "online_transformation"
    ONLINE_ATTENTION = "online_attention"
    END_TO_END_SERVING = "end_to_end_serving"


OAKEN_LAYER_DESCRIPTIONS: dict[OakenCostLayer, str] = {
    OakenCostLayer.OFFLINE_EVALUATION: (
        "Static FIDELITY measurement before/around incremental decode "
        "(tensor RMSE, attention scores, memory accounting). Not monetary cost."
    ),
    OakenCostLayer.OFFLINE_PREPROCESSING: (
        "Method-specific offline preparation (calibration, codebooks, rank search). "
        "Paid once per compressor configuration, not per generated token."
    ),
    OakenCostLayer.ONLINE_TRANSFORMATION: (
        "Per-step compress/decompress work during autoregressive generation."
    ),
    OakenCostLayer.ONLINE_ATTENTION: (
        "Attention execution with transformed cache (standard or modified kernel)."
    ),
    OakenCostLayer.END_TO_END_SERVING: (
        "User-visible decode latency / throughput including all online components."
    ),
}


@dataclass(frozen=True)
class OakenLayerSnapshot:
    layer: str
    description: str
    measured: bool
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "description": self.description,
            "measured": self.measured,
            "metrics": self.metrics,
        }


def compressor_is_stateful(compressor: KVCompressor) -> bool:
    """True when the plug-in maintains cross-token state (``reset_state`` hook)."""
    return callable(getattr(compressor, "reset_state", None))


def build_oaken_layers(
    *,
    cost: Any,
    fidelity: FidelityMetrics | None = None,
    system: SystemMetrics | None = None,
) -> list[OakenLayerSnapshot]:
    """Map FIDELITY / COST / SYSTEM blocks onto the five Oaken layers."""
    fidelity_metrics: dict[str, Any] = {}
    if fidelity is not None:
        memory = getattr(fidelity, "memory", None)
        representation = getattr(fidelity, "representation", None)
        attention = getattr(fidelity, "attention", None)
        if representation is not None:
            fidelity_metrics["representation_rmse_key"] = getattr(representation, "key_rmse", None)
            fidelity_metrics["representation_rmse_value"] = getattr(representation, "value_rmse", None)
        if attention is not None:
            fidelity_metrics["attention_rmse"] = getattr(attention, "rmse", None)
            fidelity_metrics["attention_cosine"] = getattr(attention, "cosine_similarity", None)
        if memory is not None:
            fidelity_metrics["compression_ratio"] = getattr(memory, "compression_ratio", None)
            fidelity_metrics["effective_bits_per_kv"] = getattr(
                memory, "effective_bits_per_kv_element", None
            )

    offline = cost.offline
    online = cost.online
    throughput = system.throughput if system else None

    transform_ms = online.compress_decompress_time_ms
    if transform_ms is None and online.compression_time_ms is not None:
        transform_ms = (online.compression_time_ms or 0.0) + (online.decompression_time_ms or 0.0)

    e2e_ms = online.end_to_end_decode_cost_ms
    if e2e_ms is None and throughput is not None:
        e2e_ms = throughput.end_to_end_latency_ms

    per_token_ms = None
    if throughput is not None and throughput.latency_ms_per_token is not None:
        per_token_ms = throughput.latency_ms_per_token

    return [
        OakenLayerSnapshot(
            layer=OakenCostLayer.OFFLINE_EVALUATION.value,
            description=OAKEN_LAYER_DESCRIPTIONS[OakenCostLayer.OFFLINE_EVALUATION],
            measured=bool(fidelity_metrics),
            metrics=fidelity_metrics,
        ),
        OakenLayerSnapshot(
            layer=OakenCostLayer.OFFLINE_PREPROCESSING.value,
            description=OAKEN_LAYER_DESCRIPTIONS[OakenCostLayer.OFFLINE_PREPROCESSING],
            measured=offline.calibration_required or offline.calibration_time_ms is not None,
            metrics={
                "calibration_required": offline.calibration_required,
                "calibration_dataset": offline.calibration_dataset,
                "calibration_tokens": offline.calibration_tokens,
                "calibration_time_ms": offline.calibration_time_ms,
                "calibration_memory_bytes": offline.calibration_memory_bytes,
            },
        ),
        OakenLayerSnapshot(
            layer=OakenCostLayer.ONLINE_TRANSFORMATION.value,
            description=OAKEN_LAYER_DESCRIPTIONS[OakenCostLayer.ONLINE_TRANSFORMATION],
            measured=transform_ms is not None,
            metrics={
                "compression_time_ms": online.compression_time_ms,
                "decompression_time_ms": online.decompression_time_ms,
                "compress_decompress_time_ms": online.compress_decompress_time_ms,
                "compress_decompress_overhead_frac": online.compress_decompress_overhead_frac,
            },
        ),
        OakenLayerSnapshot(
            layer=OakenCostLayer.ONLINE_ATTENTION.value,
            description=OAKEN_LAYER_DESCRIPTIONS[OakenCostLayer.ONLINE_ATTENTION],
            measured=online.attention_cost_ms is not None,
            metrics={"attention_cost_ms": online.attention_cost_ms},
        ),
        OakenLayerSnapshot(
            layer=OakenCostLayer.END_TO_END_SERVING.value,
            description=OAKEN_LAYER_DESCRIPTIONS[OakenCostLayer.END_TO_END_SERVING],
            measured=e2e_ms is not None or per_token_ms is not None,
            metrics={
                "end_to_end_decode_cost_ms": e2e_ms,
                "tokens_per_second": throughput.tokens_per_second if throughput else None,
                "latency_ms_per_token": per_token_ms,
            },
        ),
    ]
