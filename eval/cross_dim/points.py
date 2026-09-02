"""Extract cross-dimensional metric points from evaluation payloads (Phases 24–25)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from eval.runner import EvaluationResult


def _dig(data: dict[str, Any], *path: str) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _safe_log10_ratio(ratio: float, *, floor: float = 1e-12) -> float:
    return math.log10(max(ratio, floor))


def _quality_score(log10_perplexity_ratio: float | None) -> float | None:
    if log10_perplexity_ratio is None:
        return None
    return 1.0 / (1.0 + max(0.0, log10_perplexity_ratio))


@dataclass(frozen=True)
class CrossDimPoint:
    """One job configuration with metrics for correlation / trade-off plots."""

    point_id: str
    compressor: str
    context_length: int
    compression_ratio: float | None
    theoretical_compression_ratio: float | None
    perplexity_ratio: float | None
    log10_perplexity_ratio: float | None
    quality_score: float | None
    attention_rmse: float | None
    key_rmse: float | None
    value_rmse: float | None
    tokens_per_second: float | None
    latency_ms_per_token: float | None
    online_overhead_ms: float | None
    retrieval_accuracy: float | None
    instruction_compliance: float | None
    bitwidth: int | None = None
    stage: str | None = None
    label: str | None = None

    def metric(self, name: str) -> float | None:
        return getattr(self, name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "point_id": self.point_id,
            "label": self.label or self.point_id,
            "compressor": self.compressor,
            "context_length": self.context_length,
            "compression_ratio": self.compression_ratio,
            "theoretical_compression_ratio": self.theoretical_compression_ratio,
            "perplexity_ratio": self.perplexity_ratio,
            "log10_perplexity_ratio": self.log10_perplexity_ratio,
            "quality_score": self.quality_score,
            "attention_rmse": self.attention_rmse,
            "key_rmse": self.key_rmse,
            "value_rmse": self.value_rmse,
            "tokens_per_second": self.tokens_per_second,
            "latency_ms_per_token": self.latency_ms_per_token,
            "online_overhead_ms": self.online_overhead_ms,
            "retrieval_accuracy": self.retrieval_accuracy,
            "instruction_compliance": self.instruction_compliance,
            "bitwidth": self.bitwidth,
            "stage": self.stage,
        }


def _point_id(
    compressor: str,
    context_length: int,
    stage: str | None,
    bitwidth: int | None,
    label: str | None = None,
) -> str:
    if label:
        safe = str(label).replace(" ", "_")
        return f"{safe}_ctx{context_length}"
    suffix_parts: list[str] = []
    if stage:
        suffix_parts.append(str(stage))
    if bitwidth is not None:
        suffix_parts.append(f"b{bitwidth}")
    suffix = "_".join(suffix_parts) if suffix_parts else "default"
    return f"{compressor}_{suffix}_ctx{context_length}"


def extract_cross_dim_point(record: EvaluationResult | dict[str, Any]) -> CrossDimPoint | None:
    """Build a CrossDimPoint from an evaluation result or job JSON dict."""
    if isinstance(record, EvaluationResult):
        payload = record.to_dict()
    else:
        payload = record

    compressor = payload.get("compressor") or _dig(payload, "job", "compressor")
    context_length = payload.get("context_length") or _dig(payload, "job", "context_length")
    if compressor is None or context_length is None:
        return None

    bitwidth = payload.get("bitwidth")
    if bitwidth is None:
        bitwidth = _dig(payload, "job", "bitwidth")
    stage = payload.get("stage") or _dig(payload, "job", "stage")

    compression_ratio = _dig(payload, "fidelity", "memory", "compression_ratio")
    if compression_ratio is None:
        compression_ratio = _dig(payload, "section_a_fidelity", "memory", "compression_ratio")

    theoretical_compression_ratio = _dig(payload, "cost", "compression", "theoretical_compression_ratio")

    attention_rmse = _dig(payload, "fidelity", "attention", "rmse")
    if attention_rmse is None:
        attention_rmse = _dig(payload, "section_a_fidelity", "attention", "rmse")

    key_rmse = _dig(payload, "fidelity", "representation", "key_rmse")
    if key_rmse is None:
        key_rmse = _dig(payload, "section_a_fidelity", "tensor", "key_rmse")

    value_rmse = _dig(payload, "fidelity", "representation", "value_rmse")
    if value_rmse is None:
        value_rmse = _dig(payload, "section_a_fidelity", "tensor", "value_rmse")

    perplexity = _dig(payload, "behavior", "task_quality", "perplexity")
    if perplexity is None:
        perplexity = _dig(payload, "behavior", "perplexity")
    if perplexity is None:
        perplexity = _dig(payload, "section_b_inference", "perplexity")
    perplexity_baseline = _dig(payload, "behavior", "task_quality", "perplexity_baseline")
    if perplexity_baseline is None:
        perplexity_baseline = _dig(payload, "behavior", "perplexity_baseline")
    if perplexity_baseline is None:
        perplexity_baseline = _dig(payload, "section_b_inference", "perplexity_baseline")

    tokens_per_second = _dig(payload, "system", "latency_throughput", "tokens_per_second")
    latency_ms_per_token = _dig(payload, "system", "latency_throughput", "latency_ms_per_token")
    if tokens_per_second is None:
        tokens_per_second = _dig(payload, "system", "throughput", "tokens_per_second")
        latency_ms_per_token = _dig(payload, "system", "throughput", "latency_ms_per_token")
    if tokens_per_second is None:
        throughput = _dig(payload, "section_b_inference", "throughput") or {}
        if throughput.get("online_compressed_kv"):
            tokens_per_second = throughput.get("tokens_per_second")
            latency_ms_per_token = throughput.get("latency_ms_per_token")

    online_overhead_ms = _dig(payload, "cost", "online", "end_to_end_decode_cost_ms")
    if online_overhead_ms is None:
        online_overhead_ms = _dig(payload, "system", "kernel_cost", "compress_decompress_time_ms")
    if online_overhead_ms is None:
        online_overhead_ms = latency_ms_per_token

    retrieval_accuracy = _dig(payload, "behavior", "retrieval", "exact_match_accuracy")
    instruction_compliance = _dig(payload, "behavior", "instruction_following", "format_compliance_rate")

    perplexity_ratio: float | None = None
    log10_ppl_ratio: float | None = None
    if perplexity is not None and perplexity_baseline is not None and perplexity_baseline > 0:
        perplexity_ratio = float(perplexity) / float(perplexity_baseline)
        log10_ppl_ratio = _safe_log10_ratio(perplexity_ratio)

    label = payload.get("label") or _dig(payload, "job", "label")
    pid = _point_id(str(compressor), int(context_length), stage, bitwidth, label)

    return CrossDimPoint(
        point_id=pid,
        compressor=str(compressor),
        context_length=int(context_length),
        compression_ratio=None if compression_ratio is None else float(compression_ratio),
        theoretical_compression_ratio=(
            None if theoretical_compression_ratio is None else float(theoretical_compression_ratio)
        ),
        perplexity_ratio=perplexity_ratio,
        log10_perplexity_ratio=log10_ppl_ratio,
        quality_score=_quality_score(log10_ppl_ratio),
        attention_rmse=None if attention_rmse is None else float(attention_rmse),
        key_rmse=None if key_rmse is None else float(key_rmse),
        value_rmse=None if value_rmse is None else float(value_rmse),
        tokens_per_second=None if tokens_per_second is None else float(tokens_per_second),
        latency_ms_per_token=None if latency_ms_per_token is None else float(latency_ms_per_token),
        online_overhead_ms=None if online_overhead_ms is None else float(online_overhead_ms),
        retrieval_accuracy=None if retrieval_accuracy is None else float(retrieval_accuracy),
        instruction_compliance=None if instruction_compliance is None else float(instruction_compliance),
        bitwidth=bitwidth,
        stage=stage,
        label=label,
    )


def load_cross_dim_points_from_results(results: Iterable[EvaluationResult]) -> list[CrossDimPoint]:
    points: list[CrossDimPoint] = []
    for item in results:
        pt = extract_cross_dim_point(item)
        if pt is not None:
            points.append(pt)
    return points


def load_cross_dim_points_from_json(path: str | Any) -> list[CrossDimPoint]:
    import json
    from pathlib import Path

    path = Path(path)
    data = json.loads(path.read_text())
    records: list[dict[str, Any]]
    if "results" in data and isinstance(data["results"], list):
        records = data["results"]
    else:
        records = [data]
    points: list[CrossDimPoint] = []
    for record in records:
        pt = extract_cross_dim_point(record)
        if pt is not None:
            points.append(pt)
    return points
