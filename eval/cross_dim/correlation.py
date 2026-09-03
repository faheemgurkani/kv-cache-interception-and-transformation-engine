"""Cross-dimensional Pearson correlation analysis (Phase 24)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from eval.cross_dim.points import CrossDimPoint


@dataclass(frozen=True)
class CorrelationPairSpec:
    metric_x: str
    metric_y: str
    label: str
    finding_id: str | None = None


DEFAULT_CORRELATION_PAIRS: tuple[CorrelationPairSpec, ...] = (
    CorrelationPairSpec(
        "theoretical_compression_ratio",
        "compression_ratio",
        "Theoretical compression ratio ↔ measured memory compression ratio",
        "F2",
    ),
    CorrelationPairSpec(
        "compression_ratio",
        "perplexity_ratio",
        "Compression ratio ↔ perplexity ratio",
        "F2",
    ),
    CorrelationPairSpec(
        "attention_rmse",
        "perplexity_ratio",
        "Reconstruction error (attention RMSE) ↔ perplexity ratio",
        "F1",
    ),
    CorrelationPairSpec(
        "attention_rmse",
        "retrieval_accuracy",
        "Reconstruction error ↔ task accuracy (retrieval)",
        "F1",
    ),
    CorrelationPairSpec(
        "compression_ratio",
        "tokens_per_second",
        "Memory compression ratio ↔ throughput",
        "F3",
    ),
    CorrelationPairSpec(
        "online_overhead_ms",
        "tokens_per_second",
        "Online overhead ↔ throughput",
        "F3",
    ),
)


@dataclass(frozen=True)
class CorrelationResult:
    metric_x: str
    metric_y: str
    label: str
    finding_id: str | None
    sample_size: int
    pearson_r: float | None
    interpretable: bool

    def to_dict(self) -> dict:
        return {
            "metric_x": self.metric_x,
            "metric_y": self.metric_y,
            "label": self.label,
            "finding_id": self.finding_id,
            "sample_size": self.sample_size,
            "pearson_r": self.pearson_r,
            "interpretable": self.interpretable,
        }


@dataclass
class CrossDimCorrelationAnalysis:
    context_length: int | None
    point_count: int
    pairs: list[CorrelationResult] = field(default_factory=list)
    summary_question: str = "Which metrics actually predict real inference performance?"

    def to_dict(self) -> dict:
        return {
            "context_length": self.context_length,
            "point_count": self.point_count,
            "summary_question": self.summary_question,
            "pairs": [p.to_dict() for p in self.pairs],
            "weak_predictors": [
                p.label for p in self.pairs if p.interpretable and p.pearson_r is not None and abs(p.pearson_r) < 0.5
            ],
        }


def _paired_values(points: Sequence[CrossDimPoint], metric_x: str, metric_y: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for pt in points:
        x = pt.metric(metric_x)
        y = pt.metric(metric_y)
        if x is None or y is None:
            continue
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        xs.append(float(x))
        ys.append(float(y))
    return xs, ys


def pearson_r(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Sample Pearson correlation; returns None if undefined."""
    n = len(xs)
    if n != len(ys) or n < 3:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0.0 or den_y == 0.0:
        return None
    return num / (den_x * den_y)


def analyze_correlations(
    points: Sequence[CrossDimPoint],
    *,
    context_length: int | None = None,
    exclude_compressors: Sequence[str] | None = None,
    max_perplexity_ratio: float | None = 5.0,
    pair_specs: Sequence[CorrelationPairSpec] | None = None,
) -> CrossDimCorrelationAnalysis:
    filtered = list(points)
    if context_length is not None:
        filtered = [p for p in filtered if p.context_length == context_length]
    if exclude_compressors:
        blocked = {c.lower() for c in exclude_compressors}
        filtered = [p for p in filtered if p.compressor.lower() not in blocked]
    if max_perplexity_ratio is not None:
        filtered = [
            p
            for p in filtered
            if p.perplexity_ratio is None or p.perplexity_ratio <= max_perplexity_ratio
        ]

    specs = pair_specs or DEFAULT_CORRELATION_PAIRS
    results: list[CorrelationResult] = []
    for spec in specs:
        xs, ys = _paired_values(filtered, spec.metric_x, spec.metric_y)
        r = pearson_r(xs, ys)
        results.append(
            CorrelationResult(
                metric_x=spec.metric_x,
                metric_y=spec.metric_y,
                label=spec.label,
                finding_id=spec.finding_id,
                sample_size=len(xs),
                pearson_r=r,
                interpretable=len(xs) >= 3 and r is not None,
            )
        )

    return CrossDimCorrelationAnalysis(
        context_length=context_length,
        point_count=len(filtered),
        pairs=results,
    )
