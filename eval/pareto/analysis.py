"""Pareto-optimal trade-off analysis (Phase 9).

Matches the paper figure semantics at ``T=512``:
  - horizontal axis: memory compression ratio (higher is better)
  - vertical axis: ``log10(perplexity / baseline)`` (lower is better)
  - marker size proxy: tokens/sec (higher is better; optional third objective)

Works on ``EvaluationResult`` objects, ``to_dict()`` payloads, and legacy
Phase-5 bundle JSON (``section_a_fidelity`` / ``section_b_inference``).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Sequence

from eval.runner import EvaluationResult


class ParetoObjective(str, Enum):
    """Direction of optimization for one axis."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass(frozen=True)
class ParetoPoint:
    """One configuration in the quality / memory / speed trade-off space."""

    point_id: str
    compressor: str
    context_length: int
    compression_ratio: float
    perplexity: float
    perplexity_baseline: float
    perplexity_ratio: float
    log10_perplexity_ratio: float
    tokens_per_second: float | None
    bitwidth: int | None = None
    stage: str | None = None
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "point_id": self.point_id,
            "label": self.label or self.point_id,
            "compressor": self.compressor,
            "context_length": self.context_length,
            "compression_ratio": self.compression_ratio,
            "perplexity": self.perplexity,
            "perplexity_baseline": self.perplexity_baseline,
            "perplexity_ratio": self.perplexity_ratio,
            "log10_perplexity_ratio": self.log10_perplexity_ratio,
            "tokens_per_second": self.tokens_per_second,
            "bitwidth": self.bitwidth,
            "stage": self.stage,
        }


@dataclass
class ParetoAnalysis:
    """Pareto analysis for one context length (or filtered slice)."""

    context_length: int | None
    points: list[ParetoPoint]
    pareto_optimal_ids: list[str]
    frontier_2d: list[ParetoPoint]
    objectives_3d: list[tuple[str, ParetoObjective]] = field(
        default_factory=lambda: [
            ("compression_ratio", ParetoObjective.MAXIMIZE),
            ("perplexity_ratio", ParetoObjective.MINIMIZE),
            ("tokens_per_second", ParetoObjective.MAXIMIZE),
        ]
    )

    def to_dict(self) -> dict[str, Any]:
        optimal = {pid for pid in self.pareto_optimal_ids}
        return {
            "context_length": self.context_length,
            "objectives_3d": [{"metric": m, "direction": d.value} for m, d in self.objectives_3d],
            "axes_2d": {
                "x": "compression_ratio",
                "x_direction": ParetoObjective.MAXIMIZE.value,
                "y": "log10_perplexity_ratio",
                "y_direction": ParetoObjective.MINIMIZE.value,
                "marker_size": "tokens_per_second",
            },
            "point_count": len(self.points),
            "pareto_optimal_count": len(self.pareto_optimal_ids),
            "points": [
                {**p.to_dict(), "pareto_optimal": p.point_id in optimal}
                for p in self.points
            ],
            "frontier_2d": [p.to_dict() for p in self.frontier_2d],
            "pareto_optimal_ids": self.pareto_optimal_ids,
        }


def _dig(data: dict[str, Any], *path: str) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


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


def _safe_log10_ratio(ratio: float, *, floor: float = 1e-12) -> float:
    return math.log10(max(ratio, floor))


def extract_pareto_point(record: EvaluationResult | dict[str, Any]) -> ParetoPoint | None:
    """Build a ParetoPoint from an evaluation result or job JSON dict."""
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

    perplexity = _dig(payload, "behavior", "perplexity")
    if perplexity is None:
        perplexity = _dig(payload, "section_b_inference", "perplexity")
    perplexity_baseline = _dig(payload, "behavior", "perplexity_baseline")
    if perplexity_baseline is None:
        perplexity_baseline = _dig(payload, "section_b_inference", "perplexity_baseline")

    tokens_per_second = _dig(payload, "system", "throughput", "tokens_per_second")
    if tokens_per_second is None:
        throughput = _dig(payload, "section_b_inference", "throughput") or {}
        if throughput.get("online_compressed_kv"):
            tokens_per_second = throughput.get("tokens_per_second")
        else:
            tokens_per_second = _dig(payload, "section_b_inference", "throughput_baseline", "tokens_per_second")

    if compression_ratio is None or perplexity is None or perplexity_baseline is None:
        return None
    if perplexity_baseline <= 0:
        return None

    ratio = float(perplexity) / float(perplexity_baseline)
    label = payload.get("label") or _dig(payload, "job", "label")
    pid = _point_id(str(compressor), int(context_length), stage, bitwidth, label)

    return ParetoPoint(
        point_id=pid,
        compressor=str(compressor),
        context_length=int(context_length),
        compression_ratio=float(compression_ratio),
        perplexity=float(perplexity),
        perplexity_baseline=float(perplexity_baseline),
        perplexity_ratio=ratio,
        log10_perplexity_ratio=_safe_log10_ratio(ratio),
        tokens_per_second=None if tokens_per_second is None else float(tokens_per_second),
        bitwidth=bitwidth,
        stage=stage,
        label=label,
    )


def load_pareto_points_from_results(results: Iterable[EvaluationResult]) -> list[ParetoPoint]:
    points: list[ParetoPoint] = []
    for item in results:
        pt = extract_pareto_point(item)
        if pt is not None:
            points.append(pt)
    return points


def load_pareto_points_from_json(path: Path | str) -> list[ParetoPoint]:
    """Load points from a merged bundle JSON or a single job JSON."""
    path = Path(path)
    data = json.loads(path.read_text())
    records: list[dict[str, Any]]
    if "results" in data and isinstance(data["results"], list):
        records = data["results"]
    else:
        records = [data]
    points: list[ParetoPoint] = []
    for record in records:
        pt = extract_pareto_point(record)
        if pt is not None:
            points.append(pt)
    return points


def _metric_value(point: ParetoPoint, metric: str) -> float | None:
    if metric == "compression_ratio":
        return point.compression_ratio
    if metric == "perplexity_ratio":
        return point.perplexity_ratio
    if metric == "log10_perplexity_ratio":
        return point.log10_perplexity_ratio
    if metric == "tokens_per_second":
        return point.tokens_per_second
    raise KeyError(f"unknown pareto metric: {metric}")


def _dominates(
    a: ParetoPoint,
    b: ParetoPoint,
    objectives: Sequence[tuple[str, ParetoObjective]],
) -> bool:
    """True if ``a`` Pareto-dominates ``b``."""
    if a.point_id == b.point_id:
        return False

    strictly_better = False
    for metric, direction in objectives:
        av = _metric_value(a, metric)
        bv = _metric_value(b, metric)
        if av is None or bv is None:
            return False
        if direction is ParetoObjective.MAXIMIZE:
            if av < bv:
                return False
            if av > bv:
                strictly_better = True
        else:
            if av > bv:
                return False
            if av < bv:
                strictly_better = True
    return strictly_better


def compute_pareto_frontier(
    points: Sequence[ParetoPoint],
    *,
    objectives: Sequence[tuple[str, ParetoObjective]] | None = None,
) -> list[ParetoPoint]:
    """Return non-dominated points for the given objectives."""
    if not points:
        return []
    objs = objectives or [
        ("compression_ratio", ParetoObjective.MAXIMIZE),
        ("perplexity_ratio", ParetoObjective.MINIMIZE),
        ("tokens_per_second", ParetoObjective.MAXIMIZE),
    ]
    frontier: list[ParetoPoint] = []
    for candidate in points:
        if any(_dominates(other, candidate, objs) for other in points):
            continue
        frontier.append(candidate)
    return sorted(frontier, key=lambda p: (p.compression_ratio, p.log10_perplexity_ratio))


def compute_frontier_2d(
    points: Sequence[ParetoPoint],
) -> list[ParetoPoint]:
    """Paper-style 2D frontier: max compression ratio, min log10 PPL ratio."""
    return compute_pareto_frontier(
        points,
        objectives=[
            ("compression_ratio", ParetoObjective.MAXIMIZE),
            ("log10_perplexity_ratio", ParetoObjective.MINIMIZE),
        ],
    )


def analyze_pareto(
    points: Sequence[ParetoPoint],
    *,
    context_length: int | None = None,
    exclude_compressors: Sequence[str] | None = None,
) -> ParetoAnalysis:
    """Analyze a slice of points; optionally filter by context length."""
    filtered = list(points)
    if context_length is not None:
        filtered = [p for p in filtered if p.context_length == context_length]
    if exclude_compressors:
        blocked = {c.lower() for c in exclude_compressors}
        filtered = [p for p in filtered if p.compressor.lower() not in blocked]

    frontier_3d = compute_pareto_frontier(filtered)
    frontier_2d = compute_frontier_2d(filtered)

    return ParetoAnalysis(
        context_length=context_length,
        points=sorted(filtered, key=lambda p: (p.compression_ratio, p.log10_perplexity_ratio)),
        pareto_optimal_ids=[p.point_id for p in frontier_3d],
        frontier_2d=frontier_2d,
    )
