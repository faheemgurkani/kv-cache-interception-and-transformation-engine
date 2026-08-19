"""Tests for Phase 9 Pareto frontier analysis."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from eval.pareto.analysis import (
    ParetoObjective,
    ParetoPoint,
    analyze_pareto,
    compute_frontier_2d,
    compute_pareto_frontier,
    extract_pareto_point,
    load_pareto_points_from_json,
)


def _point(
    pid: str,
    *,
    ratio: float,
    ppl_ratio: float,
    tok_s: float | None = 10.0,
    ctx: int = 512,
) -> ParetoPoint:
    return ParetoPoint(
        point_id=pid,
        compressor=pid.split("_")[0],
        context_length=ctx,
        compression_ratio=ratio,
        perplexity=ppl_ratio,
        perplexity_baseline=1.0,
        perplexity_ratio=ppl_ratio,
        log10_perplexity_ratio=math.log10(max(ppl_ratio, 1e-12)),
        tokens_per_second=tok_s,
    )


def test_log10_perplexity_ratio_math():
    payload = {
        "compressor": "turboquant",
        "context_length": 512,
        "bitwidth": 4,
        "stage": "full",
        "fidelity": {"memory": {"compression_ratio": 3.1}},
        "behavior": {"perplexity": 18.6, "perplexity_baseline": 14.1},
        "system": {"throughput": {"tokens_per_second": 0.08, "online_compressed_kv": True}},
    }
    pt = extract_pareto_point(payload)
    assert pt is not None
    assert pt.compression_ratio == pytest.approx(3.1)
    assert pt.perplexity_ratio == pytest.approx(18.6 / 14.1)
    assert pt.log10_perplexity_ratio == pytest.approx(math.log10(18.6 / 14.1))


def test_2d_pareto_frontier_dominance():
    # A: high compression, low ppl ratio — should dominate C
    # B: low compression, very low ppl — on frontier
    # C: dominated by A (same ppl ratio, worse compression)
    points = [
        _point("a", ratio=3.0, ppl_ratio=1.3),
        _point("b", ratio=1.0, ppl_ratio=1.01),
        _point("c", ratio=2.0, ppl_ratio=1.3),
    ]
    front = compute_frontier_2d(points)
    ids = {p.point_id for p in front}
    assert "a" in ids
    assert "b" in ids
    assert "c" not in ids


def test_3d_pareto_requires_throughput():
    fast_bad = _point("fast_bad", ratio=2.0, ppl_ratio=100.0, tok_s=20.0)
    slow_good = _point("slow_good", ratio=2.0, ppl_ratio=1.05, tok_s=0.1)
    front = compute_pareto_frontier([fast_bad, slow_good])
    assert {p.point_id for p in front} == {"fast_bad", "slow_good"}


def test_analyze_pareto_filters_context():
    points = [
        _point("a128", ratio=1.0, ppl_ratio=1.0, ctx=128),
        _point("a512", ratio=2.0, ppl_ratio=1.1, ctx=512),
    ]
    analysis = analyze_pareto(points, context_length=512)
    assert len(analysis.points) == 1
    assert analysis.points[0].point_id == "a512"


def test_extract_legacy_bundle_json():
    root = Path(__file__).resolve().parent.parent / "results" / "phase5_modal_baseline"
    bundles = list(root.glob("phase5_modal_baseline_*.json"))
    if not bundles:
        pytest.skip("Phase-5 baseline bundle not present")
    points = load_pareto_points_from_json(bundles[-1])
    assert points
    assert all(p.compression_ratio == 1.0 for p in points)
    assert all(p.perplexity > 0 for p in points)


def test_pareto_analysis_export_roundtrip():
    points = [_point("tq", ratio=3.1, ppl_ratio=1.3), _point("rk", ratio=2.0, ppl_ratio=1e6)]
    analysis = analyze_pareto(points, context_length=512)
    payload = analysis.to_dict()
    assert payload["context_length"] == 512
    assert payload["point_count"] == 2
    assert any(row["pareto_optimal"] for row in payload["points"])
    assert payload["frontier_2d"]
