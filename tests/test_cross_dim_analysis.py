"""Tests for Phases 24-25 cross-dimensional analysis."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from eval.cross_dim.correlation import analyze_correlations, pearson_r
from eval.cross_dim.points import CrossDimPoint, extract_cross_dim_point, load_cross_dim_points_from_json


def _point(
    pid: str,
    *,
    ratio: float = 3.0,
    ppl_ratio: float = 1.3,
    attn_rmse: float = 2.5,
    tok_s: float = 0.1,
    latency: float = 10000.0,
    ctx: int = 512,
) -> CrossDimPoint:
    log10_r = math.log10(max(ppl_ratio, 1e-12))
    return CrossDimPoint(
        point_id=pid,
        compressor=pid.split("_")[0],
        context_length=ctx,
        compression_ratio=ratio,
        theoretical_compression_ratio=ratio * 0.95,
        perplexity_ratio=ppl_ratio,
        log10_perplexity_ratio=log10_r,
        quality_score=1.0 / (1.0 + max(0.0, log10_r)),
        attention_rmse=attn_rmse,
        key_rmse=1.0,
        value_rmse=2.0,
        tokens_per_second=tok_s,
        latency_ms_per_token=latency,
        online_overhead_ms=latency,
        retrieval_accuracy=0.5,
        instruction_compliance=0.6,
    )


def test_pearson_perfect_positive():
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [2.0, 4.0, 6.0, 8.0]
    assert pearson_r(xs, ys) == pytest.approx(1.0)


def test_pearson_insufficient_samples():
    assert pearson_r([1.0, 2.0], [3.0, 4.0]) is None


def test_extract_cross_dim_legacy_shape():
    payload = {
        "compressor": "turboquant",
        "context_length": 512,
        "bitwidth": 4,
        "stage": "full",
        "section_a_fidelity": {
            "tensor": {"key_rmse": 0.36, "value_rmse": 0.84},
            "attention": {"rmse": 2.49, "cosine_similarity": 0.602},
            "memory": {"compression_ratio": 3.12},
        },
        "section_b_inference": {
            "perplexity": 18.6,
            "perplexity_baseline": 14.1,
            "throughput": {
                "tokens_per_second": 0.08,
                "latency_ms_per_token": 12360.0,
                "online_compressed_kv": True,
            },
        },
    }
    pt = extract_cross_dim_point(payload)
    assert pt is not None
    assert pt.compression_ratio == pytest.approx(3.12)
    assert pt.attention_rmse == pytest.approx(2.49)
    assert pt.perplexity_ratio == pytest.approx(18.6 / 14.1)
    assert pt.tokens_per_second == pytest.approx(0.08)


def test_analyze_correlations_export():
    points = [
        _point("a", ratio=3.0, ppl_ratio=1.2, attn_rmse=2.0, tok_s=0.1),
        _point("b", ratio=2.0, ppl_ratio=10.0, attn_rmse=8.0, tok_s=5.0),
        _point("c", ratio=1.5, ppl_ratio=100.0, attn_rmse=15.0, tok_s=15.0),
    ]
    analysis = analyze_correlations(points, context_length=512)
    payload = analysis.to_dict()
    assert payload["point_count"] == 3
    assert len(payload["pairs"]) >= 6
    attn_ppl = next(
        p
        for p in payload["pairs"]
        if p["metric_x"] == "attention_rmse" and p["metric_y"] == "perplexity_ratio"
    )
    assert attn_ppl["sample_size"] == 3
    assert attn_ppl["pearson_r"] is not None


def test_load_from_phase5_bundle():
    root = Path(__file__).resolve().parent.parent / "results" / "phase5_modal_sweep_128_256_512"
    bundles = list(root.glob("phase5_modal_sweep_128_256_512_*.json"))
    if not bundles:
        pytest.skip("Phase-5 sweep bundle not present")
    points = load_cross_dim_points_from_json(bundles[-1])
    assert points
    ctx512 = [p for p in points if p.context_length == 512]
    assert ctx512
    analysis = analyze_correlations(points, context_length=512, exclude_compressors=["identity"])
    assert analysis.point_count > 0
    json.dumps(analysis.to_dict())
