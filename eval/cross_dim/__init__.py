"""Phases 24–25: cross-dimensional correlation and trade-off visualization."""

from eval.cross_dim.correlation import (
    CrossDimCorrelationAnalysis,
    CorrelationPairSpec,
    CorrelationResult,
    DEFAULT_CORRELATION_PAIRS,
    analyze_correlations,
    pearson_r,
)
from eval.cross_dim.points import CrossDimPoint, extract_cross_dim_point, load_cross_dim_points_from_json

__all__ = [
    "CrossDimCorrelationAnalysis",
    "CrossDimPoint",
    "CorrelationPairSpec",
    "CorrelationResult",
    "DEFAULT_CORRELATION_PAIRS",
    "analyze_correlations",
    "extract_cross_dim_point",
    "load_cross_dim_points_from_json",
    "pearson_r",
]
