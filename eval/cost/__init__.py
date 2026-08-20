"""COST: explicit per-method compression, offline, and online cost accounting (Phase 3).

Aggregates FIDELITY/memory storage metrics, compressor-declared offline calibration
cost, and SYSTEM timing into a unified schema for fair method comparison.
"""

from __future__ import annotations

from eval.cost.accounting import (
    CompressionCostMetrics,
    CostMetrics,
    OfflineCostMetrics,
    OnlineCostMetrics,
    evaluate_cost,
)
from eval.cost.benchmark_dimensions import BenchmarkDimensions, build_benchmark_dimensions
from eval.cost.oaken_taxonomy import OakenLayerSnapshot, OakenCostLayer, build_oaken_layers

__all__ = [
    "BenchmarkDimensions",
    "CompressionCostMetrics",
    "CostMetrics",
    "OakenCostLayer",
    "OakenLayerSnapshot",
    "OfflineCostMetrics",
    "OnlineCostMetrics",
    "build_benchmark_dimensions",
    "build_oaken_layers",
    "evaluate_cost",
]
