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

__all__ = [
    "CompressionCostMetrics",
    "CostMetrics",
    "OfflineCostMetrics",
    "OnlineCostMetrics",
    "evaluate_cost",
]
