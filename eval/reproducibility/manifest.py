"""Phase 14 reproducibility manifest extraction (RESEARCH_REDESIGN_PLAN §Phase 14)."""

from __future__ import annotations

from typing import Any

# Canonical checklist from Phase 14 YAML block.
PHASE14_FIELDS: tuple[str, ...] = (
    "model",
    "context_length",
    "generation_length",
    "hardware",
    "batch_size",
    "compression_method",
    "compression_ratio",
    "calibration",
    "dataset",
    "seed",
    "precision",
)


def extract_phase14_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    """Build the standardized Phase 14 configuration dict from ``EvaluationResult.to_dict()``."""
    controlled = payload.get("controlled_conditions") or {}
    fixed = controlled.get("fixed") or {}
    variable = controlled.get("variable") or {}
    budget = variable.get("compression_budget") or {}

    fidelity = payload.get("fidelity") or {}
    memory = fidelity.get("memory") or {}
    cost = payload.get("cost") or {}
    compression_cost = cost.get("compression") or {}
    offline = cost.get("offline") or {}

    hardware = payload.get("hardware") or fixed.get("hardware")

    return {
        "model": fixed.get("model") or payload.get("model"),
        "context_length": fixed.get("context_length", payload.get("context_length")),
        "generation_length": fixed.get("generation_length"),
        "hardware": hardware,
        "batch_size": fixed.get("batch_size"),
        "compression_method": variable.get("compressor") or budget.get("compression_method"),
        "compression_ratio": {
            "measured": memory.get("compression_ratio"),
            "theoretical": compression_cost.get("theoretical_compression_ratio"),
            "actual": compression_cost.get("actual_compression_ratio"),
        },
        "calibration": offline,
        "dataset": fixed.get("dataset"),
        "seed": budget.get("seed"),
        "precision": fixed.get("precision"),
        "compression_budget": budget,
    }


def validate_phase14_manifest(
    payload: dict[str, Any],
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> list[str]:
    """Return human-readable errors if the payload violates Phase 14 invariants."""
    errors: list[str] = []
    manifest = extract_phase14_manifest(payload)

    for field in PHASE14_FIELDS:
        if manifest.get(field) is None:
            errors.append(f"missing Phase 14 field: {field}")

    controlled = payload.get("controlled_conditions") or {}
    fixed = controlled.get("fixed") or {}
    variable = controlled.get("variable") or {}
    budget = variable.get("compression_budget") or {}

    compressor = payload.get("compressor")
    if variable.get("compressor") != compressor:
        errors.append("controlled_conditions.variable.compressor != payload.compressor")
    if budget.get("compression_method") != compressor:
        errors.append("compression_budget.compression_method != payload.compressor")
    if manifest["compression_method"] != compressor:
        errors.append("manifest.compression_method != payload.compressor")

    ctx = payload.get("context_length")
    if fixed.get("context_length") != ctx:
        errors.append("fixed.context_length != payload.context_length")

    fidelity = payload.get("fidelity") or {}
    memory = fidelity.get("memory") or {}
    cost = payload.get("cost") or {}
    compression_cost = cost.get("compression") or {}

    measured = memory.get("compression_ratio")
    uncompressed = memory.get("uncompressed_bytes")
    compressed = memory.get("compressed_bytes")
    if measured is not None and uncompressed is not None and compressed is not None and compressed > 0:
        expected_ratio = uncompressed / compressed
        if abs(measured - expected_ratio) > max(atol, rtol * abs(expected_ratio)):
            errors.append(
                f"memory.compression_ratio {measured} != uncompressed/compressed {expected_ratio}"
            )

    actual_cost_ratio = compression_cost.get("actual_compression_ratio")
    if measured is not None and actual_cost_ratio is not None and measured != actual_cost_ratio:
        errors.append("cost.compression.actual_compression_ratio != fidelity.memory.compression_ratio")

    reduction = compression_cost.get("actual_memory_reduction_bytes")
    if (
        reduction is not None
        and uncompressed is not None
        and compressed is not None
        and reduction != uncompressed - compressed
    ):
        errors.append("cost.compression.actual_memory_reduction_bytes != uncompressed - compressed")

    theoretical = compression_cost.get("theoretical_compression_ratio")
    if theoretical is not None and theoretical <= 0:
        errors.append("theoretical_compression_ratio must be positive")

    return errors
