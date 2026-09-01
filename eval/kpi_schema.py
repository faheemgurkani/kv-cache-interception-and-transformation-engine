"""KPI / result-payload completeness checks for taxonomy-coverage smokes.

Used by dummy local execution and post-Modal validation so we can tell whether
FIDELITY / BEHAVIOR / SYSTEM / COST / taxonomy / gates / Phase-7 contract /
Phase-14 manifest / Oaken layers actually landed in the collected JSON.
"""

from __future__ import annotations

import math
from typing import Any

from compressors.taxonomy import (
    CompressionCategory,
    STUB_METHODS,
    active_eval_methods,
    get_method_taxonomy,
    taxonomy_categories_covered,
)
from eval.reproducibility.manifest import validate_phase14_manifest

REQUIRED_PAYLOAD_PATHS: tuple[str, ...] = (
    "compressor",
    "context_length",
    "fidelity.representation.key_rmse",
    "fidelity.representation.value_rmse",
    "fidelity.attention.rmse",
    "fidelity.memory.compression_ratio",
    "fidelity.memory.uncompressed_bytes",
    "fidelity.memory.compressed_bytes",
    "fidelity.recurrent.applicable",
    "behavior.task_quality.perplexity",
    "behavior.retrieval.exact_match_accuracy",
    "behavior.instruction_following.format_compliance_rate",
    "system.latency_throughput.tokens_per_second",
    "system.latency_throughput.ttft_ms",
    "system.latency_throughput.itl_ms_mean",
    "system.latency_throughput.end_to_end_latency_ms",
    "cost.compression.theoretical_compression_ratio",
    "cost.offline.calibration_required",
    "cost.online.end_to_end_decode_cost_ms",
    "cost.oaken_layers",
    "cost.benchmark_dimensions",
    "taxonomy.name",
    "taxonomy.primary",
    "controlled_conditions.phase",
    "controlled_conditions.variable.compressor",
    "controlled_conditions.fixed.context_length",
    "hardware.execution_platform",
    "compatibility_gates",
)

SMOKE_EXTRA_PATHS: tuple[str, ...] = (
    "behavior.reasoning.exact_match_accuracy",
    "system.peak_memory.peak_allocated_mb",
    "system.memory_bandwidth.effective_bandwidth_gbps",
    "system.kernel_cost.compress_decompress_time_ms",
    "cost.online.kernel_cost_measured",
)

OAKEN_LAYER_NAMES: tuple[str, ...] = (
    "offline_evaluation",
    "offline_preprocessing",
    "online_transformation",
    "online_attention",
    "end_to_end_serving",
)

GATE_NAMES: tuple[str, ...] = ("loader_state", "attention", "state_semantics")


def _lookup(payload: dict[str, Any], dotted: str) -> Any:
    node: Any = payload
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def missing_paths(payload: dict[str, Any], paths: tuple[str, ...] = REQUIRED_PAYLOAD_PATHS) -> list[str]:
    missing: list[str] = []
    for path in paths:
        value = _lookup(payload, path)
        if value is None:
            missing.append(path)
    return missing


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return False


def validate_payload_invariants(
    payload: dict[str, Any],
    *,
    require_smoke_extras: bool = False,
    execution_platform: str | None = None,
) -> list[str]:
    """Return human-readable errors. Empty list means the payload is collectable + coherent."""
    errors: list[str] = []
    paths = REQUIRED_PAYLOAD_PATHS + (SMOKE_EXTRA_PATHS if require_smoke_extras else ())
    for path in missing_paths(payload, paths):
        errors.append(f"missing KPI path: {path}")

    compressor = payload.get("compressor")
    taxonomy = payload.get("taxonomy") or {}
    expected = get_method_taxonomy(str(compressor)) if compressor else None
    if compressor in STUB_METHODS:
        errors.append(f"stub compressor included in live eval: {compressor}")
    if expected is None:
        errors.append(f"no taxonomy metadata for compressor={compressor}")
    elif taxonomy.get("primary") != expected.primary.value:
        errors.append(
            f"taxonomy.primary mismatch: got {taxonomy.get('primary')} expected {expected.primary.value}"
        )
    elif taxonomy.get("name") != compressor:
        errors.append(f"taxonomy.name != compressor ({taxonomy.get('name')} != {compressor})")

    controlled = payload.get("controlled_conditions") or {}
    if controlled.get("phase") != "7":
        errors.append(f"controlled_conditions.phase={controlled.get('phase')} (expected '7')")
    variable = controlled.get("variable") or {}
    if variable.get("compressor") != compressor:
        errors.append("controlled_conditions.variable.compressor != payload.compressor")

    fidelity = payload.get("fidelity") or {}
    representation = fidelity.get("representation") or {}
    memory = fidelity.get("memory") or {}
    for key in ("key_rmse", "value_rmse"):
        if not _finite_number(representation.get(key)):
            errors.append(f"fidelity.representation.{key} is not a finite number")
    ratio = memory.get("compression_ratio")
    if not _finite_number(ratio) or float(ratio) <= 0:
        errors.append(f"compression_ratio must be finite and > 0, got {ratio}")

    ppl = _lookup(payload, "behavior.task_quality.perplexity")
    if not _finite_number(ppl) or float(ppl) <= 0:
        errors.append(f"perplexity must be finite and > 0, got {ppl}")

    tps = _lookup(payload, "system.latency_throughput.tokens_per_second")
    if not _finite_number(tps) or float(tps) <= 0:
        errors.append(f"tokens_per_second must be finite and > 0, got {tps}")

    oaken = _lookup(payload, "cost.oaken_layers") or []
    if not isinstance(oaken, list) or len(oaken) != 5:
        errors.append(f"cost.oaken_layers must have 5 entries, got {len(oaken) if isinstance(oaken, list) else oaken}")
    else:
        names = [item.get("layer") for item in oaken if isinstance(item, dict)]
        if names != list(OAKEN_LAYER_NAMES):
            errors.append(f"oaken layer order/names={names}")

    gates = payload.get("compatibility_gates") or {}
    for gate_name in GATE_NAMES:
        gate = gates.get(gate_name) if isinstance(gates, dict) else None
        if not isinstance(gate, dict) or "passed" not in gate:
            errors.append(f"compatibility_gates.{gate_name} missing")

    if compressor == "identity":
        if _finite_number(ratio) and abs(float(ratio) - 1.0) > 1e-5:
            errors.append(f"identity compression_ratio should be ~1, got {ratio}")
        key_rmse = representation.get("key_rmse")
        if _finite_number(key_rmse) and float(key_rmse) > 1e-3:
            errors.append(f"identity key_rmse should be ~0, got {key_rmse}")

    if expected is not None and expected.calibration_free is False:
        offline = _lookup(payload, "cost.offline.calibration_required")
        if offline is not True:
            errors.append(f"{compressor} should report calibration_required=true")

    if execution_platform:
        platform = _lookup(payload, "hardware.execution_platform")
        if platform != execution_platform:
            errors.append(
                f"hardware.execution_platform={platform} (expected {execution_platform})"
            )

    errors.extend(
        err
        for err in validate_phase14_manifest(payload)
        if not (
            err == "missing Phase 14 field: seed"
            and compressor not in {"qjl", "turboquant"}
        )
    )
    return errors


def validate_taxonomy_coverage(payloads: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    names = [str(item.get("compressor")) for item in payloads]
    missing_methods = [name for name in active_eval_methods() if name not in names]
    if missing_methods:
        errors.append(f"taxonomy methods missing from bundle: {missing_methods}")
    extra_stubs = sorted({name for name in names if name in STUB_METHODS})
    if extra_stubs:
        errors.append(f"stub methods present: {extra_stubs}")
    covered = taxonomy_categories_covered(tuple(names))
    required = set(CompressionCategory)
    if covered != required:
        errors.append(
            f"taxonomy categories incomplete: have {sorted(c.value for c in covered)}, "
            f"need {sorted(c.value for c in required)}"
        )
    return errors


def validate_bundle(
    payloads: list[dict[str, Any]],
    *,
    require_smoke_extras: bool = False,
    execution_platform: str | None = None,
    require_full_taxonomy: bool = True,
) -> list[str]:
    errors: list[str] = []
    if require_full_taxonomy:
        errors.extend(validate_taxonomy_coverage(payloads))
    for payload in payloads:
        label = payload.get("label") or payload.get("compressor")
        for err in validate_payload_invariants(
            payload,
            require_smoke_extras=require_smoke_extras,
            execution_platform=execution_platform,
        ):
            errors.append(f"{label}: {err}")
    return errors
