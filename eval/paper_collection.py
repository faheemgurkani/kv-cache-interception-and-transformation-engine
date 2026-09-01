"""Audit a sweep bundle against paper-required collection (redesign plans).

Maps ``RESEARCH_REDESIGN_PLAN.md`` “What to pull from codebase when writing”
and ENGINE plan Oaken/hardware/gates onto collected job JSON. Separates
*missing collection* from *quality / math anomalies* so a method defect
(e.g. QJL PPL blow-up) is reported without hiding a complete KPI harvest.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from eval.kpi_schema import (
    GATE_NAMES,
    OAKEN_LAYER_NAMES,
    SMOKE_EXTRA_PATHS,
    _lookup,
    validate_bundle,
)
from eval.reproducibility.manifest import PHASE14_FIELDS, extract_phase14_manifest

PAPER_SOURCE_MAP: tuple[tuple[str, str, str], ...] = (
    ("FIDELITY / representation", "fidelity.representation", "Phase 2 · METHODOLOGY §6.1"),
    ("FIDELITY / attention", "fidelity.attention", "Phase 2 · METHODOLOGY §6.1"),
    ("FIDELITY / memory", "fidelity.memory", "Phase 2"),
    ("FIDELITY / recurrent", "fidelity.recurrent", "ENGINE §25 (N/A on non-hybrid)"),
    ("BEHAVIOR / perplexity", "behavior.task_quality.perplexity", "Phase 2 · paper tables"),
    ("BEHAVIOR / retrieval", "behavior.retrieval", "Phase 11 engine (paper optional)"),
    ("BEHAVIOR / instruction following", "behavior.instruction_following", "Phase 11 engine"),
    ("BEHAVIOR / reasoning", "behavior.reasoning", "Phase 11 opt-in"),
    ("SYSTEM / latency", "system.latency_throughput", "Phase 2 · METHODOLOGY §6.3"),
    ("SYSTEM / peak VRAM", "system.peak_memory", "Phase 10 CUDA"),
    ("SYSTEM / GPU util", "system.gpu_utilization", "Phase 10 CUDA"),
    ("SYSTEM / kernel cost", "system.kernel_cost", "Phase 3/26 --kernel-cost"),
    ("SYSTEM / bandwidth", "system.memory_bandwidth", "Phase 3"),
    ("COST / offline", "cost.offline", "Phases 3, 26, 27"),
    ("COST / online", "cost.online", "Phases 3, 26"),
    ("COST / Oaken layers", "cost.oaken_layers", "Phase 26"),
    ("COST / benchmark dimensions", "cost.benchmark_dimensions", "Phase 27"),
    ("Taxonomy", "taxonomy", "Phase 4"),
    ("Controlled conditions", "controlled_conditions", "Phase 7"),
    ("Hardware profile", "hardware", "Phase 10"),
    ("Compatibility gates", "compatibility_gates", "ENGINE §22–24"),
)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def paper_quality_anomalies(payloads: list[dict[str, Any]]) -> list[str]:
    """Logical / mathematical issues that collection succeeded but the paper must not treat as clean."""
    anomalies: list[str] = []
    identity = next((item for item in payloads if item.get("compressor") == "identity"), None)
    identity_ppl = _lookup(identity, "behavior.task_quality.perplexity") if identity else None

    for payload in payloads:
        label = payload.get("label") or payload.get("compressor")
        compressor = payload.get("compressor")
        ratio = _lookup(payload, "fidelity.memory.compression_ratio")
        ppl = _lookup(payload, "behavior.task_quality.perplexity")
        key_rmse = _lookup(payload, "fidelity.representation.key_rmse")
        attn_rmse = _lookup(payload, "fidelity.attention.rmse")
        attn_cos = _lookup(payload, "fidelity.attention.cosine_similarity")
        key_cos = _lookup(payload, "fidelity.representation.key_cosine_similarity")

        if _finite(attn_cos) and not -1.0 <= float(attn_cos) <= 1.0:
            anomalies.append(f"{label}: attention cosine {attn_cos} outside [-1, 1]")
        if _finite(key_cos) and not -1.0 <= float(key_cos) <= 1.0:
            anomalies.append(f"{label}: key cosine {key_cos} outside [-1, 1]")

        if compressor == "identity":
            if _finite(ratio) and abs(float(ratio) - 1.0) > 1e-5:
                anomalies.append(f"{label}: identity ratio {ratio} != 1")
            if _finite(key_rmse) and float(key_rmse) > 1e-3:
                anomalies.append(f"{label}: identity key_rmse {key_rmse} != 0")
            if _finite(attn_rmse) and float(attn_rmse) > 1e-2:
                anomalies.append(f"{label}: identity attention RMSE {attn_rmse} unexpectedly large")

        if compressor == "palu" and _finite(ratio) and float(ratio) < 0.99:
            anomalies.append(f"{label}: Palu measured ratio {ratio} < 1 (cache should not expand)")

        if compressor == "qjl" and _finite(identity_ppl) and _finite(ppl) and float(ppl) > 10.0 * float(identity_ppl):
            anomalies.append(
                f"{label}: QJL PPL {ppl:.1f} > 10× identity {float(identity_ppl):.1f} "
                "(ProdQJL/MQA quality defect — do not use as a BEHAVIOR paper point)"
            )
        if compressor == "qjl" and _finite(key_cos) and float(key_cos) < 0.1:
            anomalies.append(
                f"{label}: QJL key reconstruction cosine {key_cos:.4f} ≈ 0 "
                "(1-bit reconstruct is expected-unusable; estimator must carry BEHAVIOR)"
            )

        if compressor == "rocketkv" and _finite(ratio) and float(ratio) < 0.99:
            anomalies.append(
                f"{label}: RocketKV ratio {ratio} < 1 "
                "(metadata overhead or budget ≥ context — not true expansion of KV)"
            )

        if compressor == "turboquant":
            kernel = _lookup(payload, "system.kernel_cost.compress_decompress_time_ms")
            tps = _lookup(payload, "system.latency_throughput.tokens_per_second")
            if _finite(tps) and float(tps) < 1.0:
                anomalies.append(
                    f"{label}: TurboQuant {float(tps):.2f} tok/s "
                    "(heavy WHT+quant path; kernel-cost is online compute, not calibration)"
                )
            if _lookup(payload, "cost.offline.calibration_required") is not True:
                anomalies.append(f"{label}: TurboQuant should report calibration_required=true")

        oaken = payload.get("cost", {}).get("oaken_layers") or []
        measured = [item.get("layer") for item in oaken if isinstance(item, dict) and item.get("measured")]
        if "offline_evaluation" not in measured:
            anomalies.append(f"{label}: Oaken offline_evaluation not marked measured")

    return anomalies


def paper_missing_fields(payloads: list[dict[str, Any]], *, modal: bool = False) -> list[str]:
    extras = True if modal else False
    errors = validate_bundle(
        payloads,
        require_smoke_extras=extras,
        execution_platform="modal" if modal else None,
        require_full_taxonomy=True,
    )
    if modal:
        for payload in payloads:
            label = payload.get("label") or payload.get("compressor")
            for path in SMOKE_EXTRA_PATHS:
                if _lookup(payload, path) is None:
                    errors.append(f"{label}: missing paper/CUDA extra {path}")
    return errors


def audit_paper_collection(
    payloads: list[dict[str, Any]],
    *,
    modal: bool = False,
) -> dict[str, Any]:
    missing = paper_missing_fields(payloads, modal=modal)
    anomalies = paper_quality_anomalies(payloads)
    coverage = []
    sample = payloads[0] if payloads else {}
    for name, path, source in PAPER_SOURCE_MAP:
        present = all(_lookup(item, path) is not None for item in payloads) if payloads else False
        coverage.append({"artifact": name, "path": path, "source": source, "present": present})

    phase14 = []
    if payloads:
        manifest = extract_phase14_manifest(payloads[0])
        for field in PHASE14_FIELDS:
            phase14.append({"field": field, "present": manifest.get(field) is not None})

    gates = []
    for payload in payloads:
        g = payload.get("compatibility_gates") or {}
        gates.append(
            {
                "label": payload.get("label"),
                **{name: bool((g.get(name) or {}).get("passed")) for name in GATE_NAMES},
            }
        )

    return {
        "job_count": len(payloads),
        "methods": [item.get("compressor") for item in payloads],
        "collection_ok": not missing,
        "missing": missing,
        "quality_anomalies": anomalies,
        "coverage": coverage,
        "phase14": phase14,
        "gates": gates,
        "oaken_layers": list(OAKEN_LAYER_NAMES),
    }


def write_paper_collection_report(audit: dict[str, Any], path: Path) -> Path:
    lines = [
        "# Paper collection audit",
        "",
        "Checklist from `docs/RESEARCH_REDESIGN_PLAN.md` (What to pull from codebase) "
        "and `docs/ENGINE_AND_EVALUATION_FRAMEWORKS_REDESIGN_PLAN.md` (Oaken, gates, hardware).",
        "",
        f"Jobs: **{audit['job_count']}** · methods: `{', '.join(str(m) for m in audit['methods'])}`",
        "",
        f"Collection complete: **{audit['collection_ok']}**",
        "",
        "## Coverage",
        "",
        "| artifact | JSON path | plan source | present |",
        "|---|---|---|---|",
    ]
    for row in audit["coverage"]:
        mark = "yes" if row["present"] else "NO"
        lines.append(f"| {row['artifact']} | `{row['path']}` | {row['source']} | {mark} |")
    lines.extend(["", "## Phase 14 manifest (first job)", ""])
    for row in audit["phase14"]:
        lines.append(f"- `{row['field']}`: {'present' if row['present'] else 'MISSING'}")
    lines.extend(["", "## Compatibility gates", ""])
    for row in audit["gates"]:
        lines.append(
            f"- `{row['label']}` loader={row.get('loader_state')} "
            f"attention={row.get('attention')} state={row.get('state_semantics')}"
        )
    lines.extend(["", "## Missing collection", ""])
    lines.extend([f"- {err}" for err in audit["missing"]] or ["_(none)_"])
    lines.extend(["", "## Quality / math anomalies (reported, not dropped)", ""])
    lines.extend([f"- {err}" for err in audit["quality_anomalies"]] or ["_(none)_"])
    lines.append("")
    path.write_text("\n".join(lines) + "\n")
    return path
