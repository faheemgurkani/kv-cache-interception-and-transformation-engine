"""Merge Modal worker JSON payloads into local CSV/JSON reports."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def flatten_result_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Flatten EvaluationResult.to_dict() structure for CSV export.

    Supports both the current FIDELITY/BEHAVIOR/SYSTEM payload shape and the legacy
    section_a_fidelity/section_b_inference shape still present in older results/
    bundles on disk, so both can be merged into one CSV.
    """
    is_legacy = "section_a_fidelity" in payload
    if is_legacy:
        fidelity = payload.get("section_a_fidelity") or {}
        behavior = payload.get("section_b_inference") or {}
        representation = fidelity.get("tensor") or {}
        attention = fidelity.get("attention") or {}
        memory = fidelity.get("memory") or {}
        throughput = behavior.get("throughput") or {}
        perplexity = behavior.get("perplexity")
        perplexity_baseline = behavior.get("perplexity_baseline")
    else:
        fidelity = payload.get("fidelity") or {}
        behavior = payload.get("behavior") or {}
        system = payload.get("system") or {}
        representation = fidelity.get("representation") or {}
        attention = fidelity.get("attention") or {}
        memory = fidelity.get("memory") or {}
        throughput = system.get("latency_throughput") or {}
        task_quality = behavior.get("task_quality") or {}
        perplexity = task_quality.get("perplexity")
        perplexity_baseline = task_quality.get("perplexity_baseline")

    return {
        "label": payload.get("label"),
        "compressor": payload.get("compressor"),
        "bitwidth": payload.get("bitwidth"),
        "stage": payload.get("stage"),
        "context_length": payload.get("context_length"),
        "key_rmse": representation.get("key_rmse"),
        "value_rmse": representation.get("value_rmse"),
        "attention_rmse": attention.get("rmse"),
        "attention_cosine": attention.get("cosine_similarity"),
        "attention_max_error": attention.get("max_error"),
        "uncompressed_bytes": memory.get("uncompressed_bytes"),
        "compressed_bytes": memory.get("compressed_bytes"),
        "compression_ratio": memory.get("compression_ratio"),
        "effective_bits_per_kv_element": memory.get("effective_bits_per_kv_element"),
        "shared_metadata_bytes": memory.get("shared_metadata_bytes"),
        "perplexity_compressed": perplexity,
        "perplexity_baseline": perplexity_baseline,
        "tokens_per_second": throughput.get("tokens_per_second"),
        "latency_ms_per_token": throughput.get("latency_ms_per_token"),
        "ttft_ms": throughput.get("ttft_ms"),
        "online_compressed_kv": throughput.get("online_compressed_kv"),
        "finished_at": payload.get("finished_at"),
    }


CSV_FIELDNAMES = [
    "label",
    "compressor",
    "bitwidth",
    "stage",
    "context_length",
    "key_rmse",
    "value_rmse",
    "attention_rmse",
    "attention_cosine",
    "attention_max_error",
    "uncompressed_bytes",
    "compressed_bytes",
    "compression_ratio",
    "effective_bits_per_kv_element",
    "shared_metadata_bytes",
    "perplexity_compressed",
    "perplexity_baseline",
    "tokens_per_second",
    "latency_ms_per_token",
    "ttft_ms",
    "online_compressed_kv",
    "finished_at",
]


def write_merged_reports(
    payloads: list[dict[str, Any]],
    output_dir: Path,
    stem: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"{stem}_{stamp}.json"
    csv_path = output_dir / f"{stem}_{stamp}.csv"

    report = {
        "timestamp": stamp,
        "job_count": len(payloads),
        "results": payloads,
    }
    json_path.write_text(json.dumps(report, indent=2))

    rows = [flatten_result_payload(item) for item in payloads]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    return json_path, csv_path


def load_payloads_from_directory(
    directory: Path,
    *,
    labels: set[str] | None = None,
    label_prefixes: tuple[str, ...] | None = None,
    exclude_labels: set[str] | None = None,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*.json")):
        if path.name.endswith(".error.json"):
            continue
        data = json.loads(path.read_text())
        if isinstance(data, dict) and ("section_a_fidelity" in data or "fidelity" in data):
            payload = data
        elif isinstance(data, dict) and "results" in data:
            payloads.extend(data["results"])
            continue
        else:
            continue

        label = payload.get("label") or (payload.get("job") or {}).get("label")
        if labels is not None and label not in labels:
            continue
        if exclude_labels and label in exclude_labels:
            continue
        if label_prefixes and not any(str(label or "").startswith(prefix) for prefix in label_prefixes):
            continue
        payloads.append(payload)
    return payloads
