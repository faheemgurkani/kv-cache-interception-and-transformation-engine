"""Merge Modal worker JSON payloads into local CSV/JSON reports."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def flatten_result_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Flatten EvaluationResult.to_dict() structure for CSV export."""
    section_a = payload.get("section_a_fidelity") or {}
    section_b = payload.get("section_b_inference") or {}
    tensor = section_a.get("tensor") or {}
    attention = section_a.get("attention") or {}
    memory = section_a.get("memory") or {}
    throughput = section_b.get("throughput") or {}

    return {
        "label": payload.get("label"),
        "compressor": payload.get("compressor"),
        "bitwidth": payload.get("bitwidth"),
        "stage": payload.get("stage"),
        "context_length": payload.get("context_length"),
        "key_rmse": tensor.get("key_rmse"),
        "value_rmse": tensor.get("value_rmse"),
        "attention_rmse": attention.get("rmse"),
        "attention_cosine": attention.get("cosine_similarity"),
        "attention_max_error": attention.get("max_error"),
        "uncompressed_bytes": memory.get("uncompressed_bytes"),
        "compressed_bytes": memory.get("compressed_bytes"),
        "compression_ratio": memory.get("compression_ratio"),
        "effective_bits_per_kv_element": memory.get("effective_bits_per_kv_element"),
        "shared_metadata_bytes": memory.get("shared_metadata_bytes"),
        "perplexity_compressed": section_b.get("perplexity"),
        "perplexity_baseline": section_b.get("perplexity_baseline"),
        "tokens_per_second": throughput.get("tokens_per_second"),
        "latency_ms_per_token": throughput.get("latency_ms_per_token"),
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


def load_payloads_from_directory(directory: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*.json")):
        if path.name.endswith(".error.json"):
            continue
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "section_a_fidelity" in data:
            payloads.append(data)
        elif isinstance(data, dict) and "results" in data:
            payloads.extend(data["results"])
    return payloads
