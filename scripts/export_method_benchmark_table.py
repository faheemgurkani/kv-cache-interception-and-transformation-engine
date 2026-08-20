#!/usr/bin/env python3
"""Export Phase 27 method benchmark dimension table from job bundles or plug-ins."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import setup_path  # noqa: F401
from eval.cost.benchmark_dimensions import benchmark_dimensions_from_dict
from framework.config import PROJECT_ROOT


def _load_records(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if "results" in data and isinstance(data["results"], list):
        return data["results"]
    return [data]


def _static_plugin_rows() -> list[dict]:
    from compressors.identity import IdentityCompressor
    from compressors.qjl import QJLCompressor
    from compressors.rocketkv import RocketKVCompressor
    from compressors.turboquant import TurboQuantCompressor
    from eval.cost.accounting import evaluate_cost
    from quantizers.turboquant_pipeline import TurboQuantStage

    rows: list[dict] = []
    for compressor in (
        IdentityCompressor(),
        TurboQuantCompressor(bitwidth=4, stage=TurboQuantStage.WHT_QUANT),
        QJLCompressor(),
        RocketKVCompressor(token_budget=256),
    ):
        cost = evaluate_cost(compressor, context_length=512, fidelity=None, system=None)
        dims = cost.benchmark_dimensions
        if dims is None:
            continue
        rows.append(
            {
                "compressor": compressor.name,
                "context_length": 512,
                "source": "static_plugin",
                **dims.to_dict(),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Phase 27 benchmark dimension table")
    parser.add_argument(
        "bundles",
        nargs="*",
        type=Path,
        help="Optional job JSON bundles; if omitted, exports static plug-in declarations only",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "method_benchmark_dimensions.csv",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=PROJECT_ROOT / "results" / "method_benchmark_dimensions.json",
    )
    args = parser.parse_args()

    rows: list[dict] = []
    seen: set[tuple[str, int]] = set()

    for path in args.bundles:
        for record in _load_records(path):
            compressor = record.get("compressor") or record.get("job", {}).get("compressor")
            ctx = record.get("context_length") or record.get("job", {}).get("context_length")
            if compressor is None or ctx is None:
                continue
            key = (str(compressor), int(ctx))
            if key in seen:
                continue
            dims = benchmark_dimensions_from_dict(record)
            if dims is None:
                continue
            seen.add(key)
            rows.append(
                {
                    "compressor": compressor,
                    "context_length": ctx,
                    "source": str(path.name),
                    **dims.to_dict(),
                }
            )

    if not rows:
        rows = _static_plugin_rows()

    fieldnames = [
        "compressor",
        "context_length",
        "source",
        "calibration_required",
        "calibration_dataset",
        "calibration_tokens",
        "calibration_time_ms",
        "calibration_memory_bytes",
        "stateful",
        "online_overhead_ms_per_token",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    args.json_output.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {args.output} ({len(rows)} rows)")
    print(f"Wrote {args.json_output}")


if __name__ == "__main__":
    main()
