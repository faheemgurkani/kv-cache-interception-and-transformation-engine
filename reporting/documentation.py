"""Document every collected KPI, job JSON, CSV, and log for a sweep bundle."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_TEXT_KEYS = {"label", "compressor", "stage", "finished_at", "taxonomy_primary", "taxonomy_secondary"}


def _fmt_num(value: str | float | None, precision: int = 4) -> str:
    if value is None or value == "":
        return "—"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(num) >= 1_000_000:
        return f"{num:,.0f}"
    if abs(num) >= 1000:
        return f"{num:,.2f}"
    return f"{num:.{precision}f}"


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _latest_csv(bundle_dir: Path) -> Path | None:
    candidates = sorted(bundle_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _job_jsons(bundle_dir: Path) -> list[Path]:
    jobs_dir = bundle_dir / "jobs"
    if not jobs_dir.is_dir():
        return []
    return sorted(j for j in jobs_dir.glob("*.json") if not j.name.endswith(".error.json"))


def _error_jsons(bundle_dir: Path) -> list[Path]:
    jobs_dir = bundle_dir / "jobs"
    if not jobs_dir.is_dir():
        return []
    return sorted(jobs_dir.glob("*.error.json"))


def _logs(bundle_dir: Path) -> list[Path]:
    return sorted(bundle_dir.rglob("*.log"))


def _per_layer_section(job_path: Path) -> list[str]:
    data = json.loads(job_path.read_text())
    fidelity = data.get("fidelity") or data.get("section_a_fidelity") or {}
    att = fidelity.get("attention") or {}
    per_layer = att.get("per_layer") or []
    if not per_layer:
        return []
    lines = [
        "",
        f"#### Per-layer attention fidelity — `{job_path.name}`",
        "",
        "| layer | MSE | RMSE | cosine | max error |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in per_layer:
        lines.append(
            f"| {row.get('layer')} | {_fmt_num(row.get('mse'))} | {_fmt_num(row.get('rmse'))} | "
            f"{_fmt_num(row.get('cosine_similarity'), 6)} | {_fmt_num(row.get('max_error'))} |"
        )
    return lines


def _kpi_inventory(payload: dict[str, Any], prefix: str = "") -> list[str]:
    rows: list[str] = []
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            rows.extend(_kpi_inventory(value, path))
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            rows.append(f"| `{path}` | list[{len(value)}] |")
        else:
            rows.append(f"| `{path}` | {value if not isinstance(value, float) else _fmt_num(value)} |")
    return rows


def export_bundle_documentation(
    bundle_dir: Path | str,
    output_path: Path | str,
    *,
    title: str = "Complete evaluation results",
    model_name: str = "unknown",
) -> Path:
    """Write a markdown report covering JSON/CSV/logs/KPIs in ``bundle_dir``."""
    bundle_dir = Path(bundle_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# {title}",
        "",
        f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        f"Model: **{model_name}** · Bundle: `{bundle_dir}`",
        "",
        "This file inventories every collected result, log, and KPI path from the sweep.",
        "",
    ]

    summary_path = bundle_dir / "taxonomy_smoke_local_live_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        lines.extend(["## Run summary", "", "```json", json.dumps(summary, indent=2), "```", ""])

    csv_path = _latest_csv(bundle_dir)
    if csv_path:
        rows = _read_csv_rows(csv_path)
        if rows:
            headers = list(rows[0].keys())
            lines.extend([f"## Merged CSV — `{csv_path.name}`", ""])
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("|" + "|".join(["---"] * len(headers)) + "|")
            for row in rows:
                cells = [
                    _fmt_num(row.get(h, "")) if h not in _TEXT_KEYS else row.get(h, "")
                    for h in headers
                ]
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")

    job_paths = _job_jsons(bundle_dir)
    if job_paths:
        lines.extend(["## Per-job JSON", ""])
        for jp in job_paths:
            rel = jp.relative_to(bundle_dir)
            lines.append(f"- `{rel}` ({jp.stat().st_size} bytes)")
        lines.append("")
        lines.extend(["## Per-job KPI inventory", ""])
        for jp in job_paths:
            data = json.loads(jp.read_text())
            lines.extend([f"### `{jp.name}`", "", "| path | value |", "|---|---|"])
            lines.extend(_kpi_inventory(data))
            lines.append("")
            lines.extend(_per_layer_section(jp))

    logs = _logs(bundle_dir)
    lines.extend(["## Logs", ""])
    if logs:
        for path in logs:
            rel = path.relative_to(bundle_dir)
            lines.append(f"- `{rel}` ({path.stat().st_size / 1024:.1f} KB)")
    else:
        lines.append("_(no `.log` files in this bundle)_")
    lines.append("")

    errors = _error_jsons(bundle_dir)
    lines.extend(["## Failed jobs", ""])
    if errors:
        for path in errors:
            data = json.loads(path.read_text())
            lines.append(f"- `{path.name}` — {str(data.get('error', 'unknown'))[:200]}")
    else:
        lines.append("_(none)_")
    lines.append("")

    extra_json = sorted(
        p
        for p in bundle_dir.glob("*.json")
        if p.name
        not in {
            "taxonomy_smoke_local_live_summary.json",
            "manifest.json",
        }
    )
    if extra_json:
        lines.extend(["## Other JSON artifacts", ""])
        for path in extra_json:
            lines.append(f"- `{path.name}` ({path.stat().st_size} bytes)")
        lines.append("")

    output_path.write_text("\n".join(lines) + "\n")
    return output_path
