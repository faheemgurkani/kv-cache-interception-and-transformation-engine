#!/usr/bin/env python3
"""Generate docs/RESULTS_COMPLETE.md from Phase 5 result bundles."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_OUT = PROJECT_ROOT / "docs" / "RESULTS_COMPLETE.md"

BUNDLES = [
    ("Identity baseline", PROJECT_ROOT / "results" / "phase5_modal_baseline"),
    ("TurboQuant", PROJECT_ROOT / "results" / "phase5_modal_sweep_128_256_512"),
    ("QJL", PROJECT_ROOT / "results" / "phase5_modal_qjl"),
    ("RocketKV", PROJECT_ROOT / "results" / "phase5_modal_rocketkv"),
]


def _latest_csv(bundle_dir: Path) -> Path | None:
    candidates = sorted(bundle_dir.glob("phase5_modal_*.csv"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


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


def _job_jsons(bundle_dir: Path) -> list[Path]:
    jobs_dir = bundle_dir / "jobs"
    if not jobs_dir.is_dir():
        return []
    return sorted(j for j in jobs_dir.glob("*.json") if not j.name.endswith(".error.json"))


def _manifest(bundle_dir: Path) -> dict:
    manifest_path = bundle_dir / "manifest.json"
    if manifest_path.is_file():
        return json.loads(manifest_path.read_text())
    return {}


def _log_inventory() -> list[str]:
    logs: list[str] = []
    results = PROJECT_ROOT / "results"
    for path in sorted(results.rglob("*.log")):
        rel = path.relative_to(PROJECT_ROOT)
        size_kb = path.stat().st_size / 1024
        logs.append(f"- `{rel}` ({size_kb:.1f} KB)")
    return logs


def _error_jobs() -> list[str]:
    lines: list[str] = []
    volume = PROJECT_ROOT / "results" / "modal_volume"
    if not volume.is_dir():
        return lines
    for path in sorted(volume.rglob("*.error.json")):
        data = json.loads(path.read_text())
        rel = path.relative_to(PROJECT_ROOT)
        err = data.get("error", "unknown")[:120]
        lines.append(f"- `{rel}` — {err}")
    return lines


def _per_layer_section(job_path: Path) -> list[str]:
    data = json.loads(job_path.read_text())
    att = (data.get("section_a_fidelity") or {}).get("attention") or {}
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
            f"| {row['layer']} | {_fmt_num(row['mse'])} | {_fmt_num(row['rmse'])} | "
            f"{_fmt_num(row['cosine_similarity'], 6)} | {_fmt_num(row['max_error'])} |"
        )
    return lines


def main() -> None:
    lines: list[str] = [
        "# Complete Evaluation Results",
        "",
        "Auto-generated from Phase 5 Modal bundles. Regenerate:",
        "",
        "```bash",
        "python scripts/export_results_documentation.py",
        "```",
        "",
        f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "Model: **Qwen3-1.7B** · Dataset: **WikiText-2 test** · GPU: **Modal A10G** · Contexts: **128 / 256 / 512**.",
        "",
        "Summary tables: [PHASE5_EVAL_RESULTS.md](PHASE5_EVAL_RESULTS.md) · Reproduce: [REPRODUCIBILITY.md](REPRODUCIBILITY.md).",
        "",
    ]

    for title, bundle_dir in BUNDLES:
        lines.extend(["---", "", f"## {title}", ""])
        manifest = _manifest(bundle_dir)
        if manifest:
            lines.append("**Manifest**")
            lines.append("")
            lines.append(f"- sweep_id: `{manifest.get('sweep_id', '—')}`")
            if manifest.get("modal_app_id"):
                lines.append(f"- modal_app_id: `{manifest['modal_app_id']}`")
            if manifest.get("modal_app_url"):
                lines.append(f"- modal_app_url: {manifest['modal_app_url']}")
            lines.append(f"- jobs_ok: {manifest.get('jobs_ok', '—')} / {manifest.get('jobs_total', '—')}")
            lines.append(f"- completed_at_utc: {manifest.get('completed_at_utc', '—')}")
            files = manifest.get("files") or {}
            if files.get("merged_csv"):
                lines.append(f"- merged_csv: `results/{bundle_dir.name}/{files['merged_csv']}`")
            lines.append("")

        csv_path = _latest_csv(bundle_dir)
        if csv_path:
            rows = _read_csv_rows(csv_path)
            if rows:
                headers = list(rows[0].keys())
                lines.append(f"### Full job table — `{csv_path.name}`")
                lines.append("")
                lines.append("| " + " | ".join(headers) + " |")
                lines.append("|" + "|".join(["---"] * len(headers)) + "|")
                for row in rows:
                    cells = [_fmt_num(row.get(h, "")) if h not in {"label", "compressor", "stage", "finished_at"} else row.get(h, "") for h in headers]
                    lines.append("| " + " | ".join(cells) + " |")
                lines.append("")

        job_paths = _job_jsons(bundle_dir)
        if job_paths:
            lines.append("### Per-job JSON stems")
            lines.append("")
            for jp in job_paths:
                lines.append(f"- `results/{bundle_dir.name}/jobs/{jp.name}`")
            lines.append("")
            lines.append("### Per-layer attention fidelity (Section A)")
            lines.append("")
            for jp in job_paths:
                lines.extend(_per_layer_section(jp))

    lines.extend(["---", "", "## Log inventory", ""])
    log_lines = _log_inventory()
    lines.extend(log_lines if log_lines else ["_(no `.log` files found under `results/`)_"])
    lines.append("")

    lines.extend(["## Failed / out-of-scope jobs (`.error.json` on Modal volume)", ""])
    err_lines = _error_jobs()
    lines.extend(err_lines if err_lines else ["_(none)_"])
    lines.append("")

    smoke_csvs = sorted((PROJECT_ROOT / "results").glob("phase5_modal_*smoke*.csv"))
    if smoke_csvs:
        lines.extend(["---", "", "## Smoke / debug runs (not Phase 5 official grid)", ""])
        for csv_path in smoke_csvs:
            rows = _read_csv_rows(csv_path)
            if not rows:
                continue
            headers = list(rows[0].keys())
            lines.append(f"### `{csv_path.name}`")
            lines.append("")
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("|" + "|".join(["---"] * len(headers)) + "|")
            for row in rows:
                cells = [
                    _fmt_num(row.get(h, "")) if h not in {"label", "compressor", "stage", "finished_at"} else row.get(h, "")
                    for h in headers
                ]
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")

    DOCS_OUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {DOCS_OUT} ({DOCS_OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
