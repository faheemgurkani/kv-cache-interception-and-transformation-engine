#!/usr/bin/env python3
"""Restructure OLMo2 Modal results into phase5-style bundles + docs stub."""

from __future__ import annotations

import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import setup_path  # noqa: F401

from modal_app.merge import load_payloads_from_directory, write_merged_reports

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODAL_VOLUME = PROJECT_ROOT / "results" / "modal_volume_olmo2"
BASELINE_DIR = PROJECT_ROOT / "results" / "olmo2_phase5_baseline"
TURBOQUANT_DIR = PROJECT_ROOT / "results" / "olmo2_phase5_turboquant"
ROCKETKV_DIR = PROJECT_ROOT / "results" / "olmo2_phase5_rocketkv"
QJL_DIR = PROJECT_ROOT / "results" / "olmo2_phase5_qjl"

BASELINE_LABEL = "identity_baseline"
TURBOQUANT_PREFIXES = ("tq_",)
ROCKETKV_PREFIXES = ("rocketkv_r256_", "rocketkv_r512_", "rocketkv_r1024_")
QJL_PREFIXES = ("qjl_default_",)
CONTEXT_LENGTHS = (128, 256, 512)
TURBOQUANT_CONFIGS = ("tq_full_b2", "tq_full_b3", "tq_full_b4", "tq_mse_b4")
ROCKETKV_CONFIGS = ("rocketkv_r256", "rocketkv_r512", "rocketkv_r1024")
QJL_CONFIGS = ("qjl_default",)
MODEL_NAME = "allenai/OLMo-2-0425-1B"


def _is_target_context(path: Path) -> bool:
    for length in CONTEXT_LENGTHS:
        if f"_ctx{length}_" in path.name:
            return True
    return False


def _copy_jobs(source: Path, destination: Path, prefixes: tuple[str, ...]) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for path in sorted(source.glob("*.json")):
        if path.name.endswith(".error.json"):
            continue
        if not any(path.name.startswith(prefix) for prefix in prefixes):
            continue
        if not _is_target_context(path):
            continue
        target = destination / path.name
        if not target.exists() or path.stat().st_mtime > target.stat().st_mtime:
            shutil.copy2(path, target)
        copied.append(target)
    return copied


def _write_manifest(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _bundle(
    name: str,
    out_dir: Path,
    prefixes: tuple[str, ...],
    configs: list[str],
    role: str,
) -> tuple[Path, Path, int, int]:
    jobs_dir = out_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    _copy_jobs(MODAL_VOLUME, jobs_dir, prefixes)
    payloads = load_payloads_from_directory(jobs_dir)
    ok = [p for p in payloads if p.get("status", "ok") == "ok"]
    errors = [p for p in payloads if p.get("status") == "error"]
    json_path, csv_path = write_merged_reports(ok, out_dir, name)
    _write_manifest(
        out_dir / "manifest.json",
        {
            "sweep_id": name,
            "model_name": MODEL_NAME,
            "role": role,
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "jobs_total": len(payloads),
            "jobs_ok": len(ok),
            "jobs_error": len(errors),
            "context_lengths": list(CONTEXT_LENGTHS),
            "configs": configs,
            "files": {
                "jobs_dir": "jobs/",
                "merged_csv": csv_path.name,
                "merged_json": json_path.name,
            },
        },
    )
    return json_path, csv_path, len(ok), len(errors)


def main() -> None:
    if not MODAL_VOLUME.exists():
        raise SystemExit(f"Missing fetched volume dir: {MODAL_VOLUME}")

    results = []
    results.append(
        _bundle(
            "olmo2_phase5_baseline",
            BASELINE_DIR,
            (f"{BASELINE_LABEL}_",),
            [BASELINE_LABEL],
            "shared_no_compression_baseline",
        )
    )
    results.append(
        _bundle(
            "olmo2_phase5_turboquant",
            TURBOQUANT_DIR,
            TURBOQUANT_PREFIXES,
            list(TURBOQUANT_CONFIGS),
            "turboquant_sweep",
        )
    )
    results.append(
        _bundle(
            "olmo2_phase5_qjl",
            QJL_DIR,
            QJL_PREFIXES,
            list(QJL_CONFIGS),
            "qjl_sweep",
        )
    )
    results.append(
        _bundle(
            "olmo2_phase5_rocketkv",
            ROCKETKV_DIR,
            ROCKETKV_PREFIXES,
            list(ROCKETKV_CONFIGS),
            "rocketkv_sweep",
        )
    )

    total_ok = sum(r[2] for r in results)
    total_err = sum(r[3] for r in results)
    print(f"OLMo2 restructure complete: {total_ok} ok / {total_err} error jobs across 4 bundles.")
    for json_path, csv_path, ok, err in results:
        print(f"  {json_path.parent.name}: ok={ok} err={err} → {csv_path.name}")


if __name__ == "__main__":
    main()
