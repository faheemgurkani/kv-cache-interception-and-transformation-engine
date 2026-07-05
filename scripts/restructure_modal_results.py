#!/usr/bin/env python3
"""Split shared identity baseline from method-specific Modal sweep bundles."""

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
MODAL_VOLUME = PROJECT_ROOT / "results" / "modal_volume"
BASELINE_DIR = PROJECT_ROOT / "results" / "phase5_modal_baseline"
TURBOQUANT_DIR = PROJECT_ROOT / "results" / "phase5_modal_sweep_128_256_512"
ROCKETKV_DIR = PROJECT_ROOT / "results" / "phase5_modal_rocketkv"

BASELINE_LABEL = "identity_baseline"
TURBOQUANT_PREFIXES = ("tq_",)
ROCKETKV_PREFIXES = ("rocketkv_",)
CONTEXT_LENGTHS = (128, 256, 512)
TURBOQUANT_CONFIGS = ("tq_full_b2", "tq_full_b3", "tq_full_b4", "tq_mse_b4")
ROCKETKV_CONFIGS = ("rocketkv_r25", "rocketkv_r50", "rocketkv_r75")
MODAL_APP_ID = "ap-ek9dIxujlrECcfFaOa3ok3"
MODAL_APP_URL = f"https://modal.com/apps/faheemgurkani/main/{MODAL_APP_ID}"


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


def _archive_legacy_combined(bundle_dir: Path, stem: str) -> None:
    archive_dir = bundle_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".csv", ".json"):
        for path in bundle_dir.glob(f"{stem}_*{suffix}"):
            if "20260704" in path.name:
                shutil.move(str(path), archive_dir / path.name)


def _write_manifest(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def restructure_baseline() -> tuple[Path, Path]:
    jobs_dir = BASELINE_DIR / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    _copy_jobs(MODAL_VOLUME, jobs_dir, (f"{BASELINE_LABEL}_",))

    payloads = load_payloads_from_directory(jobs_dir)
    json_path, csv_path = write_merged_reports(payloads, BASELINE_DIR, "phase5_modal_baseline")

    _write_manifest(
        BASELINE_DIR / "manifest.json",
        {
            "sweep_id": "phase5_modal_baseline",
            "role": "shared_no_compression_baseline_for_all_methods",
            "source_modal_app_id": MODAL_APP_ID,
            "source_modal_app_url": MODAL_APP_URL,
            "completed_at_utc": "2026-07-04T18:58:54Z",
            "jobs_total": len(payloads),
            "jobs_ok": len(payloads),
            "jobs_error": 0,
            "context_lengths": list(CONTEXT_LENGTHS),
            "configs": [BASELINE_LABEL],
            "reuse_for": ["turboquant", "qjl", "rocketkv", "kivi"],
            "files": {
                "jobs_dir": "jobs/",
                "merged_csv": csv_path.name,
                "merged_json": json_path.name,
            },
        },
    )
    return json_path, csv_path


def restructure_turboquant() -> tuple[Path, Path]:
    jobs_dir = TURBOQUANT_DIR / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    _copy_jobs(MODAL_VOLUME, jobs_dir, TURBOQUANT_PREFIXES)

    payloads = load_payloads_from_directory(jobs_dir)
    json_path, csv_path = write_merged_reports(
        payloads,
        TURBOQUANT_DIR,
        "phase5_modal_sweep_128_256_512",
    )
    _archive_legacy_combined(TURBOQUANT_DIR, "phase5_modal_sweep_128_256_512")

    _write_manifest(
        TURBOQUANT_DIR / "manifest.json",
        {
            "sweep_id": "phase5_modal_sweep_128_256_512",
            "method": "turboquant",
            "shared_baseline": "../phase5_modal_baseline/",
            "modal_app_id": MODAL_APP_ID,
            "modal_app_url": MODAL_APP_URL,
            "completed_at_utc": "2026-07-04T18:58:54Z",
            "runtime_minutes": 72,
            "jobs_total": len(payloads),
            "jobs_ok": len(payloads),
            "jobs_error": 0,
            "context_lengths": list(CONTEXT_LENGTHS),
            "configs": list(TURBOQUANT_CONFIGS),
            "flags": {"sync": True, "no_resume": True},
            "files": {
                "jobs_dir": "jobs/",
                "merged_csv": csv_path.name,
                "merged_json": json_path.name,
                "modal_logs": "logs/modal_app_ap-ek9dIxujlrECcfFaOa3ok3.log",
                "local_client_log": "logs/local_sweep_sync_client.log",
            },
        },
    )
    return json_path, csv_path


def restructure_rocketkv() -> tuple[Path, Path]:
    jobs_dir = ROCKETKV_DIR / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    _copy_jobs(MODAL_VOLUME, jobs_dir, ROCKETKV_PREFIXES)

    payloads = load_payloads_from_directory(jobs_dir)
    json_path, csv_path = write_merged_reports(payloads, ROCKETKV_DIR, "phase5_modal_rocketkv")

    _write_manifest(
        ROCKETKV_DIR / "manifest.json",
        {
            "sweep_id": "phase5_modal_rocketkv",
            "method": "rocketkv",
            "shared_baseline": "../phase5_modal_baseline/",
            "modal_app_id": "ap-jct37Y2ytrDK5CVMPHWTbz",
            "modal_app_url": "https://modal.com/apps/faheemgurkani/main/ap-jct37Y2ytrDK5CVMPHWTbz",
            "completed_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "jobs_total": len(payloads),
            "jobs_ok": len(payloads),
            "jobs_error": 0,
            "context_lengths": list(CONTEXT_LENGTHS),
            "configs": list(ROCKETKV_CONFIGS),
            "files": {
                "jobs_dir": "jobs/",
                "merged_csv": csv_path.name,
                "merged_json": json_path.name,
            },
        },
    )
    return json_path, csv_path


def main() -> None:
    baseline_json, baseline_csv = restructure_baseline()
    tq_json, tq_csv = restructure_turboquant()
    rk_json, rk_csv = restructure_rocketkv()
    print(f"Baseline: {len(load_payloads_from_directory(BASELINE_DIR / 'jobs'))} jobs")
    print(f"  {baseline_csv}")
    print(f"TurboQuant: {len(load_payloads_from_directory(TURBOQUANT_DIR / 'jobs'))} jobs")
    print(f"  {tq_csv}")
    print(f"RocketKV: {len(load_payloads_from_directory(ROCKETKV_DIR / 'jobs'))} jobs")
    print(f"  {rk_csv}")


if __name__ == "__main__":
    main()
