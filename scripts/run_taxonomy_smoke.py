#!/usr/bin/env python3
"""Taxonomy-coverage smoke: dummy local first, then optional Modal Gemma3-270M run.

Dummy mode does not load model weights. Modal mode runs one shortlist model
(Gemma3-270M) × every active compressor (identity, turboquant, qjl, rocketkv,
snapkv, palu) at ctx=128 with FIDELITY/BEHAVIOR/SYSTEM/COST collection enabled.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import setup_path  # noqa: F401
from eval.kpi_schema import validate_bundle
from eval.taxonomy_smoke import TAXONOMY_SMOKE_PRESET, run_dummy_collection
from framework.config import PROJECT_ROOT

DEFAULT_MODEL_CONFIG = "configs/model_gemma3_270m.yaml"
DEFAULT_MODAL_CONFIG = "configs/modal_gemma3.yaml"
DEFAULT_OUTPUT = "taxonomy_smoke_gemma3"


def _activate_shortlist_env(model_config: str, modal_config: str) -> None:
    os.environ.setdefault("KV_MODEL_CONFIG", model_config)
    os.environ.setdefault("KV_MODAL_CONFIG", modal_config)


def _resolve_modal() -> list[str]:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "modal", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            return [sys.executable, "-m", "modal"]
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    raise SystemExit("modal CLI not available in this Python environment")


def run_dummy(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = run_dummy_collection(output_dir)
    summary_path = output_dir / "taxonomy_smoke_dummy_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Dummy jobs: {summary['job_count']} methods={summary['methods']}")
    print(f"Categories: {summary['categories_covered']}")
    for item in summary["math_reports"]:
        mark = "OK" if item["ok"] else "FAIL"
        print(f"  [{mark}] {item['compressor']}: {item['detail']}")
    if summary["schema_errors"]:
        print("Schema errors:")
        for err in summary["schema_errors"]:
            print(f"  - {err}")
    print(f"Merged JSON: {summary['merged_json']}")
    print(f"Merged CSV:  {summary['merged_csv']}")
    print(f"Summary:     {summary_path}")
    if not summary["ok"]:
        raise SystemExit("Dummy taxonomy smoke failed")
    print("Dummy taxonomy smoke passed.")
    return summary


def validate_latest_modal_bundle(stem: str) -> None:
    results_dir = PROJECT_ROOT / "results"
    matches = sorted(results_dir.glob(f"{stem}_*.json"))
    if not matches:
        raise SystemExit(f"No merged bundle matching results/{stem}_*.json")
    path = matches[-1]
    data = json.loads(path.read_text())
    payloads = data.get("results") if isinstance(data, dict) else data
    if not isinstance(payloads, list):
        raise SystemExit(f"Unexpected bundle shape in {path}")
    errors = validate_bundle(
        payloads,
        require_smoke_extras=True,
        execution_platform="modal",
        require_full_taxonomy=True,
    )
    print(f"Validated {len(payloads)} jobs from {path}")
    if errors:
        for err in errors:
            print(f"  - {err}")
        raise SystemExit("Modal taxonomy smoke KPI validation failed")
    print("Modal taxonomy KPI collection passed.")


def run_modal(
    *,
    output: str,
    context_length: int,
    sync: bool,
    no_resume: bool,
) -> None:
    modal = _resolve_modal()
    cmd = [
        *modal,
        "run",
        "modal_app/sweep.py::main",
        "--preset",
        TAXONOMY_SMOKE_PRESET,
        "--context-lengths",
        str(context_length),
        "--output",
        output,
    ]
    if sync:
        cmd.append("--sync")
    if no_resume:
        cmd.append("--no-resume")
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dummy", action="store_true", help="Run local dummy execution (default if neither flag).")
    parser.add_argument("--modal", action="store_true", help="Run the Gemma3-270M taxonomy smoke on Modal.")
    parser.add_argument("--skip-dummy", action="store_true", help="Skip dummy when --modal is set.")
    parser.add_argument("--sync", action="store_true", default=True, help="Wait for Modal jobs and merge locally.")
    parser.add_argument("--detach", action="store_true", help="Spawn Modal jobs without waiting.")
    parser.add_argument("--no-resume", action="store_true", help="Re-submit jobs even if results exist on the volume.")
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--modal-config", default=DEFAULT_MODAL_CONFIG)
    parser.add_argument(
        "--dummy-output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "taxonomy_smoke_dummy",
    )
    args = parser.parse_args()

    want_dummy = args.dummy or (not args.modal) or (args.modal and not args.skip_dummy)
    if args.modal:
        _activate_shortlist_env(args.model_config, args.modal_config)

    if want_dummy:
        run_dummy(args.dummy_output_dir)

    if args.modal:
        run_modal(
            output=args.output,
            context_length=args.context_length,
            sync=not args.detach,
            no_resume=args.no_resume,
        )
        if not args.detach:
            validate_latest_modal_bundle(args.output)


if __name__ == "__main__":
    main()
