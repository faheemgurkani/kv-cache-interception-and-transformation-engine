#!/usr/bin/env python3
"""Run the full TurboQuant evaluation grid from docs/EVALUATION_PLAN.md."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import setup_path  # noqa: F401

from compressors.registry import get_compressor
from eval.runner import EvaluationResult, EvaluationRunner
from framework.config import PROJECT_ROOT, load_eval_config, load_model_config
from reporting.reporter import ResultReporter


SWEEP_CONFIGS: list[tuple[str, dict]] = [
    ("tq_full_b2", {"name": "turboquant", "stage": "full", "bitwidth": 2}),
    ("tq_full_b3", {"name": "turboquant", "stage": "full", "bitwidth": 3}),
    ("tq_full_b4", {"name": "turboquant", "stage": "full", "bitwidth": 4}),
    ("tq_mse_b4", {"name": "turboquant", "stage": "wht_quant", "bitwidth": 4}),
]


def _result_key(result: EvaluationResult) -> tuple:
    return (result.compressor, result.bitwidth, result.stage, result.context_length)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TurboQuant evaluation sweep")
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=None,
        help="Context lengths to evaluate (default: configs/model.yaml)",
    )
    parser.add_argument("--skip-perplexity", action="store_true")
    parser.add_argument("--skip-throughput", action="store_true")
    parser.add_argument("--output-stem", default="phase5_turboquant_sweep")
    parser.add_argument(
        "--resume-json",
        type=Path,
        default=None,
        help="Resume from an existing sweep JSON (skip completed rows).",
    )
    return parser.parse_args()


def load_resume_keys(path: Path) -> set[tuple]:
    payload = json.loads(path.read_text())
    keys: set[tuple] = set()
    for row in payload.get("results", []):
        keys.add(
            (
                row["compressor"],
                row.get("bitwidth"),
                row.get("stage"),
                row["context_length"],
            )
        )
    return keys


def main() -> None:
    args = parse_args()
    model_config = load_model_config()
    eval_config = load_eval_config()
    context_lengths = args.context_lengths or model_config["context_lengths"]

    if args.resume_json:
        stem = args.resume_json.stem
        payload = json.loads(args.resume_json.read_text())
        done = load_resume_keys(args.resume_json)
        all_results = []  # fresh runs append; merged on save below
        print(f"Resuming from {args.resume_json} ({len(done)} completed rows)", flush=True)
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        stem = f"{args.output_stem}_{stamp}"
        all_results = []
        done = set()

    log_dir = PROJECT_ROOT / "results" / "sweep_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    reporter = ResultReporter()

    print(f"Sweep started: stem={stem} ctx={context_lengths} configs={len(SWEEP_CONFIGS)}", flush=True)

    for ctx in context_lengths:
        print(f"\n=== context_length={ctx} ===", flush=True)
        runner = EvaluationRunner(eval_config=eval_config, model_config=model_config)

        for label, cfg in SWEEP_CONFIGS:
            params = dict(cfg)
            name = params.pop("name")
            compressor = get_compressor(name, **params)
            stage = getattr(compressor, "stage", None)
            stage_val = stage.value if hasattr(stage, "value") else stage
            bitwidth = getattr(compressor, "bitwidth", None)
            key = (compressor.name, bitwidth, stage_val, ctx)
            if key in done:
                print(f"  [{label}] skip (already done)", flush=True)
                continue

            runner.compressor = compressor
            print(f"  [{label}] bitwidth={bitwidth} stage={stage_val}", flush=True)
            result = runner.run(
                ctx,
                run_perplexity=not args.skip_perplexity,
                run_throughput=not args.skip_throughput,
                include_baselines=True,
            )
            all_results.append(result)
            done.add(_result_key(result))
            reporter.save_json(all_results, stem)
            reporter.save_summary_csv(all_results, stem)
            reporter.print_summary([result])

    print(f"\nSweep complete. Wrote results/{stem}.json and .csv", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
