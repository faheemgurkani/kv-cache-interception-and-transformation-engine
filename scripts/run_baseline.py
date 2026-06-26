"""Run a baseline compressor via the shared evaluation runner."""

from __future__ import annotations

import argparse

import setup_path  # noqa: F401
from compressors.registry import get_compressor
from eval.runner import EvaluationRunner
from framework.config import load_eval_config
from reporting.reporter import ResultReporter


def main() -> None:
    eval_config = load_eval_config()
    parser = argparse.ArgumentParser(description="Run a baseline evaluation for the KV-cache engine.")
    parser.add_argument(
        "--baseline",
        choices=["identity", "turboquant", "kivi", "qjl", "rocketkv"],
        default="identity",
        help="Compressor/baseline method to evaluate.",
    )
    parser.add_argument("--bitwidth", type=int, default=None, help="Target bitwidth.")
    parser.add_argument(
        "--context-length",
        type=int,
        default=eval_config.get("default_context_length", 512),
        help="Evaluation context length in tokens.",
    )
    parser.add_argument("--output", default=None, help="Results JSON filename stem.")
    args = parser.parse_args()

    compressor = get_compressor(args.baseline, bitwidth=args.bitwidth)
    runner = EvaluationRunner(compressor=compressor)
    result = runner.run(args.context_length)

    reporter = ResultReporter()
    output_name = args.output or f"{args.baseline}_{args.context_length}"
    reporter.save_json(result, output_name)
    reporter.print_summary([result])


if __name__ == "__main__":
    main()
