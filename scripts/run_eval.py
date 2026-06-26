"""Run evaluations for the KV-Cache Interception + Transformation Engine."""

from __future__ import annotations

import argparse

import setup_path  # noqa: F401
from compressors.registry import COMPRESSORS, get_compressor
from eval.runner import EvaluationRunner
from framework.config import load_eval_config, load_model_config
from reporting.reporter import ResultReporter


def main() -> None:
    eval_config = load_eval_config()
    model_config = load_model_config()

    parser = argparse.ArgumentParser(description="Run KV-cache interception engine evaluations.")
    parser.add_argument(
        "--compressor",
        choices=sorted(COMPRESSORS),
        default="identity",
        help="Compression method to evaluate.",
    )
    parser.add_argument("--bitwidth", type=int, default=None)
    parser.add_argument(
        "--stage",
        default=None,
        choices=["wht_only", "wht_quant", "wht_quant_residual", "full"],
        help="TurboQuant ablation stage (turboquant only).",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="Single context length. Overrides --all-context-lengths.",
    )
    parser.add_argument(
        "--all-context-lengths",
        action="store_true",
        help="Run all context lengths from configs/model.yaml.",
    )
    parser.add_argument("--skip-perplexity", action="store_true")
    parser.add_argument("--skip-throughput", action="store_true")
    parser.add_argument("--include-baselines", action="store_true", help="Also run uncompressed baseline PPL/throughput.")
    parser.add_argument("--output", default="eval_results", help="Output filename stem.")
    args = parser.parse_args()

    kwargs = {}
    if args.bitwidth is not None:
        kwargs["bitwidth"] = args.bitwidth
    if args.stage is not None:
        kwargs["stage"] = args.stage
    compressor = get_compressor(args.compressor, **kwargs)
    runner = EvaluationRunner(compressor=compressor)

    if args.all_context_lengths:
        results = runner.run_all_context_lengths(
            context_lengths=model_config["context_lengths"],
            run_perplexity=not args.skip_perplexity,
            run_throughput=not args.skip_throughput,
            include_baselines=args.include_baselines,
        )
    else:
        context_length = args.context_length or eval_config.get("default_context_length", 512)
        results = [
            runner.run(
                context_length,
                run_perplexity=not args.skip_perplexity,
                run_throughput=not args.skip_throughput,
                include_baselines=args.include_baselines,
            )
        ]

    reporter = ResultReporter()
    reporter.save_json(results, args.output)
    reporter.save_summary_csv(results, args.output)
    reporter.print_summary(results)


if __name__ == "__main__":
    main()
