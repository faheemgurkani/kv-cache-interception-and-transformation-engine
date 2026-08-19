"""Run evaluations for the KV-Cache Interception and Transformation Engine."""

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
    parser.add_argument("--skip-perplexity", action="store_true", help="Skip BEHAVIOR/task_quality (perplexity).")
    parser.add_argument("--skip-retrieval", action="store_true", help="Skip BEHAVIOR/retrieval (needle-in-haystack).")
    parser.add_argument(
        "--skip-instruction-following",
        action="store_true",
        help="Skip BEHAVIOR/instruction_following.",
    )
    parser.add_argument("--skip-fidelity", action="store_true", help="Skip FIDELITY (representation/memory/attention).")
    parser.add_argument("--skip-throughput", action="store_true", help="Skip SYSTEM/latency_throughput.")
    parser.add_argument("--reasoning", action="store_true", help="Run BEHAVIOR/reasoning (synthetic arithmetic).")
    parser.add_argument("--peak-memory", action="store_true", help="Run SYSTEM/vram (peak CUDA memory).")
    parser.add_argument("--memory-bandwidth", action="store_true", help="Run SYSTEM/memory_bandwidth.")
    parser.add_argument("--kernel-cost", action="store_true", help="Run SYSTEM/kernel_cost (compress/decompress time).")
    parser.add_argument("--gpu-utilization", action="store_true", help="Run SYSTEM/gpu_utilization (CUDA + pynvml only).")
    parser.add_argument(
        "--hardware-metrics",
        action="store_true",
        help="Enable peak VRAM + GPU util (also auto-on when KV_EXECUTION_PLATFORM=modal).",
    )
    parser.add_argument("--skip-cost", action="store_true", help="Skip COST accounting block (Phase 3).")
    parser.add_argument(
        "--max-capacity-prompt",
        type=int,
        default=None,
        help="SnapKV total KV budget including observation window.",
    )
    parser.add_argument("--window-size", type=int, default=None, help="SnapKV observation window size.")
    parser.add_argument("--kernel-size", type=int, default=None, help="SnapKV pooling kernel size.")
    parser.add_argument(
        "--compression-rate",
        type=float,
        default=None,
        help="Palu target compression rate (G-LRD, default 0.5).",
    )
    parser.add_argument("--group-size", type=int, default=None, help="Palu G-LRD group size (default 4).")
    parser.add_argument("--include-baselines", action="store_true", help="Also run uncompressed baseline PPL/throughput.")
    parser.add_argument("--output", default="eval_results", help="Output filename stem.")
    args = parser.parse_args()

    kwargs = {}
    if args.bitwidth is not None:
        kwargs["bitwidth"] = args.bitwidth
    if args.stage is not None:
        kwargs["stage"] = args.stage
    if args.max_capacity_prompt is not None:
        kwargs["max_capacity_prompt"] = args.max_capacity_prompt
    if args.window_size is not None:
        kwargs["window_size"] = args.window_size
    if args.kernel_size is not None:
        kwargs["kernel_size"] = args.kernel_size
    if args.compression_rate is not None:
        kwargs["compression_rate"] = args.compression_rate
    if args.group_size is not None:
        kwargs["group_size"] = args.group_size
    compressor = get_compressor(args.compressor, **kwargs)
    runner = EvaluationRunner(compressor=compressor)

    run_kwargs = {
        "run_fidelity": not args.skip_fidelity,
        "run_perplexity": not args.skip_perplexity,
        "run_retrieval": not args.skip_retrieval,
        "run_reasoning": args.reasoning,
        "run_instruction_following": not args.skip_instruction_following,
        "run_throughput": not args.skip_throughput,
        "run_peak_memory": args.peak_memory,
        "run_memory_bandwidth": args.memory_bandwidth,
        "run_kernel_cost": args.kernel_cost,
        "run_gpu_utilization": args.gpu_utilization,
        "collect_hardware_metrics": args.hardware_metrics or None,
        "run_cost": not args.skip_cost,
        "include_baselines": args.include_baselines,
    }

    if args.all_context_lengths:
        results = runner.run_all_context_lengths(
            context_lengths=model_config["context_lengths"],
            **run_kwargs,
        )
    else:
        context_length = args.context_length or eval_config.get("default_context_length", 512)
        results = [runner.run(context_length, **run_kwargs)]

    reporter = ResultReporter()
    reporter.save_json(results, args.output)
    reporter.save_summary_csv(results, args.output)
    reporter.print_summary(results)


if __name__ == "__main__":
    main()
