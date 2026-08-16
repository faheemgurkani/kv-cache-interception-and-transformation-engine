"""Reporting layer for KV-cache engine evaluation results."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from eval.runner import EvaluationResult
from framework.config import PROJECT_ROOT


class ResultReporter:
    """Persist evaluation outputs to results/ and plots/."""

    def __init__(self, output_dir: Path | str | None = None) -> None:
        self.output_dir = Path(output_dir or PROJECT_ROOT / "results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_json(self, results: list[EvaluationResult] | EvaluationResult, name: str) -> Path:
        payload = results if isinstance(results, list) else [results]
        report = {
            "timestamp": datetime.now(UTC).isoformat(),
            "results": [item.to_dict() for item in payload],
        }
        path = self.output_dir / f"{name}.json"
        path.write_text(json.dumps(report, indent=2))
        return path

    def save_summary_csv(self, results: list[EvaluationResult], name: str) -> Path:
        import csv

        path = self.output_dir / f"{name}.csv"
        fieldnames = [
            "compressor",
            "bitwidth",
            "stage",
            "context_length",
            # FIDELITY
            "key_rmse",
            "value_rmse",
            "key_relative_error",
            "value_relative_error",
            "key_cosine_similarity",
            "value_cosine_similarity",
            "attention_rmse",
            "attention_cosine",
            "attention_max_error",
            "attention_output_rmse",
            "attention_distribution_kl_divergence",
            "uncompressed_bytes",
            "compressed_bytes",
            "compression_ratio",
            "effective_bits_per_kv_element",
            "shared_metadata_bytes",
            # BEHAVIOR
            "perplexity_compressed",
            "perplexity_baseline",
            "retrieval_accuracy",
            "reasoning_accuracy",
            "instruction_following_compliance",
            # SYSTEM
            "ttft_ms",
            "itl_ms_mean",
            "decode_latency_ms",
            "tokens_per_second",
            "latency_ms_per_token",
            "end_to_end_latency_ms",
            "peak_vram_allocated_mb",
            "memory_bandwidth_gbps",
            "compress_decompress_time_ms",
            "online_compressed_kv",
        ]
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                fidelity = result.fidelity
                behavior = result.behavior
                system = result.system
                throughput = system.throughput if system else None
                writer.writerow(
                    {
                        "compressor": result.compressor,
                        "bitwidth": result.bitwidth,
                        "stage": result.stage,
                        "context_length": result.context_length,
                        "key_rmse": fidelity.representation.key_rmse,
                        "value_rmse": fidelity.representation.value_rmse,
                        "key_relative_error": fidelity.representation.key_relative_error,
                        "value_relative_error": fidelity.representation.value_relative_error,
                        "key_cosine_similarity": fidelity.representation.key_cosine_similarity,
                        "value_cosine_similarity": fidelity.representation.value_cosine_similarity,
                        "attention_rmse": fidelity.attention.rmse,
                        "attention_cosine": fidelity.attention.cosine_similarity,
                        "attention_max_error": fidelity.attention.max_error,
                        "attention_output_rmse": fidelity.attention.output_rmse,
                        "attention_distribution_kl_divergence": fidelity.attention.distribution_kl_divergence,
                        "uncompressed_bytes": fidelity.memory.uncompressed_bytes,
                        "compressed_bytes": fidelity.memory.compressed_bytes,
                        "compression_ratio": fidelity.memory.compression_ratio,
                        "effective_bits_per_kv_element": fidelity.memory.effective_bits_per_kv_element,
                        "shared_metadata_bytes": fidelity.memory.shared_metadata_bytes,
                        "perplexity_compressed": behavior.perplexity if behavior else None,
                        "perplexity_baseline": behavior.perplexity_baseline if behavior else None,
                        "retrieval_accuracy": (
                            behavior.retrieval.exact_match_accuracy if behavior and behavior.retrieval else None
                        ),
                        "reasoning_accuracy": (
                            behavior.reasoning.exact_match_accuracy if behavior and behavior.reasoning else None
                        ),
                        "instruction_following_compliance": (
                            behavior.instruction_following.format_compliance_rate
                            if behavior and behavior.instruction_following
                            else None
                        ),
                        "ttft_ms": throughput.ttft_ms if throughput else None,
                        "itl_ms_mean": throughput.itl_ms_mean if throughput else None,
                        "decode_latency_ms": throughput.decode_latency_ms if throughput else None,
                        "tokens_per_second": throughput.tokens_per_second if throughput else None,
                        "latency_ms_per_token": throughput.latency_ms_per_token if throughput else None,
                        "end_to_end_latency_ms": throughput.end_to_end_latency_ms if throughput else None,
                        "peak_vram_allocated_mb": (
                            system.peak_memory.peak_allocated_mb if system and system.peak_memory else None
                        ),
                        "memory_bandwidth_gbps": (
                            system.memory_bandwidth.effective_bandwidth_gbps
                            if system and system.memory_bandwidth
                            else None
                        ),
                        "compress_decompress_time_ms": (
                            system.kernel_cost.compress_decompress_time_ms
                            if system and system.kernel_cost
                            else None
                        ),
                        "online_compressed_kv": throughput.online_compressed_kv if throughput else None,
                    }
                )
        return path

    @staticmethod
    def print_summary(results: list[EvaluationResult]) -> None:
        for result in results:
            fidelity = result.fidelity
            behavior = result.behavior
            system = result.system
            parts = [
                f"[{result.compressor}] ctx={result.context_length}",
                f"stage={result.stage}" if result.stage else None,
                f"attn_rmse={fidelity.attention.rmse:.4f}",
                f"ratio={fidelity.memory.compression_ratio:.2f}x",
                f"bits/kv={fidelity.memory.effective_bits_per_kv_element:.2f}",
            ]
            parts = [p for p in parts if p]
            if behavior and behavior.perplexity is not None:
                parts.append(f"ppl={behavior.perplexity:.4f}")
            if behavior and behavior.retrieval is not None:
                parts.append(f"retrieval={behavior.retrieval.exact_match_accuracy:.2f}")
            if behavior and behavior.instruction_following is not None:
                parts.append(f"instr_follow={behavior.instruction_following.format_compliance_rate:.2f}")
            if system and system.throughput is not None:
                parts.append(f"tok/s={system.throughput.tokens_per_second:.2f}")
                if system.throughput.ttft_ms is not None:
                    parts.append(f"ttft={system.throughput.ttft_ms:.1f}ms")
            print(" ".join(parts))
