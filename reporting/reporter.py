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
            "context_length",
            "key_rmse",
            "value_rmse",
            "attention_rmse",
            "attention_cosine",
            "attention_max_error",
            "uncompressed_bytes",
            "compressed_bytes",
            "compression_ratio",
            "effective_bits_per_kv_element",
            "shared_metadata_bytes",
            "perplexity_compressed",
            "perplexity_baseline",
            "tokens_per_second",
            "latency_ms_per_token",
            "online_compressed_kv",
        ]
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                fidelity = result.fidelity
                inference = result.inference
                writer.writerow(
                    {
                        "compressor": result.compressor,
                        "bitwidth": result.bitwidth,
                        "context_length": result.context_length,
                        "key_rmse": fidelity.tensor.key_rmse,
                        "value_rmse": fidelity.tensor.value_rmse,
                        "attention_rmse": fidelity.attention.rmse,
                        "attention_cosine": fidelity.attention.cosine_similarity,
                        "attention_max_error": fidelity.attention.max_error,
                        "uncompressed_bytes": fidelity.memory.uncompressed_bytes,
                        "compressed_bytes": fidelity.memory.compressed_bytes,
                        "compression_ratio": fidelity.memory.compression_ratio,
                        "effective_bits_per_kv_element": fidelity.memory.effective_bits_per_kv_element,
                        "shared_metadata_bytes": fidelity.memory.shared_metadata_bytes,
                        "perplexity_compressed": inference.perplexity if inference else None,
                        "perplexity_baseline": inference.perplexity_baseline if inference else None,
                        "tokens_per_second": inference.throughput.tokens_per_second if inference and inference.throughput else None,
                        "latency_ms_per_token": inference.throughput.latency_ms_per_token if inference and inference.throughput else None,
                        "online_compressed_kv": inference.throughput.online_compressed_kv if inference and inference.throughput else None,
                    }
                )
        return path

    @staticmethod
    def print_summary(results: list[EvaluationResult]) -> None:
        for result in results:
            fidelity = result.fidelity
            inference = result.inference
            parts = [
                f"[{result.compressor}] ctx={result.context_length}",
                f"attn_rmse={fidelity.attention.rmse:.4f}",
                f"ratio={fidelity.memory.compression_ratio:.2f}x",
                f"bits/kv={fidelity.memory.effective_bits_per_kv_element:.2f}",
            ]
            if inference and inference.perplexity is not None:
                parts.append(f"ppl={inference.perplexity:.4f}")
            if inference and inference.throughput is not None:
                parts.append(f"tok/s={inference.throughput.tokens_per_second:.2f}")
            print(" ".join(parts))
