"""Shared evaluation framework."""

from eval.attention_score_error import AttentionMetrics, evaluate_attention_fidelity
from eval.fidelity import FidelityMetrics, evaluate_fidelity
from eval.memory import MemoryMetrics, evaluate_memory, kv_cache_bytes, process_memory_mb
from eval.perplexity import evaluate_perplexity, evaluate_perplexity_baseline
from eval.runner import EvaluationResult, EvaluationRunner, InferenceMetrics
from eval.throughput import ThroughputMetrics, evaluate_throughput, evaluate_throughput_baseline, measure_tokens_per_second

__all__ = [
    "AttentionMetrics",
    "EvaluationResult",
    "EvaluationRunner",
    "FidelityMetrics",
    "InferenceMetrics",
    "MemoryMetrics",
    "ThroughputMetrics",
    "evaluate_attention_fidelity",
    "evaluate_fidelity",
    "evaluate_memory",
    "evaluate_perplexity",
    "evaluate_perplexity_baseline",
    "evaluate_throughput",
    "evaluate_throughput_baseline",
    "kv_cache_bytes",
    "measure_tokens_per_second",
    "process_memory_mb",
]
