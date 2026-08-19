"""Shared evaluation framework: FIDELITY / BEHAVIOR / SYSTEM."""

from eval.behavior import BehaviorMetrics, evaluate_behavior
from eval.behavior.instruction_following import InstructionFollowingMetrics, evaluate_instruction_following
from eval.behavior.reasoning import ReasoningMetrics, evaluate_reasoning
from eval.behavior.retrieval import RetrievalMetrics, evaluate_retrieval
from eval.behavior.task_quality import evaluate_perplexity, evaluate_perplexity_baseline
from eval.fidelity import (
    AttentionMetrics,
    FidelityMetrics,
    MemoryMetrics,
    RepresentationMetrics,
    evaluate_attention_fidelity,
    evaluate_fidelity,
    evaluate_memory_from_cache,
    evaluate_representation,
)
from eval.fidelity.memory import evaluate_memory, kv_cache_bytes, process_memory_mb
from eval.controlled_conditions import ControlledInterceptionContract, PHASE6_PRINCIPLE, build_controlled_conditions
from eval.runner import EvaluationResult, EvaluationRunner
from eval.system import (
    GPUUtilizationMetrics,
    KernelCostMetrics,
    MemoryBandwidthMetrics,
    PeakMemoryMetrics,
    SystemMetrics,
    ThroughputMetrics,
    evaluate_gpu_utilization,
    evaluate_kernel_cost,
    evaluate_memory_bandwidth,
    evaluate_peak_vram,
    evaluate_system,
    evaluate_throughput,
    evaluate_throughput_baseline,
    measure_tokens_per_second,
)

__all__ = [
    "AttentionMetrics",
    "BehaviorMetrics",
    "ControlledInterceptionContract",
    "EvaluationResult",
    "EvaluationRunner",
    "FidelityMetrics",
    "GPUUtilizationMetrics",
    "InstructionFollowingMetrics",
    "KernelCostMetrics",
    "MemoryBandwidthMetrics",
    "MemoryMetrics",
    "PHASE6_PRINCIPLE",
    "PeakMemoryMetrics",
    "ReasoningMetrics",
    "RepresentationMetrics",
    "RetrievalMetrics",
    "SystemMetrics",
    "ThroughputMetrics",
    "build_controlled_conditions",
    "evaluate_attention_fidelity",
    "evaluate_behavior",
    "evaluate_fidelity",
    "evaluate_gpu_utilization",
    "evaluate_instruction_following",
    "evaluate_kernel_cost",
    "evaluate_memory",
    "evaluate_memory_bandwidth",
    "evaluate_memory_from_cache",
    "evaluate_peak_vram",
    "evaluate_perplexity",
    "evaluate_perplexity_baseline",
    "evaluate_reasoning",
    "evaluate_representation",
    "evaluate_retrieval",
    "evaluate_system",
    "evaluate_throughput",
    "evaluate_throughput_baseline",
    "kv_cache_bytes",
    "measure_tokens_per_second",
    "process_memory_mb",
]
