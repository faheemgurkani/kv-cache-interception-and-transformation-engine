"""BEHAVIOR: what happens to generation quality when compressed KV drives real decoding.

Sub-metrics:
- task_quality          — sliding-window perplexity (compressed vs baseline)
- retrieval             — needle-in-haystack exact-match recall
- reasoning             — synthetic multi-step arithmetic accuracy
- instruction_following — output-format compliance rate

All sub-metrics run through the same KVCacheEngine autoregressive loop used by
SYSTEM/latency, so BEHAVIOR always reflects compressed KV in actual decoding, never
a single offline forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass

from compressors.base import KVCompressor
from eval.behavior.instruction_following import InstructionFollowingMetrics, evaluate_instruction_following
from eval.behavior.reasoning import ReasoningMetrics, evaluate_reasoning
from eval.behavior.retrieval import RetrievalMetrics, evaluate_retrieval
from eval.behavior.task_quality import (
    PerplexityResult,
    evaluate_perplexity,
    evaluate_perplexity_baseline,
    evaluate_perplexity_result,
)
from framework.model import ModelLayer

__all__ = [
    "BehaviorMetrics",
    "InstructionFollowingMetrics",
    "ReasoningMetrics",
    "RetrievalMetrics",
    "evaluate_behavior",
    "evaluate_instruction_following",
    "PerplexityResult",
    "evaluate_perplexity",
    "evaluate_perplexity_baseline",
    "evaluate_perplexity_result",
    "evaluate_reasoning",
    "evaluate_retrieval",
]


@dataclass
class BehaviorMetrics:
    """Aggregate BEHAVIOR result. Every field is optional — callers opt into what they run."""

    perplexity: float | None = None
    perplexity_baseline: float | None = None
    n_tokens: int | None = None
    nll_sum: float | None = None
    prefill_tokens: int | None = None
    retrieval: RetrievalMetrics | None = None
    reasoning: ReasoningMetrics | None = None
    instruction_following: InstructionFollowingMetrics | None = None

    def to_dict(self) -> dict:
        return {
            "task_quality": {
                "perplexity": self.perplexity,
                "perplexity_baseline": self.perplexity_baseline,
                "n_tokens": self.n_tokens,
                "nll_sum": self.nll_sum,
                "prefill_tokens": self.prefill_tokens,
            },
            "retrieval": self.retrieval.to_dict() if self.retrieval else None,
            "reasoning": self.reasoning.to_dict() if self.reasoning else None,
            "instruction_following": (
                self.instruction_following.to_dict() if self.instruction_following else None
            ),
        }


def evaluate_behavior(
    model_layer: ModelLayer,
    input_ids,
    compressor: KVCompressor,
    *,
    run_task_quality: bool = True,
    run_retrieval: bool = True,
    run_reasoning: bool = False,
    run_instruction_following: bool = True,
    include_baseline: bool = False,
    perplexity_stride: int = 512,
    context_length: int | None = None,
) -> BehaviorMetrics:
    """Run the requested BEHAVIOR sub-metrics. Retrieval and instruction-following run
    by default (plan recommendation: PPL + retrieval + instruction following); reasoning
    is opt-in since it adds another generate() pass on top of task_quality."""
    perplexity_baseline = (
        evaluate_perplexity_baseline(model_layer, input_ids, stride=perplexity_stride)
        if include_baseline and run_task_quality
        else None
    )
    ppl_result = (
        evaluate_perplexity_result(
            model_layer,
            input_ids,
            compressor,
            max_length=context_length,
            stride=perplexity_stride,
        )
        if run_task_quality
        else None
    )
    perplexity = ppl_result.perplexity if ppl_result is not None else None
    retrieval = (
        evaluate_retrieval(model_layer, compressor, context_length=context_length or input_ids.size(1))
        if run_retrieval
        else None
    )
    reasoning = evaluate_reasoning(model_layer, compressor) if run_reasoning else None
    instruction_following = (
        evaluate_instruction_following(model_layer, compressor) if run_instruction_following else None
    )

    return BehaviorMetrics(
        perplexity=perplexity,
        perplexity_baseline=perplexity_baseline,
        n_tokens=ppl_result.n_tokens if ppl_result is not None else None,
        nll_sum=ppl_result.nll_sum if ppl_result is not None else None,
        prefill_tokens=ppl_result.prefill_tokens if ppl_result is not None else None,
        retrieval=retrieval,
        reasoning=reasoning,
        instruction_following=instruction_following,
    )
