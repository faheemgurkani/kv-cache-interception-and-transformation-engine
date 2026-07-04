"""Paper-independent evaluation orchestrator."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from compressors.base import KVCompressor
from compressors.registry import get_compressor
from data.loader import build_long_context_ids, load_wikitext2
from eval.fidelity import FidelityMetrics, evaluate_fidelity
from eval.memory import MemoryMetrics
from eval.perplexity import evaluate_perplexity, evaluate_perplexity_baseline
from eval.throughput import ThroughputMetrics, evaluate_throughput, evaluate_throughput_baseline
from framework.config import load_eval_config, load_model_config
from framework.model import ModelLayer


@dataclass
class InferenceMetrics:
    """Section B: online inference impact with compressed KV in the loop."""

    perplexity: float | None
    perplexity_baseline: float | None
    throughput: ThroughputMetrics | None
    throughput_baseline: ThroughputMetrics | None

    def to_dict(self) -> dict:
        return {
            "perplexity": self.perplexity,
            "perplexity_baseline": self.perplexity_baseline,
            "throughput": asdict(self.throughput) if self.throughput else None,
            "throughput_baseline": asdict(self.throughput_baseline) if self.throughput_baseline else None,
        }


@dataclass
class EvaluationResult:
    compressor: str
    bitwidth: int | None
    context_length: int
    fidelity: FidelityMetrics
    inference: InferenceMetrics | None
    stage: str | None = None

    @property
    def perplexity(self) -> float | None:
        if self.inference is None:
            return None
        return self.inference.perplexity

    @property
    def memory(self) -> MemoryMetrics:
        return self.fidelity.memory

    @property
    def throughput(self) -> ThroughputMetrics | None:
        if self.inference is None:
            return None
        return self.inference.throughput

    def to_dict(self) -> dict:
        return {
            "compressor": self.compressor,
            "bitwidth": self.bitwidth,
            "stage": self.stage,
            "context_length": self.context_length,
            "section_a_fidelity": self.fidelity.to_dict(),
            "section_b_inference": None if self.inference is None else self.inference.to_dict(),
        }


def _compressor_stage(compressor: KVCompressor) -> str | None:
    stage = getattr(compressor, "stage", None)
    if stage is None:
        return None
    return stage.value if hasattr(stage, "value") else str(stage)


class EvaluationRunner:
    """Runs Section A (offline fidelity) and Section B (online inference) metrics."""

    def __init__(
        self,
        model_layer: ModelLayer | None = None,
        compressor: KVCompressor | None = None,
        eval_config: dict | None = None,
        model_config: dict | None = None,
    ) -> None:
        self.model_config = model_config or load_model_config()
        self.eval_config = eval_config or load_eval_config()
        self.model_layer = model_layer or ModelLayer()
        self.compressor = compressor or get_compressor("identity")
        self.dataset = load_wikitext2()

    def build_context(self, context_length: int) -> torch.LongTensor:
        return build_long_context_ids(
            self.model_layer.tokenizer,
            self.dataset,
            target_length=context_length,
        ).to(self.model_layer.device)

    def run(
        self,
        context_length: int,
        run_fidelity: bool = True,
        run_perplexity: bool = True,
        run_throughput: bool = True,
        include_baselines: bool = False,
        perplexity_stride: int | None = None,
        generated_tokens: int | None = None,
    ) -> EvaluationResult:
        input_ids = self.build_context(context_length)
        stride = perplexity_stride or self.eval_config.get("perplexity_stride", 512)
        num_new_tokens = generated_tokens or self.eval_config.get("generated_tokens", 64)

        if not run_fidelity:
            raise ValueError("Section A fidelity metrics are required for every evaluation run.")

        fidelity = evaluate_fidelity(self.model_layer, input_ids, self.compressor)

        inference: InferenceMetrics | None = None
        if run_perplexity or run_throughput:
            inference = InferenceMetrics(
                perplexity=(
                    evaluate_perplexity(self.model_layer, input_ids, self.compressor, stride=stride)
                    if run_perplexity
                    else None
                ),
                perplexity_baseline=(
                    evaluate_perplexity_baseline(self.model_layer, input_ids, stride=stride)
                    if include_baselines and run_perplexity
                    else None
                ),
                throughput=(
                    evaluate_throughput(
                        self.model_layer,
                        input_ids,
                        self.compressor,
                        num_new_tokens=num_new_tokens,
                    )
                    if run_throughput
                    else None
                ),
                throughput_baseline=(
                    evaluate_throughput_baseline(
                        self.model_layer,
                        input_ids,
                        num_new_tokens=num_new_tokens,
                    )
                    if include_baselines and run_throughput
                    else None
                ),
            )

        return EvaluationResult(
            compressor=self.compressor.name,
            bitwidth=getattr(self.compressor, "bitwidth", None),
            context_length=context_length,
            fidelity=fidelity,
            inference=inference,
            stage=_compressor_stage(self.compressor),
        )

    def run_all_context_lengths(
        self,
        context_lengths: list[int] | None = None,
        **kwargs,
    ) -> list[EvaluationResult]:
        lengths = context_lengths or self.model_config.get("context_lengths", [4096])
        return [self.run(length, **kwargs) for length in lengths]
