"""Paper-independent evaluation orchestrator.

Every run produces three independent branches instead of the old offline/online
split:

    KVBench
       |
       +-- FIDELITY  -- did the transformation preserve the KV representation
       |                and attention behavior? (eval/fidelity)
       +-- BEHAVIOR  -- does the model still behave correctly after KV
       |                transformation? (eval/behavior)
       +-- SYSTEM    -- does the compression actually make inference better?
                        (eval/system)

FIDELITY runs by default (single offline forward pass). Pass ``run_fidelity=False`` or
``scripts/run_eval.py --skip-fidelity`` to collect BEHAVIOR/SYSTEM on models whose
attention adapter is not yet registered. BEHAVIOR and SYSTEM sub-metrics are opt-in
flags on ``run()`` since each adds its own generate() pass through KVCacheEngine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from compressors.base import KVCompressor
from compressors.registry import get_compressor
from data.loader import build_long_context_ids, load_wikitext2
from eval.behavior import BehaviorMetrics, evaluate_behavior
from eval.fidelity import FidelityMetrics, evaluate_fidelity
from eval.system import SystemMetrics, evaluate_system
from framework.config import load_eval_config, load_model_config
from framework.model import ModelLayer
from framework.model_capabilities import ModelCapabilities, get_model_eval_metadata, resolve_model_capabilities


@dataclass
class EvaluationResult:
    compressor: str
    bitwidth: int | None
    context_length: int
    fidelity: FidelityMetrics | None
    behavior: BehaviorMetrics | None
    system: SystemMetrics | None
    stage: str | None = None
    model_capabilities: ModelCapabilities | None = field(default=None, repr=False)

    # --- back-compat accessors (previous Section A/B field names) ---
    @property
    def perplexity(self) -> float | None:
        return self.behavior.perplexity if self.behavior else None

    @property
    def memory(self):
        return self.fidelity.memory if self.fidelity else None

    @property
    def throughput(self):
        return self.system.throughput if self.system else None

    def to_dict(self) -> dict:
        caps = self.model_capabilities
        model_meta = None
        if caps is not None:
            base = get_model_eval_metadata(self.model_layer_config) if hasattr(self, "model_layer_config") else None
            if base is None:
                base = {
                    "model_type": caps.model_type,
                    "attention_family": caps.attention_family,
                    "state_type": caps.state_type.value,
                    "rope_mode": caps.rope_mode,
                    "adapter": caps.model_type,
                    "adapter_registered": caps.adapter_registered,
                    "state_semantics_complete": caps.state_semantics_complete,
                }
            model_meta = base
        return {
            "compressor": self.compressor,
            "bitwidth": self.bitwidth,
            "stage": self.stage,
            "context_length": self.context_length,
            "model": model_meta,
            "fidelity": None if self.fidelity is None else self.fidelity.to_dict(),
            "behavior": None if self.behavior is None else self.behavior.to_dict(),
            "system": None if self.system is None else self.system.to_dict(),
        }


def _compressor_stage(compressor: KVCompressor) -> str | None:
    stage = getattr(compressor, "stage", None)
    if stage is None:
        return None
    return stage.value if hasattr(stage, "value") else str(stage)


class EvaluationRunner:
    """Runs the FIDELITY / BEHAVIOR / SYSTEM evaluation branches."""

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
        # BEHAVIOR
        run_behavior: bool = True,
        run_perplexity: bool = True,
        run_retrieval: bool = False,
        run_reasoning: bool = False,
        run_instruction_following: bool = False,
        # SYSTEM
        run_system: bool = True,
        run_throughput: bool = True,
        run_peak_memory: bool = False,
        run_memory_bandwidth: bool = False,
        run_kernel_cost: bool = False,
        run_gpu_utilization: bool = False,
        include_baselines: bool = False,
        perplexity_stride: int | None = None,
        generated_tokens: int | None = None,
    ) -> EvaluationResult:
        input_ids = self.build_context(context_length)
        stride = perplexity_stride or self.eval_config.get("perplexity_stride", 512)
        num_new_tokens = generated_tokens or self.eval_config.get("generated_tokens", 64)

        if not run_fidelity:
            fidelity = None
        else:
            fidelity = evaluate_fidelity(self.model_layer, input_ids, self.compressor)

        behavior: BehaviorMetrics | None = None
        if run_behavior:
            # Baselines must run before KVCacheEngine construction (RocketKV patches attention).
            behavior = evaluate_behavior(
                self.model_layer,
                input_ids,
                self.compressor,
                run_task_quality=run_perplexity,
                run_retrieval=run_retrieval,
                run_reasoning=run_reasoning,
                run_instruction_following=run_instruction_following,
                include_baseline=include_baselines,
                perplexity_stride=stride,
                context_length=context_length,
            )

        system: SystemMetrics | None = None
        if run_system:
            compressed_bytes = fidelity.memory.compressed_bytes if fidelity else None
            system = evaluate_system(
                self.model_layer,
                input_ids,
                self.compressor,
                run_throughput=run_throughput,
                run_peak_memory=run_peak_memory,
                run_memory_bandwidth=run_memory_bandwidth,
                run_kernel_cost=run_kernel_cost,
                run_gpu_utilization=run_gpu_utilization,
                include_baseline=include_baselines,
                num_new_tokens=num_new_tokens,
                actual_kv_memory_bytes=compressed_bytes,
            )

        return EvaluationResult(
            compressor=self.compressor.name,
            bitwidth=getattr(self.compressor, "bitwidth", None),
            context_length=context_length,
            fidelity=fidelity,
            behavior=behavior,
            system=system,
            stage=_compressor_stage(self.compressor),
            model_capabilities=resolve_model_capabilities(self.model_layer.config),
        )

    def run_all_context_lengths(
        self,
        context_lengths: list[int] | None = None,
        **kwargs,
    ) -> list[EvaluationResult]:
        lengths = context_lengths or self.model_config.get("context_lengths", [128, 256, 512])
        return [self.run(length, **kwargs) for length in lengths]
