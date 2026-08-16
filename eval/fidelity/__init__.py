"""FIDELITY: how well the compressed KV cache preserves the original tensors and attention.

Sub-metrics:
- representation — tensor-level reconstruction RMSE (compressors/representation.py)
- attention      — QK^T attention-score preservation
- memory         — KV-cache storage accounting (bytes, compression ratio, effective bitwidth)
"""

from __future__ import annotations

from dataclasses import asdict

import torch

from compressors.base import KVCompressor
from eval.fidelity.attention import AttentionMetrics, evaluate_attention_fidelity
from eval.fidelity.memory import MemoryMetrics, evaluate_memory_from_cache
from eval.fidelity.representation import RepresentationMetrics, evaluate_representation
from framework.config import load_eval_config
from framework.model import ModelLayer

__all__ = [
    "AttentionMetrics",
    "FidelityMetrics",
    "MemoryMetrics",
    "RepresentationMetrics",
    "evaluate_attention_fidelity",
    "evaluate_fidelity",
    "evaluate_memory_from_cache",
    "evaluate_representation",
]


class FidelityMetrics:
    """Aggregate FIDELITY result: representation + attention + memory."""

    def __init__(
        self,
        representation: RepresentationMetrics,
        attention: AttentionMetrics,
        memory: MemoryMetrics,
    ) -> None:
        self.representation = representation
        self.attention = attention
        self.memory = memory

    def to_dict(self) -> dict:
        return {
            "representation": self.representation.to_dict(),
            "attention": self.attention.to_dict(),
            "memory": asdict(self.memory),
        }


@torch.no_grad()
def evaluate_fidelity(
    model_layer: ModelLayer,
    input_ids: torch.Tensor,
    compressor: KVCompressor,
) -> FidelityMetrics:
    """Run the FIDELITY dimension with a single forward pass to limit GPU memory."""
    eval_config = load_eval_config()
    outputs = model_layer.model(
        input_ids.to(model_layer.device),
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    past_key_values = outputs.past_key_values
    if past_key_values is None:
        raise RuntimeError("Model did not return past_key_values.")

    representation = evaluate_representation(past_key_values, compressor)
    attention = evaluate_attention_fidelity(
        model_layer,
        input_ids,
        compressor,
        outputs=outputs,
        score_tokens=eval_config.get("attention_fidelity_tokens", 512),
    )
    memory = evaluate_memory_from_cache(
        model_layer,
        input_ids,
        compressor,
        past_key_values=past_key_values,
    )
    del outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(compressor, "reset_state"):
        compressor.reset_state()
    return FidelityMetrics(representation=representation, attention=attention, memory=memory)
