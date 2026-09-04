"""FIDELITY: how well the compressed KV cache preserves the original tensors and attention.

Sub-metrics:
- representation — tensor-level reconstruction RMSE (compressors/representation.py)
- attention      — QK^T attention-score preservation
- memory         — KV-cache storage accounting (bytes, compression ratio, effective bitwidth)
- recurrent      — hybrid Mamba state preservation R'_t = R_t (§25; hybrid models only)
"""

from __future__ import annotations

from dataclasses import asdict

import torch

from compressors.base import KVCompressor
from eval.fidelity.attention import AttentionMetrics, evaluate_attention_fidelity
from eval.fidelity.memory import MemoryMetrics, evaluate_memory_from_cache
from eval.fidelity.recurrent import RecurrentMetrics, evaluate_recurrent_fidelity
from eval.fidelity.representation import RepresentationMetrics, evaluate_representation
from framework.config import load_eval_config
from framework.model import ModelLayer

__all__ = [
    "AttentionMetrics",
    "FidelityMetrics",
    "MemoryMetrics",
    "RecurrentMetrics",
    "RepresentationMetrics",
    "evaluate_attention_fidelity",
    "evaluate_fidelity",
    "evaluate_memory_from_cache",
    "evaluate_recurrent_fidelity",
    "evaluate_representation",
]


class FidelityMetrics:
    """Aggregate FIDELITY result: representation + attention + memory."""

    def __init__(
        self,
        representation: RepresentationMetrics,
        attention: AttentionMetrics,
        memory: MemoryMetrics,
        recurrent: RecurrentMetrics,
    ) -> None:
        self.representation = representation
        self.attention = attention
        self.memory = memory
        self.recurrent = recurrent

    def to_dict(self) -> dict:
        return {
            "representation": self.representation.to_dict(),
            "attention": self.attention.to_dict(),
            "memory": asdict(self.memory),
            "recurrent": self.recurrent.to_dict(),
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

    score_tokens = min(
        int(eval_config.get("attention_fidelity_tokens", 512)),
        int(input_ids.size(1)),
        128,
    )
    representation = evaluate_representation(past_key_values, compressor)
    attention = evaluate_attention_fidelity(
        model_layer,
        input_ids,
        compressor,
        outputs=outputs,
        score_tokens=score_tokens,
    )
    memory = evaluate_memory_from_cache(
        model_layer,
        input_ids,
        compressor,
        past_key_values=past_key_values,
    )
    recurrent = evaluate_recurrent_fidelity(
        past_key_values,
        compressor,
        model_layer.config,
        device=model_layer.device,
    )
    # Drop the large hidden-state tower before BEHAVIOR/SYSTEM begin.
    if hasattr(outputs, "hidden_states"):
        try:
            outputs.hidden_states = None
        except Exception:  # noqa: BLE001
            pass
    del outputs, past_key_values
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(compressor, "reset_state"):
        compressor.reset_state()
    return FidelityMetrics(
        representation=representation,
        attention=attention,
        memory=memory,
        recurrent=recurrent,
    )
