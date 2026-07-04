"""Section A: offline compression fidelity metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from compressors.base import KVCompressor
from eval.attention_score_error import AttentionMetrics, evaluate_attention_fidelity
from eval.memory import MemoryMetrics, evaluate_memory_from_cache
from framework.config import load_eval_config
from framework.kv_cache import iter_layer_kv
from framework.model import ModelLayer


@dataclass
class TensorReconstructionMetrics:
    key_rmse: float
    value_rmse: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FidelityMetrics:
    """Offline compression fidelity (Section A)."""

    tensor: TensorReconstructionMetrics
    attention: AttentionMetrics
    memory: MemoryMetrics

    def to_dict(self) -> dict:
        return {
            "tensor": self.tensor.to_dict(),
            "attention": self.attention.to_dict(),
            "memory": asdict(self.memory),
        }


def _tensor_reconstruction_error(
    past_key_values,
    compressor: KVCompressor,
) -> TensorReconstructionMetrics:
    key_rmses: list[float] = []
    value_rmses: list[float] = []

    for layer_idx, (key, value) in enumerate(iter_layer_kv(past_key_values)):
        if hasattr(compressor, "reconstruction_error"):
            errors = compressor.reconstruction_error(key, value)
            key_rmses.append(errors["key_rmse"])
            value_rmses.append(errors["value_rmse"])
            continue

        k_hat = compressor.decompress_kv(
            compressor.compress_kv(key, layer=layer_idx, mode="key"),
            mode="key",
        ).to(key.device)
        v_hat = compressor.decompress_kv(
            compressor.compress_kv(value, layer=layer_idx, mode="value"),
            mode="value",
        ).to(value.device)
        key_rmses.append((key.float() - k_hat.float()).pow(2).mean().sqrt().item())
        value_rmses.append((value.float() - v_hat.float()).pow(2).mean().sqrt().item())

    if not key_rmses:
        raise RuntimeError("No KV tensors found for reconstruction error.")

    return TensorReconstructionMetrics(
        key_rmse=sum(key_rmses) / len(key_rmses),
        value_rmse=sum(value_rmses) / len(value_rmses),
    )


@torch.no_grad()
def evaluate_fidelity(
    model_layer: ModelLayer,
    input_ids: torch.Tensor,
    compressor: KVCompressor,
) -> FidelityMetrics:
    """Run offline fidelity with one forward pass to limit GPU memory."""
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

    tensor = _tensor_reconstruction_error(past_key_values, compressor)
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
    return FidelityMetrics(tensor=tensor, attention=attention, memory=memory)
