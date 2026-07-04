"""Section A: offline compression fidelity metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from compressors.base import KVCompressor
from eval.attention_score_error import AttentionMetrics, evaluate_attention_fidelity
from eval.memory import MemoryMetrics, evaluate_memory
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
    model_layer: ModelLayer,
    input_ids: torch.Tensor,
    compressor: KVCompressor,
) -> TensorReconstructionMetrics:
    outputs = model_layer.forward_with_cache(input_ids)
    key_rmses: list[float] = []
    value_rmses: list[float] = []

    for layer_idx, (key, value) in enumerate(iter_layer_kv(outputs.past_key_values)):
        if hasattr(compressor, "reconstruction_error"):
            errors = compressor.reconstruction_error(key, value)
            key_rmses.append(errors["key_rmse"])
            value_rmses.append(errors["value_rmse"])
            continue

        k_hat = compressor.decompress_kv(compressor.compress_kv(key, layer=layer_idx, mode="key"), mode="key").to(key.device)
        v_hat = compressor.decompress_kv(compressor.compress_kv(value, layer=layer_idx, mode="value"), mode="value").to(value.device)
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
    """Run offline fidelity: tensor RMSE, QK^T preservation, memory accounting."""
    tensor = _tensor_reconstruction_error(model_layer, input_ids, compressor)
    attention = evaluate_attention_fidelity(model_layer, input_ids, compressor)
    memory = evaluate_memory(model_layer, input_ids, compressor)
    return FidelityMetrics(tensor=tensor, attention=attention, memory=memory)
