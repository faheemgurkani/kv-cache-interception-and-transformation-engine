"""FIDELITY / Representation: tensor-level reconstruction error of compressed K/V.

RMSE uses each compressor's own `reconstruction_error` hook when available (some
methods, e.g. RocketKV, measure error on a post-selection subset of tokens rather
than a naive full round trip). Relative error and cosine similarity always use the
same explicit compress_kv -> decompress_kv round trip, since every KVCompressor
subclass must implement that pair (it's the abstract interface), which keeps those
two metrics comparable across compressors even when RMSE semantics differ.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from compressors.base import KVCompressor
from framework.kv_cache import iter_layer_kv


@dataclass
class RepresentationMetrics:
    """Tensor-level reconstruction fidelity between original and reconstructed K/V
    (per-layer averaged)."""

    key_rmse: float
    value_rmse: float
    key_relative_error: float
    value_relative_error: float
    key_cosine_similarity: float
    value_cosine_similarity: float

    def to_dict(self) -> dict:
        return asdict(self)


def _relative_error(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """||x - x_hat||_2 / ||x||_2, i.e. normalized reconstruction error."""
    original = original.float()
    reconstructed = reconstructed.float().to(original.device)
    denom = original.norm()
    if denom.item() == 0.0:
        return 0.0
    return ((original - reconstructed).norm() / denom).item()


def _cosine_similarity(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    original = original.float().flatten()
    reconstructed = reconstructed.float().to(original.device).flatten()
    value = F.cosine_similarity(original, reconstructed, dim=0).item()
    if math.isnan(value):
        return 0.0
    return max(-1.0, min(1.0, value))


def evaluate_representation(past_key_values, compressor: KVCompressor) -> RepresentationMetrics:
    """Round-trip each layer's K/V through the compressor and measure RMSE, relative
    error, and cosine similarity."""
    key_rmses: list[float] = []
    value_rmses: list[float] = []
    key_rel_errors: list[float] = []
    value_rel_errors: list[float] = []
    key_cosines: list[float] = []
    value_cosines: list[float] = []

    for layer_idx, (key, value) in enumerate(iter_layer_kv(past_key_values)):
        k_hat, v_hat, key_ref, value_ref = _layer_roundtrip(key, value, compressor, layer_idx)

        if hasattr(compressor, "reconstruction_error"):
            errors = compressor.reconstruction_error(key, value, layer=layer_idx)
            key_rmses.append(errors["key_rmse"])
            value_rmses.append(errors["value_rmse"])
        else:
            key_rmses.append((key_ref.float() - k_hat.float()).pow(2).mean().sqrt().item())
            value_rmses.append((value_ref.float() - v_hat.float()).pow(2).mean().sqrt().item())

        key_rel_errors.append(_relative_error(key_ref, k_hat))
        value_rel_errors.append(_relative_error(value_ref, v_hat))
        key_cosines.append(_cosine_similarity(key_ref, k_hat))
        value_cosines.append(_cosine_similarity(value_ref, v_hat))

    if not key_rmses:
        raise RuntimeError("No KV tensors found for reconstruction error.")

    n = len(key_rmses)
    return RepresentationMetrics(
        key_rmse=sum(key_rmses) / n,
        value_rmse=sum(value_rmses) / n,
        key_relative_error=sum(key_rel_errors) / n,
        value_relative_error=sum(value_rel_errors) / n,
        key_cosine_similarity=sum(key_cosines) / n,
        value_cosine_similarity=sum(value_cosines) / n,
    )


def _layer_roundtrip(
    key: torch.Tensor,
    value: torch.Tensor,
    compressor: KVCompressor,
    layer_idx: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return reconstructed K/V plus reference tensors aligned for comparison."""
    if getattr(compressor, "name", None) in {"rocketkv", "palu"}:
        compressed = compressor.compress(key, value, layer=layer_idx)
        k_hat, v_hat = compressor.decompress(compressed)
        payload = compressed.keys
        selected = getattr(payload, "selected_indices", None)
        if selected is not None and selected.numel() > 0:
            idx = selected.to(key.device)
            key_ref = key.index_select(2, idx)
            value_ref = value.index_select(2, idx)
        else:
            key_ref, value_ref = key, value
        return k_hat.to(key.device), v_hat.to(value.device), key_ref, value_ref

    k_hat = compressor.decompress_kv(
        compressor.compress_kv(key, layer=layer_idx, mode="key"),
        mode="key",
    ).to(key.device)
    v_hat = compressor.decompress_kv(
        compressor.compress_kv(value, layer=layer_idx, mode="value"),
        mode="value",
    ).to(value.device)
    return k_hat, v_hat, key, value
