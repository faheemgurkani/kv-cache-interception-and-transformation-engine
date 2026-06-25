"""Utilities for reading and manipulating past_key_values."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from compressors.base import CompressedKV, KVCompressor


def iter_layer_kv(past_key_values):
    if hasattr(past_key_values, "layers"):
        for layer in past_key_values.layers:
            yield layer.keys, layer.values
        return
    if hasattr(past_key_values, "key_cache"):
        for key, value in zip(past_key_values.key_cache, past_key_values.value_cache, strict=True):
            yield key, value
        return
    for layer in past_key_values:
        yield layer[0], layer[1]


def extract_layer_kv(past_key_values, layer_idx: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(past_key_values, "layers"):
        layer = past_key_values.layers[layer_idx]
        return layer.keys, layer.values
    if hasattr(past_key_values, "key_cache"):
        return past_key_values.key_cache[layer_idx], past_key_values.value_cache[layer_idx]
    key, value = past_key_values[layer_idx]
    return key, value


def get_cache_size_bytes(past_key_values) -> int:
    total = 0
    for key, value in iter_layer_kv(past_key_values):
        total += key.numel() * key.element_size()
        total += value.numel() * value.element_size()
    return total


def count_kv_elements(past_key_values) -> int:
    """Total scalar count across all K and V tensors."""
    total = 0
    for key, value in iter_layer_kv(past_key_values):
        total += key.numel() + value.numel()
    return total


def apply_compressor(
    past_key_values,
    compressor: KVCompressor,
) -> list[CompressedKV]:
    compressed_layers: list[CompressedKV] = []
    for layer_idx, (key, value) in enumerate(iter_layer_kv(past_key_values)):
        compressed_layers.append(compressor.compress(key, value, layer=layer_idx))
    return compressed_layers


def compressed_size_bytes(compressed_layers: list[CompressedKV]) -> int:
    return sum(item.nbytes for item in compressed_layers)


def decompress_cache(
    compressed_layers: list[CompressedKV],
    compressor: KVCompressor,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [compressor.decompress(item) for item in compressed_layers]


def compress_past_key_values(past_key_values, compressor: KVCompressor) -> list[CompressedKV]:
    return apply_compressor(past_key_values, compressor)


def decompress_to_legacy_cache(
    compressed_layers: list[CompressedKV],
    compressor: KVCompressor,
    model_config,
    device: torch.device | None = None,
):
    """Rebuild past_key_values from compressed layers for the next forward pass."""
    layer_pairs = []
    for item in compressed_layers:
        key, value = compressor.decompress(item)
        if device is not None:
            key = key.to(device)
            value = value.to(device)
        layer_pairs.append((key, value))
    legacy = tuple(layer_pairs)
    try:
        from transformers.cache_utils import DynamicCache

        return DynamicCache(ddp_cache_data=legacy, config=model_config)
    except (ImportError, TypeError):
        return legacy
