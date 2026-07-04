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


def is_incremental_compressed(compressed: CompressedKV) -> bool:
    return isinstance(compressed.keys, list)


def incremental_seq_length(compressed_layers: list[CompressedKV]) -> int:
    if not compressed_layers:
        return 0
    first = compressed_layers[0]
    if is_incremental_compressed(first):
        return len(first.keys)  # type: ignore[arg-type]
    return first.original_shape[2]


def payload_list_bytes(payloads: list[object], compressor: KVCompressor) -> int:
    return sum(compressor._payload_bytes(item) for item in payloads)


def decompress_compressed_layer(
    compressed: CompressedKV,
    compressor: KVCompressor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decompress one layer; supports batch (single payload) or incremental (payload lists)."""
    if is_incremental_compressed(compressed):
        key_parts = [compressor.decompress_kv(item, mode="key") for item in compressed.keys]  # type: ignore[union-attr]
        value_parts = [compressor.decompress_kv(item, mode="value") for item in compressed.values]  # type: ignore[union-attr]
        return torch.cat(key_parts, dim=2), torch.cat(value_parts, dim=2)
    return compressor.decompress(compressed)


def apply_compressor(
    past_key_values,
    compressor: KVCompressor,
) -> list[CompressedKV]:
    compressed_layers: list[CompressedKV] = []
    for layer_idx, (key, value) in enumerate(iter_layer_kv(past_key_values)):
        compressed_layers.append(compressor.compress(key, value, layer=layer_idx))
    return compressed_layers


def compressed_layer_bytes(compressed: CompressedKV, compressor: KVCompressor) -> int:
    if is_incremental_compressed(compressed):
        key_bytes = payload_list_bytes(compressed.keys, compressor)  # type: ignore[arg-type]
        value_bytes = payload_list_bytes(compressed.values, compressor)  # type: ignore[arg-type]
        return key_bytes + value_bytes
    return compressed.nbytes


def compressed_size_bytes(compressed_layers: list[CompressedKV], compressor: KVCompressor | None = None) -> int:
    if compressor is None:
        return sum(item.nbytes for item in compressed_layers)
    return sum(compressed_layer_bytes(item, compressor) for item in compressed_layers)


def decompress_cache(
    compressed_layers: list[CompressedKV],
    compressor: KVCompressor,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [decompress_compressed_layer(item, compressor) for item in compressed_layers]


def compress_past_key_values(past_key_values, compressor: KVCompressor) -> list[CompressedKV]:
    return apply_compressor(past_key_values, compressor)


def compress_token_slice(
    key: torch.Tensor,
    value: torch.Tensor,
    token_idx: int,
    layer_idx: int,
    compressor: KVCompressor,
) -> tuple[object, object]:
    """Compress a single token position along the sequence dimension."""
    key_slice = key[:, :, token_idx : token_idx + 1, :]
    value_slice = value[:, :, token_idx : token_idx + 1, :]
    return (
        compressor.compress_kv(key_slice, layer=layer_idx, mode="key"),
        compressor.compress_kv(value_slice, layer=layer_idx, mode="value"),
    )


def build_incremental_layer(
    key: torch.Tensor,
    value: torch.Tensor,
    key_payloads: list[object],
    value_payloads: list[object],
    layer_idx: int,
    compressor: KVCompressor,
) -> CompressedKV:
    from compressors.base import CompressedKV

    nbytes = payload_list_bytes(key_payloads, compressor) + payload_list_bytes(value_payloads, compressor)
    return CompressedKV(
        keys=key_payloads,
        values=value_payloads,
        original_shape=tuple(key.shape),
        nbytes=nbytes,
        bitwidth=getattr(compressor, "bitwidth", None),
        layer=layer_idx,
    )


def trim_compressed_cache(
    compressed_cache,
    drop_tokens: int,
    compressor: KVCompressor,
):
    """Drop oldest token payloads and refresh byte counts."""
    from compressors.base import CompressedKV
    from framework.kv_engine import CompressedCache

    if drop_tokens <= 0:
        return compressed_cache

    trimmed_layers: list[CompressedKV] = []
    for layer in compressed_cache.layers:
        if not is_incremental_compressed(layer):
            raise ValueError("trim_compressed_cache requires incremental payload lists.")
        key_payloads = list(layer.keys)[drop_tokens:]  # type: ignore[index]
        value_payloads = list(layer.values)[drop_tokens:]  # type: ignore[index]
        if not key_payloads:
            raise ValueError("Cannot trim entire compressed cache.")
        nbytes = payload_list_bytes(key_payloads, compressor) + payload_list_bytes(value_payloads, compressor)
        trimmed_layers.append(
            CompressedKV(
                keys=key_payloads,
                values=value_payloads,
                original_shape=layer.original_shape,
                nbytes=nbytes,
                bitwidth=layer.bitwidth,
                layer=layer.layer,
            )
        )
    return CompressedCache(layers=trimmed_layers)


def decompress_to_legacy_cache(
    compressed_layers: list[CompressedKV],
    compressor: KVCompressor,
    model_config,
    device: torch.device | None = None,
):
    """Rebuild past_key_values from compressed layers for the next forward pass."""
    layer_pairs = []
    for item in compressed_layers:
        key, value = decompress_compressed_layer(item, compressor)
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
