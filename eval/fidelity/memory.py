"""FIDELITY / Memory: KV-cache storage accounting (paper-independent)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from compressors.base import KVCompressor
from framework.kv_cache import apply_compressor, compressed_size_bytes, get_cache_size_bytes
from framework.model import ModelLayer
from framework.model_capabilities import resolve_model_capabilities
from framework.state_interface import (
    attention_kv_bytes,
    count_visible_state_elements,
    recurrent_state_bytes,
    visible_state_bytes,
)
from framework.storage_accounting import effective_bits_per_element


@dataclass
class MemoryMetrics:
    context_length: int
    num_kv_elements: int
    uncompressed_bytes: int
    compressed_bytes: int
    shared_metadata_bytes: int
    compression_ratio: float
    effective_bits_per_kv_element: float
    process_memory_mb: float
    attention_kv_bytes: int | None = None
    recurrent_state_bytes: int | None = None
    compressed_kv_bytes: int | None = None
    kv_compression_ratio: float | None = None
    total_visible_state_bytes: int | None = None


def process_memory_mb() -> float:
    import psutil

    return psutil.Process().memory_info().rss / (1024 * 1024)


def kv_cache_bytes(
    num_layers: int,
    seq_len: int,
    num_kv_heads: int,
    head_dim: int,
    batch_size: int = 1,
    bytes_per_element: int = 2,
    *,
    value_head_dim: int | None = None,
) -> int:
    """Analytical KV-cache size.

    Standard (symmetric K/V):
        B × L × T × H_KV × (D_k + D_v) × b, with D_v = D_k when omitted.

    Examples:
        OLMo2/Qwen3/Gemma3 — pass one ``head_dim`` (D_k = D_v).
        TinyDeepSeek — pass ``head_dim=64, value_head_dim=32``.
    """
    value_dim = head_dim if value_head_dim is None else value_head_dim
    elements = batch_size * num_layers * seq_len * num_kv_heads * (head_dim + value_dim)
    return elements * bytes_per_element


@torch.no_grad()
def evaluate_memory(
    model_layer: ModelLayer,
    input_ids: torch.Tensor,
    compressor: KVCompressor,
) -> MemoryMetrics:
    outputs = model_layer.forward_with_cache(input_ids)
    past_key_values = outputs.past_key_values
    if past_key_values is None:
        raise RuntimeError("Model did not return past_key_values.")
    return evaluate_memory_from_cache(model_layer, input_ids, compressor, past_key_values)


@torch.no_grad()
def evaluate_memory_from_cache(
    model_layer: ModelLayer,
    input_ids: torch.Tensor,
    compressor: KVCompressor,
    past_key_values,
) -> MemoryMetrics:
    caps = resolve_model_capabilities(model_layer.config)
    attn_bytes = attention_kv_bytes(past_key_values)
    recurrent_bytes = recurrent_state_bytes(past_key_values)
    visible_bytes = visible_state_bytes(past_key_values)
    num_elements = count_visible_state_elements(past_key_values)

    compressed_layers = apply_compressor(past_key_values, compressor)
    payload_bytes = compressed_size_bytes(compressed_layers, compressor)
    shared_metadata_bytes = compressor.shared_storage_bytes()
    compressed_kv_bytes = payload_bytes + shared_metadata_bytes
    compressed_total_bytes = compressed_kv_bytes + recurrent_bytes
    ratio = visible_bytes / compressed_total_bytes if compressed_total_bytes > 0 else 1.0
    kv_ratio = attn_bytes / compressed_kv_bytes if compressed_kv_bytes > 0 else 1.0
    effective_bits = effective_bits_per_element(compressed_total_bytes * 8, num_elements)

    return MemoryMetrics(
        context_length=input_ids.size(1),
        num_kv_elements=num_elements,
        uncompressed_bytes=visible_bytes,
        compressed_bytes=compressed_total_bytes,
        shared_metadata_bytes=shared_metadata_bytes,
        compression_ratio=ratio,
        effective_bits_per_kv_element=effective_bits,
        process_memory_mb=process_memory_mb(),
        attention_kv_bytes=attn_bytes,
        recurrent_state_bytes=recurrent_bytes if caps.has_recurrent_state else None,
        compressed_kv_bytes=compressed_kv_bytes,
        kv_compression_ratio=kv_ratio if caps.has_recurrent_state else None,
        total_visible_state_bytes=visible_bytes,
    )
