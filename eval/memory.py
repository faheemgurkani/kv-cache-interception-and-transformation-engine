"""KV-cache memory evaluation (paper-independent)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from compressors.base import KVCompressor
from framework.storage_accounting import effective_bits_per_element
from framework.kv_cache import (
    apply_compressor,
    compressed_size_bytes,
    count_kv_elements,
    get_cache_size_bytes,
)
from framework.model import ModelLayer


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
) -> int:
    """Analytical KV-cache size: B × 2 × layers × tokens × kv_heads × head_dim × bytes."""
    elements = batch_size * 2 * num_layers * seq_len * num_kv_heads * head_dim
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

    uncompressed_bytes = get_cache_size_bytes(past_key_values)
    num_kv_elements = count_kv_elements(past_key_values)
    compressed_layers = apply_compressor(past_key_values, compressor)
    payload_bytes = compressed_size_bytes(compressed_layers, compressor)
    shared_metadata_bytes = compressor.shared_storage_bytes()
    compressed_bytes = payload_bytes + shared_metadata_bytes
    ratio = uncompressed_bytes / compressed_bytes if compressed_bytes > 0 else 1.0
    effective_bits = effective_bits_per_element(compressed_bytes * 8, num_kv_elements)

    return MemoryMetrics(
        context_length=input_ids.size(1),
        num_kv_elements=num_kv_elements,
        uncompressed_bytes=uncompressed_bytes,
        compressed_bytes=compressed_bytes,
        shared_metadata_bytes=shared_metadata_bytes,
        compression_ratio=ratio,
        effective_bits_per_kv_element=effective_bits,
        process_memory_mb=process_memory_mb(),
    )
