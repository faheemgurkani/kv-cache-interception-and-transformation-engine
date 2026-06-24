"""KV-cache memory footprint evaluation."""

from __future__ import annotations

import psutil


def kv_cache_bytes(num_layers: int, seq_len: int, num_kv_heads: int, head_dim: int, bitwidth: int = 16) -> int:
    """Estimate KV-cache size in bytes."""
    elements = 2 * num_layers * seq_len * num_kv_heads * head_dim
    return elements * (bitwidth / 8)


def process_memory_mb() -> float:
    """Current process RSS in megabytes."""
    return psutil.Process().memory_info().rss / (1024 * 1024)
