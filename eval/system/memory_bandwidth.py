"""SYSTEM / Memory bandwidth: effective KV bytes moved per second during decode.

Estimated analytically rather than via a profiler (which would require CUDA and a
per-backend kernel trace): every decode step in KVCacheEngine.step decompresses the
existing compressed cache to a legacy K/V cache (a read of `cache.nbytes`) and then
recompresses the appended token (a write). Summing `cache.nbytes` across steps and
dividing by wall-clock time gives an effective GB/s figure comparable across
compressors — smaller caches move fewer bytes per step at the cost of more (or fewer)
total steps, which is exactly the tradeoff this metric is meant to expose.

Accounting (audit S3): per step, *read* = the cache stored before the step (what gets
decompressed); *write* = the newly appended payload bytes (the whole prompt cache on the
prefill step). Counted on compressed, stored bytes only: the forward pass itself also
streams the decompressed K/V and the weights, which this figure does not include. On small
models the wall-clock is Python/launch bound, so the effective GB/s is far below device
peak and should be read as a *relative* number across compressors, not a bandwidth
utilisation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from compressors.base import KVCompressor
from framework.model import ModelLayer


@dataclass
class MemoryBandwidthMetrics:
    context_length: int
    generated_tokens: int
    elapsed_seconds: float
    total_kv_bytes_moved: int
    effective_bandwidth_gbps: float
    kv_bytes_read: int = 0
    kv_bytes_written: int = 0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@torch.no_grad()
def evaluate_memory_bandwidth(
    model_layer: ModelLayer,
    input_ids: torch.Tensor,
    compressor: KVCompressor,
    num_new_tokens: int = 64,
) -> MemoryBandwidthMetrics:
    if hasattr(compressor, "reset_state"):
        compressor.reset_state()
    engine = model_layer.make_kv_engine(compressor)
    device = model_layer.device

    cache = None
    generated = input_ids
    bytes_read = 0
    bytes_written = 0
    prev_nbytes = 0

    start = time.perf_counter()
    for _ in range(num_new_tokens):
        step_input = generated if cache is None else generated[:, -1:]
        logits, cache = engine.step(step_input, compressed_cache=cache)
        # Read the prior stored cache; write only what this step appended.
        bytes_read += prev_nbytes
        bytes_written += max(cache.nbytes - prev_nbytes, 0)
        prev_nbytes = cache.nbytes
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)
    elapsed = time.perf_counter() - start

    total_bytes_moved = bytes_read + bytes_written
    bandwidth_gbps = (total_bytes_moved / elapsed / 1e9) if elapsed > 0 else 0.0

    return MemoryBandwidthMetrics(
        context_length=input_ids.size(1),
        generated_tokens=num_new_tokens,
        elapsed_seconds=elapsed,
        total_kv_bytes_moved=total_bytes_moved,
        effective_bandwidth_gbps=bandwidth_gbps,
        kv_bytes_read=bytes_read,
        kv_bytes_written=bytes_written,
    )
