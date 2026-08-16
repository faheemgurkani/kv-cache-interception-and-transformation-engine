"""SYSTEM / Kernel cost: compress/decompress overhead vs. model forward time.

Wall-clock only (no CUDA event profiler dependency, so this also works on MPS/CPU).
Compress/decompress time is measured by temporarily wrapping the compressor's
`compress_kv` / `decompress_kv` — the two methods every KVCompressor must implement
— so this works uniformly across plug-ins without per-compressor instrumentation.
`attention_execution_time_ms` is a proxy: total step time minus measured
compress/decompress time, i.e. "everything else in the forward pass" (attention +
MLP + norms) — the compression engine cannot isolate attention alone without a
CUDA kernel trace.

Caveat: RocketKV's online path (framework/rocketkv_online.py) calls
`compress_layer_from_kv` directly instead of `compress_kv`, so its per-step
compression cost is not captured here and will read as pure "attention" time.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from compressors.base import KVCompressor
from framework.model import ModelLayer


@dataclass
class KernelCostMetrics:
    context_length: int
    generated_tokens: int
    total_step_time_ms: float
    compress_time_ms: float
    decompress_time_ms: float
    compress_decompress_time_ms: float
    attention_execution_time_ms: float
    compress_decompress_overhead_frac: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@contextmanager
def _timed_methods(compressor: KVCompressor):
    """Monkeypatch compress_kv/decompress_kv to accumulate wall-clock time, then restore."""
    totals = {"compress": 0.0, "decompress": 0.0}
    orig_compress = compressor.compress_kv
    orig_decompress = compressor.decompress_kv

    def timed_compress(*args, **kwargs):
        start = time.perf_counter()
        result = orig_compress(*args, **kwargs)
        totals["compress"] += time.perf_counter() - start
        return result

    def timed_decompress(*args, **kwargs):
        start = time.perf_counter()
        result = orig_decompress(*args, **kwargs)
        totals["decompress"] += time.perf_counter() - start
        return result

    compressor.compress_kv = timed_compress  # type: ignore[method-assign]
    compressor.decompress_kv = timed_decompress  # type: ignore[method-assign]
    try:
        yield totals
    finally:
        compressor.compress_kv = orig_compress  # type: ignore[method-assign]
        compressor.decompress_kv = orig_decompress  # type: ignore[method-assign]


@torch.no_grad()
def evaluate_kernel_cost(
    model_layer: ModelLayer,
    input_ids: torch.Tensor,
    compressor: KVCompressor,
    num_new_tokens: int = 32,
) -> KernelCostMetrics:
    if hasattr(compressor, "reset_state"):
        compressor.reset_state()
    engine = model_layer.make_kv_engine(compressor)

    generated = input_ids
    cache = None

    with _timed_methods(compressor) as totals:
        start = time.perf_counter()
        for _ in range(num_new_tokens):
            step_input = generated if cache is None else generated[:, -1:]
            logits, cache = engine.step(step_input, compressed_cache=cache)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
        total_elapsed = time.perf_counter() - start

    compress_ms = totals["compress"] * 1000
    decompress_ms = totals["decompress"] * 1000
    total_ms = total_elapsed * 1000
    compress_decompress_ms = compress_ms + decompress_ms

    return KernelCostMetrics(
        context_length=input_ids.size(1),
        generated_tokens=num_new_tokens,
        total_step_time_ms=total_ms,
        compress_time_ms=compress_ms,
        decompress_time_ms=decompress_ms,
        compress_decompress_time_ms=compress_decompress_ms,
        attention_execution_time_ms=max(total_ms - compress_decompress_ms, 0.0),
        compress_decompress_overhead_frac=(compress_decompress_ms / total_ms) if total_ms > 0 else 0.0,
    )
