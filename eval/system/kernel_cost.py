"""SYSTEM / Kernel cost: compress/decompress overhead vs. model forward time.

Wall-clock only (no CUDA event profiler dependency, so this also works on MPS/CPU).
Compress/decompress time is measured by temporarily wrapping the compressor's
``compress_kv`` / ``decompress_kv``, optional ``compress_layer_from_kv`` (RocketKV),
and layer ``decompress`` — restored after measurement.
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
    """Monkeypatch compressor hooks to accumulate wall-clock time, then restore."""
    totals = {"compress": 0.0, "decompress": 0.0}
    originals: dict[str, object] = {}
    in_layer_decompress = {"active": False}

    def _wrap(name: str, fn, bucket: str):
        originals[name] = fn

        def timed(*args, **kwargs):
            if bucket == "decompress" and in_layer_decompress["active"]:
                return fn(*args, **kwargs)
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            totals[bucket] += time.perf_counter() - start
            return result

        return timed

    compressor.compress_kv = _wrap("compress_kv", compressor.compress_kv, "compress")  # type: ignore[method-assign]

    orig_decompress_kv = compressor.decompress_kv

    def timed_decompress_kv(*args, **kwargs):
        if in_layer_decompress["active"]:
            return orig_decompress_kv(*args, **kwargs)
        start = time.perf_counter()
        result = orig_decompress_kv(*args, **kwargs)
        totals["decompress"] += time.perf_counter() - start
        return result

    compressor.decompress_kv = timed_decompress_kv  # type: ignore[method-assign]

    if hasattr(compressor, "compress_layer_from_kv"):
        originals["compress_layer_from_kv"] = compressor.compress_layer_from_kv  # type: ignore[attr-defined]
        compressor.compress_layer_from_kv = _wrap(  # type: ignore[method-assign]
            "compress_layer_from_kv",
            compressor.compress_layer_from_kv,  # type: ignore[attr-defined]
            "compress",
        )

    if hasattr(compressor, "decompress"):
        originals["decompress"] = compressor.decompress
        orig_decompress = compressor.decompress

        def timed_decompress_bound(compressed):
            in_layer_decompress["active"] = True
            start = time.perf_counter()
            try:
                return orig_decompress(compressed)
            finally:
                totals["decompress"] += time.perf_counter() - start
                in_layer_decompress["active"] = False

        compressor.decompress = timed_decompress_bound  # type: ignore[method-assign]

    try:
        yield totals
    finally:
        compressor.compress_kv = originals["compress_kv"]  # type: ignore[method-assign]
        compressor.decompress_kv = orig_decompress_kv  # type: ignore[method-assign]
        if "compress_layer_from_kv" in originals:
            compressor.compress_layer_from_kv = originals["compress_layer_from_kv"]  # type: ignore[method-assign]
        if "decompress" in originals:
            compressor.decompress = originals["decompress"]  # type: ignore[method-assign]


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
