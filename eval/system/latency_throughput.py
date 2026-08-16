"""SYSTEM / Latency & throughput: TTFT, inter-token latency, and tokens/sec.

Timed as a manual step loop (rather than calling KVCacheEngine.generate as a black
box) so TTFT and per-token latency can be split out:
- TTFT (time to first token): the first engine.step() call — prefill forward pass
  plus compressing the full prompt's KV in one shot.
- ITL (inter-token latency): each subsequent engine.step() call — one decode
  forward pass plus incremental compress/decompress of the KV cache.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import torch

from compressors.base import KVCompressor
from framework.model import ModelLayer


@dataclass
class ThroughputMetrics:
    context_length: int
    generated_tokens: int
    elapsed_seconds: float
    tokens_per_second: float
    latency_ms_per_token: float
    ttft_ms: float | None = None
    itl_ms_mean: float | None = None
    itl_ms_p50: float | None = None
    itl_ms_p99: float | None = None
    decode_latency_ms: float | None = None
    end_to_end_latency_ms: float | None = None
    online_compressed_kv: bool = True


def _percentile(samples: list[float], pct: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    idx = min(int(len(ordered) * pct), len(ordered) - 1)
    return ordered[idx]


@torch.no_grad()
def evaluate_throughput(
    model_layer: ModelLayer,
    input_ids: torch.Tensor,
    compressor: KVCompressor,
    num_new_tokens: int = 128,
) -> ThroughputMetrics:
    """Measure TTFT/ITL/throughput using KVCacheEngine (compress/decompress each step)."""
    if hasattr(compressor, "reset_state"):
        compressor.reset_state()
    engine = model_layer.make_kv_engine(compressor)

    generated = input_ids
    cache = None
    itl_samples_ms: list[float] = []
    ttft_ms: float | None = None

    start = time.perf_counter()
    for step_idx in range(num_new_tokens):
        step_input = generated if cache is None else generated[:, -1:]
        step_start = time.perf_counter()
        logits, cache = engine.step(step_input, compressed_cache=cache)
        step_ms = (time.perf_counter() - step_start) * 1000
        if step_idx == 0:
            ttft_ms = step_ms
        else:
            itl_samples_ms.append(step_ms)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)
    elapsed = time.perf_counter() - start

    tokens_per_second = num_new_tokens / elapsed if elapsed > 0 else 0.0
    latency_ms = (elapsed / num_new_tokens) * 1000 if num_new_tokens > 0 else 0.0
    itl_mean = statistics.fmean(itl_samples_ms) if itl_samples_ms else None

    return ThroughputMetrics(
        context_length=input_ids.size(1),
        generated_tokens=num_new_tokens,
        elapsed_seconds=elapsed,
        tokens_per_second=tokens_per_second,
        latency_ms_per_token=latency_ms,
        ttft_ms=ttft_ms,
        itl_ms_mean=itl_mean,
        itl_ms_p50=_percentile(itl_samples_ms, 0.50) if itl_samples_ms else None,
        itl_ms_p99=_percentile(itl_samples_ms, 0.99) if itl_samples_ms else None,
        decode_latency_ms=itl_mean,
        end_to_end_latency_ms=elapsed * 1000,
        online_compressed_kv=True,
    )


@torch.no_grad()
def evaluate_throughput_baseline(
    model_layer: ModelLayer,
    input_ids: torch.Tensor,
    num_new_tokens: int = 128,
) -> ThroughputMetrics:
    """Uncompressed HF generate baseline (reference only; no per-step split available
    since model.generate() is a single opaque call)."""
    start = time.perf_counter()
    model_layer.generate(input_ids, max_new_tokens=num_new_tokens)
    elapsed = time.perf_counter() - start

    tokens_per_second = num_new_tokens / elapsed if elapsed > 0 else 0.0
    latency_ms = (elapsed / num_new_tokens) * 1000 if num_new_tokens > 0 else 0.0

    return ThroughputMetrics(
        context_length=input_ids.size(1),
        generated_tokens=num_new_tokens,
        elapsed_seconds=elapsed,
        tokens_per_second=tokens_per_second,
        latency_ms_per_token=latency_ms,
        end_to_end_latency_ms=elapsed * 1000,
        online_compressed_kv=False,
    )


def measure_tokens_per_second(generate_fn, num_tokens: int = 128) -> float:
    start = time.perf_counter()
    generate_fn(num_tokens)
    elapsed = time.perf_counter() - start
    return num_tokens / elapsed if elapsed > 0 else 0.0
