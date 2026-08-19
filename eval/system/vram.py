"""SYSTEM / Peak memory: device allocator peak (CUDA/MPS) or process RSS (CPU)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from compressors.base import KVCompressor
from eval.system.device_metrics import PeakMemoryTracker
from framework.model import ModelLayer


@dataclass
class PeakMemoryMetrics:
    context_length: int
    generated_tokens: int
    peak_allocated_mb: float | None
    peak_reserved_mb: float | None
    peak_process_rss_mb: float | None
    memory_backend: str
    cuda_available: bool

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@torch.no_grad()
def evaluate_peak_vram(
    model_layer: ModelLayer,
    input_ids: torch.Tensor,
    compressor: KVCompressor,
    num_new_tokens: int = 128,
) -> PeakMemoryMetrics:
    """Peak memory during compressed-KV generate().

    CUDA: ``max_memory_allocated`` / ``max_memory_reserved``.
    MPS: ``current_allocated_memory`` / ``driver_allocated_memory`` (polled peak).
    CPU: process RSS peak (``psutil``).
    """
    if hasattr(compressor, "reset_state"):
        compressor.reset_state()
    engine = model_layer.make_kv_engine(compressor)
    tracker = PeakMemoryTracker(model_layer.device)
    tracker.reset()

    generated = input_ids
    cache = None
    for _ in range(num_new_tokens):
        tracker.sample()
        step_input = generated if cache is None else generated[:, -1:]
        logits, cache = engine.step(step_input, compressed_cache=cache)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)
        tracker.sample()

    snap = tracker.snapshot()
    return PeakMemoryMetrics(
        context_length=input_ids.size(1),
        generated_tokens=num_new_tokens,
        peak_allocated_mb=snap.peak_allocated_mb,
        peak_reserved_mb=snap.peak_reserved_mb,
        peak_process_rss_mb=snap.peak_process_rss_mb,
        memory_backend=snap.memory_backend,
        cuda_available=torch.cuda.is_available(),
    )
