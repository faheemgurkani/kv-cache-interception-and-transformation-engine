"""SYSTEM / Peak VRAM: peak CUDA allocator usage during compressed-KV generation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from compressors.base import KVCompressor
from framework.model import ModelLayer


@dataclass
class PeakMemoryMetrics:
    context_length: int
    generated_tokens: int
    peak_allocated_mb: float | None
    peak_reserved_mb: float | None
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
    """Peak allocated/reserved CUDA memory across a compressed-KV generate() call.

    Returns None for the peak-* fields off CUDA (MPS/CPU expose no equivalent
    allocator-peak API), but still reports the attempted config for the record.
    """
    cuda_available = torch.cuda.is_available()
    if hasattr(compressor, "reset_state"):
        compressor.reset_state()
    engine = model_layer.make_kv_engine(compressor)

    if cuda_available:
        torch.cuda.reset_peak_memory_stats(model_layer.device)

    engine.generate(input_ids, max_new_tokens=num_new_tokens)

    peak_allocated_mb = None
    peak_reserved_mb = None
    if cuda_available:
        peak_allocated_mb = torch.cuda.max_memory_allocated(model_layer.device) / (1024 * 1024)
        peak_reserved_mb = torch.cuda.max_memory_reserved(model_layer.device) / (1024 * 1024)

    return PeakMemoryMetrics(
        context_length=input_ids.size(1),
        generated_tokens=num_new_tokens,
        peak_allocated_mb=peak_allocated_mb,
        peak_reserved_mb=peak_reserved_mb,
        cuda_available=cuda_available,
    )
