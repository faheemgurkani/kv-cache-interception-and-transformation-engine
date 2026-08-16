"""SYSTEM / GPU utilization: best-effort NVML sampling during compressed-KV generation.

Optional: requires `pynvml` and a real NVIDIA GPU (Modal/CUDA hosts). Returns None on
MPS/CPU or when pynvml isn't installed, rather than raising, since this metric is
explicitly "if possible" — the rest of SYSTEM must still run without it.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import torch

from compressors.base import KVCompressor
from framework.model import ModelLayer


@dataclass
class GPUUtilizationMetrics:
    available: bool
    samples: int
    mean_utilization_pct: float | None
    max_utilization_pct: float | None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@torch.no_grad()
def evaluate_gpu_utilization(
    model_layer: ModelLayer,
    input_ids: torch.Tensor,
    compressor: KVCompressor,
    num_new_tokens: int = 32,
    sample_interval_s: float = 0.05,
) -> GPUUtilizationMetrics:
    try:
        import pynvml
    except ImportError:
        return GPUUtilizationMetrics(available=False, samples=0, mean_utilization_pct=None, max_utilization_pct=None)

    if not torch.cuda.is_available():
        return GPUUtilizationMetrics(available=False, samples=0, mean_utilization_pct=None, max_utilization_pct=None)

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(model_layer.device.index or 0)

    samples: list[float] = []
    stop = threading.Event()

    def _sample() -> None:
        while not stop.is_set():
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            samples.append(float(util.gpu))
            time.sleep(sample_interval_s)

    if hasattr(compressor, "reset_state"):
        compressor.reset_state()
    engine = model_layer.make_kv_engine(compressor)

    sampler = threading.Thread(target=_sample, daemon=True)
    sampler.start()
    try:
        engine.generate(input_ids, max_new_tokens=num_new_tokens)
    finally:
        stop.set()
        sampler.join(timeout=1.0)
        pynvml.nvmlShutdown()

    if not samples:
        return GPUUtilizationMetrics(available=True, samples=0, mean_utilization_pct=None, max_utilization_pct=None)

    return GPUUtilizationMetrics(
        available=True,
        samples=len(samples),
        mean_utilization_pct=sum(samples) / len(samples),
        max_utilization_pct=max(samples),
    )
