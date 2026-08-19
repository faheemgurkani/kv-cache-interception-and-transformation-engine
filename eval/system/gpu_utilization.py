"""SYSTEM / Utilization: NVML GPU util on CUDA; process CPU util on MPS/CPU."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from compressors.base import KVCompressor
from eval.system.device_metrics import UtilizationSampler
from framework.model import ModelLayer


@dataclass
class GPUUtilizationMetrics:
    available: bool
    utilization_backend: str
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
    if hasattr(compressor, "reset_state"):
        compressor.reset_state()
    engine = model_layer.make_kv_engine(compressor)

    sampler = UtilizationSampler(model_layer.device, sample_interval_s=sample_interval_s)
    sampler.start()
    try:
        engine.generate(input_ids, max_new_tokens=num_new_tokens)
    finally:
        sampler.stop()

    if not sampler.samples:
        return GPUUtilizationMetrics(
            available=sampler.available,
            utilization_backend=sampler.utilization_backend,
            samples=0,
            mean_utilization_pct=None,
            max_utilization_pct=None,
        )

    return GPUUtilizationMetrics(
        available=True,
        utilization_backend=sampler.utilization_backend,
        samples=len(sampler.samples),
        mean_utilization_pct=sampler.mean_pct(),
        max_utilization_pct=sampler.max_pct(),
    )
