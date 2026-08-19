"""Lloyd-Max vector quantization for TurboQuant."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from sklearn.cluster import KMeans

CALIBRATION_DATASET = "gaussian_synthetic"
CALIBRATION_TOKENS = 1_000_000


@dataclass(frozen=True)
class CentroidCalibrationStats:
    num_bits: int
    seed: int
    dataset: str
    calibration_tokens: int
    calibration_time_ms: float
    calibration_memory_bytes: int


_last_centroid_calibration: CentroidCalibrationStats | None = None


def last_centroid_calibration() -> CentroidCalibrationStats | None:
    """Return stats from the most recent Lloyd-Max centroid build (cache miss only)."""
    return _last_centroid_calibration


@lru_cache(maxsize=8)
def build_centroids(num_bits: int, seed: int = 42) -> torch.Tensor:
    """Fit Lloyd-Max centroids on Gaussian samples (offline-style)."""
    global _last_centroid_calibration
    k = 2**num_bits
    rng = np.random.default_rng(seed)
    samples = rng.standard_normal(CALIBRATION_TOKENS).reshape(-1, 1)

    import psutil

    mem_before = psutil.Process().memory_info().rss
    start = time.perf_counter()
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=seed)
    kmeans.fit(samples)
    elapsed_ms = (time.perf_counter() - start) * 1000
    mem_after = psutil.Process().memory_info().rss

    centroids = np.sort(kmeans.cluster_centers_.flatten())
    _last_centroid_calibration = CentroidCalibrationStats(
        num_bits=num_bits,
        seed=seed,
        dataset=CALIBRATION_DATASET,
        calibration_tokens=CALIBRATION_TOKENS,
        calibration_time_ms=elapsed_ms,
        calibration_memory_bytes=max(mem_after - mem_before, 0),
    )
    return torch.tensor(centroids, dtype=torch.float32)


def normalize_features(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    d = x.shape[dim]
    return x / math.sqrt(d)


def compute_gamma(y: torch.Tensor, centroids: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Per-vector amax scale so rotated coefficients fit the Lloyd-Max codebook."""
    c_max = centroids.abs().max().item()
    amax = y.abs().amax(dim=dim, keepdim=True).clamp(min=1e-8)
    return amax / c_max


def quantize(x: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """Nearest-centroid Lloyd-Max quantization; returns int32 indices."""
    c = centroids.to(device=x.device, dtype=x.dtype)
    diffs = (x.unsqueeze(-1) - c).abs()
    return diffs.argmin(dim=-1).to(torch.int32)


def dequantize(indices: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    c = centroids.to(device=indices.device, dtype=torch.float32)
    return c[indices.long()]


def correct_rotated_norm(y: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Renormalize reconstructed rotated coefficients to unit L2 norm."""
    norm = y.norm(dim=dim, keepdim=True).clamp(min=1e-8)
    return y / norm
