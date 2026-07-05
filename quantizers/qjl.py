"""QJL random-projection sign encoding (TurboQuant residual + standalone keys)."""

from __future__ import annotations

import math

import torch


def projection_matrix(
    head_dim: int,
    proj_dim: int | None = None,
    seed: int = 42,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Fixed Gaussian projection S ∈ R^{m×d} with deterministic seed per head_dim."""
    m = proj_dim or head_dim
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + head_dim)
    s = torch.randn(m, head_dim, generator=gen)
    return s.to(device=device) if device else s


def qjl_encode(residual: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
    """b = sign(S @ r) — strict ±1 bits, not round/threshold."""
    z = torch.einsum("ij,...j->...i", projection, residual)
    return torch.where(z >= 0, 1, -1).to(torch.int8)


def qjl_decode(
    bits: torch.Tensor,
    projection: torch.Tensor,
    norm_r: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct vector: r_hat = sqrt(pi/2) / m * S^T b * ||r||."""
    device = bits.device
    projection = projection.to(device)
    b = bits.to(projection.dtype)
    r_hat = torch.einsum("ji,...i->...j", projection, b)
    scale = math.sqrt(math.pi / 2.0) / projection.shape[0]
    return r_hat * scale * norm_r.to(device)
