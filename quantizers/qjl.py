"""QJL random-projection residual encoding for TurboQuant."""

from __future__ import annotations

import math

import torch


def projection_matrix(dim: int, seed: int = 42, device: torch.device | None = None) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + dim)
    s = torch.randn(dim, dim, generator=gen)
    return s.to(device=device) if device else s


def qjl_encode(residual: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
    """b = sign(S @ r) for vectors in the last dimension."""
    z = torch.einsum("ij,...j->...i", projection, residual)
    return torch.sign(z).to(torch.int8)


def qjl_decode(
    bits: torch.Tensor,
    projection: torch.Tensor,
    norm_r: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct residual: r_hat = sqrt(pi/2) / d * S^T b * ||r||."""
    device = bits.device
    projection = projection.to(device)
    b = bits.to(projection.dtype)
    r_hat = torch.einsum("ji,...i->...j", projection, b)
    scale = math.sqrt(math.pi / 2.0) / projection.shape[0]
    r_hat = r_hat * scale
    return r_hat * norm_r.to(device)
