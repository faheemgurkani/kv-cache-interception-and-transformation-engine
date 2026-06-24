"""Walsh-Hadamard transform for TurboQuant (with MPS/CPU fallback)."""

from __future__ import annotations

import math

import torch

try:
    from fast_hadamard_transform import hadamard_transform as _fht_cuda

    _HAS_FHT = True
except ImportError:
    _HAS_FHT = False


def next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def pad_to_power_of_two(x: torch.Tensor, dim: int = -1) -> tuple[torch.Tensor, int]:
    size = x.shape[dim]
    target = next_power_of_two(size)
    if target == size:
        return x, size
    pad_amount = target - size
    pad_dims = [0] * (2 * x.ndim)
    pad_idx = 2 * (x.ndim - 1 - (dim if dim >= 0 else x.ndim + dim))
    pad_dims[pad_idx] = pad_amount
    return torch.nn.functional.pad(x, pad_dims), size


def unpad(x: torch.Tensor, original_dim: int, dim: int = -1) -> torch.Tensor:
    if original_dim == x.shape[dim]:
        return x
    return x.narrow(dim, 0, original_dim)


def _hadamard_scipy(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    import scipy.linalg

    n = x.shape[dim]
    h = torch.tensor(scipy.linalg.hadamard(n), dtype=x.dtype, device=x.device) / math.sqrt(n)
    return torch.einsum("ij,...j->...i", h, x.movedim(dim, -1)).movedim(-1, dim)


def _apply_wht(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if _HAS_FHT and x.is_cuda:
        flat = x.movedim(dim, -1)
        out = _fht_cuda(flat.reshape(-1, flat.shape[-1]))
        return out.reshape(flat.shape).movedim(-1, dim)
    return _hadamard_scipy(x, dim=dim)


def hadamard_transform(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Orthonormal WHT: y = Hx with H^T H = I on the last (padded) dimension."""
    x_pad, _ = pad_to_power_of_two(x, dim=dim)
    return _apply_wht(x_pad, dim=dim)


def inverse_hadamard_transform(y: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Inverse WHT (self-adjoint for orthonormal Hadamard)."""
    return _apply_wht(y, dim=dim)
