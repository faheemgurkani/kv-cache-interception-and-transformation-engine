"""TurboQuant per-tensor compression pipeline (WHT → Lloyd-Max → QJL)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import torch

from framework.storage_accounting import (
    bits_to_bytes,
    float32_storage_bits,
    index_storage_bits,
    sign_storage_bits,
)
from quantizers.hadamard import (
    hadamard_transform,
    inverse_hadamard_transform,
    pad_to_power_of_two,
    unpad,
)
from quantizers.lloyd_max import build_centroids, dequantize, normalize_features, quantize
from quantizers.qjl import projection_matrix, qjl_decode, qjl_encode


class TurboQuantStage(str, Enum):
    WHT_ONLY = "wht_only"
    WHT_QUANT = "wht_quant"
    WHT_QUANT_RESIDUAL = "wht_quant_residual"
    FULL = "full"


# Fixed per-tensor metadata: dims, stage, bitwidth, shape (conservative packed estimate).
TURBOQUANT_METADATA_BYTES = 32


@dataclass
class TurboQuantTensorPayload:
    """Compressed payload for a single K or V tensor."""

    indices: torch.Tensor | None
    qjl_bits: torch.Tensor | None
    norm_r: torch.Tensor | None
    original_dim: int
    padded_dim: int
    original_shape: tuple[int, ...]
    original_dtype: torch.dtype
    stage: TurboQuantStage
    bitwidth: int
    wht_only: torch.Tensor | None = None

    def storage_bits(self) -> int:
        """True bits stored (bit-packed QJL signs, bitwidth-sized indices, metadata)."""
        bits = TURBOQUANT_METADATA_BYTES * 8
        if self.indices is not None:
            bits += index_storage_bits(self.indices.numel(), self.bitwidth)
        if self.qjl_bits is not None:
            bits += sign_storage_bits(self.qjl_bits.numel())
        if self.norm_r is not None:
            bits += float32_storage_bits(self.norm_r.numel())
        if self.wht_only is not None:
            bits += float32_storage_bits(self.wht_only.numel())
        return bits

    def storage_bytes(self) -> int:
        return bits_to_bytes(self.storage_bits())

    @property
    def nbytes(self) -> int:
        return self.storage_bytes()


class TurboQuantPipeline:
    """Mathematical TurboQuant pipeline operating on one KV tensor."""

    def __init__(
        self,
        bitwidth: int = 4,
        stage: TurboQuantStage = TurboQuantStage.FULL,
        seed: int = 42,
    ) -> None:
        self.bitwidth = bitwidth
        self.stage = stage
        self.seed = seed
        self.centroids = build_centroids(bitwidth, seed=seed)
        self._projections: dict[int, torch.Tensor] = {}

    def _get_projection(self, dim: int, device: torch.device) -> torch.Tensor:
        if dim not in self._projections:
            self._projections[dim] = projection_matrix(dim, seed=self.seed, device=device)
        return self._projections[dim]

    def compress_tensor(self, x: torch.Tensor) -> TurboQuantTensorPayload:
        original_shape = tuple(x.shape)
        original_dtype = x.dtype
        x = x.float()
        x_pad, original_dim = pad_to_power_of_two(x, dim=-1)
        padded_dim = x_pad.shape[-1]

        y = hadamard_transform(x_pad, dim=-1)
        y = normalize_features(y, dim=-1)

        if self.stage == TurboQuantStage.WHT_ONLY:
            return TurboQuantTensorPayload(
                indices=None,
                qjl_bits=None,
                norm_r=None,
                original_dim=original_dim,
                padded_dim=padded_dim,
                original_shape=original_shape,
                original_dtype=original_dtype,
                stage=self.stage,
                bitwidth=self.bitwidth,
                wht_only=y.detach().cpu(),
            )

        indices = quantize(y, self.centroids).to(torch.int8)
        y_mse = dequantize(indices, self.centroids)

        if self.stage == TurboQuantStage.WHT_QUANT:
            return TurboQuantTensorPayload(
                indices=indices.detach().cpu(),
                qjl_bits=None,
                norm_r=None,
                original_dim=original_dim,
                padded_dim=padded_dim,
                original_shape=original_shape,
                original_dtype=original_dtype,
                stage=self.stage,
                bitwidth=self.bitwidth,
            )

        residual = y - y_mse

        if self.stage == TurboQuantStage.WHT_QUANT_RESIDUAL:
            return TurboQuantTensorPayload(
                indices=indices.detach().cpu(),
                qjl_bits=None,
                norm_r=residual.norm(dim=-1, keepdim=True).detach().cpu(),
                original_dim=original_dim,
                padded_dim=padded_dim,
                original_shape=original_shape,
                original_dtype=original_dtype,
                stage=self.stage,
                bitwidth=self.bitwidth,
            )

        norm_r = residual.norm(dim=-1, keepdim=True)
        proj = self._get_projection(padded_dim, device=x.device)
        bits = qjl_encode(residual, proj)

        return TurboQuantTensorPayload(
            indices=indices.detach().cpu(),
            qjl_bits=bits.detach().cpu(),
            norm_r=norm_r.detach().cpu(),
            original_dim=original_dim,
            padded_dim=padded_dim,
            original_shape=original_shape,
            original_dtype=original_dtype,
            stage=TurboQuantStage.FULL,
            bitwidth=self.bitwidth,
        )

    def decompress_tensor(self, payload: TurboQuantTensorPayload) -> torch.Tensor:
        """Decompress on CPU for cross-device stability (MPS/CUDA/CPU)."""
        cpu = torch.device("cpu")

        if payload.stage == TurboQuantStage.WHT_ONLY:
            y = payload.wht_only
            assert y is not None
            y = y.float().to(cpu)
            x_pad = inverse_hadamard_transform(y * math.sqrt(payload.padded_dim), dim=-1)
            x_pad = unpad(x_pad, payload.original_dim, dim=-1)
            return x_pad.to(payload.original_dtype).reshape(payload.original_shape)

        assert payload.indices is not None
        indices = payload.indices.to(cpu)
        y_mse = dequantize(indices, self.centroids.to(cpu))

        if payload.stage == TurboQuantStage.WHT_QUANT:
            y = y_mse
        elif payload.stage == TurboQuantStage.WHT_QUANT_RESIDUAL:
            y = y_mse
        else:
            assert payload.qjl_bits is not None and payload.norm_r is not None
            proj = self._get_projection(payload.padded_dim, cpu)
            bits = payload.qjl_bits.to(cpu)
            norm_r = payload.norm_r.to(cpu)
            r_hat = qjl_decode(bits, proj, norm_r)
            y = y_mse + r_hat

        x_pad = inverse_hadamard_transform(y * math.sqrt(payload.padded_dim), dim=-1)
        x_pad = unpad(x_pad, payload.original_dim, dim=-1)
        return x_pad.to(payload.original_dtype).reshape(payload.original_shape)

    def reconstruction_error(self, x: torch.Tensor) -> float:
        restored = self.decompress_tensor(self.compress_tensor(x))
        return (x.float() - restored.float()).pow(2).mean().sqrt().item()
