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
from quantizers.lloyd_max import (
    build_centroids,
    compute_gamma,
    correct_rotated_norm,
    dequantize,
    normalize_features,
    quantize,
)
from quantizers.qjl import projection_matrix, qjl_decode, qjl_encode


class TurboQuantStage(str, Enum):
    WHT_ONLY = "wht_only"
    WHT_QUANT = "wht_quant"
    WHT_QUANT_RESIDUAL = "wht_quant_residual"
    FULL = "full"


TURBOQUANT_METADATA_BYTES = 32


@dataclass
class TurboQuantTensorPayload:
    """Compressed payload for a single K or V tensor."""

    indices: torch.Tensor | None
    qjl_bits: torch.Tensor | None
    norm_r: torch.Tensor | None
    vector_norm: torch.Tensor | None
    gamma: torch.Tensor | None
    original_dim: int
    padded_dim: int
    original_shape: tuple[int, ...]
    original_dtype: torch.dtype
    stage: TurboQuantStage
    bitwidth: int
    wht_only: torch.Tensor | None = None

    def storage_bits(self) -> int:
        bits = TURBOQUANT_METADATA_BYTES * 8
        if self.indices is not None:
            bits += index_storage_bits(self.indices.numel(), self.bitwidth)
        if self.qjl_bits is not None:
            bits += sign_storage_bits(self.qjl_bits.numel())
        if self.norm_r is not None:
            bits += float32_storage_bits(self.norm_r.numel())
        if self.vector_norm is not None:
            bits += float32_storage_bits(self.vector_norm.numel())
        if self.gamma is not None:
            bits += float32_storage_bits(self.gamma.numel())
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

    @staticmethod
    def _store_tensor(tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Keep payloads on CUDA; use CPU for MPS/CPU for cross-device stability."""
        pinned = tensor.detach()
        if reference.device.type == "cuda":
            return pinned.to(reference.device)
        return pinned.cpu()

    @staticmethod
    def _payload_device(payload: TurboQuantTensorPayload) -> torch.device:
        for field in (
            payload.indices,
            payload.vector_norm,
            payload.gamma,
            payload.qjl_bits,
            payload.norm_r,
            payload.wht_only,
        ):
            if field is not None:
                return field.device
        return torch.device("cpu")

    def _decompress_device(self, payload: TurboQuantTensorPayload, target_device: torch.device | None) -> torch.device:
        device = target_device or self._payload_device(payload)
        if device.type == "mps":
            return torch.device("cpu")
        return device

    def _to_rotated(self, x_pad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unit-norm + WHT + feature normalize; returns y, vector_norm, gamma."""
        vector_norm = x_pad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x_unit = x_pad / vector_norm
        y = normalize_features(hadamard_transform(x_unit, dim=-1), dim=-1)
        gamma = compute_gamma(y, self.centroids, dim=-1)
        return y, vector_norm, gamma

    def _from_rotated(
        self,
        y: torch.Tensor,
        vector_norm: torch.Tensor,
        padded_dim: int,
        original_dim: int,
        original_shape: tuple[int, ...],
        original_dtype: torch.dtype,
        apply_norm_correction: bool = False,
    ) -> torch.Tensor:
        if apply_norm_correction:
            y = correct_rotated_norm(y, dim=-1)
        x_unit_pad = inverse_hadamard_transform(y * math.sqrt(padded_dim), dim=-1)
        x_pad = x_unit_pad * vector_norm
        x_pad = unpad(x_pad, original_dim, dim=-1)
        return x_pad.to(original_dtype).reshape(original_shape)

    def compress_tensor(self, x: torch.Tensor, use_qjl: bool = True) -> TurboQuantTensorPayload:
        original_shape = tuple(x.shape)
        original_dtype = x.dtype
        x = x.float()
        x_pad, original_dim = pad_to_power_of_two(x, dim=-1)
        padded_dim = x_pad.shape[-1]

        y, vector_norm, gamma = self._to_rotated(x_pad)
        y_scaled = y / gamma

        if self.stage == TurboQuantStage.WHT_ONLY:
            return TurboQuantTensorPayload(
                indices=None,
                qjl_bits=None,
                norm_r=None,
                vector_norm=self._store_tensor(vector_norm, x),
                gamma=None,
                original_dim=original_dim,
                padded_dim=padded_dim,
                original_shape=original_shape,
                original_dtype=original_dtype,
                stage=self.stage,
                bitwidth=self.bitwidth,
                wht_only=self._store_tensor(y, x),
            )

        indices = quantize(y_scaled, self.centroids).to(torch.int8)
        y_mse = dequantize(indices, self.centroids) * gamma

        if self.stage == TurboQuantStage.WHT_QUANT:
            return TurboQuantTensorPayload(
                indices=self._store_tensor(indices, x),
                qjl_bits=None,
                norm_r=None,
                vector_norm=self._store_tensor(vector_norm, x),
                gamma=self._store_tensor(gamma, x),
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
                indices=self._store_tensor(indices, x),
                qjl_bits=None,
                norm_r=self._store_tensor(residual.norm(dim=-1, keepdim=True), x),
                vector_norm=self._store_tensor(vector_norm, x),
                gamma=self._store_tensor(gamma, x),
                original_dim=original_dim,
                padded_dim=padded_dim,
                original_shape=original_shape,
                original_dtype=original_dtype,
                stage=self.stage,
                bitwidth=self.bitwidth,
            )

        if not use_qjl:
            return TurboQuantTensorPayload(
                indices=self._store_tensor(indices, x),
                qjl_bits=None,
                norm_r=None,
                vector_norm=self._store_tensor(vector_norm, x),
                gamma=self._store_tensor(gamma, x),
                original_dim=original_dim,
                padded_dim=padded_dim,
                original_shape=original_shape,
                original_dtype=original_dtype,
                stage=TurboQuantStage.WHT_QUANT,
                bitwidth=self.bitwidth,
            )

        norm_r = residual.norm(dim=-1, keepdim=True)
        proj = self._get_projection(padded_dim, device=x.device)
        bits = qjl_encode(residual, proj)

        return TurboQuantTensorPayload(
            indices=self._store_tensor(indices, x),
            qjl_bits=self._store_tensor(bits, x),
            norm_r=self._store_tensor(norm_r, x),
            vector_norm=self._store_tensor(vector_norm, x),
            gamma=self._store_tensor(gamma, x),
            original_dim=original_dim,
            padded_dim=padded_dim,
            original_shape=original_shape,
            original_dtype=original_dtype,
            stage=TurboQuantStage.FULL,
            bitwidth=self.bitwidth,
        )

    def decompress_tensor(
        self,
        payload: TurboQuantTensorPayload,
        target_device: torch.device | None = None,
    ) -> torch.Tensor:
        """Decompress on CPU for MPS; stay on CUDA when payloads are GPU-resident."""
        device = self._decompress_device(payload, target_device)

        if payload.stage == TurboQuantStage.WHT_ONLY:
            y = payload.wht_only
            assert y is not None and payload.vector_norm is not None
            return self._from_rotated(
                y.float().to(device),
                payload.vector_norm.to(device),
                payload.padded_dim,
                payload.original_dim,
                payload.original_shape,
                payload.original_dtype,
                apply_norm_correction=False,
            )

        assert payload.indices is not None
        assert payload.vector_norm is not None and payload.gamma is not None
        indices = payload.indices.to(device)
        vector_norm = payload.vector_norm.to(device)
        gamma = payload.gamma.to(device)
        y_mse = dequantize(indices, self.centroids.to(device)) * gamma

        if payload.stage == TurboQuantStage.WHT_QUANT:
            y = y_mse
        elif payload.stage == TurboQuantStage.WHT_QUANT_RESIDUAL:
            y = y_mse
        else:
            assert payload.qjl_bits is not None and payload.norm_r is not None
            proj = self._get_projection(payload.padded_dim, device)
            r_hat = qjl_decode(payload.qjl_bits.to(device), proj, payload.norm_r.to(device))
            y = y_mse + r_hat

        return self._from_rotated(
            y,
            vector_norm,
            payload.padded_dim,
            payload.original_dim,
            payload.original_shape,
            payload.original_dtype,
            apply_norm_correction=False,
        )

    def reconstruction_error(self, x: torch.Tensor, use_qjl: bool = True) -> float:
        restored = self.decompress_tensor(self.compress_tensor(x, use_qjl=use_qjl))
        restored = restored.to(device=x.device)
        return (x.float() - restored.float()).pow(2).mean().sqrt().item()
