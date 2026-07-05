"""Standalone QJL pipeline: random Gaussian projection + sign quantization on keys."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from framework.storage_accounting import (
    bits_to_bytes,
    float32_storage_bits,
    sign_storage_bits,
)
from quantizers.qjl import projection_matrix, qjl_decode, qjl_encode

QJL_METADATA_BYTES = 24


@dataclass
class QJLTensorPayload:
    """Compressed QJL payload for a single key vector (sign bits + norm)."""

    sign_bits: torch.Tensor
    vector_norm: torch.Tensor
    proj_dim: int
    head_dim: int
    original_shape: tuple[int, ...]
    original_dtype: torch.dtype
    original_device: str = "cpu"
    passthrough: torch.Tensor | None = None

    def storage_bits(self) -> int:
        if self.passthrough is not None:
            return self.passthrough.numel() * 16
        bits = QJL_METADATA_BYTES * 8
        bits += sign_storage_bits(self.sign_bits.numel())
        bits += float32_storage_bits(self.vector_norm.numel())
        return bits

    def storage_bytes(self) -> int:
        return bits_to_bytes(self.storage_bits())

    @property
    def nbytes(self) -> int:
        return self.storage_bytes()


class QJLPipeline:
    """QJL key compression: sign(S @ k) + ||k||; values stored uncompressed."""

    def __init__(self, seed: int = 42, proj_dim: int | None = None) -> None:
        self.seed = seed
        self.proj_dim = proj_dim
        self._projections: dict[tuple[int, str], torch.Tensor] = {}

    def _get_projection(self, head_dim: int, device: torch.device) -> torch.Tensor:
        cache_key = (head_dim, str(device))
        if cache_key not in self._projections:
            m = self.proj_dim or head_dim
            self._projections[cache_key] = projection_matrix(
                head_dim,
                proj_dim=m,
                seed=self.seed,
                device=device,
            )
        return self._projections[cache_key]

    def compress_tensor(self, x: torch.Tensor, mode: str = "key") -> QJLTensorPayload:
        original_shape = tuple(x.shape)
        original_dtype = x.dtype
        head_dim = x.shape[-1]

        if mode != "key":
            return QJLTensorPayload(
                sign_bits=torch.empty(0, dtype=torch.int8),
                vector_norm=torch.empty(0),
                proj_dim=0,
                head_dim=head_dim,
                original_shape=original_shape,
                original_dtype=original_dtype,
                original_device=str(x.device),
                passthrough=x.detach().clone(),
            )

        x_float = x.float()
        vector_norm = x_float.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        proj = self._get_projection(head_dim, device=x.device)
        sign_bits = qjl_encode(x_float, proj)

        return QJLTensorPayload(
            sign_bits=sign_bits.detach().cpu(),
            vector_norm=vector_norm.detach().cpu(),
            proj_dim=proj.shape[0],
            head_dim=head_dim,
            original_shape=original_shape,
            original_dtype=original_dtype,
            original_device=str(x.device),
        )

    def decompress_tensor(self, payload: QJLTensorPayload) -> torch.Tensor:
        if payload.passthrough is not None:
            return payload.passthrough

        target = torch.device(payload.original_device)
        proj = self._get_projection(payload.head_dim, target)[: payload.proj_dim]
        k_hat = qjl_decode(
            payload.sign_bits.to(target),
            proj,
            payload.vector_norm.to(target),
        )
        return k_hat.to(payload.original_dtype).reshape(payload.original_shape)

    def _estimate_from_signs(
        self,
        query: torch.Tensor,
        sign_bits: torch.Tensor,
        vector_norm: torch.Tensor,
        proj: torch.Tensor,
        num_kv_heads: int,
    ) -> torch.Tensor:
        """Core asymmetric estimator with per-query-head GQA mapping."""
        q = query.float()
        num_q = q.shape[1]
        group = 1 if num_q == num_kv_heads else num_q // num_kv_heads
        m = proj.shape[0]
        scale = math.sqrt(math.pi / 2.0) / m

        head_scores: list[torch.Tensor] = []
        for qi in range(num_q):
            kv = qi // group
            q_h = q[:, qi, :, :]
            sq = torch.where(torch.einsum("md,btd->btm", proj, q_h) >= 0, 1.0, -1.0)
            k_signs = sign_bits[:, kv, :, :]
            k_norms = vector_norm[:, kv, :]
            dots = torch.einsum("btm,bkm->btk", sq, k_signs)
            head_scores.append(scale * dots * k_norms.unsqueeze(1))
        return torch.stack(head_scores, dim=1)

    def estimate_inner_products(
        self,
        query: torch.Tensor,
        payload: QJLTensorPayload,
    ) -> torch.Tensor:
        """
        Asymmetric QJL estimator for q · k without reconstructing k.

        q · k ≈ sqrt(pi/2) * (||k|| / m) * <Sq, sign(Sk)>

        Returns scores of shape [B, H_q, Tq, Tk].
        """
        if payload.passthrough is not None:
            k = payload.passthrough.float()
            q = query.float()
            if q.shape[1] != k.shape[1]:
                group = q.shape[1] // k.shape[1]
                k = k.repeat_interleave(group, dim=1)
            return torch.einsum("bhqd,bhkd->bhqk", q, k)

        device = query.device
        proj = self._get_projection(payload.head_dim, device)[: payload.proj_dim]
        kt = payload.sign_bits.to(device).to(proj.dtype)
        norms = payload.vector_norm.to(device).squeeze(-1)
        return self._estimate_from_signs(query, kt, norms, proj, payload.original_shape[1])

    def estimate_attention_scores(
        self,
        query: torch.Tensor,
        payloads: list[QJLTensorPayload] | QJLTensorPayload,
        head_dim: int,
        num_q_heads: int | None = None,
    ) -> torch.Tensor:
        """Estimate QK^T / sqrt(d) from compressed key payloads."""
        if isinstance(payloads, list):
            if not payloads:
                device = query.device
                return torch.empty(
                    query.shape[0],
                    query.shape[1],
                    query.shape[2],
                    0,
                    device=device,
                    dtype=query.dtype,
                )
            device = query.device
            first = payloads[0]
            proj = self._get_projection(first.head_dim, device)[: first.proj_dim]
            sign_bits = torch.cat(
                [item.sign_bits.to(device).to(proj.dtype) for item in payloads],
                dim=2,
            )
            norms = torch.cat(
                [item.vector_norm.to(device) for item in payloads],
                dim=2,
            ).squeeze(-1)
            scores = self._estimate_from_signs(
                query,
                sign_bits,
                norms,
                proj,
                first.original_shape[1],
            )
        else:
            scores = self.estimate_inner_products(query, payloads)
            if scores.dim() == query.dim() - 1:
                scores = scores.unsqueeze(-2)

        return scores / math.sqrt(head_dim)

    def reconstruction_error(self, x: torch.Tensor, mode: str = "key") -> float:
        restored = self.decompress_tensor(self.compress_tensor(x, mode=mode))
        return (x.float() - restored.float()).pow(2).mean().sqrt().item()
