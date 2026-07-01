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
        self._projections: dict[int, torch.Tensor] = {}

    def _get_projection(self, head_dim: int, device: torch.device) -> torch.Tensor:
        if head_dim not in self._projections:
            m = self.proj_dim or head_dim
            gen = torch.Generator(device="cpu")
            gen.manual_seed(self.seed + head_dim)
            s = torch.randn(m, head_dim, generator=gen)
            self._projections[head_dim] = s.to(device)
        return self._projections[head_dim]

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
        )

    def decompress_tensor(self, payload: QJLTensorPayload) -> torch.Tensor:
        if payload.passthrough is not None:
            return payload.passthrough

        cpu = torch.device("cpu")
        proj = self._get_projection(payload.head_dim, cpu)[: payload.proj_dim]
        k_hat = qjl_decode(
            payload.sign_bits.to(cpu),
            proj,
            payload.vector_norm.to(cpu),
        )
        return k_hat.to(payload.original_dtype).reshape(payload.original_shape)

    def estimate_inner_products(
        self,
        query: torch.Tensor,
        payload: QJLTensorPayload,
    ) -> torch.Tensor:
        """
        Asymmetric QJL estimator for q · k without reconstructing k.

        q · k ≈ sqrt(pi/2) * (||k|| / m) * <Sq, sign(Sk)>

        Returns scores of shape [B, H, Tq, Tk] (or [B, H, Tk] when Tq=1).
        """
        if payload.passthrough is not None:
            k = payload.passthrough.float()
            q = query.float()
            if q.shape[1] != k.shape[1]:
                group = q.shape[1] // k.shape[1]
                q = q.view(q.shape[0], k.shape[1], group, q.shape[2], q.shape[3]).mean(dim=2)
            return torch.einsum("...qd,...kd->...qk", q, k)

        device = query.device
        proj = self._get_projection(payload.head_dim, device)[: payload.proj_dim]
        q = query.float()
        num_kv_heads = payload.original_shape[1]
        if q.shape[1] != num_kv_heads:
            group = q.shape[1] // num_kv_heads
            q = q.view(q.shape[0], num_kv_heads, group, q.shape[2], q.shape[3]).mean(dim=2)

        sq = torch.sign(torch.einsum("md,...d->...m", proj, q))
        kt = payload.sign_bits.to(device).to(proj.dtype)
        dots = torch.einsum("...qm,...km->...qk", sq, kt)
        m = proj.shape[0]
        scale = math.sqrt(math.pi / 2.0) / m
        norms = payload.vector_norm.to(device).squeeze(-1)
        return scale * norms.unsqueeze(-2) * dots

    def estimate_attention_scores(
        self,
        query: torch.Tensor,
        payloads: list[QJLTensorPayload] | QJLTensorPayload,
        head_dim: int,
        num_q_heads: int | None = None,
    ) -> torch.Tensor:
        """Estimate QK^T / sqrt(d) from compressed key payloads."""
        if isinstance(payloads, list):
            parts = []
            for item in payloads:
                part = self.estimate_inner_products(query, item)
                if part.dim() == query.dim() - 1:
                    part = part.unsqueeze(-1)
                parts.append(part)
            scores = torch.cat(parts, dim=-1)
        else:
            scores = self.estimate_inner_products(query, payloads)
            if scores.dim() == query.dim() - 1:
                scores = scores.unsqueeze(-1)

        if num_q_heads is not None and scores.shape[1] != num_q_heads:
            repeats = num_q_heads // scores.shape[1]
            scores = scores.repeat_interleave(repeats, dim=1)

        return scores / math.sqrt(head_dim)

    def reconstruction_error(self, x: torch.Tensor, mode: str = "key") -> float:
        restored = self.decompress_tensor(self.compress_tensor(x, mode=mode))
        return (x.float() - restored.float()).pow(2).mean().sqrt().item()
