"""Palu: G-LRD low-rank latent KV cache (Category C + E under RoPE)."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from framework.storage_accounting import bits_to_bytes, float32_storage_bits

PALU_METADATA_BYTES = 32


@dataclass
class PaluGroupFactors:
    """Low-rank factors for one head group."""

    a_key: torch.Tensor
    b_key: torch.Tensor
    a_value: torch.Tensor
    b_value: torch.Tensor
    rank: int
    group_size: int


@dataclass
class PaluLatentPayload:
    """Low-rank latent H^k and H^v for one layer."""

    h_key: torch.Tensor
    h_value: torch.Tensor
    original_seq_len: int
    rank: int
    group_size: int

    def storage_bits(self) -> int:
        bits = PALU_METADATA_BYTES * 8
        bits += float32_storage_bits(self.h_key.numel())
        bits += float32_storage_bits(self.h_value.numel())
        return bits

    def storage_bytes(self) -> int:
        return bits_to_bytes(self.storage_bits())

    @property
    def nbytes(self) -> int:
        return self.storage_bytes()


@dataclass
class PaluLayerFactors:
    """Per-layer grouped low-rank projection factors."""

    groups: list[PaluGroupFactors] = field(default_factory=list)
    num_kv_heads: int = 0
    head_dim: int = 0


def truncated_svd_factors(
    weight: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """W ≈ A @ B with A ∈ R^{out×r}, B ∈ R^{r×in} (truncated SVD)."""
    w = weight.float()
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    rank = max(1, min(rank, s.numel()))
    s_r = s[:rank]
    u_r = u[:, :rank]
    vh_r = vh[:rank, :]
    sqrt_s = torch.sqrt(s_r)
    a = u_r * sqrt_s.unsqueeze(0)
    b = sqrt_s.unsqueeze(1) * vh_r
    return a, b


def decompose_projection_g_lrd(
    weight: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    group_size: int,
    compression_rate: float,
) -> PaluLayerFactors:
    """Group-head (G-LRD) truncated SVD on a K or V projection weight matrix."""
    out_features, in_features = weight.shape
    expected = num_heads * head_dim
    if out_features != expected:
        raise ValueError(f"Expected out_features={expected}, got {out_features}")

    rank_per_group = max(1, int(head_dim * compression_rate))
    groups: list[PaluGroupFactors] = []
    for group_start in range(0, num_heads, group_size):
        group_heads = min(group_size, num_heads - group_start)
        row_start = group_start * head_dim
        row_end = row_start + group_heads * head_dim
        w_group = weight[row_start:row_end, :]
        group_rank = max(1, int(group_heads * head_dim * compression_rate))
        a, b = truncated_svd_factors(w_group, group_rank)
        groups.append(
            PaluGroupFactors(
                a_key=a,
                b_key=b,
                a_value=torch.empty(0),
                b_value=torch.empty(0),
                rank=group_rank,
                group_size=group_heads,
            )
        )
    return PaluLayerFactors(groups=groups, num_kv_heads=num_heads, head_dim=head_dim)


def build_layer_factors_from_projections(
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    *,
    num_kv_heads: int,
    head_dim: int,
    group_size: int,
    compression_rate: float,
) -> PaluLayerFactors:
    """Build G-LRD factors for both K and V projection weights."""
    k_factors = decompose_projection_g_lrd(
        k_weight,
        num_heads=num_kv_heads,
        head_dim=head_dim,
        group_size=group_size,
        compression_rate=compression_rate,
    )
    v_factors = decompose_projection_g_lrd(
        v_weight,
        num_heads=num_kv_heads,
        head_dim=head_dim,
        group_size=group_size,
        compression_rate=compression_rate,
    )
    merged: list[PaluGroupFactors] = []
    for kg, vg in zip(k_factors.groups, v_factors.groups, strict=True):
        merged.append(
            PaluGroupFactors(
                a_key=kg.a_key,
                b_key=kg.b_key,
                a_value=vg.a_key,
                b_value=vg.b_key,
                rank=max(kg.rank, vg.rank),
                group_size=kg.group_size,
            )
        )
    return PaluLayerFactors(groups=merged, num_kv_heads=num_kv_heads, head_dim=head_dim)


def project_hidden_to_latent(hidden: torch.Tensor, factors: PaluLayerFactors) -> tuple[torch.Tensor, torch.Tensor]:
    """x @ A for each group → latent H^k, H^v with shape (B, H, T, r_g) per group."""
    batch, seq_len, hidden_dim = hidden.shape
    h_key_parts: list[torch.Tensor] = []
    h_value_parts: list[torch.Tensor] = []
    for group in factors.groups:
        hk = torch.matmul(hidden, group.a_key.t())
        hv = torch.matmul(hidden, group.a_value.t())
        h_key_parts.append(hk.view(batch, seq_len, group.group_size, -1).transpose(1, 2))
        h_value_parts.append(hv.view(batch, seq_len, group.group_size, -1).transpose(1, 2))
    h_key = torch.cat(h_key_parts, dim=1)
    h_value = torch.cat(h_value_parts, dim=1)
    return h_key, h_value


def reconstruct_kv_from_latent(
    h_key: torch.Tensor,
    h_value: torch.Tensor,
    factors: PaluLayerFactors,
) -> tuple[torch.Tensor, torch.Tensor]:
    """H @ B → full K/V tensors (pre-RoPE layout)."""
    batch, num_heads, seq_len, _ = h_key.shape
    key_parts: list[torch.Tensor] = []
    value_parts: list[torch.Tensor] = []
    head_offset = 0
    for group in factors.groups:
        gh = group.group_size
        hk = h_key[:, head_offset : head_offset + gh, :, :]
        hv = h_value[:, head_offset : head_offset + gh, :, :]
        hk_flat = hk.transpose(1, 2).reshape(batch, seq_len, -1)
        hv_flat = hv.transpose(1, 2).reshape(batch, seq_len, -1)
        key_parts.append(torch.matmul(hk_flat, group.b_key))
        value_parts.append(torch.matmul(hv_flat, group.b_value))
        head_offset += gh
    key = torch.cat(key_parts, dim=-1).view(batch, seq_len, num_heads, factors.head_dim).transpose(1, 2)
    value = torch.cat(value_parts, dim=-1).view(batch, seq_len, num_heads, factors.head_dim).transpose(1, 2)
    return key, value


def compress_kv_lowrank(
    key: torch.Tensor,
    value: torch.Tensor,
    rank: int,
) -> PaluLatentPayload:
    """Post-hoc truncated SVD on cached K/V (offline FIDELITY path)."""
    batch, num_heads, seq_len, head_dim = key.shape
    flat_k = key.transpose(1, 2).reshape(batch, seq_len, num_heads * head_dim)
    flat_v = value.transpose(1, 2).reshape(batch, seq_len, num_heads * head_dim)
    u_k, s_k, _ = torch.linalg.svd(flat_k.float(), full_matrices=False)
    u_v, s_v, _ = torch.linalg.svd(flat_v.float(), full_matrices=False)
    r = max(1, min(rank, s_k.numel(), s_v.numel()))
    h_key = (u_k[:, :, :r] * torch.sqrt(s_k[:r])).to(key.dtype)
    h_value = (u_v[:, :, :r] * torch.sqrt(s_v[:r])).to(value.dtype)
    return PaluLatentPayload(
        h_key=h_key.detach().cpu(),
        h_value=h_value.detach().cpu(),
        original_seq_len=seq_len,
        rank=r,
        group_size=num_heads,
    )


def decompress_kv_lowrank(payload: PaluLatentPayload, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct approximate K/V from latent payload (offline path)."""
    batch, num_heads, _seq_len, head_dim = key.shape
    out_dim = num_heads * head_dim
    k_proj = payload.h_key.to(key.device) @ payload.h_key.to(key.device).transpose(-2, -1)
    v_proj = payload.h_value.to(value.device) @ payload.h_value.to(value.device).transpose(-2, -1)
    flat_k = k_proj
    flat_v = v_proj
    if flat_k.shape[-1] != out_dim:
        flat_k = flat_k[..., :out_dim]
        flat_v = flat_v[..., :out_dim]
    k2 = flat_k.view(batch, -1, num_heads, head_dim).transpose(1, 2)
    v2 = flat_v.view(batch, -1, num_heads, head_dim).transpose(1, 2)
    return k2.to(key.dtype), v2.to(value.dtype)
