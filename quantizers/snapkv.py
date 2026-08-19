"""SnapKV: prefill-only observation-window voting + pooled top-k eviction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from framework.storage_accounting import bits_to_bytes, float32_storage_bits

SNAPKV_METADATA_BYTES = 24


@dataclass
class SnapKVLayerPayload:
    """Compact FP16 K/V after SnapKV prefill compression."""

    keys: torch.Tensor
    values: torch.Tensor
    original_seq_len: int
    compressed: bool = True

    def storage_bits(self) -> int:
        bits = SNAPKV_METADATA_BYTES * 8
        bits += self.keys.numel() * 16
        bits += self.values.numel() * 16
        return bits

    def storage_bytes(self) -> int:
        return bits_to_bytes(self.storage_bits())

    @property
    def nbytes(self) -> int:
        return self.storage_bytes()


def _compute_attn_weights(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scaled dot-product attention weights (softmax over keys)."""
    head_dim = query_states.shape[-1]
    scores = torch.matmul(query_states, key_states.transpose(-2, -1)) / (head_dim**0.5)
    if attention_mask is not None:
        if attention_mask.dim() == 2:
            mask = attention_mask[:, None, None, :].to(scores.dtype)
            scores = scores + (1.0 - mask) * torch.finfo(scores.dtype).min
        elif attention_mask.dim() == 4:
            scores = scores + attention_mask[..., : scores.shape[-1]]
    return F.softmax(scores.float(), dim=-1).to(query_states.dtype)


def _align_query_to_kv_heads(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
) -> torch.Tensor:
    """Reduce GQA query heads to KV heads by mean-pooling query groups."""
    num_q = query_states.shape[1]
    num_kv = key_states.shape[1]
    if num_q == num_kv:
        return query_states
    if num_q % num_kv != 0:
        return query_states[:, :num_kv]
    n_rep = num_q // num_kv
    bsz, _, q_len, dim = query_states.shape
    grouped = query_states.view(bsz, num_kv, n_rep, q_len, dim)
    return grouped.mean(dim=2)


def snap_kv(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *,
    window_size: int,
    max_capacity_prompt: int,
    kernel_size: int = 5,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Listing 1 — SnapKV prefill compression (per-head top-k on prefix)."""
    _bsz, _num_heads, q_len, _head_dim = query_states.shape
    kv_len = key_states.shape[2]
    if q_len != kv_len or q_len < max_capacity_prompt:
        return key_states, value_states

    prefix_len = kv_len - window_size
    if prefix_len <= 0:
        return key_states, value_states

    prefix_budget = max_capacity_prompt - window_size
    if prefix_budget <= 0:
        return key_states[..., -max_capacity_prompt:, :], value_states[..., -max_capacity_prompt:, :]

    query_aligned = _align_query_to_kv_heads(query_states, key_states)
    attn_weights = _compute_attn_weights(
        query_aligned[..., -window_size:, :],
        key_states,
        attention_mask=None,
    )
    vote = attn_weights[..., :, :prefix_len].sum(dim=-2)

    bsz, num_heads, prefix_len = vote.shape
    pooled = vote.reshape(bsz * num_heads, 1, prefix_len)
    pool_vote = F.max_pool1d(
        pooled,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    ).reshape(bsz, num_heads, prefix_len)

    num_keep = min(prefix_budget, prefix_len)
    _, indices = pool_vote.topk(num_keep, dim=-1)
    indices = indices.sort(dim=-1).values

    gather_idx = indices.unsqueeze(-1).expand(-1, -1, -1, key_states.shape[-1])
    k_past = key_states[..., :prefix_len, :].gather(dim=2, index=gather_idx)
    v_past = value_states[..., :prefix_len, :].gather(dim=2, index=gather_idx)
    k_obs = key_states[..., -window_size:, :]
    v_obs = value_states[..., -window_size:, :]
    return torch.cat([k_past, k_obs], dim=2), torch.cat([v_past, v_obs], dim=2)
