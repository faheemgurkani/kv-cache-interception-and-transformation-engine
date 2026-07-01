"""RocketKV token selection: permanent filtering + hybrid sparse attention."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from framework.storage_accounting import bits_to_bytes, float32_storage_bits

ROCKETKV_METADATA_BYTES = 32


@dataclass
class RocketKVLayerPayload:
    """Layer-level RocketKV cache: selected indices + retained K/V tensors."""

    selected_indices: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor
    original_seq_len: int
    selection_mode: str = "permanent"

    def storage_bits(self) -> int:
        bits = ROCKETKV_METADATA_BYTES * 8
        bits += float32_storage_bits(self.selected_indices.numel() * 8)
        bits += self.keys.numel() * 16
        bits += self.values.numel() * 16
        return bits

    def storage_bytes(self) -> int:
        return bits_to_bytes(self.storage_bits())

    @property
    def nbytes(self) -> int:
        return self.storage_bytes()


class TokenSelector:
    """
    Permanent token filtering (SnapKV-inspired heuristic).

    Keeps a configurable fraction of prefix tokens by importance score plus
    all tokens in the trailing observation window.
    """

    def __init__(
        self,
        keep_ratio: float = 0.5,
        window_size: int = 32,
        min_keep: int = 1,
    ) -> None:
        self.keep_ratio = keep_ratio
        self.window_size = window_size
        self.min_keep = min_keep

    def score_prefix_tokens(
        self,
        prefix_keys: torch.Tensor,
        window_keys: torch.Tensor,
    ) -> torch.Tensor:
        """Score prefix keys by max dot-product with mean window key (per head)."""
        window_anchor = window_keys.float().mean(dim=2, keepdim=True)
        scores = (prefix_keys.float() * window_anchor).sum(dim=-1)
        return scores.mean(dim=1)

    def select(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select important tokens along the sequence dimension.

        Returns (selected_indices, kept_key, kept_value).
        """
        seq_len = key.shape[2]
        if seq_len == 0:
            empty = torch.empty(0, dtype=torch.long, device=key.device)
            return empty, key, value

        if seq_len <= self.window_size:
            indices = torch.arange(seq_len, device=key.device)
            return indices, key, value

        window_start = seq_len - self.window_size
        prefix_keys = key[:, :, :window_start, :]
        window_keys = key[:, :, window_start:, :]

        scores = self.score_prefix_tokens(prefix_keys, window_keys)
        num_keep = max(self.min_keep, int(window_start * self.keep_ratio))
        num_keep = min(num_keep, window_start)

        _, top_local = scores[0].topk(num_keep, dim=0)
        top_local = top_local.sort().values
        prefix_indices = top_local

        window_indices = torch.arange(window_start, seq_len, device=key.device)
        selected_indices = torch.cat([prefix_indices, window_indices], dim=0)

        kept_key = key.index_select(2, selected_indices)
        kept_value = value.index_select(2, selected_indices)
        return selected_indices, kept_key, kept_value


class HybridSparseAttention:
    """
    Hybrid Sparse Attention (HSA): dynamic top-k token selection per query step.

    Approximates token importance by reducing over heads before final top-k.
    """

    def __init__(self, dynamic_top_k: int = 64) -> None:
        self.dynamic_top_k = dynamic_top_k

    def approximate_scores(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> torch.Tensor:
        """Reduce over heads, then compute query-key dot products."""
        q_reduced = query.float().mean(dim=1, keepdim=True)
        k_reduced = key.float().mean(dim=1, keepdim=True)
        return torch.matmul(q_reduced, k_reduced.transpose(-2, -1)).squeeze(1)

    def select_top_k(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        permanent_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select top-k tokens for the current query, unioned with permanent indices.
        """
        seq_len = key.shape[2]
        if seq_len == 0:
            return key, value, torch.empty(0, dtype=torch.long, device=key.device)

        scores = self.approximate_scores(query, key)
        k = min(self.dynamic_top_k, seq_len)
        _, top_indices = scores[0, -1].topk(k, dim=-1)
        top_indices = top_indices.sort().values

        if permanent_indices is not None and permanent_indices.numel() > 0:
            combined = torch.unique(torch.cat([permanent_indices.to(key.device), top_indices]))
            combined = combined.sort().values
        else:
            combined = top_indices

        sparse_key = key.index_select(2, combined)
        sparse_value = value.index_select(2, combined)
        return sparse_key, sparse_value, combined
