"""RocketKV token selection: permanent filtering + hybrid sparse attention."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from framework.storage_accounting import bits_to_bytes, float32_storage_bits

ROCKETKV_METADATA_BYTES = 32


@dataclass
class RocketKVLayerPayload:
    """Layer-level RocketKV cache: global indices + retained K/V tensors."""

    selected_indices: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor
    original_seq_len: int
    stage1_locked: bool = False
    permanent_prefix_global: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
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
    Stage-1 permanent token filtering (SnapKV-inspired).

    Keeps an observation window at the sequence tail plus top-scoring prefix
    tokens up to ``max_tokens`` total.
    """

    def __init__(
        self,
        window_size: int = 32,
        min_keep: int = 1,
        *,
        keep_ratio: float | None = None,
    ) -> None:
        self.window_size = window_size
        self.min_keep = min_keep
        self.keep_ratio = keep_ratio  # legacy only

    def score_prefix_tokens(
        self,
        prefix_keys: torch.Tensor,
        window_keys: torch.Tensor,
    ) -> torch.Tensor:
        """Score prefix keys by dot-product with mean window key (per head)."""
        window_anchor = window_keys.float().mean(dim=2, keepdim=True)
        scores = (prefix_keys.float() * window_anchor).sum(dim=-1)
        return scores.mean(dim=1)

    def select_with_budget(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        max_tokens: int,
        global_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select up to ``max_tokens`` along the sequence dimension.

        Returns (local_indices, kept_key, kept_value, kept_global_indices).
        """
        seq_len = key.shape[2]
        device = key.device
        if global_indices is None:
            global_indices = torch.arange(seq_len, device=device)
        if seq_len == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, key, value, empty

        if seq_len <= max_tokens:
            local = torch.arange(seq_len, device=device)
            return local, key, value, global_indices

        window_size = min(self.window_size, max_tokens)
        window_start = seq_len - window_size
        window_local = torch.arange(window_start, seq_len, device=device)
        prefix_budget = max_tokens - window_size

        if prefix_budget <= 0:
            local = window_local
        else:
            prefix_keys = key[:, :, :window_start, :]
            window_keys = key[:, :, window_start:, :]
            scores = self.score_prefix_tokens(prefix_keys, window_keys)
            num_keep = min(prefix_budget, window_start)
            num_keep = max(self.min_keep, num_keep)
            _, top_local = scores[0].topk(num_keep, dim=0)
            prefix_local = top_local.sort().values
            local = torch.cat([prefix_local, window_local], dim=0).sort().values

        kept_key = key.index_select(2, local)
        kept_value = value.index_select(2, local)
        kept_global = global_indices[local]
        return local, kept_key, kept_value, kept_global

    def maintain_with_permanent(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        permanent_global: torch.Tensor,
        max_tokens: int,
        global_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Keep locked permanent globals plus the trailing window, capped at ``max_tokens``."""
        seq_len = key.shape[2]
        device = key.device
        if global_indices is None:
            global_indices = torch.arange(seq_len, device=device)
        if seq_len <= max_tokens:
            local = torch.arange(seq_len, device=device)
            return local, key, value, global_indices

        window_size = min(self.window_size, max_tokens)
        window_local = torch.arange(max(0, seq_len - window_size), seq_len, device=device)
        perm = permanent_global.to(device)
        perm_mask = torch.isin(global_indices, perm)
        perm_local = torch.nonzero(perm_mask, as_tuple=False).squeeze(-1)

        combined_local = torch.unique(torch.cat([perm_local, window_local], dim=0)).sort().values
        if combined_local.numel() > max_tokens:
            keep_perm = max_tokens - window_local.numel()
            if keep_perm > 0 and perm_local.numel() > 0:
                perm_local = perm_local.sort().values[-keep_perm:]
                combined_local = torch.unique(torch.cat([perm_local, window_local], dim=0)).sort().values
            else:
                combined_local = window_local[-max_tokens:]

        kept_key = key.index_select(2, combined_local)
        kept_value = value.index_select(2, combined_local)
        kept_global = global_indices[combined_local]
        return combined_local, kept_key, kept_value, kept_global

    def select(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        max_tokens: int | None = None,
        *,
        keep_ratio: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Backward-compatible wrapper returning (indices, kept_key, kept_value)."""
        if max_tokens is None:
            if keep_ratio is None:
                keep_ratio = self.keep_ratio if self.keep_ratio is not None else 0.5
            seq_len = key.shape[2]
            if seq_len <= self.window_size:
                max_tokens = seq_len
            else:
                prefix = seq_len - self.window_size
                max_tokens = max(self.min_keep, int(prefix * keep_ratio)) + self.window_size
                max_tokens = min(max_tokens, seq_len)
        _, kept_key, kept_value, kept_global = self.select_with_budget(key, value, max_tokens)
        return kept_global, kept_key, kept_value


class HybridSparseAttention:
    """
    Stage-2 Hybrid Sparse Attention (HSA): dynamic top-k unioned with permanent tokens.
    """

    def __init__(
        self,
        attention_budget: int = 512,
        *,
        dynamic_top_k: int | None = None,
    ) -> None:
        self.attention_budget = dynamic_top_k if dynamic_top_k is not None else attention_budget

    def approximate_scores(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> torch.Tensor:
        """Per-head query-key scores reduced over KV heads for top-k selection."""
        batch, num_q_heads, q_len, head_dim = query.shape
        _, num_kv_heads, k_len, _ = key.shape
        if num_q_heads != num_kv_heads:
            group = num_q_heads // num_kv_heads
            query = query.view(batch, num_kv_heads, group, q_len, head_dim).mean(dim=2)
        scores = torch.matmul(query.float(), key.float().transpose(-2, -1))
        return scores.mean(dim=1)

    def select_with_budget(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        max_tokens: int,
        permanent_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select up to ``max_tokens`` for attention, always unioning permanent indices."""
        seq_len = key.shape[2]
        if seq_len == 0:
            empty = torch.empty(0, dtype=torch.long, device=key.device)
            return key, value, empty

        if seq_len <= max_tokens and permanent_indices is None:
            indices = torch.arange(seq_len, device=key.device)
            return key, value, indices

        scores = self.approximate_scores(query, key)
        q_pos = query.shape[2] - 1
        k = min(max_tokens, seq_len)
        _, top_indices = scores[0, q_pos].topk(k, dim=-1)
        top_indices = top_indices.sort().values

        if permanent_indices is not None and permanent_indices.numel() > 0:
            perm = permanent_indices.to(key.device)
            perm = perm[perm < seq_len]
            combined = torch.unique(torch.cat([perm, top_indices])).sort().values
        else:
            combined = top_indices

        if combined.numel() > max_tokens:
            perm = permanent_indices.to(key.device) if permanent_indices is not None else torch.empty(0, device=key.device)
            perm = perm[perm < seq_len]
            if perm.numel() >= max_tokens:
                score_perm = scores[0, q_pos, perm]
                _, order = score_perm.topk(max_tokens, dim=-1)
                combined = perm[order].sort().values
            else:
                perm_set = set(perm.tolist())
                dynamic_only = [idx.item() for idx in top_indices if idx.item() not in perm_set]
                slots = max_tokens - perm.numel()
                extra: list[int] = dynamic_only[:slots]
                if extra:
                    extra_t = torch.tensor(extra, device=key.device, dtype=torch.long)
                    combined = torch.unique(torch.cat([perm, extra_t])).sort().values
                else:
                    combined = perm.sort().values

        sparse_key = key.index_select(2, combined)
        sparse_value = value.index_select(2, combined)
        return sparse_key, sparse_value, combined

    def select_top_k(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        permanent_indices: torch.Tensor | None = None,
        *,
        dynamic_top_k: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        budget = dynamic_top_k if dynamic_top_k is not None else self.attention_budget
        return self.select_with_budget(query, key, value, budget, permanent_indices)
