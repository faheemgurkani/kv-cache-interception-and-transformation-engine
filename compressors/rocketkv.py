"""RocketKV compressor: two-stage token selection + physical cache eviction."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from compressors.base import CompressedKV, KVCompressor
from quantizers.rocketkv import HybridSparseAttention, RocketKVLayerPayload, TokenSelector


@dataclass
class RocketKVLayerState:
    stage1_locked: bool = False
    permanent_prefix_global: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    current_global: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    logical_seq_len: int = 0


class RocketKVCompressor(KVCompressor):
    """
    RocketKV plug-in with paper-style token budgets.

    Stage 1 — permanent filtering (SnapKV-style), locked once the cache reaches
    ``token_budget``.
    Stage 2 — Hybrid Sparse Attention at decode time, capped by ``hsa_budget``.
    """

    name = "rocketkv"

    def __init__(
        self,
        bitwidth: int = 16,
        token_budget: int = 512,
        hsa_budget: int | None = None,
        window_size: int = 32,
        *,
        keep_ratio: float | None = None,
        dynamic_top_k: int | None = None,
    ) -> None:
        self.bitwidth = bitwidth
        self.window_size = window_size
        if keep_ratio is not None and dynamic_top_k is not None and token_budget == 512:
            token_budget = dynamic_top_k
        if dynamic_top_k is not None and hsa_budget is None:
            hsa_budget = dynamic_top_k
        self.token_budget = int(token_budget)
        self.hsa_budget = int(hsa_budget or self.token_budget)
        self.token_selector = TokenSelector(window_size=window_size, keep_ratio=keep_ratio)
        self.hsa = HybridSparseAttention(
            attention_budget=self.hsa_budget,
            dynamic_top_k=dynamic_top_k,
        )
        self._layer_state: dict[int, RocketKVLayerState] = {}

    def reset_state(self) -> None:
        self._layer_state.clear()

    def _state(self, layer: int) -> RocketKVLayerState:
        return self._layer_state.setdefault(layer, RocketKVLayerState())

    def _permanent_prefix_global(self, layer: int) -> torch.Tensor:
        state = self._state(layer)
        return state.permanent_prefix_global

    def _global_to_local(
        self,
        global_indices: torch.Tensor,
        stored_global: torch.Tensor,
    ) -> torch.Tensor:
        if global_indices.numel() == 0 or stored_global.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=stored_global.device)
        mapping = {int(g): idx for idx, g in enumerate(stored_global.tolist())}
        local = [mapping[int(g.item())] for g in global_indices if int(g.item()) in mapping]
        if not local:
            return torch.empty(0, dtype=torch.long, device=stored_global.device)
        return torch.tensor(local, dtype=torch.long, device=stored_global.device)

    def sync_layer_from_payload(self, layer: int, payload: RocketKVLayerPayload) -> None:
        self.restore_state_from_payload(layer, payload)

    def extend_global_indices(self, layer: int, seq_len: int, device: torch.device) -> torch.Tensor:
        state = self._state(layer)
        if state.current_global.numel() == 0:
            fallback = torch.arange(seq_len, device=device)
            state.current_global = fallback.detach().cpu()
            return fallback
        if state.current_global.numel() == seq_len:
            return state.current_global.to(device)
        if state.current_global.numel() == seq_len - 1:
            next_global = int(state.current_global.max().item()) + 1
            extended = torch.cat(
                [state.current_global, torch.tensor([next_global], dtype=torch.long)],
                dim=0,
            )
            state.current_global = extended.detach().cpu()
            return extended.to(device)
        fallback = torch.arange(seq_len, device=device)
        state.current_global = fallback.detach().cpu()
        return fallback

    def _resolve_global_indices(
        self,
        key: torch.Tensor,
        layer: int,
        prior_payload: RocketKVLayerPayload | None = None,
    ) -> torch.Tensor:
        state = self._state(layer)
        seq_len = key.shape[2]
        device = key.device

        if prior_payload is not None:
            prior_global = prior_payload.selected_indices.to(device)
            prior_len = prior_global.numel()
            if seq_len == prior_len:
                return prior_global
            if seq_len == prior_len + 1:
                next_global = int(prior_global.max().item()) + 1
                return torch.cat(
                    [prior_global, torch.tensor([next_global], device=device, dtype=torch.long)],
                    dim=0,
                )
            return torch.arange(seq_len, device=device)

        return self.extend_global_indices(layer, seq_len, device)

    def restore_state_from_payload(self, layer: int, payload: RocketKVLayerPayload) -> None:
        state = self._state(layer)
        state.stage1_locked = payload.stage1_locked
        state.current_global = payload.selected_indices.clone()
        state.logical_seq_len = payload.original_seq_len
        if payload.permanent_prefix_global.numel() > 0:
            state.permanent_prefix_global = payload.permanent_prefix_global.clone()
        elif payload.stage1_locked and payload.selected_indices.numel() > 0:
            max_global = int(payload.selected_indices.max().item())
            window_size = min(self.window_size, payload.selected_indices.numel())
            window_start_global = max_global - window_size + 1
            state.permanent_prefix_global = payload.selected_indices[
                payload.selected_indices < window_start_global
            ].clone()

    def apply_stage1(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer: int = 0,
        global_indices: torch.Tensor | None = None,
        prior_payload: RocketKVLayerPayload | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Permanent selection with lock-after-budget semantics."""
        state = self._state(layer)
        seq_len = key.shape[2]
        if global_indices is None:
            global_indices = self._resolve_global_indices(key, layer, prior_payload)

        global_seq_len = int(global_indices.max().item()) + 1 if global_indices.numel() else seq_len
        should_lock = (not state.stage1_locked) and global_seq_len >= self.token_budget

        if should_lock:
            stage1_local, kept_key, kept_value, kept_global = self.token_selector.select_with_budget(
                key,
                value,
                self.token_budget,
                global_indices=global_indices,
            )
            state.stage1_locked = True
            window_size = min(self.window_size, kept_global.numel())
            max_global = int(kept_global.max().item())
            window_start_global = max_global - window_size + 1
            state.permanent_prefix_global = kept_global[kept_global < window_start_global].detach().cpu()
            state.current_global = kept_global.detach().cpu()
            return kept_global, kept_key, kept_value, kept_global, stage1_local

        if state.stage1_locked:
            stage1_local, kept_key, kept_value, kept_global = self.token_selector.maintain_with_permanent(
                key,
                value,
                state.permanent_prefix_global,
                self.token_budget,
                global_indices=global_indices,
            )
            state.current_global = kept_global.detach().cpu()
            return kept_global, kept_key, kept_value, kept_global, stage1_local

        stage1_local = torch.arange(seq_len, device=key.device)
        state.current_global = global_indices.detach().cpu()
        return global_indices, key, value, global_indices, stage1_local

    def apply_stage2(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        stored_global: torch.Tensor,
        layer: int = 0,
        stage1_local: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """HSA dynamic selection unioned with permanent prefix tokens."""
        state = self._state(layer)
        if stage1_local is None:
            stage1_local = torch.arange(key.shape[2], device=key.device)

        if not state.stage1_locked or query.shape[2] != 1:
            return key, value, stage1_local

        perm_prefix = self._permanent_prefix_global(layer)
        perm_local = self._global_to_local(perm_prefix, stored_global)
        sparse_key, sparse_value, stage2_local = self.hsa.select_with_budget(
            query,
            key,
            value,
            self.hsa_budget,
            permanent_indices=perm_local,
        )
        mask_local = stage1_local[stage2_local.to(stage1_local.device)]
        return sparse_key, sparse_value, mask_local

    def _build_payload(
        self,
        indices: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        original_seq_len: int,
        layer: int,
    ) -> RocketKVLayerPayload:
        state = self._state(layer)
        return RocketKVLayerPayload(
            selected_indices=indices.detach().cpu(),
            keys=key.detach().cpu(),
            values=value.detach().cpu(),
            original_seq_len=original_seq_len,
            stage1_locked=state.stage1_locked,
            permanent_prefix_global=state.permanent_prefix_global.detach().cpu(),
        )

    def compress_kv(
        self,
        tensor: torch.Tensor,
        layer: int = 0,
        mode: str = "key",
    ) -> object:
        if tensor.shape[2] != 1:
            raise ValueError("RocketKV incremental compress_kv expects one token per payload.")
        return tensor.detach().clone()

    def decompress_kv(self, payload: object, mode: str = "key") -> torch.Tensor:
        if isinstance(payload, RocketKVLayerPayload):
            return payload.keys if mode == "key" else payload.values
        if not isinstance(payload, torch.Tensor):
            raise TypeError(f"Expected tensor payload, got {type(payload)}")
        return payload

    def compress(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer: int = 0,
    ) -> CompressedKV:
        saved = self._layer_state.get(layer)
        self._layer_state.pop(layer, None)
        try:
            _, kept_key, kept_value, kept_global, _ = self.apply_stage1(key, value, layer=layer)
            payload = self._build_payload(kept_global, kept_key, kept_value, key.shape[2], layer)
            return CompressedKV(
                keys=payload,
                values=payload,
                original_shape=tuple(key.shape),
                nbytes=payload.nbytes,
                bitwidth=self.bitwidth,
                layer=layer,
            )
        finally:
            if saved is not None:
                self._layer_state[layer] = saved
            else:
                self._layer_state.pop(layer, None)

    def decompress(self, compressed: CompressedKV) -> tuple[torch.Tensor, torch.Tensor]:
        payload = compressed.keys
        if isinstance(payload, RocketKVLayerPayload):
            return payload.keys, payload.values
        if isinstance(payload, list):
            return self.decompress_incremental_layer(compressed)
        key = self.decompress_kv(compressed.keys, mode="key")
        value = self.decompress_kv(compressed.values, mode="value")
        return key, value

    def decompress_incremental_layer(
        self,
        compressed: CompressedKV,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(compressed.keys, RocketKVLayerPayload):
            payload = compressed.keys
            return payload.keys, payload.values

        key_parts = [self.decompress_kv(item, mode="key") for item in compressed.keys]  # type: ignore[union-attr]
        value_parts = [self.decompress_kv(item, mode="value") for item in compressed.values]  # type: ignore[union-attr]
        key = torch.cat(key_parts, dim=2)
        value = torch.cat(value_parts, dim=2)
        _, kept_key, kept_value, _, _ = self.apply_stage1(key, value, layer=compressed.layer)
        return kept_key, kept_value

    def compress_layer_from_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer: int,
        original_seq_len: int | None = None,
        prior_payload: RocketKVLayerPayload | None = None,
    ) -> CompressedKV:
        """Physical eviction entry point used by KVCacheEngine after each step."""
        orig_len = original_seq_len or key.shape[2]
        _, kept_key, kept_value, kept_global, _ = self.apply_stage1(
            key,
            value,
            layer=layer,
            prior_payload=prior_payload,
        )
        state = self._state(layer)
        state.logical_seq_len = orig_len
        payload = self._build_payload(kept_global, kept_key, kept_value, orig_len, layer)
        return CompressedKV(
            keys=payload,
            values=payload,
            original_shape=(key.shape[0], key.shape[1], orig_len, key.shape[3]),
            nbytes=payload.nbytes,
            bitwidth=self.bitwidth,
            layer=layer,
        )

    def trim_layer(self, compressed: CompressedKV, drop_tokens: int) -> CompressedKV:
        """Drop oldest stored tokens from a physically evicted layer payload."""
        payload = compressed.keys
        if not isinstance(payload, RocketKVLayerPayload):
            raise ValueError("RocketKV trim_layer requires RocketKVLayerPayload.")
        if drop_tokens <= 0:
            return compressed
        if payload.keys.shape[2] <= drop_tokens:
            raise ValueError("Cannot trim entire RocketKV layer payload.")

        new_payload = RocketKVLayerPayload(
            selected_indices=payload.selected_indices[drop_tokens:],
            keys=payload.keys[:, :, drop_tokens:, :],
            values=payload.values[:, :, drop_tokens:, :],
            original_seq_len=payload.original_seq_len,
            stage1_locked=payload.stage1_locked,
            permanent_prefix_global=payload.permanent_prefix_global,
        )
        return CompressedKV(
            keys=new_payload,
            values=new_payload,
            original_shape=compressed.original_shape,
            nbytes=new_payload.nbytes,
            bitwidth=compressed.bitwidth,
            layer=compressed.layer,
        )

    def select_dynamic_tokens(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stored_global, kept_key, kept_value, _, stage1_local = self.apply_stage1(key, value, layer=layer)
        return self.apply_stage2(
            query,
            kept_key,
            kept_value,
            stored_global,
            layer=layer,
            stage1_local=stage1_local,
        )

    def reconstruction_error(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer: int = 0,
    ) -> dict[str, float]:
        """Section A tensor fidelity on post-selection retained tokens."""
        saved = self._layer_state.get(layer)
        self.reset_state()
        try:
            _, kept_key, kept_value, kept_global, _ = self.apply_stage1(key, value, layer=layer)
            key_rmse = (kept_key.float() - kept_key.float()).pow(2).mean().sqrt().item()
            value_rmse = (kept_value.float() - kept_value.float()).pow(2).mean().sqrt().item()
            return {
                "key_rmse": key_rmse,
                "value_rmse": value_rmse,
                "tokens_retained_ratio": kept_key.shape[2] / max(key.shape[2], 1),
                "tokens_retained": float(kept_key.shape[2]),
                "tokens_dropped": float(key.shape[2] - kept_key.shape[2]),
            }
        finally:
            self._layer_state.clear()
            if saved is not None:
                self._layer_state[layer] = saved

    def attention_fidelity(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        head_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        layer: int = 0,
    ) -> tuple[float, float, float, float]:
        """Compare QK^T on full keys vs stage-1 selected keys (aligned positions)."""
        import math

        import torch.nn.functional as F

        from eval.attention_score_error import attention_scores, expand_kv_heads

        saved = self._layer_state.get(layer)
        self.reset_state()
        try:
            _, kept_key, _, kept_global, _ = self.apply_stage1(key, value, layer=layer)
            key_exp = expand_kv_heads(key, num_q_heads, num_kv_heads)
            kept_exp = expand_kv_heads(kept_key, num_q_heads, num_kv_heads)
            scores_fp = attention_scores(query, key_exp, head_dim)
            scores_kept = attention_scores(query, kept_exp, head_dim)
            projected = torch.zeros_like(scores_fp)
            projected[..., kept_global.to(scores_fp.device)] = scores_kept
            diff = scores_fp.float() - projected.float()
            mse = diff.pow(2).mean().item()
            rmse = math.sqrt(mse)
            cosine = F.cosine_similarity(scores_fp.flatten(), projected.flatten(), dim=0).item()
            max_error = diff.abs().max().item()
            return mse, rmse, cosine, max_error
        finally:
            self._layer_state.clear()
            if saved is not None:
                self._layer_state[layer] = saved
