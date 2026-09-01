"""SnapKV compressor: prefill-only token eviction (Category A)."""

from __future__ import annotations

import torch

from compressors.base import CompressedKV, KVCompressor, OfflineCostMetadata
from compressors.taxonomy import METHOD_TAXONOMY
from quantizers.snapkv import SnapKVLayerPayload, snap_kv


class SnapKVCompressor(KVCompressor):
    """SnapKV plug-in — observation-window voting, pooled top-k, standard attention."""

    name = "snapkv"
    bitwidth = 16

    def __init__(
        self,
        max_capacity_prompt: int = 1024,
        window_size: int = 16,
        kernel_size: int = 5,
    ) -> None:
        self.max_capacity_prompt = int(max_capacity_prompt)
        self.window_size = int(window_size)
        self.kernel_size = int(kernel_size)
        self._prefill_done: dict[int, bool] = {}

    @property
    def taxonomy(self):
        return METHOD_TAXONOMY[self.name]

    def reset_state(self) -> None:
        self._prefill_done.clear()

    def theoretical_compression_ratio(self, *, context_length: int | None = None) -> float | None:
        if context_length is None or context_length <= 0:
            return None
        if context_length < self.max_capacity_prompt:
            return 1.0
        return context_length / self.max_capacity_prompt

    def offline_cost_metadata(self) -> OfflineCostMetadata:
        return OfflineCostMetadata(calibration_required=False)

    def compress_kv(
        self,
        tensor: torch.Tensor,
        layer: int = 0,
        mode: str = "key",
    ) -> object:
        return tensor.detach().clone()

    def decompress_kv(self, payload: object, mode: str = "key") -> torch.Tensor:
        if isinstance(payload, SnapKVLayerPayload):
            return payload.keys if mode == "key" else payload.values
        if not isinstance(payload, torch.Tensor):
            raise TypeError(f"Expected tensor or SnapKVLayerPayload, got {type(payload)}")
        return payload

    def compress(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer: int = 0,
        query_states: torch.Tensor | None = None,
    ) -> CompressedKV:
        """Offline/batch compress. Uses key observation window as query proxy when needed."""
        seq_len = key.shape[2]
        if query_states is None:
            query_states = key
        if query_states.shape[1] != key.shape[1]:
            # GQA: SnapKV voting is defined per KV head; use key-as-query proxy.
            query_states = key
        if seq_len >= self.max_capacity_prompt:
            key, value = snap_kv(
                query_states,
                key,
                value,
                window_size=self.window_size,
                max_capacity_prompt=self.max_capacity_prompt,
                kernel_size=self.kernel_size,
            )
        payload = SnapKVLayerPayload(
            keys=key.detach().cpu(),
            values=value.detach().cpu(),
            original_seq_len=seq_len,
            compressed=seq_len >= self.max_capacity_prompt,
        )
        return CompressedKV(
            keys=payload,
            values=payload,
            original_shape=tuple(key.shape),
            nbytes=payload.nbytes,
            bitwidth=self.bitwidth,
            layer=layer,
        )

    def decompress(self, compressed: CompressedKV) -> tuple[torch.Tensor, torch.Tensor]:
        payload = compressed.keys
        if isinstance(payload, SnapKVLayerPayload):
            return payload.keys, payload.values
        return self.decompress_kv(compressed.keys, mode="key"), self.decompress_kv(compressed.values, mode="value")

    def wrap_layer_from_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer: int,
        *,
        original_seq_len: int | None = None,
    ) -> CompressedKV:
        """Package runtime K/V after optional online prefill compression."""
        orig_len = original_seq_len or key.shape[2]
        payload = SnapKVLayerPayload(
            keys=key.detach().cpu(),
            values=value.detach().cpu(),
            original_seq_len=orig_len,
            compressed=key.shape[2] < orig_len or orig_len >= self.max_capacity_prompt,
        )
        self._prefill_done[layer] = payload.compressed
        return CompressedKV(
            keys=payload,
            values=payload,
            original_shape=(key.shape[0], key.shape[1], orig_len, key.shape[3]),
            nbytes=payload.nbytes,
            bitwidth=self.bitwidth,
            layer=layer,
        )

    def reconstruction_error(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer: int = 0,
    ) -> dict[str, float]:
        compressed = self.compress(key, value, layer=layer)
        k2, v2 = self.decompress(compressed)
        key_rmse = (k2.float() - k2.float()).pow(2).mean().sqrt().item()
        value_rmse = (v2.float() - v2.float()).pow(2).mean().sqrt().item()
        return {
            "key_rmse": key_rmse,
            "value_rmse": value_rmse,
            "tokens_retained_ratio": k2.shape[2] / max(key.shape[2], 1),
            "tokens_retained": float(k2.shape[2]),
            "tokens_dropped": float(max(key.shape[2] - k2.shape[2], 0)),
        }

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
        import math

        import torch.nn.functional as F

        from eval.fidelity.attention import attention_scores, expand_kv_heads

        saved = dict(self._prefill_done)
        self.reset_state()
        try:
            compressed = self.compress(key, value, layer=layer, query_states=query)
            kept_key, kept_value = self.decompress(compressed)
            kept_key = kept_key.to(key.device)
            kept_value = kept_value.to(value.device)
            key_exp = expand_kv_heads(key, num_q_heads, num_kv_heads)
            kept_exp = expand_kv_heads(kept_key, num_q_heads, num_kv_heads)
            scores_fp = attention_scores(query, key_exp, head_dim)
            scores_kept = attention_scores(query, kept_exp, head_dim)
            min_len = min(scores_fp.shape[-1], scores_kept.shape[-1])
            diff = scores_fp[..., :min_len].float() - scores_kept[..., :min_len].float()
            mse = diff.pow(2).mean().item()
            rmse = math.sqrt(mse)
            cosine = F.cosine_similarity(
                scores_fp[..., :min_len].float().flatten(),
                scores_kept[..., :min_len].float().flatten(),
                dim=0,
            ).item()
            cosine = max(-1.0, min(1.0, float(cosine)))
            max_error = diff.abs().max().item()
            return mse, rmse, cosine, max_error
        finally:
            self._prefill_done = saved
