"""RocketKV compressor: token selection + eviction (no vector quantization)."""

from __future__ import annotations

import torch

from compressors.base import CompressedKV, KVCompressor
from quantizers.rocketkv import HybridSparseAttention, RocketKVLayerPayload, TokenSelector


class RocketKVCompressor(KVCompressor):
    """
    RocketKV plug-in: drop tokens instead of quantizing vectors.

    Stage 1 — permanent filtering via ``TokenSelector`` (SnapKV-inspired).
    Stage 2 — dynamic top-k via ``HybridSparseAttention`` at decode time.

    ``compress()`` / ``decompress()`` operate on full layer tensors.
    Per-token ``compress_kv()`` stores raw slices for incremental engine
    compatibility; selection is applied when decompressing full layers.
    """

    name = "rocketkv"

    def __init__(
        self,
        bitwidth: int = 16,
        keep_ratio: float = 0.5,
        window_size: int = 32,
        dynamic_top_k: int = 64,
    ) -> None:
        self.bitwidth = bitwidth
        self.keep_ratio = keep_ratio
        self.token_selector = TokenSelector(
            keep_ratio=keep_ratio,
            window_size=window_size,
        )
        self.hsa = HybridSparseAttention(dynamic_top_k=dynamic_top_k)
        self._permanent_indices: dict[int, torch.Tensor] = {}

    def compress_kv(
        self,
        tensor: torch.Tensor,
        layer: int = 0,
        mode: str = "key",
    ) -> torch.Tensor:
        return tensor.detach().clone()

    def decompress_kv(self, payload: object, mode: str = "key") -> torch.Tensor:
        if not isinstance(payload, torch.Tensor):
            raise TypeError(f"Expected raw tensor payload, got {type(payload)}")
        return payload

    def compress(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer: int = 0,
    ) -> CompressedKV:
        indices, kept_key, kept_value = self.token_selector.select(key, value)
        self._permanent_indices[layer] = indices.detach().cpu()

        payload = RocketKVLayerPayload(
            selected_indices=indices.detach().cpu(),
            keys=kept_key.detach().cpu(),
            values=kept_value.detach().cpu(),
            original_seq_len=key.shape[2],
        )
        return CompressedKV(
            keys=payload,
            values=payload,
            original_shape=tuple(key.shape),
            nbytes=payload.nbytes,
            bitwidth=self.bitwidth,
            layer=layer,
        )

    def decompress_incremental_layer(
        self,
        compressed: CompressedKV,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Concatenate per-token payloads; sparsity is applied in the online attention hook."""
        key_parts = [self.decompress_kv(item, mode="key") for item in compressed.keys]  # type: ignore[union-attr]
        value_parts = [self.decompress_kv(item, mode="value") for item in compressed.values]  # type: ignore[union-attr]
        return torch.cat(key_parts, dim=2), torch.cat(value_parts, dim=2)

    def decompress(self, compressed: CompressedKV) -> tuple[torch.Tensor, torch.Tensor]:
        payload = compressed.keys
        if isinstance(payload, list):
            key_parts = [self.decompress_kv(item, mode="key") for item in payload]
            value_parts = [self.decompress_kv(item, mode="value") for item in compressed.values]
            key = torch.cat(key_parts, dim=2)
            value = torch.cat(value_parts, dim=2)
            _, kept_key, kept_value = self.token_selector.select(key, value)
            return kept_key, kept_value

        if isinstance(payload, RocketKVLayerPayload):
            return payload.keys, payload.values

        key = self.decompress_kv(compressed.keys, mode="key")
        value = self.decompress_kv(compressed.values, mode="value")
        return key, value

    def select_dynamic_tokens(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stage 2 HSA: dynamic top-k selection for the current decode step."""
        permanent = self._permanent_indices.get(layer)
        return self.hsa.select_top_k(query, key, value, permanent_indices=permanent)
