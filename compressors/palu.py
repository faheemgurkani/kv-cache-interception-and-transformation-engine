"""Palu compressor: G-LRD low-rank latent KV cache (Category C + E)."""

from __future__ import annotations

import torch

from compressors.base import CompressedKV, KVCompressor, OfflineCostMetadata
from compressors.taxonomy import METHOD_TAXONOMY
from quantizers.palu import (
    PaluLatentPayload,
    PaluLayerFactors,
    build_layer_factors_from_projections,
    compress_kv_lowrank,
    decompress_kv_lowrank,
    reconstruct_kv_from_latent,
)


class PaluCompressor(KVCompressor):
    """Palu plug-in — group-head low-rank latent cache with RoPE-aware online path."""

    name = "palu"
    bitwidth = 16

    def __init__(
        self,
        compression_rate: float = 0.5,
        group_size: int = 4,
        calibration_samples: int = 2048,
        calibration_seq_len: int = 1024,
    ) -> None:
        self.compression_rate = float(compression_rate)
        self.group_size = int(group_size)
        self.calibration_samples = int(calibration_samples)
        self.calibration_seq_len = int(calibration_seq_len)
        self._layer_factors: dict[int, PaluLayerFactors] = {}
        self._model_bound = False

    @property
    def taxonomy(self):
        return METHOD_TAXONOMY[self.name]

    def reset_state(self) -> None:
        return

    @property
    def uses_weight_factors(self) -> bool:
        return self._model_bound and bool(self._layer_factors)

    def bind_model(self, model) -> None:
        """Offline G-LRD decomposition of k_proj / v_proj weights (once per model)."""
        if self._model_bound:
            return
        for layer_idx, layer in enumerate(model.model.layers):
            attn = layer.self_attn
            k_weight = attn.k_proj.weight.detach().cpu()
            v_weight = attn.v_proj.weight.detach().cpu()
            num_kv_heads = getattr(attn, "num_key_value_heads", k_weight.shape[0] // 128)
            head_dim = k_weight.shape[0] // max(num_kv_heads, 1)
            self._layer_factors[layer_idx] = build_layer_factors_from_projections(
                k_weight,
                v_weight,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                group_size=self.group_size,
                compression_rate=self.compression_rate,
            )
        self._model_bound = True

    def layer_factors(self, layer: int) -> PaluLayerFactors | None:
        return self._layer_factors.get(layer)

    def effective_rank(self, layer: int = 0, *, seq_len: int | None = None, head_dim: int | None = None) -> int:
        factors = self._layer_factors.get(layer)
        if factors and factors.groups:
            return factors.groups[0].rank
        base = max(1, int((head_dim or 128) * self.compression_rate))
        if seq_len is not None:
            return max(1, min(base, seq_len))
        return base

    def theoretical_compression_ratio(self, *, context_length: int | None = None) -> float | None:
        if self.compression_rate <= 0:
            return None
        return 1.0 / self.compression_rate

    def offline_cost_metadata(self) -> OfflineCostMetadata:
        return OfflineCostMetadata(
            calibration_required=True,
            calibration_dataset="wikitext-2",
            calibration_tokens=self.calibration_samples * self.calibration_seq_len,
        )

    def shared_storage_bytes(self) -> int:
        total = 0
        for factors in self._layer_factors.values():
            for group in factors.groups:
                for tensor in (group.a_key, group.b_key, group.a_value, group.b_value):
                    total += tensor.numel() * tensor.element_size()
        return total

    def compress_kv(
        self,
        tensor: torch.Tensor,
        layer: int = 0,
        mode: str = "key",
    ) -> object:
        return tensor.detach().clone()

    def decompress_kv(self, payload: object, mode: str = "key") -> torch.Tensor:
        if isinstance(payload, PaluLatentPayload):
            key, value = decompress_kv_lowrank(payload)
            return key if mode == "key" else value
        if not isinstance(payload, torch.Tensor):
            raise TypeError(f"Expected tensor or PaluLatentPayload, got {type(payload)}")
        return payload

    def compress(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer: int = 0,
    ) -> CompressedKV:
        rank = self.effective_rank(layer, seq_len=key.shape[2], head_dim=key.shape[3])
        payload = compress_kv_lowrank(key, value, rank=rank)
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
        if isinstance(payload, PaluLatentPayload):
            if payload.b_key is not None:
                return decompress_kv_lowrank(payload)
            factors = self._layer_factors.get(compressed.layer)
            if factors is not None:
                return reconstruct_kv_from_latent(
                    payload.h_key.to(torch.float32),
                    payload.h_value.to(torch.float32),
                    factors,
                )
            return decompress_kv_lowrank(payload)
        return self.decompress_kv(compressed.keys, mode="key"), self.decompress_kv(compressed.values, mode="value")

    def wrap_latent_layer(
        self,
        h_key: torch.Tensor,
        h_value: torch.Tensor,
        layer: int,
        *,
        original_seq_len: int,
    ) -> CompressedKV:
        payload = PaluLatentPayload(
            h_key=h_key.detach().cpu(),
            h_value=h_value.detach().cpu(),
            original_seq_len=original_seq_len,
            rank=h_key.shape[-1],
            group_size=h_key.shape[1],
        )
        return CompressedKV(
            keys=payload,
            values=payload,
            original_shape=(h_key.shape[0], h_key.shape[1], original_seq_len, 0),
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
        key_rmse = (k2.float() - key.float()).pow(2).mean().sqrt().item()
        value_rmse = (v2.float() - value.float()).pow(2).mean().sqrt().item()
        return {"key_rmse": key_rmse, "value_rmse": value_rmse}
