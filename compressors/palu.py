"""Palu compressor: G-LRD low-rank latent KV cache (Category C + E)."""

from __future__ import annotations

import time

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


def _allocator_bytes() -> int | None:
    if torch.cuda.is_available():
        return int(torch.cuda.memory_allocated())
    return None


def _iter_self_attn_modules(model):
    """Yield per-layer attention modules across nested HF causal-LM wrappers."""
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        inner = getattr(inner, "language_model", inner)
        layers = getattr(inner, "layers", None)
    if layers is None:
        raise AttributeError("Could not locate decoder layers for Palu bind_model")
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            raise AttributeError("Decoder layer has no self_attn for Palu bind_model")
        yield attn


def _kv_projection_geometry(attn, k_weight: torch.Tensor) -> tuple[int, int]:
    """Resolve (num_kv_heads, head_dim) without the unsafe ``out // 128`` fallback.

    Gemma3-270M stores ``head_dim=256`` and ``num_key_value_heads=1`` on the
    config, not as ``attn.num_key_value_heads``. Using ``out_features // 128``
    would invent two heads of dim 128 and break G-LRD.
    """
    config = getattr(attn, "config", None)
    head_dim = getattr(attn, "head_dim", None)
    if head_dim is None and config is not None:
        head_dim = getattr(config, "head_dim", None)
    num_kv_heads = getattr(attn, "num_key_value_heads", None)
    if num_kv_heads is None and config is not None:
        num_kv_heads = getattr(config, "num_key_value_heads", None)
    if head_dim is None and num_kv_heads:
        head_dim = k_weight.shape[0] // max(int(num_kv_heads), 1)
    if num_kv_heads is None and head_dim:
        num_kv_heads = k_weight.shape[0] // max(int(head_dim), 1)
    if head_dim is None or num_kv_heads is None:
        raise ValueError(
            "Cannot resolve KV head geometry for Palu bind_model "
            f"(k_proj out_features={tuple(k_weight.shape)})"
        )
    return int(num_kv_heads), int(head_dim)


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
        self._calibration_time_ms: float | None = None
        self._calibration_memory_bytes: int | None = None

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
        start = time.perf_counter()
        mem_before = _allocator_bytes()
        for layer_idx, attn in enumerate(_iter_self_attn_modules(model)):
            k_weight = attn.k_proj.weight.detach().cpu()
            v_weight = attn.v_proj.weight.detach().cpu()
            num_kv_heads, head_dim = _kv_projection_geometry(attn, k_weight)
            self._layer_factors[layer_idx] = build_layer_factors_from_projections(
                k_weight,
                v_weight,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                group_size=self.group_size,
                compression_rate=self.compression_rate,
            )
        self._calibration_time_ms = (time.perf_counter() - start) * 1000.0
        mem_after = _allocator_bytes()
        if mem_before is not None and mem_after is not None:
            self._calibration_memory_bytes = max(0, mem_after - mem_before)
        self._model_bound = True

    def fidelity_rank_capped_by_seq(self, seq_len: int, head_dim: int, layer: int = 0) -> bool:
        """True when post-hoc SVD rank >= min(seq, d) so reconstruction is algebraically exact."""
        rank = self.effective_rank(layer, seq_len=seq_len, head_dim=head_dim)
        return rank >= min(max(seq_len, 1), max(head_dim, 1))

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
            calibration_time_ms=self._calibration_time_ms,
            calibration_memory_bytes=self._calibration_memory_bytes,
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
        k2 = k2.to(key.device)
        v2 = v2.to(value.device)
        key_rmse = (k2.float() - key.float()).pow(2).mean().sqrt().item()
        value_rmse = (v2.float() - value.float()).pow(2).mean().sqrt().item()
        return {"key_rmse": key_rmse, "value_rmse": value_rmse}
