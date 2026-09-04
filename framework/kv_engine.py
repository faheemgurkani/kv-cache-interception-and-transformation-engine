"""KV-cache interception engine — compresses KV flow between transformer steps.

Phase 6 contract: this engine is the fixed decode path. Model weights, input
tokens, attention implementation (eager), and incremental no-recompression
semantics are shared across runs; only the bound ``KVCompressor`` plug-in varies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from compressors.base import CompressedKV, KVCompressor
from framework.kv_cache import (
    append_decompressed_tokens,
    build_incremental_layer,
    compress_token_slice,
    decompress_cache,
    decompress_to_legacy_cache,
    incremental_seq_length,
    is_incremental_compressed,
    iter_layer_kv,
)
from quantizers.rocketkv import RocketKVLayerPayload
from quantizers.snapkv import SnapKVLayerPayload


@dataclass
class CompressedCache:
    """Full-model compressed KV state (one entry per layer)."""

    layers: list[CompressedKV] = field(default_factory=list)

    @property
    def nbytes(self) -> int:
        return sum(layer.nbytes for layer in self.layers)

    @property
    def seq_length(self) -> int:
        return incremental_seq_length(self.layers)


class KVCacheEngine:
    """
    Intercepts past_key_values after each forward pass, runs the plug-in
    compressor, and decompresses before the next step.

    This is the controlled interception point (Phase 6): every compressor shares
    the same incremental decode loop; swap ``compressor`` only.

    Online mode stores **incremental** compressed payloads: each token's K/V is
    compressed once when it is produced and never re-compressed on later steps.
    """

    def __init__(self, model, compressor: KVCompressor) -> None:
        self.model = model
        self.compressor = compressor
        self.compressed_cache: CompressedCache | None = None
        self._last_full_cache = None
        self._decode_prefix: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        self._decode_prefix_len: int = 0
        from framework.attention_patches import ensure_vanilla_attention

        try:
            ensure_vanilla_attention(model)
        except AttributeError:
            pass
        if getattr(compressor, "name", "") == "rocketkv":
            from framework.rocketkv_online import enable_rocketkv_online

            enable_rocketkv_online(model, compressor)
        elif getattr(compressor, "name", "") == "snapkv":
            from framework.snapkv_online import enable_snapkv_online

            enable_snapkv_online(model, compressor)
        elif getattr(compressor, "name", "") == "qjl":
            from framework.qjl_online import enable_qjl_online

            enable_qjl_online(model, compressor)
        elif getattr(compressor, "name", "") == "palu":
            from framework.palu_online import enable_palu_online

            enable_palu_online(model, compressor)

    def _compress_new_tokens(
        self,
        past_key_values,
        prev_seq: int,
        prior_layers: list[CompressedKV] | None,
    ) -> list[CompressedKV]:
        """Compress only newly appended token positions (incremental append)."""
        new_layers: list[CompressedKV] = []
        for layer_idx, (key, value) in enumerate(iter_layer_kv(past_key_values)):
            total_seq = key.shape[2]
            if prior_layers is None:
                key_payloads: list[object] = []
                value_payloads: list[object] = []
                start = 0
            else:
                prior = prior_layers[layer_idx]
                key_payloads = list(prior.keys)  # type: ignore[arg-type]
                value_payloads = list(prior.values)  # type: ignore[arg-type]
                start = prev_seq

            remaining = total_seq - start
            splitter = getattr(self.compressor, "split_seq_payload", None)
            if remaining > 1 and splitter is not None:
                key_full = self.compressor.compress_kv(key[:, :, start:total_seq, :], layer_idx, "key")
                value_full = self.compressor.compress_kv(
                    value[:, :, start:total_seq, :], layer_idx, "value"
                )
                key_payloads.extend(splitter(key_full))
                value_payloads.extend(splitter(value_full))
            else:
                for token_idx in range(start, total_seq):
                    key_payload, value_payload = compress_token_slice(
                        key, value, token_idx, layer_idx, self.compressor
                    )
                    key_payloads.append(key_payload)
                    value_payloads.append(value_payload)

            new_layers.append(
                build_incremental_layer(
                    key,
                    value,
                    key_payloads,
                    value_payloads,
                    layer_idx,
                    self.compressor,
                )
            )
        return new_layers

    def _invalidate_decode_prefix(self) -> None:
        self._decode_prefix = None
        self._decode_prefix_len = 0

    def _refresh_decode_prefix(self, new_layers: list[CompressedKV], prev_seq: int) -> None:
        if not new_layers or not is_incremental_compressed(new_layers[0]):
            self._invalidate_decode_prefix()
            return
        new_seq = incremental_seq_length(new_layers)
        if (
            self._decode_prefix is not None
            and self._decode_prefix_len == prev_seq
            and new_seq >= prev_seq
        ):
            self._decode_prefix = append_decompressed_tokens(
                self._decode_prefix,
                new_layers,
                prev_seq,
                self.compressor,
            )
            self._decode_prefix_len = new_seq
            return
        self._decode_prefix = decompress_cache(new_layers, self.compressor)
        self._decode_prefix_len = new_seq

    def _legacy_cache_from_prefix(self, cache: CompressedCache, device: torch.device):
        if (
            self._decode_prefix is None
            or self._decode_prefix_len == 0
            or cache.seq_length < self._decode_prefix_len
        ):
            self._invalidate_decode_prefix()
            return decompress_to_legacy_cache(
                cache.layers,
                self.compressor,
                self.model.config,
                device=device,
                template_cache=self._last_full_cache,
            )
        if cache.seq_length > self._decode_prefix_len:
            self._decode_prefix = append_decompressed_tokens(
                self._decode_prefix,
                cache.layers,
                self._decode_prefix_len,
                self.compressor,
            )
            self._decode_prefix_len = cache.seq_length
        from framework.kv_cache import merge_decompressed_kv_into_cache
        from framework.state_interface import hybrid_layer_detected

        if self._last_full_cache is not None and hybrid_layer_detected(self._last_full_cache):
            return merge_decompressed_kv_into_cache(self._last_full_cache, self._decode_prefix)
        try:
            from transformers.cache_utils import DynamicCache

            # Do NOT use DynamicCache(ddp_cache_data=...). In current transformers that
            # constructor calls layer.update() which torch.cats onto existing tensors and
            # can balloon GPU memory (A10G OOM at ctx≥256 for identity/TQ/QJL).
            if hasattr(DynamicCache, "from_legacy_cache"):
                return DynamicCache.from_legacy_cache(tuple(self._decode_prefix))
            legacy = DynamicCache()
            for layer_idx, (key_states, value_states) in enumerate(self._decode_prefix):
                legacy.update(key_states, value_states, layer_idx)
            return legacy
        except (ImportError, TypeError, AttributeError, ValueError):
            return tuple(self._decode_prefix)

    @torch.no_grad()
    def step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        compressed_cache: CompressedCache | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, CompressedCache]:
        cache = compressed_cache or self.compressed_cache
        prev_seq = cache.seq_length if cache is not None else 0
        if cache is None or cache.seq_length < self._decode_prefix_len:
            self._invalidate_decode_prefix()

        if attention_mask is None:
            attention_mask = torch.ones(
                input_ids.shape[0],
                prev_seq + input_ids.shape[1],
                device=input_ids.device,
                dtype=torch.long,
            )

        past_kv = None
        if cache is not None and cache.layers:
            if getattr(self.compressor, "name", "") == "rocketkv":
                for layer_idx, layer in enumerate(cache.layers):
                    payload = layer.keys
                    if isinstance(payload, RocketKVLayerPayload):
                        self.compressor.restore_state_from_payload(layer_idx, payload)  # type: ignore[attr-defined]
            elif getattr(self.compressor, "name", "") == "qjl":
                self.compressor.sync_key_payloads_from_cache(cache.layers)  # type: ignore[attr-defined]
            past_kv = self._legacy_cache_from_prefix(cache, input_ids.device)
        elif getattr(self.compressor, "name", "") == "qjl" and hasattr(self.compressor, "reset_state"):
            self.compressor.reset_state()  # type: ignore[attr-defined]
            self._invalidate_decode_prefix()

        forward_mask = attention_mask
        _uses_layer_types = bool(getattr(self.model.config, "layer_types", None))
        _needs_native_mask = getattr(self.compressor, "name", "") in {"rocketkv", "qjl"}
        if _uses_layer_types and _needs_native_mask:
            # Gemma3 builds per-layer sliding/full masks internally; a flat mask
            # desynchronizes once RocketKV sparsifies keys during decode.
            forward_mask = None

        outputs = self.model(
            input_ids,
            attention_mask=forward_mask,
            past_key_values=past_kv,
            position_ids=position_ids,
            use_cache=True,
        )

        prior_layers = cache.layers if cache is not None else None
        if getattr(self.compressor, "name", "") == "rocketkv":
            new_layers: list[CompressedKV] = []
            for layer_idx, (key, value) in enumerate(iter_layer_kv(outputs.past_key_values)):
                prior_payload = None
                if prior_layers is not None:
                    prior = prior_layers[layer_idx].keys
                    if isinstance(prior, RocketKVLayerPayload):
                        prior_payload = prior
                orig_len = key.shape[2]
                if prior_payload is not None and prior_payload.selected_indices.numel():
                    orig_len = max(orig_len, int(prior_payload.selected_indices.max().item()) + 1)
                new_layers.append(
                    self.compressor.compress_layer_from_kv(  # type: ignore[attr-defined]
                        key,
                        value,
                        layer_idx,
                        original_seq_len=orig_len,
                        prior_payload=prior_payload,
                    )
                )
            new_cache = CompressedCache(layers=new_layers)
            self.compressed_cache = new_cache
            self._last_full_cache = outputs.past_key_values
            return outputs.logits, new_cache

        if getattr(self.compressor, "name", "") == "snapkv":
            new_layers: list[CompressedKV] = []
            logical_seq = prev_seq + input_ids.shape[1]
            for layer_idx, (key, value) in enumerate(iter_layer_kv(outputs.past_key_values)):
                new_layers.append(
                    self.compressor.wrap_layer_from_kv(  # type: ignore[attr-defined]
                        key,
                        value,
                        layer_idx,
                        original_seq_len=logical_seq,
                    )
                )
            new_cache = CompressedCache(layers=new_layers)
            self.compressed_cache = new_cache
            self._last_full_cache = outputs.past_key_values
            return outputs.logits, new_cache

        if getattr(self.compressor, "name", "") == "palu":
            new_layers: list[CompressedKV] = []
            for layer_idx, (key, value) in enumerate(iter_layer_kv(outputs.past_key_values)):
                new_layers.append(self.compressor.compress(key, value, layer=layer_idx))
            new_cache = CompressedCache(layers=new_layers)
            self.compressed_cache = new_cache
            self._last_full_cache = outputs.past_key_values
            return outputs.logits, new_cache

        new_layers = self._compress_new_tokens(outputs.past_key_values, prev_seq, prior_layers)
        new_cache = CompressedCache(layers=new_layers)
        self.compressed_cache = new_cache
        # Only hybrid models need the raw HF cache retained for state merge.
        from framework.state_interface import hybrid_layer_detected

        if hybrid_layer_detected(outputs.past_key_values):
            self._last_full_cache = outputs.past_key_values
        else:
            self._last_full_cache = None
        self._refresh_decode_prefix(new_layers, prev_seq)
        return outputs.logits, new_cache

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Manual greedy loop with KV compression on every step."""
        generated = input_ids
        attn = attention_mask
        cache: CompressedCache | None = None
        self._invalidate_decode_prefix()

        for _ in range(max_new_tokens):
            logits, cache = self.step(generated if cache is None else generated[:, -1:], attn, cache)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
            if attn is not None:
                attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

        return generated

    def compress_existing_cache(self, past_key_values) -> CompressedCache:
        """Compress a full KV snapshot incrementally (one payload per token)."""
        layers = self._compress_new_tokens(past_key_values, prev_seq=0, prior_layers=None)
        self.compressed_cache = CompressedCache(layers=layers)
        return self.compressed_cache
