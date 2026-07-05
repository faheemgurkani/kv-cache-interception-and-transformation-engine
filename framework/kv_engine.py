"""KV-cache interception engine — compresses KV flow between transformer steps."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from compressors.base import CompressedKV, KVCompressor
from framework.kv_cache import (
    build_incremental_layer,
    compress_token_slice,
    decompress_to_legacy_cache,
    incremental_seq_length,
    iter_layer_kv,
)


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

    Online mode stores **incremental** compressed payloads: each token's K/V is
    compressed once when it is produced and never re-compressed on later steps.
    """

    def __init__(self, model, compressor: KVCompressor) -> None:
        self.model = model
        self.compressor = compressor
        self.compressed_cache: CompressedCache | None = None
        if getattr(compressor, "name", "") == "rocketkv":
            from framework.rocketkv_online import enable_rocketkv_online

            enable_rocketkv_online(model, compressor)

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

    @torch.no_grad()
    def step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        compressed_cache: CompressedCache | None = None,
    ) -> tuple[torch.Tensor, CompressedCache]:
        cache = compressed_cache or self.compressed_cache
        prev_seq = cache.seq_length if cache is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(
                input_ids.shape[0],
                prev_seq + input_ids.shape[1],
                device=input_ids.device,
                dtype=torch.long,
            )

        past_kv = None
        if cache is not None and cache.layers:
            past_kv = decompress_to_legacy_cache(
                cache.layers, self.compressor, self.model.config, device=input_ids.device
            )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_kv,
            use_cache=True,
        )

        prior_layers = cache.layers if cache is not None else None
        new_layers = self._compress_new_tokens(outputs.past_key_values, prev_seq, prior_layers)
        new_cache = CompressedCache(layers=new_layers)
        self.compressed_cache = new_cache
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
