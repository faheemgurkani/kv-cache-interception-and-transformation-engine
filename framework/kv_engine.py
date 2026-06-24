"""KV-cache interception engine — compresses KV flow between transformer steps."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from compressors.base import CompressedKV, KVCompressor
from framework.kv_cache import compress_past_key_values, decompress_to_legacy_cache, iter_layer_kv


@dataclass
class CompressedCache:
    """Full-model compressed KV state (one entry per layer)."""

    layers: list[CompressedKV] = field(default_factory=list)

    @property
    def nbytes(self) -> int:
        return sum(layer.nbytes for layer in self.layers)


class KVCacheEngine:
    """
    Intercepts past_key_values after each forward pass, runs the plug-in
    compressor, and decompresses before the next step.
    """

    def __init__(self, model, compressor: KVCompressor) -> None:
        self.model = model
        self.compressor = compressor
        self.compressed_cache: CompressedCache | None = None

    @torch.no_grad()
    def step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        compressed_cache: CompressedCache | None = None,
    ) -> tuple[torch.Tensor, CompressedCache]:
        past_kv = None
        cache = compressed_cache or self.compressed_cache
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

        new_layers: list[CompressedKV] = []
        for layer_idx, (key, value) in enumerate(iter_layer_kv(outputs.past_key_values)):
            new_layers.append(self.compressor.compress(key, value, layer=layer_idx))

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
        layers = compress_past_key_values(past_key_values, self.compressor)
        self.compressed_cache = CompressedCache(layers=layers)
        return self.compressed_cache
