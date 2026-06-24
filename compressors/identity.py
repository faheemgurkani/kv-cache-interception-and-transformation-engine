"""Identity (no-compression) baseline compressor."""

from __future__ import annotations

import torch

from compressors.base import CompressedKV, KVCompressor


class IdentityCompressor(KVCompressor):
    """Passthrough compressor used to validate the evaluation pipeline."""

    name = "identity"
    bitwidth = 16

    def compress_kv(self, tensor: torch.Tensor, layer: int = 0, mode: str = "key") -> torch.Tensor:
        return tensor.detach().clone()

    def decompress_kv(self, payload: object, mode: str = "key") -> torch.Tensor:
        return payload  # type: ignore[return-value]

    def compress(self, key: torch.Tensor, value: torch.Tensor, layer: int = 0) -> CompressedKV:
        key_copy = key.detach().clone()
        value_copy = value.detach().clone()
        nbytes = key_copy.numel() * key_copy.element_size() + value_copy.numel() * value_copy.element_size()
        return CompressedKV(
            keys=key_copy,
            values=value_copy,
            original_shape=tuple(key.shape),
            nbytes=nbytes,
            bitwidth=self.bitwidth,
            layer=layer,
        )
