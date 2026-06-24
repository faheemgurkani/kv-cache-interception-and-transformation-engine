"""RocketKV compressor: token selection + eviction."""

from __future__ import annotations

import torch

from compressors.base import KVCompressor


class RocketKVCompressor(KVCompressor):
    """RocketKV baseline placeholder."""

    name = "rocketkv"

    def __init__(self, bitwidth: int = 16) -> None:
        self.bitwidth = bitwidth

    def compress_kv(self, tensor: torch.Tensor, layer: int = 0, mode: str = "key"):
        raise NotImplementedError("RocketKV compressor pending implementation.")

    def decompress_kv(self, payload: object, mode: str = "key") -> torch.Tensor:
        raise NotImplementedError("RocketKV decompress pending implementation.")
