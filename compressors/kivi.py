"""KIVI compressor: asymmetric INT2 quantization."""

from __future__ import annotations

import torch

from compressors.base import KVCompressor


class KIVICompressor(KVCompressor):
    """KIVI baseline placeholder."""

    name = "kivi"

    def __init__(self, bitwidth: int = 2) -> None:
        self.bitwidth = bitwidth

    def compress_kv(self, tensor: torch.Tensor, layer: int = 0, mode: str = "key"):
        raise NotImplementedError("KIVI compressor pending implementation.")

    def decompress_kv(self, payload: object, mode: str = "key") -> torch.Tensor:
        raise NotImplementedError("KIVI decompress pending implementation.")
