"""QJL compressor: random projection + 1-bit quantization."""

from __future__ import annotations

import torch

from compressors.base import KVCompressor


class QJLCompressor(KVCompressor):
    """QJL baseline placeholder."""

    name = "qjl"

    def __init__(self, bitwidth: int = 1) -> None:
        self.bitwidth = bitwidth

    def compress_kv(self, tensor: torch.Tensor, layer: int = 0, mode: str = "key"):
        raise NotImplementedError("QJL compressor pending implementation.")

    def decompress_kv(self, payload: object, mode: str = "key") -> torch.Tensor:
        raise NotImplementedError("QJL decompress pending implementation.")
