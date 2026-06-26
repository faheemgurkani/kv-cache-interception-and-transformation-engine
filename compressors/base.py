"""KVCompressor plug-in interface for the KV-cache interception engine."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class CompressedKV:
    """Compressed representation of one layer's key/value tensors."""

    keys: object
    values: object
    original_shape: tuple[int, ...]
    nbytes: int
    bitwidth: int | None = None
    layer: int = 0


class KVCompressor(ABC):
    """Paper-independent interface for KV-cache compression plug-ins."""

    name: str = "base"
    bitwidth: int = 16

    @abstractmethod
    def compress_kv(
        self,
        tensor: torch.Tensor,
        layer: int = 0,
        mode: str = "key",
    ) -> object:
        """Compress a single K or V tensor (mode is 'key' or 'value')."""

    @abstractmethod
    def decompress_kv(self, payload: object, mode: str = "key") -> torch.Tensor:
        """Restore a single K or V tensor from its compressed payload."""

    def compress(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer: int = 0,
    ) -> CompressedKV:
        key_payload = self.compress_kv(key, layer=layer, mode="key")
        value_payload = self.compress_kv(value, layer=layer, mode="value")
        nbytes = self._payload_bytes(key_payload) + self._payload_bytes(value_payload)
        return CompressedKV(
            keys=key_payload,
            values=value_payload,
            original_shape=tuple(key.shape),
            nbytes=nbytes,
            bitwidth=getattr(self, "bitwidth", None),
            layer=layer,
        )

    def decompress(self, compressed: CompressedKV) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(compressed.keys, list):
            key_parts = [self.decompress_kv(item, mode="key") for item in compressed.keys]
            value_parts = [self.decompress_kv(item, mode="value") for item in compressed.values]
            return torch.cat(key_parts, dim=2), torch.cat(value_parts, dim=2)
        key = self.decompress_kv(compressed.keys, mode="key")
        value = self.decompress_kv(compressed.values, mode="value")
        return key, value

    @staticmethod
    def _payload_bytes(payload: object) -> int:
        if hasattr(payload, "storage_bytes"):
            return int(payload.storage_bytes())
        if hasattr(payload, "nbytes"):
            return int(payload.nbytes)
        if isinstance(payload, torch.Tensor):
            return payload.numel() * payload.element_size()
        return 0

    def shared_storage_bytes(self) -> int:
        """Optional shared tables (centroids, codebooks) counted once per model run."""
        return 0

    def compression_ratio(self, key: torch.Tensor, value: torch.Tensor) -> float:
        original_bytes = key.numel() * key.element_size() + value.numel() * value.element_size()
        compressed = self.compress(key, value)
        if compressed.nbytes == 0:
            return 1.0
        return original_bytes / compressed.nbytes
