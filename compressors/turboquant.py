"""TurboQuant KV compression: WHT → Lloyd-Max → QJL residual."""

from __future__ import annotations

import torch

from compressors.base import KVCompressor
from quantizers.turboquant_pipeline import TurboQuantPipeline, TurboQuantStage, TurboQuantTensorPayload


class TurboQuantCompressor(KVCompressor):
    """TurboQuant plug-in for the KV compression layer."""

    name = "turboquant"

    def __init__(
        self,
        bitwidth: int = 4,
        stage: TurboQuantStage | str = TurboQuantStage.FULL,
        seed: int = 42,
    ) -> None:
        self.bitwidth = bitwidth
        if isinstance(stage, str):
            stage = TurboQuantStage(stage)
        self.stage = stage
        self.pipeline = TurboQuantPipeline(bitwidth=bitwidth, stage=stage, seed=seed)

    def compress_kv(
        self,
        tensor: torch.Tensor,
        layer: int = 0,
        mode: str = "key",
    ) -> TurboQuantTensorPayload:
        use_qjl = self.stage == TurboQuantStage.FULL and mode == "value"
        return self.pipeline.compress_tensor(tensor, use_qjl=use_qjl)

    def decompress_kv(self, payload: object, mode: str = "key") -> torch.Tensor:
        if not isinstance(payload, TurboQuantTensorPayload):
            raise TypeError(f"Expected TurboQuantTensorPayload, got {type(payload)}")
        return self.pipeline.decompress_tensor(payload)

    def shared_storage_bytes(self) -> int:
        """Shared Lloyd-Max centroid table (one copy per compressor, not per layer)."""
        centroids = self.pipeline.centroids
        return centroids.numel() * centroids.element_size()

    def reconstruction_error(self, key: torch.Tensor, value: torch.Tensor) -> dict[str, float]:
        return {
            "key_rmse": self.pipeline.reconstruction_error(key, use_qjl=False),
            "value_rmse": self.pipeline.reconstruction_error(
                value,
                use_qjl=self.stage == TurboQuantStage.FULL,
            ),
        }
