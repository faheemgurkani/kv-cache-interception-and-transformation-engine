"""QJL compressor: random projection + 1-bit sign quantization on keys."""

from __future__ import annotations

import torch

from compressors.base import CompressedKV, KVCompressor
from quantizers.qjl_pipeline import QJLPipeline, QJLTensorPayload


class QJLCompressor(KVCompressor):
    """
    QJL plug-in: compress keys via sign(S @ k) + ||k||; values stored uncompressed.

    Attention inner products are estimated via the asymmetric QJL estimator
    (see ``estimate_attention_scores``) without reconstructing keys.
    """

    name = "qjl"

    def __init__(self, bitwidth: int = 1, seed: int = 42, proj_dim: int | None = None) -> None:
        self.bitwidth = bitwidth
        self.pipeline = QJLPipeline(seed=seed, proj_dim=proj_dim)

    def compress_kv(
        self,
        tensor: torch.Tensor,
        layer: int = 0,
        mode: str = "key",
    ) -> QJLTensorPayload:
        return self.pipeline.compress_tensor(tensor, mode=mode)

    def decompress_kv(self, payload: object, mode: str = "key") -> torch.Tensor:
        if not isinstance(payload, QJLTensorPayload):
            raise TypeError(f"Expected QJLTensorPayload, got {type(payload)}")
        return self.pipeline.decompress_tensor(payload)

    def estimate_attention_scores(
        self,
        query: torch.Tensor,
        key_payload: object,
        head_dim: int,
    ) -> torch.Tensor:
        """Estimate QK^T / sqrt(d) using the QJL asymmetric inner-product estimator."""
        num_q_heads = query.shape[1]
        if isinstance(key_payload, list):
            return self.pipeline.estimate_attention_scores(
                query, key_payload, head_dim, num_q_heads=num_q_heads
            )
        return self.pipeline.estimate_attention_scores(
            query, key_payload, head_dim, num_q_heads=num_q_heads  # type: ignore[arg-type]
        )

    def reconstruction_error(self, key: torch.Tensor, value: torch.Tensor) -> dict[str, float]:
        return {
            "key_rmse": self.pipeline.reconstruction_error(key, mode="key"),
            "value_rmse": self.pipeline.reconstruction_error(value, mode="value"),
        }

    def shared_storage_bytes(self) -> int:
        """Count regenerated projection matrices once per model run."""
        total = 0
        for projection in self.pipeline._projections.values():
            total += projection.numel() * projection.element_size()
        return total
