"""QJL compressor: random projection + 1-bit sign quantization on keys."""

from __future__ import annotations

import torch

from compressors.base import CompressedKV, KVCompressor
from quantizers.qjl_pipeline import QJLPipeline, QJLTensorPayload


class QJLCompressor(KVCompressor):
    """
    QJL plug-in: compress keys via sign(S @ k) + ||k||; values stored uncompressed.

    Online inference uses the asymmetric QJL attention estimator (see
    ``framework/qjl_online.py``), not reconstructed keys in standard attention.
    """

    name = "qjl"

    def __init__(self, bitwidth: int = 1, seed: int = 42, proj_dim: int | None = None) -> None:
        self.bitwidth = bitwidth
        self.pipeline = QJLPipeline(seed=seed, proj_dim=proj_dim)
        self._online_key_payloads: dict[int, list[QJLTensorPayload]] = {}

    def reset_state(self) -> None:
        self._online_key_payloads.clear()

    def online_key_payloads(self, layer: int) -> list[QJLTensorPayload]:
        return self._online_key_payloads.setdefault(layer, [])

    def sync_key_payloads_from_cache(self, layers: list[CompressedKV]) -> None:
        """Restore per-token key payloads from the incremental compressed cache."""
        self._online_key_payloads.clear()
        for layer_idx, layer in enumerate(layers):
            keys = layer.keys
            if isinstance(keys, list):
                self._online_key_payloads[layer_idx] = list(keys)  # type: ignore[arg-type]

    def compress_key_token(self, layer: int, key_slice: torch.Tensor) -> QJLTensorPayload:
        payload = self.pipeline.compress_tensor(key_slice, mode="key")
        self._online_key_payloads.setdefault(layer, []).append(payload)
        return payload

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

    def attention_fidelity(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        head_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        layer: int = 0,
    ) -> tuple[float, float, float, float]:
        """Section A: compare exact QK^T to QJL estimator on compressed keys."""
        import math

        import torch.nn.functional as F

        from eval.attention_score_error import attention_scores, expand_kv_heads

        key_exp = expand_kv_heads(key, num_q_heads, num_kv_heads)
        scores_fp = attention_scores(query, key_exp, head_dim)
        payload = self.compress_kv(key, layer=layer, mode="key")
        scores_est = self.estimate_attention_scores(query, payload, head_dim)
        diff = scores_fp.float() - scores_est.float()
        mse = diff.pow(2).mean().item()
        rmse = math.sqrt(mse)
        cosine = F.cosine_similarity(scores_fp.flatten(), scores_est.flatten(), dim=0).item()
        max_error = diff.abs().max().item()
        return mse, rmse, cosine, max_error

    def reconstruction_error(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer: int = 0,
    ) -> dict[str, float]:
        _ = layer
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
