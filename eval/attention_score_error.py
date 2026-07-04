"""Attention-score preservation metrics (QK^T fidelity, offline)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from compressors.base import KVCompressor
from framework.kv_cache import iter_layer_kv
from framework.model import ModelLayer


@dataclass
class LayerAttentionMetrics:
    layer: int
    mse: float
    rmse: float
    cosine_similarity: float
    max_error: float


@dataclass
class AttentionMetrics:
    """Aggregate attention-score distortion after KV compression."""

    mse: float
    rmse: float
    cosine_similarity: float
    max_error: float
    per_layer: list[LayerAttentionMetrics]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["per_layer"] = [asdict(item) for item in self.per_layer]
        return payload


def expand_kv_heads(key: torch.Tensor, num_q_heads: int, num_kv_heads: int) -> torch.Tensor:
    """Repeat KV heads to match query head count (GQA)."""
    if num_q_heads == num_kv_heads:
        return key
    repeats = num_q_heads // num_kv_heads
    return key.repeat_interleave(repeats, dim=1)


def attention_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """Scaled dot-product scores: QK^T / sqrt(d). Shapes (B, H, Tq, Tk)."""
    return torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)


def _score_distortion(scores_fp: torch.Tensor, scores_quant: torch.Tensor) -> tuple[float, float, float, float]:
    diff = scores_fp.float() - scores_quant.float()
    mse = diff.pow(2).mean().item()
    rmse = math.sqrt(mse)
    cosine = F.cosine_similarity(scores_fp.flatten(), scores_quant.flatten(), dim=0).item()
    max_error = diff.abs().max().item()
    return mse, rmse, cosine, max_error


def _compute_layer_queries(
    model_layer: ModelLayer,
    layer_idx: int,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Recompute RoPE-applied query states for one decoder layer."""
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    layer = model_layer.model.model.layers[layer_idx]
    attn = layer.self_attn
    config = model_layer.config

    normed = layer.input_layernorm(hidden_states)
    batch, seq_len, _ = normed.shape
    head_dim = config.head_dim
    num_q_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads

    query = attn.q_proj(normed).view(batch, seq_len, num_q_heads, head_dim).transpose(1, 2)
    key = attn.k_proj(normed).view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    cos, sin = position_embeddings
    query, _ = apply_rotary_pos_emb(query, key, cos, sin)
    return query


@torch.no_grad()
def evaluate_attention_fidelity(
    model_layer: ModelLayer,
    input_ids: torch.Tensor,
    compressor: KVCompressor,
) -> AttentionMetrics:
    """
    Offline inner-product preservation: compare QK^T before and after K compression.

    Uses float queries from a reference forward pass and keys from past_key_values
    (the tensors the compressor actually stores).
    """
    model = model_layer.model
    device = model_layer.device
    input_ids = input_ids.to(device)

    outputs = model(input_ids, use_cache=True, output_hidden_states=True, return_dict=True)
    hidden_states = outputs.hidden_states
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    position_embeddings = model.model.rotary_emb(hidden_states[0], position_ids)

    config = model_layer.config
    head_dim = config.head_dim
    num_q_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads

    per_layer: list[LayerAttentionMetrics] = []
    mse_sum = 0.0
    rmse_sum = 0.0
    cosine_sum = 0.0
    max_error = 0.0
    layers = 0

    for layer_idx, (key, _value) in enumerate(iter_layer_kv(outputs.past_key_values)):
        query = _compute_layer_queries(model_layer, layer_idx, hidden_states[layer_idx], position_embeddings)
        key_exp = expand_kv_heads(key, num_q_heads, num_kv_heads)

        key_payload = compressor.compress_kv(key, layer=layer_idx, mode="key")

        scores_fp = attention_scores(query, key_exp, head_dim)

        if hasattr(compressor, "estimate_attention_scores"):
            scores_quant = compressor.estimate_attention_scores(query, key_payload, head_dim)
        else:
            key_hat = compressor.decompress_kv(key_payload, mode="key").to(device=query.device)
            key_hat_exp = expand_kv_heads(key_hat, num_q_heads, num_kv_heads)
            scores_quant = attention_scores(query, key_hat_exp, head_dim)

        mse, rmse, cosine, layer_max = _score_distortion(scores_fp, scores_quant)
        per_layer.append(
            LayerAttentionMetrics(
                layer=layer_idx,
                mse=mse,
                rmse=rmse,
                cosine_similarity=cosine,
                max_error=layer_max,
            )
        )
        mse_sum += mse
        rmse_sum += rmse
        cosine_sum += cosine
        max_error = max(max_error, layer_max)
        layers += 1

    if layers == 0:
        raise RuntimeError("No KV layers available for attention fidelity evaluation.")

    return AttentionMetrics(
        mse=mse_sum / layers,
        rmse=rmse_sum / layers,
        cosine_similarity=cosine_sum / layers,
        max_error=max_error,
        per_layer=per_layer,
    )
