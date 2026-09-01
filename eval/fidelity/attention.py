"""FIDELITY / Attention: QK^T score preservation after KV compression."""

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
    output_rmse: float | None
    distribution_kl_divergence: float | None


@dataclass
class AttentionMetrics:
    """Aggregate attention-score distortion after KV compression.

    `output_rmse` / `distribution_kl_divergence` are averaged only over layers where
    they could be computed (some compressors expose a distortion-only hook that
    doesn't return the raw quantized score tensor needed for those two metrics).
    """

    mse: float
    rmse: float
    cosine_similarity: float
    max_error: float
    output_rmse: float | None
    distribution_kl_divergence: float | None
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


def clamp_cosine(value: float) -> float:
    """bf16/fp16 cosine can land slightly outside [-1, 1]; KPI schema forbids that."""
    if math.isnan(value):
        return 0.0
    return max(-1.0, min(1.0, float(value)))


def _score_distortion(scores_fp: torch.Tensor, scores_quant: torch.Tensor) -> tuple[float, float, float, float]:
    diff = scores_fp.float() - scores_quant.float()
    mse = diff.pow(2).mean().item()
    rmse = math.sqrt(mse)
    cosine = F.cosine_similarity(scores_fp.float().flatten(), scores_quant.float().flatten(), dim=0).item()
    cosine = clamp_cosine(cosine)
    max_error = diff.abs().max().item()
    return mse, rmse, cosine, max_error


def _attention_output_and_divergence(
    scores_fp: torch.Tensor,
    scores_quant: torch.Tensor,
    value_hat: torch.Tensor,
) -> tuple[float, float]:
    """Attention-output RMSE and KL(softmax(fp) || softmax(quant)), the two
    downstream-facing views of score distortion: what actually reaches the next
    layer (weighted V), and how much the attention distribution itself shifted."""
    probs_fp = F.softmax(scores_fp.float(), dim=-1)
    probs_quant = F.softmax(scores_quant.float(), dim=-1)

    output_fp = torch.matmul(probs_fp, value_hat.float())
    output_quant = torch.matmul(probs_quant, value_hat.float())
    output_rmse = (output_fp - output_quant).pow(2).mean().sqrt().item()

    eps = 1e-8
    kl = (probs_fp * ((probs_fp + eps).log() - (probs_quant + eps).log())).sum(dim=-1).mean().item()
    return output_rmse, kl


def _compute_layer_queries(
    model_layer: ModelLayer,
    layer_idx: int,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Recompute RoPE-applied query states for one decoder layer."""
    from framework.model_adapter import (
        load_attention_ops,
        pre_attention_hidden,
        project_attention_states,
        resolve_key_head_dim,
    )

    layer = model_layer.model.model.layers[layer_idx]
    attn = layer.self_attn
    ops = load_attention_ops(model_layer.config)

    normed = pre_attention_hidden(layer, hidden_states, ops)
    cos, sin = position_embeddings
    query, _key, _value = project_attention_states(
        attn, normed, ops, cos, sin, config=model_layer.config
    )
    return query


def _slice_score_window(
    query: torch.Tensor,
    key: torch.Tensor,
    score_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the trailing token window for QK^T fidelity to avoid O(n^2) VRAM blowup."""
    seq_len = query.shape[-2]
    if score_tokens <= 0 or seq_len <= score_tokens:
        return query, key
    return query[..., -score_tokens:, :], key[..., -score_tokens:, :]


@torch.no_grad()
def evaluate_attention_fidelity(
    model_layer: ModelLayer,
    input_ids: torch.Tensor,
    compressor: KVCompressor,
    outputs=None,
    score_tokens: int = 512,
) -> AttentionMetrics:
    """
    Offline inner-product preservation: compare QK^T before and after K compression.

    Uses float queries from a reference forward pass and keys from past_key_values
    (the tensors the compressor actually stores).
    """
    model = model_layer.model
    device = model_layer.device
    input_ids = input_ids.to(device)

    if outputs is None:
        outputs = model(input_ids, use_cache=True, output_hidden_states=True, return_dict=True)

    hidden_states = outputs.hidden_states
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    from framework.model_adapter import resolve_key_head_dim
    from framework.rope import build_rope_context

    rope_context = build_rope_context(model, hidden_states[0], position_ids, config=model.config)

    config = model_layer.config
    score_head_dim = resolve_key_head_dim(config)
    num_q_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads

    per_layer: list[LayerAttentionMetrics] = []
    mse_sum = 0.0
    rmse_sum = 0.0
    cosine_sum = 0.0
    max_error = 0.0
    output_rmse_sum = 0.0
    kl_sum = 0.0
    output_layers = 0
    layers = 0

    for layer_idx, (key, value) in enumerate(iter_layer_kv(outputs.past_key_values)):
        query = _compute_layer_queries(
            model_layer,
            layer_idx,
            hidden_states[layer_idx],
            rope_context.get_rope(layer_idx),
        )
        key_exp = expand_kv_heads(key, num_q_heads, num_kv_heads)
        query, key_exp = _slice_score_window(query, key_exp, score_tokens)

        scores_fp = attention_scores(query, key_exp, score_head_dim)
        scores_quant = None

        if hasattr(compressor, "attention_fidelity"):
            # Distortion-only hook: no raw quantized score tensor is returned, so
            # output_rmse / distribution_kl_divergence can't be computed for this layer.
            mse, rmse, cosine, layer_max = compressor.attention_fidelity(
                query,
                key,
                value,
                score_head_dim,
                num_q_heads,
                num_kv_heads,
                layer=layer_idx,
            )
        elif hasattr(compressor, "estimate_attention_scores"):
            key_payload = compressor.compress_kv(key, layer=layer_idx, mode="key")
            scores_quant = compressor.estimate_attention_scores(query, key_payload, score_head_dim)
            mse, rmse, cosine, layer_max = _score_distortion(scores_fp, scores_quant)
        else:
            key_payload = compressor.compress_kv(key, layer=layer_idx, mode="key")
            key_hat = compressor.decompress_kv(key_payload, mode="key").to(device=query.device)
            key_hat_exp = expand_kv_heads(key_hat, num_q_heads, num_kv_heads)
            if score_tokens > 0 and key_hat_exp.shape[-2] > score_tokens:
                key_hat_exp = key_hat_exp[..., -score_tokens:, :]
            scores_quant = attention_scores(query, key_hat_exp, score_head_dim)
            mse, rmse, cosine, layer_max = _score_distortion(scores_fp, scores_quant)

        output_rmse: float | None = None
        kl: float | None = None
        if scores_quant is not None:
            value_payload = compressor.compress_kv(value, layer=layer_idx, mode="value")
            value_hat = compressor.decompress_kv(value_payload, mode="value").to(device=query.device)
            value_hat_exp = expand_kv_heads(value_hat, num_q_heads, num_kv_heads)
            if score_tokens > 0 and value_hat_exp.shape[-2] > score_tokens:
                value_hat_exp = value_hat_exp[..., -score_tokens:, :]
            if value_hat_exp.shape[-2] == scores_quant.shape[-1]:
                output_rmse, kl = _attention_output_and_divergence(scores_fp, scores_quant, value_hat_exp)
                output_rmse_sum += output_rmse
                kl_sum += kl
                output_layers += 1
            del value_payload, value_hat, value_hat_exp

        cosine = clamp_cosine(float(cosine))
        per_layer.append(
            LayerAttentionMetrics(
                layer=layer_idx,
                mse=mse,
                rmse=rmse,
                cosine_similarity=cosine,
                max_error=layer_max,
                output_rmse=output_rmse,
                distribution_kl_divergence=kl,
            )
        )
        mse_sum += mse
        rmse_sum += rmse
        cosine_sum += cosine
        max_error = max(max_error, layer_max)
        layers += 1

        del query, key_exp, scores_fp
        if "key_payload" in locals():
            del key_payload
        if scores_quant is not None:
            del scores_quant
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if layers == 0:
        raise RuntimeError("No KV layers available for attention fidelity evaluation.")

    return AttentionMetrics(
        mse=mse_sum / layers,
        rmse=rmse_sum / layers,
        cosine_similarity=clamp_cosine(cosine_sum / layers),
        max_error=max_error,
        output_rmse=(output_rmse_sum / output_layers) if output_layers else None,
        distribution_kl_divergence=(kl_sum / output_layers) if output_layers else None,
        per_layer=per_layer,
    )
