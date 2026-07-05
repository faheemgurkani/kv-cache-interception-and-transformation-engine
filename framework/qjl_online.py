"""QJL online inference: asymmetric attention estimator (no key reconstruction)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from compressors.qjl import QJLCompressor
from quantizers.qjl_pipeline import QJLTensorPayload


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def _ensure_key_payloads(
    compressor: QJLCompressor,
    layer_idx: int,
    key_states: torch.Tensor,
) -> list[QJLTensorPayload]:
    """Ensure one QJL payload per key position along the sequence axis."""
    payloads = compressor.online_key_payloads(layer_idx)
    seq_len = key_states.shape[2]
    if len(payloads) < seq_len:
        for token_idx in range(len(payloads), seq_len):
            slice_k = key_states[:, :, token_idx : token_idx + 1, :]
            payloads.append(compressor.compress_key_token(layer_idx, slice_k))
    return payloads[:seq_len]


def qjl_eager_attention_forward(
    query: torch.Tensor,
    value: torch.Tensor,
    key_payloads: list[QJLTensorPayload],
    compressor: QJLCompressor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    num_key_value_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention using QJL-estimated QK^T; values remain exact FP16."""
    head_dim = query.shape[-1]
    scores = compressor.estimate_attention_scores(query, key_payloads, head_dim)
    # estimate_attention_scores applies 1/sqrt(d); match module scaling if needed
    expected = head_dim**-0.5
    if abs(scaling - expected) > 1e-6:
        scores = scores * (scaling / expected)

    if attention_mask is not None:
        scores = scores + attention_mask

    attn_weights = F.softmax(scores.float(), dim=-1).to(query.dtype)
    value_states = _repeat_kv(value, num_key_value_groups)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def enable_qjl_online(model, compressor: QJLCompressor) -> None:
    """Patch Qwen3 eager attention to score keys with the QJL estimator."""
    if getattr(model, "_qjl_online_enabled", False):
        return

    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn

        def make_forward(layer_index: int):
            def forward(
                hidden_states: torch.Tensor,
                position_embeddings: tuple[torch.Tensor, torch.Tensor],
                attention_mask: torch.Tensor | None,
                past_key_values=None,
                **kwargs,
            ):
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, attn.head_dim)

                query_states = attn.q_norm(attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
                key_states = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
                value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

                if past_key_values is not None:
                    key_states, value_states = past_key_values.update(
                        key_states,
                        value_states,
                        layer_index,
                    )

                key_payloads = _ensure_key_payloads(compressor, layer_index, key_states)
                attn_output, attn_weights = qjl_eager_attention_forward(
                    query_states,
                    value_states,
                    key_payloads,
                    compressor,
                    attention_mask,
                    scaling=attn.scaling,
                    num_key_value_groups=attn.num_key_value_groups,
                )

                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                attn_output = attn.o_proj(attn_output)
                return attn_output, attn_weights

            return forward

        attn.forward = make_forward(layer_idx)  # type: ignore[method-assign]

    model._qjl_online_enabled = True
