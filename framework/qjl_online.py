"""QJL online inference: literature ProdQJL attention (float Sq, signed keys only)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from compressors.qjl import QJLCompressor
from framework.attention_patches import align_attention_mask
from framework.model_adapter import load_attention_ops, project_attention_states
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


def append_key_payloads(
    compressor: QJLCompressor,
    layer_idx: int,
    new_key_states: torch.Tensor,
) -> list[QJLTensorPayload]:
    """Encode freshly projected keys before they mix with reconstructed past."""
    for token_idx in range(new_key_states.shape[2]):
        slice_k = new_key_states[:, :, token_idx : token_idx + 1, :]
        compressor.compress_key_token(layer_idx, slice_k)
    return compressor.online_key_payloads(layer_idx)


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
    expected = head_dim**-0.5
    if abs(scaling - expected) > 1e-6:
        scores = scores * (scaling / expected)

    q_len = query.shape[2]
    k_len = scores.shape[-1]
    attention_mask = align_attention_mask(attention_mask, q_len=q_len, k_len=k_len)
    if attention_mask is not None:
        scores = scores + attention_mask.to(device=scores.device, dtype=scores.dtype)

    attn_weights = F.softmax(scores.float(), dim=-1).to(query.dtype)
    value_states = _repeat_kv(value, num_key_value_groups)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def enable_qjl_online(model, compressor: QJLCompressor) -> None:
    """Patch eager attention to score keys with the QJL estimator."""
    if getattr(model, "_qjl_online_enabled", False):
        return

    ops = load_attention_ops(model.config)

    from framework.attention_patches import decoder_layers

    for layer_idx, layer in enumerate(decoder_layers(model)):
        attn = layer.self_attn

        def make_forward(layer_index: int, attn_module=attn, attn_ops=ops):
            def forward(
                hidden_states: torch.Tensor,
                position_embeddings: tuple[torch.Tensor, torch.Tensor],
                attention_mask: torch.Tensor | None,
                past_key_values=None,
                **kwargs,
            ):
                input_shape = hidden_states.shape[:-1]
                cos, sin = position_embeddings
                query_states, key_states, value_states = project_attention_states(
                    attn_module,
                    hidden_states,
                    attn_ops,
                    cos,
                    sin,
                    config=model.config,
                )

                # Sketch the *new* keys before update() concatenates reconstructed past.
                # Re-encoding from decompressed 1-bit keys (cosine ≈ 0) is what made
                # Gemma3 MQA / Qwen3 GQA BEHAVIOR PPL explode.
                already = len(compressor.online_key_payloads(layer_index))
                if already == 0 or past_key_values is None:
                    key_payloads = _ensure_key_payloads(compressor, layer_index, key_states)
                else:
                    key_payloads = append_key_payloads(compressor, layer_index, key_states)

                if past_key_values is not None:
                    cache_kwargs = kwargs.get("cache_kwargs")
                    if cache_kwargs:
                        key_states, value_states = past_key_values.update(
                            key_states,
                            value_states,
                            layer_index,
                            cache_kwargs,
                        )
                    else:
                        key_states, value_states = past_key_values.update(
                            key_states,
                            value_states,
                            layer_index,
                        )

                if len(key_payloads) != key_states.shape[2]:
                    key_payloads = _ensure_key_payloads(compressor, layer_index, key_states)

                attn_output, attn_weights = qjl_eager_attention_forward(
                    query_states,
                    value_states,
                    key_payloads,
                    compressor,
                    attention_mask,
                    scaling=attn_module.scaling,
                    num_key_value_groups=attn_module.num_key_value_groups,
                )

                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                attn_output = attn_module.o_proj(attn_output)
                return attn_output, attn_weights

            return forward

        attn.forward = make_forward(layer_idx)  # type: ignore[method-assign]

    model._qjl_online_enabled = True
