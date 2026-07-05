"""RocketKV online inference: stage-1 filtering + HSA in the attention path."""

from __future__ import annotations

import torch

from compressors.rocketkv import RocketKVCompressor


def apply_online_kv_sparsity(
    compressor: RocketKVCompressor,
    layer_idx: int,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply RocketKV stage-1 permanent filtering then stage-2 HSA.

    Returns sparse K/V plus ``kept_indices`` into the pre-sparsity sequence axis.
    """
    if key.shape[2] == 0:
        empty = torch.empty(0, dtype=torch.long, device=key.device)
        return key, value, empty

    stage1_indices, key, value = compressor.token_selector.select(key, value)
    compressor._permanent_indices[layer_idx] = stage1_indices.detach().cpu()
    key, value, stage2_indices = compressor.hsa.select_top_k(
        query,
        key,
        value,
        permanent_indices=None,
    )
    if stage1_indices.numel() == 0:
        return key, value, stage2_indices
    kept_indices = stage1_indices[stage2_indices]
    return key, value, kept_indices


def align_attention_mask(
    attention_mask: torch.Tensor | None,
    kept_indices: torch.Tensor,
    key_seq_len: int,
) -> torch.Tensor | None:
    """Slice the additive/causal mask to match sparse key/value length."""
    if attention_mask is None:
        return None
    if attention_mask.shape[-1] == key_seq_len:
        return attention_mask
    if kept_indices.numel() != key_seq_len:
        return attention_mask[..., :key_seq_len]

    if attention_mask.dim() == 4:
        return attention_mask.index_select(-1, kept_indices)
    if attention_mask.dim() == 2:
        return attention_mask.index_select(1, kept_indices)
    return attention_mask[..., :key_seq_len]


def _resolve_attention_interface(attn_module, config):
    from transformers.models.qwen3.modeling_qwen3 import ALL_ATTENTION_FUNCTIONS, eager_attention_forward

    return ALL_ATTENTION_FUNCTIONS.get_interface(
        config._attn_implementation,
        eager_attention_forward,
    )


def enable_rocketkv_online(model, compressor: RocketKVCompressor) -> None:
    """Patch Qwen3 eager attention to apply RocketKV sparsity before softmax."""
    if getattr(model, "_rocketkv_online_enabled", False):
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

                key_states, value_states, kept_indices = apply_online_kv_sparsity(
                    compressor,
                    layer_index,
                    query_states,
                    key_states,
                    value_states,
                )
                attention_mask = align_attention_mask(
                    attention_mask,
                    kept_indices,
                    key_states.shape[2],
                )

                attention_interface = _resolve_attention_interface(attn, model.config)
                attn_output, attn_weights = attention_interface(
                    attn,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    dropout=0.0 if not attn.training else attn.attention_dropout,
                    scaling=attn.scaling,
                    sliding_window=attn.sliding_window,
                    **kwargs,
                )

                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                attn_output = attn.o_proj(attn_output)
                return attn_output, attn_weights

            return forward

        attn.forward = make_forward(layer_idx)  # type: ignore[method-assign]

    model._rocketkv_online_enabled = True
