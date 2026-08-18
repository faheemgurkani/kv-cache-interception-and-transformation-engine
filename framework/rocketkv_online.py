"""RocketKV online inference: stage-1 filtering + HSA in the attention path."""

from __future__ import annotations

import torch

from compressors.rocketkv import RocketKVCompressor
from framework.model_adapter import (
    attention_call_kwargs,
    load_attention_ops,
    project_attention_states,
    resolve_attention_interface,
)


def apply_online_kv_sparsity(
    compressor: RocketKVCompressor,
    layer_idx: int,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply RocketKV stage-1 (permanent) then stage-2 (HSA) sparsity.

    Returns sparse K/V plus ``kept_indices`` into the pre-sparsity sequence axis.
    """
    if key.shape[2] == 0:
        empty = torch.empty(0, dtype=torch.long, device=key.device)
        return key, value, empty

    global_indices = compressor.extend_global_indices(layer_idx, key.shape[2], key.device)
    stored_global, key, value, _, stage1_local = compressor.apply_stage1(
        key,
        value,
        layer=layer_idx,
        global_indices=global_indices,
    )
    key, value, mask_local = compressor.apply_stage2(
        query,
        key,
        value,
        stored_global,
        layer=layer_idx,
        stage1_local=stage1_local,
    )
    if mask_local.numel() != key.shape[2]:
        mask_local = mask_local[: key.shape[2]]
    return key, value, mask_local


def align_attention_mask(
    attention_mask: torch.Tensor | None,
    kept_indices: torch.Tensor,
    key_seq_len: int,
    query_seq_len: int | None = None,
) -> torch.Tensor | None:
    """Slice the additive/causal mask to match sparse key/value length."""
    if attention_mask is None:
        return None

    q_len = query_seq_len or 1

    if attention_mask.dim() == 4:
        if attention_mask.shape[-1] == key_seq_len and attention_mask.shape[-2] == q_len:
            return attention_mask
        base = attention_mask[..., -q_len:, :]
        if (
            kept_indices.numel() > 0
            and kept_indices.numel() == key_seq_len
            and int(kept_indices.max().item()) < base.shape[-1]
        ):
            aligned = base.index_select(-1, kept_indices.to(base.device))
            if aligned.shape[-1] == key_seq_len:
                return aligned
        return attention_mask[..., -q_len:, -key_seq_len:]

    if attention_mask.shape[-1] == key_seq_len:
        if query_seq_len is None or attention_mask.shape[-2] == query_seq_len:
            return attention_mask
    if kept_indices.numel() == 0:
        aligned = attention_mask[..., :key_seq_len]
    else:
        idx = kept_indices.to(attention_mask.device)
        if idx.max().item() >= attention_mask.shape[-1]:
            idx = idx[idx < attention_mask.shape[-1]]
        if attention_mask.dim() == 2:
            aligned = attention_mask.index_select(1, idx)
        else:
            aligned = attention_mask[..., :key_seq_len]

    if query_seq_len is not None and aligned.dim() >= 2 and aligned.shape[-2] > query_seq_len:
        aligned = aligned[..., -query_seq_len:, :]
    if aligned.shape[-1] != key_seq_len:
        aligned = aligned[..., :key_seq_len]
    return aligned


def _write_sparse_cache(
    past_key_values,
    layer_index: int,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> None:
    """Physically evict tokens from the runtime DynamicCache."""
    if past_key_values is None:
        return
    if hasattr(past_key_values, "layers"):
        past_key_values.layers[layer_index].keys = key_states
        past_key_values.layers[layer_index].values = value_states
        return
    if hasattr(past_key_values, "key_cache"):
        past_key_values.key_cache[layer_index] = key_states
        past_key_values.value_cache[layer_index] = value_states


def enable_rocketkv_online(model, compressor: RocketKVCompressor) -> None:
    """Patch eager attention to apply RocketKV sparsity before softmax."""
    if getattr(model, "_rocketkv_online_enabled", False):
        return

    ops = load_attention_ops(model.config)

    for layer_idx, layer in enumerate(model.model.layers):
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
                _write_sparse_cache(
                    past_key_values,
                    layer_index,
                    key_states,
                    value_states,
                )
                attention_mask = align_attention_mask(
                    attention_mask,
                    kept_indices,
                    key_states.shape[2],
                    query_seq_len=query_states.shape[2],
                )
                if attention_mask is not None and attention_mask.dim() == 4:
                    q_len = query_states.shape[2]
                    k_len = key_states.shape[2]
                    if attention_mask.shape[-2] != q_len or attention_mask.shape[-1] != k_len:
                        attention_mask = attention_mask[..., -q_len:, :k_len]
                    if attention_mask.shape[-1] < k_len:
                        key_states = key_states[..., : attention_mask.shape[-1], :]
                        value_states = value_states[..., : attention_mask.shape[-1], :]

                attention_interface = resolve_attention_interface(
                    attn_module, model.config, attn_ops
                )
                call_kwargs = attention_call_kwargs(attn_module, attn_ops)
                attn_output, attn_weights = attention_interface(
                    attn_module,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    **call_kwargs,
                    **kwargs,
                )

                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                attn_output = attn_module.o_proj(attn_output)
                return attn_output, attn_weights

            return forward

        attn.forward = make_forward(layer_idx)  # type: ignore[method-assign]

    model._rocketkv_online_enabled = True
