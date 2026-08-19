"""SnapKV online inference: prefill-only compression inside the attention path."""

from __future__ import annotations

from compressors.snapkv import SnapKVCompressor
from framework.model_adapter import (
    attention_call_kwargs,
    load_attention_ops,
    project_attention_states,
    resolve_attention_interface,
)
from quantizers.snapkv import snap_kv


def _write_cache_kv(
    past_key_values,
    layer_index: int,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> None:
    if past_key_values is None:
        return
    if hasattr(past_key_values, "layers"):
        past_key_values.layers[layer_index].keys = key_states
        past_key_values.layers[layer_index].values = value_states
        return
    if hasattr(past_key_values, "key_cache"):
        past_key_values.key_cache[layer_index] = key_states
        past_key_values.value_cache[layer_index] = value_states


def enable_snapkv_online(model, compressor: SnapKVCompressor) -> None:
    """Patch eager attention to apply SnapKV once during prefill."""
    if getattr(model, "_snapkv_online_enabled", False):
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

                q_len = query_states.shape[2]
                kv_len = key_states.shape[2]
                if q_len == kv_len and kv_len >= compressor.max_capacity_prompt:
                    key_states, value_states = snap_kv(
                        query_states,
                        key_states,
                        value_states,
                        window_size=compressor.window_size,
                        max_capacity_prompt=compressor.max_capacity_prompt,
                        kernel_size=compressor.kernel_size,
                        attention_mask=attention_mask,
                    )
                    _write_cache_kv(past_key_values, layer_index, key_states, value_states)

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

    model._snapkv_online_enabled = True
