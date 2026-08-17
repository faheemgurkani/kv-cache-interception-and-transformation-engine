"""Model-family adapters for KV interception (Qwen3, OLMo2, …)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn


@dataclass(frozen=True)
class AttentionOps:
    """Family-specific attention helpers used by online patches and Section A."""

    model_type: str
    apply_rotary_pos_emb: Callable
    eager_attention_forward: Callable
    all_attention_functions: Any
    qk_norm_layout: str  # "per_head" (Qwen3) | "flat" (OLMo2) | "none" (Falcon-H1, future)
    has_input_layernorm: bool
    passes_sliding_window: bool
    layer_types: tuple[str, ...] | None = None


AttentionAdapterBuilder = Callable[[str], AttentionOps]


def resolve_model_type(config) -> str:
    return str(getattr(config, "model_type", "") or "").lower()


def resolve_head_dim(config, attn_module: nn.Module | None = None) -> int:
    """Return head dim; OLMo2 config omits ``head_dim`` (derive from hidden/heads)."""
    if attn_module is not None:
        head_dim = getattr(attn_module, "head_dim", None)
        if head_dim is not None:
            return int(head_dim)
    head_dim = getattr(config, "head_dim", None)
    if head_dim is not None:
        return int(head_dim)
    hidden = int(config.hidden_size)
    heads = int(config.num_attention_heads)
    if heads <= 0 or hidden % heads != 0:
        raise ValueError(f"Cannot derive head_dim from hidden_size={hidden}, heads={heads}")
    return hidden // heads


def _build_qwen_ops(model_type: str) -> AttentionOps:
    from transformers.models.qwen3.modeling_qwen3 import (
        ALL_ATTENTION_FUNCTIONS,
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    return AttentionOps(
        model_type=model_type,
        apply_rotary_pos_emb=apply_rotary_pos_emb,
        eager_attention_forward=eager_attention_forward,
        all_attention_functions=ALL_ATTENTION_FUNCTIONS,
        qk_norm_layout="per_head",
        has_input_layernorm=True,
        passes_sliding_window=True,
    )


def _build_olmo2_ops(_model_type: str) -> AttentionOps:
    from transformers.models.olmo2.modeling_olmo2 import (
        ALL_ATTENTION_FUNCTIONS,
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    return AttentionOps(
        model_type="olmo2",
        apply_rotary_pos_emb=apply_rotary_pos_emb,
        eager_attention_forward=eager_attention_forward,
        all_attention_functions=ALL_ATTENTION_FUNCTIONS,
        qk_norm_layout="flat",
        has_input_layernorm=False,
        passes_sliding_window=False,
    )


ATTENTION_ADAPTER_REGISTRY: dict[str, AttentionAdapterBuilder] = {
    "qwen3": _build_qwen_ops,
    "qwen2": _build_qwen_ops,
    "olmo2": _build_olmo2_ops,
}


def is_attention_adapter_registered(config) -> bool:
    return resolve_model_type(config) in ATTENTION_ADAPTER_REGISTRY


def load_attention_ops(config) -> AttentionOps:
    """Import RoPE / eager attention symbols for the active model family."""
    model_type = resolve_model_type(config)
    builder = ATTENTION_ADAPTER_REGISTRY.get(model_type)
    if builder is None:
        raise NotImplementedError(
            f"Online attention adapters are not implemented for model_type={model_type!r}. "
            f"Supported: {', '.join(sorted(ATTENTION_ADAPTER_REGISTRY))}."
        )
    ops = builder(model_type)
    layer_types = getattr(config, "layer_types", None)
    if layer_types:
        return AttentionOps(
            model_type=ops.model_type,
            apply_rotary_pos_emb=ops.apply_rotary_pos_emb,
            eager_attention_forward=ops.eager_attention_forward,
            all_attention_functions=ops.all_attention_functions,
            qk_norm_layout=ops.qk_norm_layout,
            has_input_layernorm=ops.has_input_layernorm,
            passes_sliding_window=ops.passes_sliding_window,
            layer_types=tuple(layer_types),
        )
    return ops


def project_qkv(
    attn: nn.Module,
    hidden_states: torch.Tensor,
    ops: AttentionOps,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project + (optional) Q/K-norm + reshape to ``[B, H, T, D]``."""
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)

    if ops.qk_norm_layout == "flat":
        query_states = attn.q_norm(attn.q_proj(hidden_states))
        key_states = attn.k_norm(attn.k_proj(hidden_states))
        value_states = attn.v_proj(hidden_states)
        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)
        return query_states, key_states, value_states

    if ops.qk_norm_layout == "none":
        query_states = attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        return query_states, key_states, value_states

    query_states = attn.q_norm(attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    return query_states, key_states, value_states


def pre_attention_hidden(layer: nn.Module, hidden_states: torch.Tensor, ops: AttentionOps) -> torch.Tensor:
    """Hidden states fed into Q/K projections (pre-norm vs post-norm families)."""
    if ops.has_input_layernorm and hasattr(layer, "input_layernorm"):
        return layer.input_layernorm(hidden_states)
    return hidden_states


def resolve_attention_interface(attn: nn.Module, config, ops: AttentionOps):
    return ops.all_attention_functions.get_interface(
        config._attn_implementation,
        ops.eager_attention_forward,
    )


def attention_call_kwargs(attn: nn.Module, ops: AttentionOps) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "dropout": 0.0 if not attn.training else attn.attention_dropout,
        "scaling": attn.scaling,
    }
    if ops.passes_sliding_window and hasattr(attn, "sliding_window"):
        kwargs["sliding_window"] = attn.sliding_window
    return kwargs
