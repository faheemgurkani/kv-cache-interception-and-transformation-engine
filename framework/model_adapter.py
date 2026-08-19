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
        qk_head_dim = getattr(attn_module, "qk_head_dim", None)
        if qk_head_dim is not None:
            return int(qk_head_dim)
    head_dim = getattr(config, "head_dim", None)
    if head_dim is not None:
        return int(head_dim)
    qk_head_dim = getattr(config, "qk_head_dim", None)
    if qk_head_dim is not None:
        return int(qk_head_dim)
    hidden = int(config.hidden_size)
    heads = int(config.num_attention_heads)
    if heads <= 0 or hidden % heads != 0:
        raise ValueError(f"Cannot derive head_dim from hidden_size={hidden}, heads={heads}")
    return hidden // heads


def resolve_key_head_dim(config, attn_module: nn.Module | None = None) -> int:
    """Return per-head key dimension (MLA expanded K uses ``qk_head_dim``)."""
    if attn_module is not None and getattr(attn_module, "qk_head_dim", None) is not None:
        return int(attn_module.qk_head_dim)
    if getattr(config, "qk_head_dim", None) is not None:
        return int(config.qk_head_dim)
    return resolve_head_dim(config, attn_module)


def resolve_value_head_dim(config, attn_module: nn.Module | None = None) -> int:
    """Return per-head value dimension (MLA expanded V may differ from K)."""
    if attn_module is not None and getattr(attn_module, "v_head_dim", None) is not None:
        return int(attn_module.v_head_dim)
    if getattr(config, "v_head_dim", None) is not None:
        return int(config.v_head_dim)
    return resolve_head_dim(config, attn_module)


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


def _build_gemma3_ops(model_type: str) -> AttentionOps:
    from transformers.models.gemma3.modeling_gemma3 import (
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


def _build_falcon_h1_ops(model_type: str) -> AttentionOps:
    from transformers.models.falcon_h1.modeling_falcon_h1 import (
        ALL_ATTENTION_FUNCTIONS,
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    return AttentionOps(
        model_type=model_type,
        apply_rotary_pos_emb=apply_rotary_pos_emb,
        eager_attention_forward=eager_attention_forward,
        all_attention_functions=ALL_ATTENTION_FUNCTIONS,
        qk_norm_layout="none",
        has_input_layernorm=True,
        passes_sliding_window=False,
    )


def _build_deepseek_v3_ops(model_type: str) -> AttentionOps:
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
        ALL_ATTENTION_FUNCTIONS,
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    return AttentionOps(
        model_type=model_type,
        apply_rotary_pos_emb=apply_rotary_pos_emb,
        eager_attention_forward=eager_attention_forward,
        all_attention_functions=ALL_ATTENTION_FUNCTIONS,
        qk_norm_layout="mla",
        has_input_layernorm=True,
        passes_sliding_window=False,
    )


ATTENTION_ADAPTER_REGISTRY: dict[str, AttentionAdapterBuilder] = {
    "qwen3": _build_qwen_ops,
    "qwen2": _build_qwen_ops,
    "olmo2": _build_olmo2_ops,
    "gemma3_text": _build_gemma3_ops,
    "deepseek_v3": _build_deepseek_v3_ops,
    "falcon_h1": _build_falcon_h1_ops,
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

    if ops.qk_norm_layout == "mla":
        raise RuntimeError("Use project_attention_states() for MLA (deepseek_v3) attention modules.")

    query_states = attn.q_norm(attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    return query_states, key_states, value_states


def project_mla_qkv(
    attn: nn.Module,
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    config,
    ops: AttentionOps,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DeepSeek MLA: latent KV projection + split nope/RoPE (expanded cache path)."""
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
        apply_rotary_pos_emb,
        apply_rotary_pos_emb_interleave,
    )

    batch_size, seq_length = hidden_states.shape[:-1]
    query_shape = (batch_size, seq_length, -1, attn.qk_head_dim)
    key_shape = (batch_size, seq_length, -1, attn.qk_nope_head_dim + attn.v_head_dim)

    if attn.q_lora_rank is None:
        q_states = attn.q_proj(hidden_states)
    else:
        q_states = attn.q_b_proj(attn.q_a_layernorm(attn.q_a_proj(hidden_states)))
    q_states = q_states.view(query_shape).transpose(1, 2)
    q_pass, q_rot = torch.split(q_states, [attn.qk_nope_head_dim, attn.qk_rope_head_dim], dim=-1)

    compressed_kv = attn.kv_a_proj_with_mqa(hidden_states)
    k_pass, k_rot = torch.split(compressed_kv, [attn.kv_lora_rank, attn.qk_rope_head_dim], dim=-1)
    k_pass = attn.kv_b_proj(attn.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
    k_pass, value_states = torch.split(k_pass, [attn.qk_nope_head_dim, attn.v_head_dim], dim=-1)
    k_rot = k_rot.view(batch_size, 1, seq_length, attn.qk_rope_head_dim)

    if getattr(config, "rope_interleave", False):
        q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
    else:
        q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
    k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

    query_states = torch.cat((q_pass, q_rot), dim=-1)
    key_states = torch.cat((k_pass, k_rot), dim=-1)
    del ops  # layout marker only; MLA uses native HF projection path
    return query_states, key_states, value_states


def project_attention_states(
    attn: nn.Module,
    hidden_states: torch.Tensor,
    ops: AttentionOps,
    cos: torch.Tensor,
    sin: torch.Tensor,
    config=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project Q/K/V and apply RoPE (family-specific, including MLA split-RoPE)."""
    if ops.qk_norm_layout == "mla":
        return project_mla_qkv(attn, hidden_states, cos, sin, config or attn.config, ops)
    query_states, key_states, value_states = project_qkv(attn, hidden_states, ops)
    query_states, key_states = ops.apply_rotary_pos_emb(query_states, key_states, cos, sin)
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
    }
    if ops.qk_norm_layout == "mla":
        kwargs["scaling"] = attn.scaling
        return kwargs
    kwargs["scaling"] = attn.scaling
    if ops.passes_sliding_window and hasattr(attn, "sliding_window"):
        kwargs["sliding_window"] = attn.sliding_window
    return kwargs
