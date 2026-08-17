"""Model capability metadata for the five-model architecture matrix."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from framework.model_adapter import resolve_model_type


class StateType(str, Enum):
    """Primary inference-state family for a model checkpoint."""

    CONVENTIONAL_KV = "conventional_kv"
    MLA = "mla"
    HYBRID = "hybrid"


class CompatibilityGate(str, Enum):
    """Three independent compatibility gates used by KVBench."""

    LOADER_STATE = "loader_state"
    ATTENTION = "attention"
    STATE_SEMANTICS = "state_semantics"


@dataclass(frozen=True)
class ModelCapabilities:
    """Lightweight internal contract for every supported model family."""

    model_type: str
    attention_family: str
    kv_layout: str
    qk_norm_layout: str
    rope_mode: str
    has_recurrent_state: bool
    native_latent_cache: bool
    per_layer_attention_type: bool
    state_type: StateType
    adapter_registered: bool
    state_semantics_complete: bool
    expanded_kv_disclosure: str | None = None

    def supports_gate(self, gate: CompatibilityGate) -> bool:
        if gate is CompatibilityGate.LOADER_STATE:
            return True
        if gate is CompatibilityGate.ATTENTION:
            return self.adapter_registered
        if gate is CompatibilityGate.STATE_SEMANTICS:
            return self.state_semantics_complete
        raise ValueError(f"Unknown gate: {gate!r}")


CAPABILITIES_BY_MODEL_TYPE: dict[str, ModelCapabilities] = {
    "olmo2": ModelCapabilities(
        model_type="olmo2",
        attention_family="mha",
        kv_layout="standard",
        qk_norm_layout="flat",
        rope_mode="global",
        has_recurrent_state=False,
        native_latent_cache=False,
        per_layer_attention_type=False,
        state_type=StateType.CONVENTIONAL_KV,
        adapter_registered=True,
        state_semantics_complete=True,
    ),
    "qwen3": ModelCapabilities(
        model_type="qwen3",
        attention_family="gqa",
        kv_layout="standard",
        qk_norm_layout="per_head",
        rope_mode="global",
        has_recurrent_state=False,
        native_latent_cache=False,
        per_layer_attention_type=False,
        state_type=StateType.CONVENTIONAL_KV,
        adapter_registered=True,
        state_semantics_complete=True,
    ),
    "qwen2": ModelCapabilities(
        model_type="qwen2",
        attention_family="gqa",
        kv_layout="standard",
        qk_norm_layout="per_head",
        rope_mode="global",
        has_recurrent_state=False,
        native_latent_cache=False,
        per_layer_attention_type=False,
        state_type=StateType.CONVENTIONAL_KV,
        adapter_registered=True,
        state_semantics_complete=True,
    ),
    "gemma3_text": ModelCapabilities(
        model_type="gemma3_text",
        attention_family="mqa",
        kv_layout="standard",
        qk_norm_layout="per_head",
        rope_mode="per_layer_type",
        has_recurrent_state=False,
        native_latent_cache=False,
        per_layer_attention_type=True,
        state_type=StateType.CONVENTIONAL_KV,
        adapter_registered=False,
        state_semantics_complete=True,
    ),
    "deepseek_v3": ModelCapabilities(
        model_type="deepseek_v3",
        attention_family="mla",
        kv_layout="expanded",
        qk_norm_layout="mla",
        rope_mode="architecture_specific",
        has_recurrent_state=False,
        native_latent_cache=True,
        per_layer_attention_type=False,
        state_type=StateType.MLA,
        adapter_registered=False,
        state_semantics_complete=True,
        expanded_kv_disclosure=(
            "HF eager DeepseekV3Attention materializes expanded per-head K/V in the "
            "cache, not the native kv_lora_rank latent representation."
        ),
    ),
    "falcon_h1": ModelCapabilities(
        model_type="falcon_h1",
        attention_family="gqa",
        kv_layout="attention_only",
        qk_norm_layout="none",
        rope_mode="architecture_specific",
        has_recurrent_state=True,
        native_latent_cache=False,
        per_layer_attention_type=True,
        state_type=StateType.HYBRID,
        adapter_registered=False,
        state_semantics_complete=False,
        expanded_kv_disclosure=(
            "Attention K/V is visible in the cache; Mamba recurrent/conv state is "
            "present but not yet included in default memory accounting."
        ),
    ),
}


def resolve_model_capabilities(config) -> ModelCapabilities:
    """Return capability metadata for a HuggingFace model config."""
    model_type = resolve_model_type(config)
    if model_type in CAPABILITIES_BY_MODEL_TYPE:
        return CAPABILITIES_BY_MODEL_TYPE[model_type]
    return ModelCapabilities(
        model_type=model_type,
        attention_family="unknown",
        kv_layout="unknown",
        qk_norm_layout="unknown",
        rope_mode="unknown",
        has_recurrent_state=False,
        native_latent_cache=False,
        per_layer_attention_type=False,
        state_type=StateType.CONVENTIONAL_KV,
        adapter_registered=False,
        state_semantics_complete=False,
    )


def is_attention_adapter_registered(config) -> bool:
    return resolve_model_capabilities(config).adapter_registered


@dataclass(frozen=True)
class LayerAttentionMetadata:
    """Per-layer attention metadata for evaluation and RoPE selection."""

    layer_idx: int
    attention_type: str | None
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    rope_type: str | None = None
    window_size: int | None = None
    is_sliding: bool | None = None


def get_layer_attention_metadata(config, layer_idx: int) -> LayerAttentionMetadata:
    """Return attention metadata for one decoder layer."""
    layer_types = getattr(config, "layer_types", None) or []
    attention_type = layer_types[layer_idx] if layer_idx < len(layer_types) else None
    rope_type = attention_type if attention_type in {"sliding_attention", "full_attention"} else None
    window_size = getattr(config, "sliding_window", None)
    return LayerAttentionMetadata(
        layer_idx=layer_idx,
        attention_type=attention_type,
        num_q_heads=int(config.num_attention_heads),
        num_kv_heads=int(config.num_key_value_heads),
        head_dim=int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)),
        rope_type=rope_type,
        window_size=int(window_size) if window_size is not None else None,
        is_sliding=attention_type == "sliding_attention" if attention_type else None,
    )
