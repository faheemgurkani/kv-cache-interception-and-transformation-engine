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
        adapter_registered=True,
        state_semantics_complete=True,
    ),
    "deepseek_v3": ModelCapabilities(
        model_type="deepseek_v3",
        attention_family="mla",
        kv_layout="expanded",
        qk_norm_layout="mla",
        rope_mode="split_nope_rope",
        has_recurrent_state=False,
        native_latent_cache=True,
        per_layer_attention_type=False,
        state_type=StateType.MLA,
        adapter_registered=True,
        state_semantics_complete=False,
        expanded_kv_disclosure=(
            "HF eager DeepseekV3Attention materializes expanded per-head K/V in the "
            "cache (D_k=64, D_v=32), not the native kv_lora_rank latent representation."
        ),
    ),
    "falcon_h1": ModelCapabilities(
        model_type="falcon_h1",
        attention_family="gqa",
        kv_layout="hybrid_visible",
        qk_norm_layout="none",
        rope_mode="global",
        has_recurrent_state=True,
        native_latent_cache=False,
        per_layer_attention_type=False,
        state_type=StateType.HYBRID,
        adapter_registered=True,
        state_semantics_complete=True,
        expanded_kv_disclosure=(
            "Every layer is hybrid (attention GQA + Mamba2). Memory accounting counts "
            "attention K/V and Mamba recurrent/conv state; compression targets K/V only."
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
    from framework.model_adapter import resolve_head_dim

    layer_types = getattr(config, "layer_types", None) or []
    attention_type = layer_types[layer_idx] if layer_idx < len(layer_types) else None
    rope_type = attention_type if attention_type in {"sliding_attention", "full_attention"} else None
    window_size = getattr(config, "sliding_window", None)
    return LayerAttentionMetadata(
        layer_idx=layer_idx,
        attention_type=attention_type,
        num_q_heads=int(config.num_attention_heads),
        num_kv_heads=int(getattr(config, "num_key_value_heads", config.num_attention_heads)),
        head_dim=resolve_head_dim(config),
        rope_type=rope_type,
        window_size=int(window_size) if window_size is not None else None,
        is_sliding=attention_type == "sliding_attention" if attention_type else None,
    )


def get_model_eval_metadata(config, *, local_path: str | None = None) -> dict[str, object]:
    """Reference metadata recorded with every evaluation run (OLMo2 baseline template)."""
    from dataclasses import asdict

    from framework.model_adapter import (
        resolve_head_dim,
        resolve_key_head_dim,
        resolve_model_type,
        resolve_value_head_dim,
    )

    caps = resolve_model_capabilities(config)
    num_q_heads = int(config.num_attention_heads)
    num_kv_heads = int(getattr(config, "num_key_value_heads", num_q_heads))
    head_dim = resolve_head_dim(config)
    key_head_dim = resolve_key_head_dim(config)
    value_head_dim = resolve_value_head_dim(config)
    metadata: dict[str, object] = {
        "model_type": resolve_model_type(config),
        "local_path": local_path,
        "attention_family": caps.attention_family,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "key_head_dim": key_head_dim,
        "value_head_dim": value_head_dim,
        "num_layers": int(config.num_hidden_layers),
        "hidden_size": int(config.hidden_size),
        "rope_mode": caps.rope_mode,
        "state_type": caps.state_type.value,
        "qk_norm_layout": caps.qk_norm_layout,
        "adapter": resolve_model_type(config),
        "adapter_registered": caps.adapter_registered,
        "state_semantics_complete": caps.state_semantics_complete,
    }
    if caps.per_layer_attention_type:
        num_layers = int(config.num_hidden_layers)
        metadata["layer_attention"] = [
            asdict(get_layer_attention_metadata(config, layer_idx))
            for layer_idx in range(num_layers)
        ]
        metadata["sliding_window"] = getattr(config, "sliding_window", None)
    if caps.attention_family == "mla":
        metadata["cache_representation"] = "expanded_kv"
        metadata["kv_lora_rank"] = getattr(config, "kv_lora_rank", None)
        metadata["qk_nope_head_dim"] = getattr(config, "qk_nope_head_dim", None)
        metadata["qk_rope_head_dim"] = getattr(config, "qk_rope_head_dim", None)
        metadata["qk_head_dim"] = getattr(config, "qk_head_dim", None)
        if caps.expanded_kv_disclosure:
            metadata["expanded_kv_disclosure"] = caps.expanded_kv_disclosure
    if caps.state_type == StateType.HYBRID:
        metadata["compression_policy"] = {"attention": "compressible", "recurrent": "passthrough"}
        if caps.expanded_kv_disclosure:
            metadata["hybrid_state_disclosure"] = caps.expanded_kv_disclosure
    return metadata


def build_compatibility_manifest(config, *, yaml_section: dict | None = None) -> dict[str, object]:
    """Declarative compatibility manifest (§30), derived from capabilities + config."""
    caps = resolve_model_capabilities(config)
    num_q_heads = int(config.num_attention_heads)
    num_kv_heads = int(getattr(config, "num_key_value_heads", num_q_heads))
    manifest: dict[str, object] = {
        "model_type": resolve_model_type(config),
        "architecture": {
            "family": (
                caps.state_type.value
                if caps.state_type in {StateType.HYBRID, StateType.MLA}
                else caps.attention_family
            ),
            "state_type": caps.state_type.value,
            "q_heads": num_q_heads,
            "kv_heads": num_kv_heads,
        },
        "attention": {
            "adapter": resolve_model_type(config),
            "qk_norm": caps.qk_norm_layout,
            "rope": caps.rope_mode,
            "adapter_registered": caps.adapter_registered,
        },
        "cache": {
            "type": "hybrid_visible" if caps.state_type == StateType.HYBRID else caps.kv_layout,
            "native_latent": caps.native_latent_cache,
        },
        "evaluation": {
            "fidelity_representation": True,
            "fidelity_attention": caps.adapter_registered,
            "fidelity_memory": True,
            "behavior_identity": caps.supports_gate(CompatibilityGate.LOADER_STATE),
            "behavior_turboquant": caps.supports_gate(CompatibilityGate.LOADER_STATE),
            "behavior_qjl": caps.adapter_registered,
            "behavior_rocketkv": caps.adapter_registered,
            "system": caps.supports_gate(CompatibilityGate.LOADER_STATE),
            "total_state_accounting": caps.has_recurrent_state or caps.native_latent_cache,
        },
    }
    if caps.state_type == StateType.HYBRID:
        manifest["state"] = {"attention": "compressible", "recurrent": "passthrough"}
    if caps.per_layer_attention_type:
        manifest["cache"] = {
            **manifest["cache"],  # type: ignore[dict-item]
            "sliding_layers": True,
        }
    if yaml_section:
        _merge_manifest_section(manifest, yaml_section)
    return manifest


def load_compatibility_manifest(
    config,
    *,
    yaml_config: dict | None = None,
) -> dict[str, object]:
    """Load manifest from YAML ``compatibility:`` block when present, else derive."""
    yaml_section = (yaml_config or {}).get("compatibility")
    return build_compatibility_manifest(config, yaml_section=yaml_section)


def validate_manifest(manifest: dict[str, object], caps: ModelCapabilities, config) -> None:
    """Ensure declarative manifest matches live capability metadata (§30)."""
    architecture = manifest.get("architecture", {})
    attention = manifest.get("attention", {})
    if not isinstance(architecture, dict) or not isinstance(attention, dict):
        raise ValueError("Compatibility manifest requires architecture and attention sections.")

    expected_family = (
        caps.state_type.value
        if caps.state_type in {StateType.HYBRID, StateType.MLA}
        else caps.attention_family
    )
    if architecture.get("family") != expected_family:
        raise ValueError(
            f"Manifest family {architecture.get('family')!r} != {expected_family!r}."
        )
    if attention.get("adapter") != resolve_model_type(config):
        raise ValueError(
            f"Manifest adapter {attention.get('adapter')!r} != {resolve_model_type(config)!r}."
        )
    if int(architecture.get("q_heads", -1)) != int(config.num_attention_heads):
        raise ValueError("Manifest q_heads does not match model config.")
    num_kv = int(getattr(config, "num_key_value_heads", config.num_attention_heads))
    if int(architecture.get("kv_heads", -1)) != num_kv:
        raise ValueError("Manifest kv_heads does not match model config.")


def _merge_manifest_section(base: dict[str, object], override: dict[str, object]) -> None:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_manifest_section(base[key], value)  # type: ignore[arg-type]
        else:
            base[key] = value
