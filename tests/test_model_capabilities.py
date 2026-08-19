"""Unit tests for model capability metadata."""

from types import SimpleNamespace

from framework.model_capabilities import (
    CAPABILITIES_BY_MODEL_TYPE,
    CompatibilityGate,
    resolve_model_capabilities,
)


def test_all_shortlist_families_have_capabilities():
    expected = {"olmo2", "qwen3", "gemma3_text", "deepseek_v3", "falcon_h1"}
    assert expected.issubset(set(CAPABILITIES_BY_MODEL_TYPE))


def test_olmo2_capabilities():
    caps = CAPABILITIES_BY_MODEL_TYPE["olmo2"]
    assert caps.attention_family == "mha"
    assert caps.rope_mode == "global"
    assert caps.adapter_registered is True
    assert caps.supports_gate(CompatibilityGate.ATTENTION)
    assert caps.supports_gate(CompatibilityGate.STATE_SEMANTICS)


def test_qwen3_capabilities():
    caps = CAPABILITIES_BY_MODEL_TYPE["qwen3"]
    assert caps.attention_family == "gqa"
    assert caps.qk_norm_layout == "per_head"
    assert caps.adapter_registered is True


def test_gemma3_capabilities():
    caps = CAPABILITIES_BY_MODEL_TYPE["gemma3_text"]
    assert caps.per_layer_attention_type is True
    assert caps.rope_mode == "per_layer_type"
    assert caps.adapter_registered is True
    assert caps.attention_family == "mqa"


def test_falcon_h1_capabilities():
    caps = CAPABILITIES_BY_MODEL_TYPE["falcon_h1"]
    assert caps.has_recurrent_state is True
    assert caps.state_semantics_complete is True
    assert caps.rope_mode == "global"
    assert caps.per_layer_attention_type is False
    assert caps.qk_norm_layout == "none"
    assert caps.adapter_registered is True
    assert caps.supports_gate(CompatibilityGate.ATTENTION)
    assert caps.supports_gate(CompatibilityGate.STATE_SEMANTICS)


def test_tinydeepseek_capabilities():
    caps = CAPABILITIES_BY_MODEL_TYPE["deepseek_v3"]
    assert caps.native_latent_cache is True
    assert caps.state_semantics_complete is False
    assert caps.adapter_registered is True
    assert caps.rope_mode == "split_nope_rope"
    assert caps.expanded_kv_disclosure is not None
    assert caps.supports_gate(CompatibilityGate.ATTENTION)
    assert not caps.supports_gate(CompatibilityGate.STATE_SEMANTICS)


def test_unknown_model_type_defaults_to_unsupported():
    config = SimpleNamespace(model_type="unknown_arch")
    caps = resolve_model_capabilities(config)
    assert caps.adapter_registered is False
    assert caps.state_semantics_complete is False
