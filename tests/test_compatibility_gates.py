"""Tests for compatibility gate evaluation."""

from types import SimpleNamespace

from framework.compatibility import (
    CompatibilityGate,
    check_attention_gate,
    check_state_semantics_gate,
)


def test_olmo2_attention_gate_passes():
    config = SimpleNamespace(model_type="olmo2")
    result = check_attention_gate(config)
    assert result.passed is True
    assert result.gate is CompatibilityGate.ATTENTION


def test_falcon_attention_gate_fails_until_adapter_registered():
    config = SimpleNamespace(model_type="falcon_h1")
    result = check_attention_gate(config)
    assert result.passed is False
    assert "falcon_h1" in result.detail


def test_tinydeepseek_state_semantics_gate_fails_for_native_latent_scope():
    config = SimpleNamespace(model_type="deepseek_v3")
    past_key_values = SimpleNamespace(
        layers=[
            SimpleNamespace(
                keys=object(),
                values=object(),
            )
        ]
    )
    result = check_state_semantics_gate(config, past_key_values)
    assert result.passed is False
    assert "latent" in result.detail.lower()


def test_gemma3_attention_gate_passes():
    config = SimpleNamespace(model_type="gemma3_text", layer_types=["sliding_attention"] * 5 + ["full_attention"] * 1)
    result = check_attention_gate(config)
    assert result.passed is True
