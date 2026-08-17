"""Tests for compatibility gate evaluation."""

from types import SimpleNamespace

from framework.compatibility import (
    CompatibilityGate,
    check_attention_gate,
    check_state_semantics_gate,
)
from framework.model_capabilities import resolve_model_capabilities


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


def test_falcon_state_semantics_gate_fails_with_recurrent_cache():
    config = SimpleNamespace(model_type="falcon_h1")
    caps = resolve_model_capabilities(config)
    past_key_values = SimpleNamespace(
        layers=[
            SimpleNamespace(
                keys=object(),
                values=object(),
                recurrent_states=object(),
                conv_states=object(),
            )
        ]
    )
    result = check_state_semantics_gate(config, past_key_values)
    assert result.passed is False
