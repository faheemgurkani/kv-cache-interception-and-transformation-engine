"""Tests for attention adapter registry (WP1 — no behavior change for existing families)."""

from types import SimpleNamespace

import pytest

from framework.model_adapter import ATTENTION_ADAPTER_REGISTRY, load_attention_ops, resolve_model_type


def test_registry_contains_existing_families():
    assert {"qwen3", "qwen2", "olmo2", "gemma3_text", "deepseek_v3"}.issubset(set(ATTENTION_ADAPTER_REGISTRY))


def test_qwen3_and_olmo2_ops_load():
    for model_type in ("qwen3", "qwen2", "olmo2", "gemma3_text", "deepseek_v3"):
        config = SimpleNamespace(model_type=model_type, layer_types=["full_attention"] * 4)
        ops = load_attention_ops(config)
        assert ops.model_type in {model_type, "olmo2"}
        assert ops.qk_norm_layout in {"per_head", "flat", "mla"}


def test_unregistered_family_raises():
    config = SimpleNamespace(model_type="falcon_h1")
    with pytest.raises(NotImplementedError, match="falcon_h1"):
        load_attention_ops(config)


def test_resolve_model_type_normalizes_case():
    config = SimpleNamespace(model_type="Qwen3")
    assert resolve_model_type(config) == "qwen3"
