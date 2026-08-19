"""Tests for compression method taxonomy (Phase 4)."""

from compressors.taxonomy import METHOD_TAXONOMY, CompressionCategory, get_method_taxonomy


def test_all_registered_methods_have_taxonomy():
    from compressors.registry import COMPRESSORS

    for name in COMPRESSORS:
        assert name in METHOD_TAXONOMY, f"Missing taxonomy for {name}"


def test_rocketkv_is_hybrid_with_eviction_secondary():
    tax = get_method_taxonomy("rocketkv")
    assert tax.primary == CompressionCategory.HYBRID
    assert CompressionCategory.EVICTION in tax.secondary


def test_qjl_modifies_attention():
    tax = get_method_taxonomy("qjl")
    assert tax.primary == CompressionCategory.QUANTIZATION
    assert tax.modifies_attention is True


def test_taxonomy_to_dict():
    tax = get_method_taxonomy("snapkv")
    payload = tax.to_dict()
    assert payload["primary"] == "eviction"
    assert payload["name"] == "snapkv"
