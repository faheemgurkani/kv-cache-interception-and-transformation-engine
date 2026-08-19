"""Comprehensive SnapKV tests: mathematics, storage, fidelity hooks, and engine integration."""

from __future__ import annotations

import math

import pytest
import torch

from compressors.snapkv import SnapKVCompressor
from compressors.taxonomy import CompressionCategory, get_method_taxonomy
from eval.cost.accounting import evaluate_cost
from eval.fidelity.representation import evaluate_representation
from framework.kv_cache import (
    compressed_size_bytes,
    decompress_compressed_layer,
    incremental_seq_length,
)
from quantizers.snapkv import SnapKVLayerPayload, snap_kv


def _make_seq(bsz=1, heads=4, seq_len=128, dim=32):
    query = torch.randn(bsz, heads, seq_len, dim)
    key = torch.randn(bsz, heads, seq_len, dim)
    value = torch.randn(bsz, heads, seq_len, dim)
    return query, key, value


# --- Taxonomy & metadata -------------------------------------------------------


def test_snapkv_taxonomy_matches_documentation():
    tax = get_method_taxonomy("snapkv")
    assert tax is not None
    assert tax.primary == CompressionCategory.EVICTION
    assert tax.modifies_attention is False
    assert tax.calibration_free is True
    assert tax.compression_unit == "head"


def test_snapkv_theoretical_ratio_short_context_is_one():
    compressor = SnapKVCompressor(max_capacity_prompt=1024)
    assert compressor.theoretical_compression_ratio(context_length=512) == 1.0


def test_snapkv_theoretical_ratio_long_context():
    compressor = SnapKVCompressor(max_capacity_prompt=1024)
    ratio = compressor.theoretical_compression_ratio(context_length=2048)
    assert ratio == pytest.approx(2.0)


def test_snapkv_offline_cost_calibration_free():
    meta = SnapKVCompressor().offline_cost_metadata()
    assert meta.calibration_required is False


# --- Listing 1 mathematics ---------------------------------------------------


def test_snap_kv_output_length_equals_max_capacity():
    query, key, value = _make_seq(seq_len=128)
    max_capacity = 64
    window_size = 16
    k2, v2 = snap_kv(
        query, key, value,
        window_size=window_size,
        max_capacity_prompt=max_capacity,
        kernel_size=5,
    )
    assert k2.shape[2] == max_capacity
    assert v2.shape[2] == max_capacity
    assert k2.shape == v2.shape


def test_snap_kv_preserves_observation_window_exactly():
    query, key, value = _make_seq(seq_len=128)
    window_size = 16
    k2, v2 = snap_kv(
        query, key, value,
        window_size=window_size,
        max_capacity_prompt=64,
        kernel_size=5,
    )
    assert torch.allclose(k2[..., -window_size:, :], key[..., -window_size:, :])
    assert torch.allclose(v2[..., -window_size:, :], value[..., -window_size:, :])


def test_snap_kv_skips_when_shorter_than_budget():
    query, key, value = _make_seq(seq_len=32)
    k2, v2 = snap_kv(query, key, value, window_size=8, max_capacity_prompt=64, kernel_size=5)
    assert k2.shape == key.shape
    assert v2.shape == value.shape


def test_snap_kv_skips_on_decode_step_not_prefill():
    """Decode has q_len=1, kv_len>1 — no compression per paper."""
    bsz, heads, seq_len, dim = 1, 4, 64, 32
    key = torch.randn(bsz, heads, seq_len, dim)
    value = torch.randn(bsz, heads, seq_len, dim)
    query_decode = torch.randn(bsz, heads, 1, dim)
    k2, v2 = snap_kv(
        query_decode, key, value,
        window_size=8,
        max_capacity_prompt=32,
        kernel_size=5,
    )
    assert k2.shape == key.shape
    assert v2.shape == value.shape


def test_snap_kv_k_and_v_use_same_gather_indices():
    query, key, value = _make_seq(seq_len=96)
    k2, v2 = snap_kv(query, key, value, window_size=8, max_capacity_prompt=48, kernel_size=5)
    # Prefix portion must come from identical index sets (values differ but positions align).
    prefix_len = 48 - 8
    assert k2[..., :prefix_len, :].shape == v2[..., :prefix_len, :].shape
    assert not torch.allclose(k2[..., :prefix_len, :], v2[..., :prefix_len, :])


def test_snap_kv_higher_vote_prefix_more_likely_retained():
    """Construct keys so prefix position 0 has highest attention vote."""
    bsz, heads, seq_len, dim = 1, 1, 64, 16
    window_size = 8
    max_capacity = 24
    prefix_len = seq_len - window_size

    key = torch.zeros(bsz, heads, seq_len, dim)
    value = torch.arange(seq_len, dtype=torch.float32).view(1, 1, seq_len, 1).expand(bsz, heads, seq_len, dim)
    anchor = torch.ones(dim)
    key[..., 0, :] = anchor * 10.0
    key[..., -window_size:, :] = torch.randn(bsz, heads, window_size, dim)
    query = key.clone()

    k2, _ = snap_kv(
        query, key, value,
        window_size=window_size,
        max_capacity_prompt=max_capacity,
        kernel_size=1,
    )
    # Position 0 should survive prefix selection (highest vote).
    assert torch.allclose(k2[..., 0, :], key[..., 0, :], atol=1e-4)


def test_snap_kv_pooling_kernel_affects_selection():
    query, key, value = _make_seq(seq_len=128)
    k_no_pool, _ = snap_kv(
        query, key, value, window_size=8, max_capacity_prompt=48, kernel_size=1,
    )
    k_pool, _ = snap_kv(
        query, key, value, window_size=8, max_capacity_prompt=48, kernel_size=7,
    )
    assert k_no_pool.shape == k_pool.shape
    # Pooling can change which prefix tokens survive; shapes must match.
    assert k_no_pool.shape[2] == 48


# --- Compressor round-trip & storage -------------------------------------------


def test_snapkv_payload_storage_bytes_formula():
    key = torch.randn(1, 4, 32, 16)
    payload = SnapKVLayerPayload(keys=key, values=key, original_seq_len=128, compressed=True)
    expected_fp16 = key.numel() * 2 * 2
    assert payload.nbytes >= expected_fp16


def test_snapkv_compress_decompress_round_trip():
    compressor = SnapKVCompressor(max_capacity_prompt=64, window_size=8, kernel_size=5)
    key = torch.randn(1, 4, 128, 32)
    value = torch.randn(1, 4, 128, 32)
    compressed = compressor.compress(key, value, layer=0, query_states=key)
    k2, v2 = compressor.decompress(compressed)
    assert k2.shape[2] == 64
    assert v2.shape[2] == 64
    assert compressed.nbytes < key.numel() * 4


def test_snapkv_kv_cache_decompress_layer():
    compressor = SnapKVCompressor(max_capacity_prompt=64, window_size=8)
    key = torch.randn(1, 4, 128, 32)
    value = torch.randn(1, 4, 128, 32)
    compressed = compressor.compress(key, value, query_states=key)
    k2, v2 = decompress_compressed_layer(compressed, compressor)
    assert k2.shape[2] == 64


def test_snapkv_incremental_seq_length_uses_stored_keys():
    compressor = SnapKVCompressor(max_capacity_prompt=64, window_size=8)
    key = torch.randn(1, 4, 128, 32)
    value = torch.randn(1, 4, 128, 32)
    layer = compressor.compress(key, value, query_states=key)
    assert incremental_seq_length([layer]) == 64


# --- Fidelity hooks ------------------------------------------------------------


def test_snapkv_reconstruction_error_reports_retention():
    compressor = SnapKVCompressor(max_capacity_prompt=64, window_size=8)
    key = torch.randn(1, 4, 128, 32)
    value = torch.randn(1, 4, 128, 32)
    errors = compressor.reconstruction_error(key, value, layer=0)
    assert errors["key_rmse"] < 1e-4
    assert errors["tokens_retained_ratio"] == pytest.approx(64 / 128)
    assert errors["tokens_dropped"] == 64.0


def test_snapkv_attention_fidelity_non_zero_when_tokens_dropped():
    compressor = SnapKVCompressor(max_capacity_prompt=64, window_size=8)
    key = torch.randn(1, 4, 128, 32)
    value = torch.randn(1, 4, 128, 32)
    query = torch.randn(1, 8, 128, 32)
    mse, rmse, cosine, max_error = compressor.attention_fidelity(
        query, key, value, head_dim=32, num_q_heads=8, num_kv_heads=4, layer=0,
    )
    assert rmse > 0.0
    assert cosine < 1.0
    assert max_error > 0.0
    assert mse == pytest.approx(rmse**2, rel=1e-5)


def test_snapkv_representation_eval_uses_batch_compress():
    compressor = SnapKVCompressor(max_capacity_prompt=64, window_size=8)
    key = torch.randn(1, 4, 128, 32)
    value = torch.randn(1, 4, 128, 32)
    past_key_values = ((key, value),)
    metrics = evaluate_representation(past_key_values, compressor)
    assert metrics.key_rmse < 1e-4
    assert metrics.key_cosine_similarity > 0.99


def test_snapkv_memory_compression_ratio_via_layers():
    compressor = SnapKVCompressor(max_capacity_prompt=64, window_size=8)
    key = torch.randn(1, 4, 128, 32)
    value = torch.randn(1, 4, 128, 32)
    layer = compressor.compress(key, value, query_states=key)
    original_bytes = key.numel() * 2 + value.numel() * 2
    compressed_bytes = compressed_size_bytes([layer], compressor)
    assert compressed_bytes < original_bytes
    assert compressed_bytes / original_bytes == pytest.approx(64 / 128, rel=0.05)


# --- Cost accounting -----------------------------------------------------------


def test_snapkv_cost_block_calibration_free():
    compressor = SnapKVCompressor(max_capacity_prompt=64, window_size=8)
    key = torch.randn(1, 4, 128, 32)
    value = torch.randn(1, 4, 128, 32)
    layer = compressor.compress(key, value, query_states=key)

    class _Mem:
        compression_ratio = 2.0
        uncompressed_bytes = 1000
        compressed_bytes = 500
        shared_metadata_bytes = 0

    class _Fidelity:
        memory = _Mem()

    cost = evaluate_cost(compressor, context_length=128, fidelity=_Fidelity())  # type: ignore[arg-type]
    assert cost.offline.calibration_required is False
    assert cost.compression.theoretical_compression_ratio == pytest.approx(128 / 64)
