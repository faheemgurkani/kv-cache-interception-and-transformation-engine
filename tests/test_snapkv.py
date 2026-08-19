"""Tests for SnapKV compression (Category A: eviction)."""

from __future__ import annotations

import torch

from compressors.snapkv import SnapKVCompressor
from compressors.taxonomy import CompressionCategory, get_method_taxonomy
from quantizers.snapkv import snap_kv


def test_snapkv_taxonomy():
    tax = get_method_taxonomy("snapkv")
    assert tax is not None
    assert tax.primary == CompressionCategory.EVICTION
    assert tax.modifies_attention is False


def test_snap_kv_reduces_long_prefill():
    bsz, heads, seq_len, dim = 1, 4, 128, 32
    window_size = 16
    max_capacity = 64
    query = torch.randn(bsz, heads, seq_len, dim)
    key = torch.randn(bsz, heads, seq_len, dim)
    value = torch.randn(bsz, heads, seq_len, dim)
    k2, v2 = snap_kv(
        query,
        key,
        value,
        window_size=window_size,
        max_capacity_prompt=max_capacity,
        kernel_size=5,
    )
    assert k2.shape[2] == max_capacity
    assert v2.shape[2] == max_capacity


def test_snap_kv_skips_short_sequences():
    query = torch.randn(1, 4, 32, 32)
    key = torch.randn(1, 4, 32, 32)
    value = torch.randn(1, 4, 32, 32)
    k2, v2 = snap_kv(query, key, value, window_size=8, max_capacity_prompt=64, kernel_size=5)
    assert k2.shape == key.shape
    assert v2.shape == value.shape


def test_snapkv_compressor_round_trip():
    compressor = SnapKVCompressor(max_capacity_prompt=64, window_size=8, kernel_size=5)
    key = torch.randn(1, 4, 128, 32)
    value = torch.randn(1, 4, 128, 32)
    compressed = compressor.compress(key, value, layer=0, query_states=key)
    k2, v2 = compressor.decompress(compressed)
    assert k2.shape[2] == 64
    assert v2.shape[2] == 64
    assert compressed.nbytes < key.numel() * 2 + value.numel() * 2


def test_snapkv_offline_cost_calibration_free():
    compressor = SnapKVCompressor()
    meta = compressor.offline_cost_metadata()
    assert meta.calibration_required is False
