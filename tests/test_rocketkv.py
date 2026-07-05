"""RocketKV compression layer tests."""

import torch

from compressors.rocketkv import RocketKVCompressor
from quantizers.rocketkv import HybridSparseAttention, TokenSelector


def test_token_selector_budget_reduces_sequence():
    selector = TokenSelector(window_size=8)
    key = torch.randn(1, 8, 32, 128)
    value = torch.randn(1, 8, 32, 128)
    indices, kept_key, kept_value, _ = selector.select_with_budget(key, value, max_tokens=16)
    assert kept_key.shape[2] == 16
    assert kept_value.shape[2] == 16
    assert indices.numel() == 16


def test_token_selector_keeps_short_sequences():
    selector = TokenSelector(window_size=32)
    key = torch.randn(1, 8, 16, 128)
    value = torch.randn(1, 8, 16, 128)
    indices, kept_key, kept_value, _ = selector.select_with_budget(key, value, max_tokens=64)
    assert kept_key.shape[2] == 16
    assert indices.numel() == 16


def test_hybrid_sparse_attention_respects_budget_and_permanent():
    hsa = HybridSparseAttention(attention_budget=8)
    query = torch.randn(1, 16, 1, 128)
    key = torch.randn(1, 8, 64, 128)
    value = torch.randn(1, 8, 64, 128)
    permanent = torch.tensor([0, 5, 10])
    sparse_key, sparse_value, indices = hsa.select_with_budget(
        query, key, value, max_tokens=8, permanent_indices=permanent
    )
    assert sparse_key.shape[2] <= 8
    assert sparse_value.shape[2] == sparse_key.shape[2]
    assert set(permanent.tolist()).issubset(set(indices.tolist()))


def test_rocketkv_compress_decompress():
    compressor = RocketKVCompressor(token_budget=16, window_size=8)
    key = torch.randn(1, 8, 32, 128)
    value = torch.randn(1, 8, 32, 128)
    compressed = compressor.compress(key, value, layer=0)
    k2, v2 = compressor.decompress(compressed)
    assert k2.shape[2] == 16
    assert v2.shape == k2.shape
    assert compressed.nbytes < key.numel() * 2 + value.numel() * 2


def test_rocketkv_stage1_locks_once():
    compressor = RocketKVCompressor(token_budget=16, window_size=8)
    key = torch.randn(1, 8, 32, 128)
    value = torch.randn(1, 8, 32, 128)
    idx1, _, _, _, _ = compressor.apply_stage1(key, value, layer=0)
    assert compressor._state(0).stage1_locked
    prefix = compressor._state(0).permanent_prefix_global.clone()

    key2 = torch.randn(1, 8, 33, 128)
    value2 = torch.randn(1, 8, 33, 128)
    idx2, k2, _, _, _ = compressor.apply_stage1(key2, value2, layer=0)
    assert torch.equal(compressor._state(0).permanent_prefix_global, prefix)
    assert k2.shape[2] <= 16
    assert idx2.numel() <= 16


def test_rocketkv_reconstruction_error_reports_retention():
    compressor = RocketKVCompressor(token_budget=16, window_size=8)
    key = torch.randn(1, 8, 32, 128)
    value = torch.randn(1, 8, 32, 128)
    errors = compressor.reconstruction_error(key, value, layer=0)
    assert errors["key_rmse"] < 1e-5
    assert errors["tokens_retained_ratio"] == 16 / 32
    assert errors["tokens_dropped"] == 16


def test_rocketkv_attention_fidelity_not_zero():
    compressor = RocketKVCompressor(token_budget=16, window_size=8)
    key = torch.randn(1, 8, 32, 128)
    value = torch.randn(1, 8, 32, 128)
    query = torch.randn(1, 16, 32, 128)
    mse, rmse, cosine, max_error = compressor.attention_fidelity(
        query, key, value, head_dim=128, num_q_heads=16, num_kv_heads=8, layer=0
    )
    assert rmse > 0.0
    assert cosine < 1.0
    assert max_error > 0.0


def test_rocketkv_dynamic_selection():
    compressor = RocketKVCompressor(token_budget=16, hsa_budget=12, window_size=8)
    key = torch.randn(1, 8, 32, 128)
    value = torch.randn(1, 8, 32, 128)
    compressor.apply_stage1(key, value, layer=0)
    query = torch.randn(1, 16, 1, 128)
    sparse_k, sparse_v, indices = compressor.select_dynamic_tokens(query, key, value, layer=0)
    assert sparse_k.shape[2] <= 12
    assert sparse_v.shape[2] == sparse_k.shape[2]
    assert indices.numel() == sparse_k.shape[2]
