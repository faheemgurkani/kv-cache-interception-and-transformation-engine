"""RocketKV online sparsity tests (no full model load)."""

import torch

from compressors.rocketkv import RocketKVCompressor
from framework.rocketkv_online import align_attention_mask, apply_online_kv_sparsity


def test_online_kv_sparsity_respects_budget():
    compressor = RocketKVCompressor(token_budget=16, hsa_budget=12, window_size=8)
    key = torch.randn(1, 8, 32, 128)
    value = torch.randn(1, 8, 32, 128)
    query = torch.randn(1, 16, 1, 128)

    sparse_k, sparse_v, kept = apply_online_kv_sparsity(compressor, 0, query, key, value)
    assert sparse_k.shape[2] <= 12
    assert sparse_v.shape == sparse_k.shape
    assert kept.numel() == sparse_k.shape[2]


def test_online_stage1_locks_permanent_prefix():
    compressor = RocketKVCompressor(token_budget=16, hsa_budget=16, window_size=8)
    key = torch.randn(1, 8, 32, 128)
    value = torch.randn(1, 8, 32, 128)
    query = torch.randn(1, 16, 1, 128)

    apply_online_kv_sparsity(compressor, 0, query, key, value)
    prefix = compressor._state(0).permanent_prefix_global.clone()

    key2 = torch.randn(1, 8, 33, 128)
    value2 = torch.randn(1, 8, 33, 128)
    apply_online_kv_sparsity(compressor, 0, query, key2, value2)
    assert torch.equal(compressor._state(0).permanent_prefix_global, prefix)


def test_align_attention_mask_truncates_oversized_indices():
    kept = torch.arange(139)
    mask = torch.zeros(1, 1, 1, 139)
    aligned = align_attention_mask(mask, kept, key_seq_len=136)
    assert aligned is not None
    assert aligned.shape == (1, 1, 1, 136)


def test_align_attention_mask_matches_sparse_keys():
    kept = torch.tensor([0, 2, 5])
    mask_2d = torch.zeros(1, 6)
    aligned = align_attention_mask(mask_2d, kept, key_seq_len=3)
    assert aligned is not None
    assert aligned.shape == (1, 3)

    mask_4d = torch.zeros(1, 1, 1, 6)
    aligned_4d = align_attention_mask(mask_4d, kept, key_seq_len=3)
    assert aligned_4d is not None
    assert aligned_4d.shape == (1, 1, 1, 3)
