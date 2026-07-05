"""RocketKV online sparsity tests (no full model load)."""

import torch

from compressors.rocketkv import RocketKVCompressor
from framework.rocketkv_online import align_attention_mask, apply_online_kv_sparsity


def test_incremental_layer_concatenates_without_selection():
    compressor = RocketKVCompressor(keep_ratio=0.5, window_size=8)
    key = torch.randn(1, 8, 32, 128)
    value = torch.randn(1, 8, 32, 128)
    key_payloads = []
    value_payloads = []
    for t in range(key.shape[2]):
        key_payloads.append(compressor.compress_kv(key[:, :, t : t + 1, :], layer=0, mode="key"))
        value_payloads.append(compressor.compress_kv(value[:, :, t : t + 1, :], layer=0, mode="value"))

    from compressors.base import CompressedKV

    layer = CompressedKV(
        keys=key_payloads,
        values=value_payloads,
        original_shape=tuple(key.shape),
        nbytes=0,
        layer=0,
    )
    k2, v2 = compressor.decompress_incremental_layer(layer)
    assert k2.shape == key.shape
    assert v2.shape == value.shape


def test_online_kv_sparsity_applies_both_stages():
    compressor = RocketKVCompressor(keep_ratio=0.5, window_size=8, dynamic_top_k=8)
    key = torch.randn(1, 8, 32, 128)
    value = torch.randn(1, 8, 32, 128)
    query = torch.randn(1, 16, 1, 128)

    sparse_k, sparse_v, kept = apply_online_kv_sparsity(compressor, 0, query, key, value)
    assert sparse_k.shape[2] < key.shape[2]
    assert sparse_v.shape == sparse_k.shape
    assert kept.numel() == sparse_k.shape[2]


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
