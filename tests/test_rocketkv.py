"""RocketKV compression layer tests."""

import torch

from compressors.rocketkv import RocketKVCompressor
from quantizers.rocketkv import HybridSparseAttention, TokenSelector


def test_token_selector_reduces_sequence():
    selector = TokenSelector(keep_ratio=0.5, window_size=8)
    key = torch.randn(1, 8, 32, 128)
    value = torch.randn(1, 8, 32, 128)
    indices, kept_key, kept_value = selector.select(key, value)
    assert kept_key.shape[2] < key.shape[2]
    assert kept_value.shape[2] == kept_key.shape[2]
    assert indices.numel() == kept_key.shape[2]


def test_token_selector_keeps_short_sequences():
    selector = TokenSelector(keep_ratio=0.5, window_size=32)
    key = torch.randn(1, 8, 16, 128)
    value = torch.randn(1, 8, 16, 128)
    indices, kept_key, kept_value = selector.select(key, value)
    assert kept_key.shape[2] == 16
    assert indices.numel() == 16


def test_hybrid_sparse_attention_top_k():
    hsa = HybridSparseAttention(dynamic_top_k=8)
    query = torch.randn(1, 16, 1, 128)
    key = torch.randn(1, 8, 64, 128)
    value = torch.randn(1, 8, 64, 128)
    sparse_key, sparse_value, indices = hsa.select_top_k(query, key, value)
    assert sparse_key.shape[2] <= 8
    assert sparse_value.shape[2] == sparse_key.shape[2]


def test_rocketkv_compress_decompress():
    compressor = RocketKVCompressor(keep_ratio=0.5, window_size=8)
    key = torch.randn(1, 8, 32, 128)
    value = torch.randn(1, 8, 32, 128)
    compressed = compressor.compress(key, value, layer=0)
    k2, v2 = compressor.decompress(compressed)
    assert k2.shape[2] < key.shape[2]
    assert v2.shape == k2.shape
    assert compressed.nbytes < key.numel() * 2 + value.numel() * 2


def test_rocketkv_incremental_decompress_applies_selection():
    compressor = RocketKVCompressor(keep_ratio=0.5, window_size=8)
    key = torch.randn(1, 8, 32, 128)
    value = torch.randn(1, 8, 32, 128)
    key_payloads = []
    value_payloads = []
    for t in range(key.shape[2]):
        kp, vp = (
            compressor.compress_kv(key[:, :, t : t + 1, :], layer=0, mode="key"),
            compressor.compress_kv(value[:, :, t : t + 1, :], layer=0, mode="value"),
        )
        key_payloads.append(kp)
        value_payloads.append(vp)

    from compressors.base import CompressedKV

    layer = CompressedKV(
        keys=key_payloads,
        values=value_payloads,
        original_shape=tuple(key.shape),
        nbytes=0,
        layer=0,
    )
    k2, v2 = compressor.decompress(layer)
    assert k2.shape[2] < key.shape[2]


def test_rocketkv_dynamic_selection():
    compressor = RocketKVCompressor(keep_ratio=0.5, window_size=8, dynamic_top_k=16)
    key = torch.randn(1, 8, 32, 128)
    value = torch.randn(1, 8, 32, 128)
    compressor.compress(key, value, layer=0)
    query = torch.randn(1, 16, 1, 128)
    sparse_k, sparse_v, indices = compressor.select_dynamic_tokens(query, key, value, layer=0)
    assert sparse_k.shape[2] <= key.shape[2]
    assert sparse_v.shape[2] == sparse_k.shape[2]
    assert indices.numel() == sparse_k.shape[2]
