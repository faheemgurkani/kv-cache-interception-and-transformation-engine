"""QJL compression layer tests."""

import math

import torch

from compressors.qjl import QJLCompressor
from eval.attention_score_error import attention_scores


def test_qjl_key_roundtrip_shape():
    compressor = QJLCompressor()
    key = torch.randn(1, 8, 4, 128)
    payload = compressor.compress_kv(key, layer=0, mode="key")
    restored = compressor.decompress_kv(payload, mode="key")
    assert restored.shape == key.shape


def test_qjl_values_passthrough():
    compressor = QJLCompressor()
    value = torch.randn(1, 8, 4, 128)
    payload = compressor.compress_kv(value, layer=0, mode="value")
    restored = compressor.decompress_kv(payload, mode="value")
    assert torch.allclose(value, restored)


def test_qjl_compression_smaller_than_fp16():
    compressor = QJLCompressor()
    key = torch.randn(1, 8, 16, 128)
    value = torch.randn(1, 8, 16, 128)
    compressed = compressor.compress(key, value)
    raw_bytes = key.numel() * 2 + value.numel() * 2
    assert compressed.nbytes < raw_bytes


def test_qjl_attention_estimator_shape_and_correlation():
    compressor = QJLCompressor()
    query = torch.randn(1, 8, 4, 128)
    key = torch.randn(1, 8, 4, 128)
    payload = compressor.compress_kv(key, layer=0, mode="key")

    scores_est = compressor.estimate_attention_scores(query, payload, head_dim=128)
    scores_true = attention_scores(query, key, head_dim=128)

    assert scores_est.shape == scores_true.shape
    corr = torch.corrcoef(
        torch.stack([scores_est.flatten(), scores_true.flatten()])
    )[0, 1]
    assert corr > 0.2


def test_qjl_layer_compress_decompress():
    compressor = QJLCompressor()
    key = torch.randn(1, 8, 4, 128)
    value = torch.randn(1, 8, 4, 128)
    compressed = compressor.compress(key, value, layer=0)
    k2, v2 = compressor.decompress(compressed)
    assert k2.shape == key.shape
    assert v2.shape == value.shape
    assert torch.allclose(value, v2)
    errors = compressor.reconstruction_error(key, value)
    assert errors["key_rmse"] > 0
    assert errors["value_rmse"] < 1e-5
