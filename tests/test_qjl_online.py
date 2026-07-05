"""QJL compressor readiness tests (lightweight)."""

import torch

from compressors.qjl import QJLCompressor


def test_qjl_shared_storage_counts_projection_once():
    compressor = QJLCompressor(seed=42)
    key = torch.randn(1, 8, 4, 128)
    compressor.compress_kv(key, layer=0, mode="key")
    assert compressor.shared_storage_bytes() > 0


def test_qjl_shared_storage_stable_across_layers():
    compressor = QJLCompressor(seed=42)
    key = torch.randn(1, 8, 4, 128)
    compressor.compress_kv(key, layer=0, mode="key")
    first = compressor.shared_storage_bytes()
    compressor.compress_kv(key, layer=1, mode="key")
    assert compressor.shared_storage_bytes() == first
