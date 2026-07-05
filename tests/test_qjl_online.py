"""QJL online estimator tests."""

import torch

from compressors.qjl import QJLCompressor
from framework.qjl_online import qjl_eager_attention_forward


def test_qjl_online_payload_sync_and_estimator_attention():
    compressor = QJLCompressor(seed=42)
    query = torch.randn(1, 16, 1, 128)
    key = torch.randn(1, 8, 4, 128)
    value = torch.randn(1, 8, 4, 128)

    payloads = []
    for t in range(key.shape[2]):
        payloads.append(compressor.compress_key_token(0, key[:, :, t : t + 1, :]))

    out, weights = qjl_eager_attention_forward(
        query,
        value,
        payloads,
        compressor,
        attention_mask=None,
        scaling=128**-0.5,
        num_key_value_groups=2,
    )
    assert out.shape == (1, 1, 16, 128)
    assert weights.shape[-1] == key.shape[2]


def test_qjl_sync_from_incremental_cache():
    from compressors.base import CompressedKV
    from framework.kv_cache import build_incremental_layer

    compressor = QJLCompressor(seed=42)
    key = torch.randn(1, 8, 3, 128)
    value = torch.randn(1, 8, 3, 128)
    key_payloads = [compressor.compress_kv(key[:, :, t : t + 1, :], layer=0) for t in range(3)]
    value_payloads = [compressor.compress_kv(value[:, :, t : t + 1, :], layer=0, mode="value") for t in range(3)]
    layer = build_incremental_layer(key, value, key_payloads, value_payloads, 0, compressor)

    compressor.reset_state()
    compressor.sync_key_payloads_from_cache([layer])
    assert len(compressor.online_key_payloads(0)) == 3


def test_qjl_shared_storage_counts_projection_once():
    compressor = QJLCompressor(seed=42)
    key = torch.randn(1, 8, 4, 128)
    compressor.compress_kv(key, layer=0, mode="key")
    assert compressor.shared_storage_bytes() > 0

    first = compressor.shared_storage_bytes()
    compressor.compress_kv(key, layer=1, mode="key")
    assert compressor.shared_storage_bytes() == first
