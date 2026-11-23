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


def test_qjl_attention_aligns_prefill_mask_on_decode():
    compressor = QJLCompressor(seed=42)
    query = torch.randn(1, 4, 1, 64)
    key = torch.randn(1, 1, 8, 64)
    value = torch.randn(1, 1, 8, 64)
    payloads = [compressor.compress_key_token(0, key[:, :, t : t + 1, :]) for t in range(8)]
    # Prefill-sized 4D mask (Gemma3 decode leak) — must not raise or broadcast to 8 queries.
    mask = torch.zeros(1, 1, 8, 8)
    mask[..., -1] = -1e9
    out, weights = qjl_eager_attention_forward(
        query,
        value,
        payloads,
        compressor,
        attention_mask=mask,
        scaling=64**-0.5,
        num_key_value_groups=4,
    )
    assert out.shape == (1, 1, 4, 64)
    assert weights.shape == (1, 4, 1, 8)
    assert torch.isfinite(out).all()


def test_ensure_key_payloads_one_payload_per_position():
    """Regression (audit B3): payloads must align 1:1 with key positions, no duplicates."""
    from framework.qjl_online import _ensure_key_payloads

    compressor = QJLCompressor(seed=42)
    key = torch.randn(1, 1, 16, 64)
    payloads = _ensure_key_payloads(compressor, 0, key)

    assert len(payloads) == 16
    assert len(compressor.online_key_payloads(0)) == 16  # no hidden duplicates

    batch = compressor.compress_kv(key, layer=0, mode="key")
    for t in (0, 1, 7, 15):
        assert torch.equal(payloads[t].sign_bits, batch.sign_bits[:, :, t : t + 1, :])
        assert torch.allclose(payloads[t].vector_norm, batch.vector_norm[:, :, t : t + 1, :])

    # Extending with later tokens (decode) keeps alignment.
    key2 = torch.cat([key, torch.randn(1, 1, 3, 64)], dim=2)
    payloads2 = _ensure_key_payloads(compressor, 0, key2)
    assert len(payloads2) == 19 and len(compressor.online_key_payloads(0)) == 19


def test_qjl_online_attention_matches_exact_with_wide_sketch_mqa():
    """Regression (audit B3): MQA (1 KV head, 4 Q heads) online path tracks exact attention."""
    import torch.nn.functional as F

    from framework.qjl_online import _ensure_key_payloads

    torch.manual_seed(0)
    compressor = QJLCompressor(seed=42, proj_dim=8192)
    q = torch.randn(1, 4, 12, 64)
    k = torch.randn(1, 1, 12, 64)
    v = torch.randn(1, 1, 12, 64)
    payloads = _ensure_key_payloads(compressor, 0, k)
    mask = torch.full((1, 1, 12, 12), float("-inf")).triu(1)
    out, _ = qjl_eager_attention_forward(
        q, v, payloads, compressor, attention_mask=mask, scaling=64**-0.5, num_key_value_groups=4
    )
    scores = torch.matmul(q, k.expand(1, 4, 12, 64).transpose(-2, -1)) * 64**-0.5 + mask
    exact = torch.matmul(F.softmax(scores, dim=-1), v.expand(1, 4, 12, 64)).transpose(1, 2)
    rel = ((out - exact).norm() / exact.norm()).item()
    assert rel < 0.25, rel
