"""QJL compression layer tests."""

import math

import torch

from compressors.qjl import QJLCompressor
from eval.fidelity.attention import attention_scores
from quantizers.qjl import projection_matrix, qjl_decode, qjl_encode


def test_qjl_projection_is_deterministic_and_reused():
    s1 = projection_matrix(128, seed=42)
    s2 = projection_matrix(128, seed=42)
    s3 = projection_matrix(128, seed=99)
    assert torch.equal(s1, s2)
    assert not torch.equal(s1, s3)


def test_qjl_encode_uses_sign_not_round():
    proj = projection_matrix(8, proj_dim=8, seed=1)
    residual = torch.linspace(-1, 1, 8)
    bits = qjl_encode(residual, proj)
    z = torch.einsum("ij,j->i", proj, residual)
    expected = torch.where(z >= 0, 1, -1).to(torch.int8)
    assert torch.equal(bits, expected)
    assert set(bits.unique().tolist()).issubset({-1, 1})


def test_qjl_decode_scaling():
    proj = projection_matrix(16, proj_dim=16, seed=7)
    x = torch.randn(16)
    norm = x.norm().unsqueeze(0)
    bits = qjl_encode(x, proj)
    restored = qjl_decode(bits, proj, norm)
    manual = torch.einsum("ji,i->j", proj, bits.float())
    manual = manual * (math.sqrt(math.pi / 2.0) / proj.shape[0]) * norm
    assert torch.allclose(restored, manual)


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


def test_qjl_estimator_keeps_query_projection_float():
    """ProdQJL: float Sq = S@q, binary only on keys (literature Def. 3.1 / Eq. 4)."""
    from quantizers.qjl_pipeline import QJLPipeline

    torch.manual_seed(0)
    d, m = 32, 64
    pipeline = QJLPipeline(seed=7, proj_dim=m)
    proj = pipeline._get_projection(d, torch.device("cpu"))
    assert proj.shape == (m, d)

    q = torch.randn(1, 1, 1, d)
    k = torch.randn(1, 1, 3, d)
    payload = pipeline.compress_tensor(k, mode="key")

    # Manual literature estimator
    sq = torch.einsum("md,btd->btm", proj, q[:, 0].float())  # float, not signed
    assert sq.abs().amax() > 1.0 + 1e-5  # would be ±1 if signed
    bk = payload.sign_bits.float()
    norms = payload.vector_norm.squeeze(-1)
    scale = math.sqrt(math.pi / 2.0) / m
    manual = scale * torch.einsum("btm,bkm->btk", sq, bk[:, 0]) * norms[:, 0].unsqueeze(1)

    got = pipeline.estimate_inner_products(q, payload)[:, 0]
    assert torch.allclose(got, manual, atol=1e-5)

    # Signing both sides is a different (biased) estimator — must not match code.
    sq_signed = torch.where(sq >= 0, 1.0, -1.0)
    both_signed = scale * torch.einsum("btm,bkm->btk", sq_signed, bk[:, 0]) * norms[:, 0].unsqueeze(1)
    assert not torch.allclose(got, both_signed, atol=1e-4)


def test_qjl_float_sq_lower_mae_than_both_signed():
    """Float-Sq ProdQJL should track q·k better than signing both vectors."""
    from quantizers.qjl_pipeline import QJLPipeline

    torch.manual_seed(123)
    d, m, n = 64, 128, 200
    pipeline = QJLPipeline(seed=11, proj_dim=m)
    proj = pipeline._get_projection(d, torch.device("cpu"))
    scale = math.sqrt(math.pi / 2.0) / m

    true_dots = []
    float_sq_errs = []
    both_signed_errs = []
    for _ in range(n):
        q = torch.randn(d)
        k = torch.randn(d)
        kn = k.norm().clamp(min=1e-8)
        sq = proj @ q
        bk = torch.where((proj @ k) >= 0, 1.0, -1.0)
        est_float = scale * kn * torch.dot(sq, bk)
        est_both = scale * kn * torch.dot(torch.where(sq >= 0, 1.0, -1.0), bk)
        true = torch.dot(q, k)
        true_dots.append(true.item())
        float_sq_errs.append(abs(est_float.item() - true.item()))
        both_signed_errs.append(abs(est_both.item() - true.item()))

    mae_float = sum(float_sq_errs) / n
    mae_both = sum(both_signed_errs) / n
    # Unbiased ProdQJL must beat angle-style both-signed hashing on MAE.
    assert mae_float < mae_both
    # Also check the pipeline path against true dots on a batched case.
    q_b = torch.randn(1, 2, 1, d)
    k_b = torch.randn(1, 2, 8, d)
    payload = pipeline.compress_tensor(k_b, mode="key")
    est = pipeline.estimate_inner_products(q_b, payload)
    true_b = torch.einsum("bhqd,bhkd->bhqk", q_b.float(), k_b.float())
    corr = torch.corrcoef(torch.stack([est.flatten(), true_b.flatten()]))[0, 1]
    assert corr > 0.5


def test_qjl_attention_fidelity_uses_estimator():
    compressor = QJLCompressor()
    query = torch.randn(1, 8, 4, 128)
    key = torch.randn(1, 8, 4, 128)
    value = torch.randn(1, 8, 4, 128)
    _, rmse, cosine, _ = compressor.attention_fidelity(
        query, key, value, head_dim=128, num_q_heads=8, num_kv_heads=8, layer=0
    )
    assert rmse > 0
    assert cosine < 1.0


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
