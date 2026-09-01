"""Comprehensive Palu tests: SVD mathematics, G-LRD, storage, fidelity, and integration."""

from __future__ import annotations

import pytest
import torch

from compressors.palu import PaluCompressor
from compressors.taxonomy import CompressionCategory, get_method_taxonomy
from eval.cost.accounting import evaluate_cost
from framework.kv_cache import compressed_size_bytes, decompress_compressed_layer, incremental_seq_length
from quantizers.palu import (
    PaluLatentPayload,
    build_layer_factors_from_projections,
    compress_kv_lowrank,
    decompress_kv_lowrank,
    project_hidden_to_latent,
    reconstruct_kv_from_latent,
    truncated_svd_factors,
)


# --- Taxonomy & metadata -------------------------------------------------------


def test_palu_taxonomy_matches_documentation():
    tax = get_method_taxonomy("palu")
    assert tax is not None
    assert tax.primary == CompressionCategory.PROJECTION
    assert CompressionCategory.MODIFIED_ATTENTION in tax.secondary
    assert tax.modifies_attention is True
    assert tax.calibration_free is False


def test_palu_theoretical_compression_ratio():
    compressor = PaluCompressor(compression_rate=0.5)
    assert compressor.theoretical_compression_ratio() == pytest.approx(2.0)


def test_palu_offline_cost_wikitext_calibration():
    compressor = PaluCompressor(calibration_samples=2048, calibration_seq_len=1024)
    meta = compressor.offline_cost_metadata()
    assert meta.calibration_required is True
    assert meta.calibration_dataset == "wikitext-2"
    assert meta.calibration_tokens == 2048 * 1024


# --- Truncated SVD mathematics (Eq. 4) ----------------------------------------


def test_truncated_svd_factors_shapes():
    weight = torch.randn(256, 512)
    a, b = truncated_svd_factors(weight, rank=32)
    assert a.shape == (256, 32)
    assert b.shape == (32, 512)
    assert (a @ b).shape == weight.shape


def test_truncated_svd_reconstruction_error_decreases_with_rank():
    weight = torch.randn(128, 64)
    err_r4 = torch.norm(weight - truncated_svd_factors(weight, 4)[0] @ truncated_svd_factors(weight, 4)[1], p="fro")
    err_r32 = torch.norm(weight - truncated_svd_factors(weight, 32)[0] @ truncated_svd_factors(weight, 32)[1], p="fro")
    assert err_r32 < err_r4


def test_truncated_svd_latent_forward_matches_full_projection():
    """h = x @ B^T, y = h @ A^T ≈ x @ W^T where W ≈ A @ B."""
    in_dim, out_dim, rank = 64, 128, 8
    w = torch.randn(out_dim, in_dim)
    a, b = truncated_svd_factors(w, rank)
    hidden = torch.randn(2, 16, in_dim)
    y_full = hidden @ w.t()
    h = hidden @ b.t()
    y_latent = h @ a.t()
    rel_err = (y_full - y_latent).norm() / y_full.norm()
    assert rel_err < 0.95


# --- G-LRD group decomposition (Section 3.2) -----------------------------------


def test_g_lrd_build_layer_factors_group_count():
    num_heads, head_dim, hidden = 8, 32, 512
    k_w = torch.randn(num_heads * head_dim, hidden)
    v_w = torch.randn(num_heads * head_dim, hidden)
    factors = build_layer_factors_from_projections(
        k_w, v_w,
        num_kv_heads=num_heads,
        head_dim=head_dim,
        group_size=4,
        compression_rate=0.5,
    )
    assert len(factors.groups) == 2
    assert factors.num_kv_heads == num_heads
    assert factors.head_dim == head_dim


def test_g_lrd_project_hidden_reconstruct_round_trip():
    num_heads, head_dim, hidden_dim = 4, 16, 64
    k_w = torch.randn(num_heads * head_dim, hidden_dim)
    v_w = torch.randn(num_heads * head_dim, hidden_dim)
    factors = build_layer_factors_from_projections(
        k_w, v_w,
        num_kv_heads=num_heads,
        head_dim=head_dim,
        group_size=2,
        compression_rate=0.5,
    )
    hidden = torch.randn(1, 10, hidden_dim)
    h_key, h_value = project_hidden_to_latent(hidden, factors)
    assert h_key.shape[1] == num_heads
    key, value = reconstruct_kv_from_latent(h_key, h_value, factors)
    assert key.shape == (1, num_heads, 10, head_dim)
    assert value.shape == key.shape
    y_k_ref = hidden @ k_w.t()
    y_k_ref = y_k_ref.view(1, 10, num_heads, head_dim).transpose(1, 2)
    rel_k = (key.float() - y_k_ref.float()).norm() / y_k_ref.norm()
    assert rel_k < 0.6


# --- Post-hoc KV low-rank (FIDELITY path) --------------------------------------


def test_palu_lowrank_exact_reconstruction_at_full_rank():
    key = torch.randn(1, 2, 8, 16)
    value = torch.randn(1, 2, 8, 16)
    full_rank = min(8, 2 * 16)
    payload = compress_kv_lowrank(key, value, rank=full_rank)
    k2, v2 = decompress_kv_lowrank(payload)
    assert torch.allclose(k2, key, atol=1e-4)
    assert torch.allclose(v2, value, atol=1e-4)


def test_palu_lowrank_rmse_decreases_with_rank():
    key = torch.randn(1, 4, 32, 64)
    value = torch.randn(1, 4, 32, 64)
    p4 = compress_kv_lowrank(key, value, rank=4)
    p16 = compress_kv_lowrank(key, value, rank=16)
    k4, _ = decompress_kv_lowrank(p4)
    k16, _ = decompress_kv_lowrank(p16)
    rmse4 = (k4 - key).pow(2).mean().sqrt()
    rmse16 = (k16 - key).pow(2).mean().sqrt()
    assert rmse16 < rmse4


def test_palu_payload_storage_bytes_count_latents_not_factors():
    key = torch.randn(1, 4, 16, 32)
    value = torch.randn(1, 4, 16, 32)
    payload = compress_kv_lowrank(key, value, rank=4)
    assert payload.b_key is not None
    assert payload.b_value is not None
    latent_bytes = payload.h_key.numel() * payload.h_key.element_size()
    latent_bytes += payload.h_value.numel() * payload.h_value.element_size()
    assert payload.factor_storage_bits() > 0
    assert payload.nbytes <= latent_bytes + 32


def test_palu_compressor_round_trip():
    compressor = PaluCompressor(compression_rate=0.25, group_size=2)
    key = torch.randn(1, 4, 32, 64)
    value = torch.randn(1, 4, 32, 64)
    compressed = compressor.compress(key, value, layer=0)
    k2, v2 = compressor.decompress(compressed)
    assert k2.shape == key.shape
    assert v2.shape == value.shape
    errors = compressor.reconstruction_error(key, value, layer=0)
    assert errors["key_rmse"] >= 0.0
    assert errors["value_rmse"] >= 0.0


def test_palu_kv_cache_decompress_layer():
    compressor = PaluCompressor(compression_rate=0.5)
    key = torch.randn(1, 4, 16, 32)
    value = torch.randn(1, 4, 16, 32)
    compressed = compressor.compress(key, value)
    k2, v2 = decompress_compressed_layer(compressed, compressor)
    assert k2.shape == key.shape


def test_palu_incremental_seq_length_from_latent():
    compressor = PaluCompressor(compression_rate=0.5)
    key = torch.randn(1, 4, 16, 32)
    value = torch.randn(1, 4, 16, 32)
    layer = compressor.compress(key, value)
    assert incremental_seq_length([layer]) == 16


def test_palu_memory_payload_uses_low_rank_latent():
    compressor = PaluCompressor(compression_rate=0.25, group_size=4)
    key = torch.randn(1, 8, 64, 128)
    value = torch.randn(1, 8, 64, 128)
    layer = compressor.compress(key, value)
    payload = layer.keys
    assert isinstance(payload, PaluLatentPayload)
    assert payload.rank <= 32
    assert payload.h_key.shape[1] == 64


# --- Cost accounting -----------------------------------------------------------


def test_palu_cost_block_reports_calibration():
    compressor = PaluCompressor(compression_rate=0.5)
    cost = evaluate_cost(compressor, context_length=128, fidelity=None, system=None)
    assert cost.offline.calibration_required is True
    assert cost.compression.theoretical_compression_ratio == pytest.approx(2.0)


# --- Registry ------------------------------------------------------------------


def test_palu_registry_factory():
    from compressors.registry import get_compressor

    c = get_compressor("palu", compression_rate=0.3, group_size=2)
    assert c.name == "palu"
    assert c.compression_rate == 0.3
