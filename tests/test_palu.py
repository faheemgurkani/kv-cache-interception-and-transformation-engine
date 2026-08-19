"""Tests for Palu compression (Category C + E)."""

from __future__ import annotations

import torch

from compressors.palu import PaluCompressor
from compressors.taxonomy import CompressionCategory, get_method_taxonomy
from quantizers.palu import compress_kv_lowrank, decompress_kv_lowrank, truncated_svd_factors


def test_palu_taxonomy():
    tax = get_method_taxonomy("palu")
    assert tax is not None
    assert tax.primary == CompressionCategory.PROJECTION
    assert CompressionCategory.MODIFIED_ATTENTION in tax.secondary
    assert tax.modifies_attention is True


def test_truncated_svd_factors_shape():
    weight = torch.randn(256, 512)
    a, b = truncated_svd_factors(weight, rank=32)
    assert a.shape == (256, 32)
    assert b.shape == (32, 512)
    approx = a @ b
    assert approx.shape == weight.shape


def test_palu_lowrank_round_trip():
    key = torch.randn(1, 4, 32, 64)
    value = torch.randn(1, 4, 32, 64)
    payload = compress_kv_lowrank(key, value, rank=4)
    k2, v2 = decompress_kv_lowrank(payload)
    assert k2.shape == key.shape
    assert v2.shape == value.shape
    assert payload.rank == 4


def test_palu_compressor_reconstruction_error():
    compressor = PaluCompressor(compression_rate=0.5, group_size=2)
    key = torch.randn(1, 4, 16, 32)
    value = torch.randn(1, 4, 16, 32)
    errors = compressor.reconstruction_error(key, value, layer=0)
    assert errors["key_rmse"] >= 0.0
    assert errors["value_rmse"] >= 0.0


def test_palu_offline_cost_requires_calibration():
    compressor = PaluCompressor()
    meta = compressor.offline_cost_metadata()
    assert meta.calibration_required is True
    assert meta.calibration_dataset == "wikitext-2"
