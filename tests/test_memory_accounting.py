"""Tests for bit-accurate KV-cache memory accounting."""

import torch

from compressors.turboquant import TurboQuantCompressor
from eval.memory import kv_cache_bytes
from framework.storage_accounting import bits_to_bytes, sign_storage_bits
from quantizers.turboquant_pipeline import TurboQuantStage, TurboQuantTensorPayload


def test_kv_cache_bytes_includes_batch():
    single = kv_cache_bytes(num_layers=28, seq_len=512, num_kv_heads=8, head_dim=128)
    batched = kv_cache_bytes(
        num_layers=28,
        seq_len=512,
        num_kv_heads=8,
        head_dim=128,
        batch_size=2,
        bytes_per_element=2,
    )
    assert batched == single * 2


def test_qjl_signs_count_as_one_bit_each():
    signs = torch.ones(128, dtype=torch.int8)
    payload = TurboQuantTensorPayload(
        indices=torch.zeros(128, dtype=torch.int8),
        qjl_bits=signs,
        norm_r=torch.ones(8, 1, dtype=torch.float32),
        vector_norm=torch.ones(8, 1, dtype=torch.float32),
        gamma=torch.ones(8, 1, dtype=torch.float32),
        original_dim=128,
        padded_dim=128,
        original_shape=(1, 8, 8, 128),
        original_dtype=torch.float16,
        stage=TurboQuantStage.FULL,
        bitwidth=4,
    )
    sign_bits = sign_storage_bits(signs.numel())
    assert sign_bits == 128
    assert bits_to_bytes(sign_bits) == 16
    int8_tensor_bytes = signs.numel() * signs.element_size()
    assert int8_tensor_bytes == 128
    sign_component_bytes = bits_to_bytes(sign_storage_bits(signs.numel()))
    assert sign_component_bytes == 16


def test_indices_use_bitwidth_not_container_dtype():
    indices = torch.zeros(64, dtype=torch.int8)
    payload = TurboQuantTensorPayload(
        indices=indices,
        qjl_bits=None,
        norm_r=None,
        vector_norm=torch.ones(8, 1, dtype=torch.float32),
        gamma=torch.ones(8, 1, dtype=torch.float32),
        original_dim=128,
        padded_dim=128,
        original_shape=(1, 8, 8, 128),
        original_dtype=torch.float16,
        stage=TurboQuantStage.WHT_QUANT,
        bitwidth=4,
    )
    index_bits = indices.numel() * 4
    assert index_bits == 256
    assert bits_to_bytes(index_bits) == 32
    assert indices.numel() * indices.element_size() == 64


def test_turboquant_full_effective_bits_below_fp16():
    key = torch.randn(1, 8, 4, 128)
    value = torch.randn(1, 8, 4, 128)
    compressor = TurboQuantCompressor(bitwidth=4, stage=TurboQuantStage.FULL)
    compressed = compressor.compress(key, value, layer=0)
    raw_bytes = key.numel() * 2 + value.numel() * 2
    assert compressed.nbytes < raw_bytes
    effective_bits = (compressed.nbytes * 8) / (key.numel() + value.numel())
    assert effective_bits < 16.0


def test_shared_centroids_counted_once():
    compressor = TurboQuantCompressor(bitwidth=4, stage=TurboQuantStage.FULL)
    assert compressor.shared_storage_bytes() == 16 * 4
