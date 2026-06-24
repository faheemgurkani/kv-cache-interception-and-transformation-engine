"""TurboQuant compression layer tests."""

import torch

from compressors.turboquant import TurboQuantCompressor
from quantizers.turboquant_pipeline import TurboQuantStage


def test_turboquant_wht_roundtrip():
    compressor = TurboQuantCompressor(bitwidth=4, stage=TurboQuantStage.WHT_ONLY)
    x = torch.randn(1, 8, 4, 128)
    payload = compressor.compress_kv(x, layer=0, mode="key")
    restored = compressor.decompress_kv(payload, mode="key")
    assert restored.shape == x.shape
    assert torch.allclose(x, restored, atol=1e-4)


def test_turboquant_full_roundtrip():
    compressor = TurboQuantCompressor(bitwidth=4, stage=TurboQuantStage.FULL)
    key = torch.randn(1, 8, 4, 128)
    value = torch.randn(1, 8, 4, 128)
    compressed = compressor.compress(key, value, layer=0)
    k2, v2 = compressor.decompress(compressed)
    assert k2.shape == key.shape
    assert v2.shape == value.shape
    errors = compressor.reconstruction_error(key, value)
    assert errors["key_rmse"] < 2.0
    assert errors["value_rmse"] < 2.0


def test_turboquant_stages_increasing_compression():
    key = torch.randn(1, 8, 2, 128)
    value = torch.randn(1, 8, 2, 128)
    wht = TurboQuantCompressor(bitwidth=4, stage=TurboQuantStage.WHT_ONLY)
    quant = TurboQuantCompressor(bitwidth=4, stage=TurboQuantStage.WHT_QUANT)
    full = TurboQuantCompressor(bitwidth=4, stage=TurboQuantStage.FULL)
    wht_bytes = wht.compress(key, value).nbytes
    quant_bytes = quant.compress(key, value).nbytes
    full_bytes = full.compress(key, value).nbytes
    assert wht_bytes > quant_bytes
    assert full_bytes > quant_bytes
