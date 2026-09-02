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
    assert errors["key_rmse"] < 3.0
    assert errors["value_rmse"] < 3.0


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


def test_turboquant_split_seq_roundtrip_matches_full():
    compressor = TurboQuantCompressor(bitwidth=4, stage=TurboQuantStage.FULL)
    x = torch.randn(1, 2, 8, 32)
    full = compressor.compress_kv(x, layer=0, mode="key")
    parts = compressor.split_seq_payload(full)
    assert len(parts) == 8
    restored = torch.cat([compressor.decompress_kv(part, mode="key") for part in parts], dim=2)
    batched = compressor.decompress_kv(full, mode="key")
    assert restored.shape == x.shape
    assert torch.allclose(restored, batched, atol=1e-5)


def test_hadamard_matrix_is_cached():
    from quantizers.hadamard import _HADAMARD_MATRIX_CACHE, hadamard_transform

    _HADAMARD_MATRIX_CACHE.clear()
    x = torch.randn(2, 16)
    hadamard_transform(x)
    assert _HADAMARD_MATRIX_CACHE
    first = next(iter(_HADAMARD_MATRIX_CACHE.values()))
    hadamard_transform(x)
    assert next(iter(_HADAMARD_MATRIX_CACHE.values())) is first


def test_incremental_decompress_appends_only_new_tokens():
    from compressors.base import CompressedKV
    from framework.kv_cache import append_decompressed_tokens, build_incremental_layer

    compressor = TurboQuantCompressor(bitwidth=4, stage=TurboQuantStage.WHT_QUANT)
    key = torch.randn(1, 2, 4, 16)
    value = torch.randn(1, 2, 4, 16)
    key_payloads = [compressor.compress_kv(key[:, :, t : t + 1], layer=0, mode="key") for t in range(4)]
    value_payloads = [compressor.compress_kv(value[:, :, t : t + 1], layer=0, mode="value") for t in range(4)]
    layer = build_incremental_layer(key, value, key_payloads, value_payloads, 0, compressor)

    prefix_k = torch.cat([compressor.decompress_kv(p, mode="key") for p in key_payloads[:2]], dim=2)
    prefix_v = torch.cat([compressor.decompress_kv(p, mode="value") for p in value_payloads[:2]], dim=2)

    calls = {"n": 0}
    orig = compressor.decompress_kv

    def counted(payload, mode="key"):
        calls["n"] += 1
        return orig(payload, mode=mode)

    compressor.decompress_kv = counted  # type: ignore[method-assign]
    out = append_decompressed_tokens([(prefix_k, prefix_v)], [layer], prefix_len=2, compressor=compressor)
    assert calls["n"] == 4  # 2 new keys + 2 new values
    assert out[0][0].shape[2] == 4
