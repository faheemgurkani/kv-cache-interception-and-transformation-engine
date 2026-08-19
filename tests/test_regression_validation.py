"""WP5 regression validation: legacy paths unchanged after five-model refactor (§36)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from compressors.identity import IdentityCompressor
from compressors.registry import get_compressor
from eval.runner import EvaluationRunner
from framework.config import load_model_config
from framework.kv_cache import apply_compressor, iter_layer_kv
from framework.model import ModelLayer
from framework.model_capabilities import load_compatibility_manifest
from framework.state_compression import compress_state, compressed_attention_layers

OLMO2 = Path(__file__).resolve().parent.parent / "models" / "olmo2_1b"
QWEN3_06 = Path(__file__).resolve().parent.parent / "models" / "qwen3_0.6b"
QWEN3_17 = Path(__file__).resolve().parent.parent / "models" / "legacy" / "qwen3_1.7b"
CONFIG_ROOT = Path(__file__).resolve().parent.parent / "configs"


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _identity_smoke(model_path: Path, config_name: str) -> None:
    yaml_cfg = load_model_config(CONFIG_ROOT / config_name)
    model_layer = ModelLayer(model_path=model_path, device=_device())
    runner = EvaluationRunner(
        model_layer=model_layer,
        compressor=IdentityCompressor(),
        model_config=yaml_cfg,
    )
    result = runner.run(
        context_length=128,
        run_fidelity=True,
        run_behavior=True,
        run_perplexity=True,
        run_system=True,
        run_throughput=True,
        generated_tokens=4,
        perplexity_stride=64,
    )
    assert result.compatibility_probe is not None
    assert result.compatibility_probe.gate_passed("loader_state")
    assert result.compatibility_probe.gate_passed("attention")
    assert result.fidelity is not None
    assert result.fidelity.attention.cosine_similarity > 0.99
    assert result.fidelity.representation.key_cosine_similarity > 0.99
    assert result.fidelity.memory.compression_ratio == pytest.approx(1.0, rel=1e-3)
    assert result.behavior.perplexity is not None and result.behavior.perplexity > 0
    assert result.system.throughput.tokens_per_second > 0
    assert result.compatibility_probe.manifest is not None


@pytest.mark.skipif(not OLMO2.exists(), reason="OLMo2-1B not downloaded")
def test_wp5_olmo2_identity_regression():
    _identity_smoke(OLMO2, "model_olmo2_1b.yaml")


@pytest.mark.skipif(not QWEN3_06.exists(), reason="Qwen3-0.6B not downloaded")
def test_wp5_qwen3_0_6b_identity_regression():
    _identity_smoke(QWEN3_06, "model_qwen3_0.6b.yaml")


@pytest.mark.skipif(not QWEN3_17.exists(), reason="Qwen3-1.7B legacy not downloaded")
def test_wp5_qwen3_1_7b_identity_regression():
    _identity_smoke(QWEN3_17, "model_qwen3.yaml")


@pytest.mark.skipif(not OLMO2.exists(), reason="OLMo2-1B not downloaded")
def test_wp5_compress_state_matches_apply_compressor():
    model_layer = ModelLayer(model_path=OLMO2, device=_device())
    input_ids = model_layer.tokenize("Backward compatibility check")
    with torch.no_grad():
        outputs = model_layer.forward_with_cache(input_ids, use_cache=True)
    compressor = IdentityCompressor()
    legacy = apply_compressor(outputs.past_key_values, compressor)
    stateful = compressed_attention_layers(compress_state(outputs.past_key_values, compressor))
    assert len(stateful) == len(legacy)
    for a, b in zip(stateful, legacy, strict=True):
        assert a.nbytes == b.nbytes


@pytest.mark.skipif(not OLMO2.exists(), reason="OLMo2-1B not downloaded")
def test_wp5_iter_layer_kv_unchanged():
    model_layer = ModelLayer(model_path=OLMO2, device=_device())
    input_ids = model_layer.tokenize("Iterator stability")
    with torch.no_grad():
        outputs = model_layer.forward_with_cache(input_ids, use_cache=True)
    pairs = list(iter_layer_kv(outputs.past_key_values))
    assert len(pairs) == model_layer.config.num_hidden_layers
    key, value = pairs[0]
    assert key.ndim == 4 and value.ndim == 4
