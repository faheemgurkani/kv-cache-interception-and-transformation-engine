"""Smoke tests for the generic evaluation runner."""

from pathlib import Path

import pytest

from compressors.identity import IdentityCompressor
from eval.runner import EvaluationRunner
from framework.config import load_model_config
from framework.model import ModelLayer

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "legacy" / "qwen3_1.7b"
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "model_qwen3.yaml"


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="Model not downloaded")
def test_evaluation_runner_identity_smoke():
    model_layer = ModelLayer(model_path=MODEL_DIR)
    yaml_cfg = load_model_config(CONFIG_PATH)
    runner = EvaluationRunner(
        model_layer=model_layer,
        compressor=IdentityCompressor(),
        model_config=yaml_cfg,
    )
    result = runner.run(
        context_length=128,
        run_perplexity=True,
        run_throughput=True,
        generated_tokens=8,
        perplexity_stride=64,
    )

    assert result.compressor == "identity"
    assert result.perplexity is not None and result.perplexity > 0
    assert result.behavior.retrieval is not None
    assert result.behavior.instruction_following is not None
    assert result.memory.compression_ratio == 1.0
    assert result.fidelity.attention.rmse < 1e-3
    assert result.fidelity.recurrent.applicable is False
    assert result.throughput is not None and result.throughput.tokens_per_second > 0
    assert result.throughput.online_compressed_kv is True

    assert result.compatibility_probe is not None
    assert result.compatibility_probe.gate_passed("loader_state")
    assert result.compatibility_probe.gate_passed("attention")
    assert result.compatibility_probe.gate_passed("state_semantics")
    assert result.compatibility_probe.manifest is not None
    assert result.compatibility_probe.manifest["architecture"]["family"] == "gqa"

    payload = result.to_dict()
    assert payload["compatibility_gates"] is not None
    assert payload["compatibility_manifest"] is not None
    assert payload["fidelity"]["recurrent"]["applicable"] is False
    assert payload["cost"] is not None
    assert payload["cost"]["compression"]["theoretical_compression_ratio"] == 1.0
    assert payload["cost"]["offline"]["calibration_required"] is False
    assert payload["cost"]["online"]["end_to_end_decode_cost_ms"] is not None
    assert payload["controlled_conditions"] is not None
    assert payload["controlled_conditions"]["phase"] == "7"
    assert payload["controlled_conditions"]["variable"]["compressor"] == "identity"
    assert payload["controlled_conditions"]["fixed"]["context_length"] == 128
    assert payload["controlled_conditions"]["fixed"]["decoding_configuration"]["strategy"] == "greedy"
    assert payload["controlled_conditions"]["fixed"]["hardware"]["device_type"] in {"cpu", "mps", "cuda"}
