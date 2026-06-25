"""Smoke tests for the generic evaluation runner."""

from pathlib import Path

import pytest

from compressors.identity import IdentityCompressor
from eval.runner import EvaluationRunner
from framework.model import ModelLayer

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "qwen3_1.7b"


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="Model not downloaded")
def test_evaluation_runner_identity_smoke():
    model_layer = ModelLayer()
    runner = EvaluationRunner(model_layer=model_layer, compressor=IdentityCompressor())
    result = runner.run(
        context_length=128,
        run_perplexity=True,
        run_throughput=True,
        generated_tokens=8,
        perplexity_stride=64,
    )

    assert result.compressor == "identity"
    assert result.perplexity is not None and result.perplexity > 0
    assert result.memory.compression_ratio == 1.0
    assert result.fidelity.attention.rmse < 1e-3
    assert result.throughput is not None and result.throughput.tokens_per_second > 0
    assert result.throughput.online_compressed_kv is True
