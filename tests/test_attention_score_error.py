"""Tests for QK^T attention fidelity metrics."""

from pathlib import Path

import pytest
import torch

from compressors.identity import IdentityCompressor
from compressors.turboquant import TurboQuantCompressor
from eval.attention_score_error import evaluate_attention_fidelity
from framework.model import ModelLayer
from quantizers.turboquant_pipeline import TurboQuantStage

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "qwen3_1.7b"


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="Model not downloaded")
def test_identity_attention_fidelity_is_near_zero():
    model_layer = ModelLayer()
    ids = model_layer.tokenize("Attention fidelity identity baseline check.")[:, :32]
    metrics = evaluate_attention_fidelity(model_layer, ids, IdentityCompressor())
    assert metrics.rmse < 1e-3
    assert metrics.cosine_similarity > 0.999
    assert metrics.max_error < 1e-2
    assert len(metrics.per_layer) == model_layer.config.num_hidden_layers


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="Model not downloaded")
def test_turboquant_attention_fidelity_is_measured():
    model_layer = ModelLayer()
    ids = model_layer.tokenize("Attention fidelity TurboQuant measurement check.")[:, :32]
    compressor = TurboQuantCompressor(bitwidth=4, stage=TurboQuantStage.FULL)
    metrics = evaluate_attention_fidelity(model_layer, ids, compressor)
    assert metrics.rmse >= 0.0
    assert -1.0 <= metrics.cosine_similarity <= 1.0
    assert metrics.max_error >= 0.0
    assert len(metrics.per_layer) == model_layer.config.num_hidden_layers


def test_attention_scores_shape():
    from eval.attention_score_error import attention_scores

    q = torch.randn(1, 16, 4, 128)
    k = torch.randn(1, 16, 4, 128)
    scores = attention_scores(q, k, head_dim=128)
    assert scores.shape == (1, 16, 4, 4)
