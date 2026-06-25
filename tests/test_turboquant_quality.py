"""TurboQuant online quality regression tests."""

import math
from pathlib import Path

import pytest

from compressors.identity import IdentityCompressor
from compressors.identity import IdentityCompressor
from compressors.turboquant import TurboQuantCompressor
from data.loader import build_long_context_ids, load_wikitext2
from eval.perplexity import evaluate_perplexity
from framework.model import ModelLayer
from quantizers.turboquant_pipeline import TurboQuantStage

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "qwen3_1.7b"
CONTEXT_LENGTH = 128


def _eval_ids(model_layer: ModelLayer) -> "torch.Tensor":
    dataset = load_wikitext2()
    return build_long_context_ids(model_layer.tokenizer, dataset, CONTEXT_LENGTH).to(model_layer.device)


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="Model not downloaded")
def test_wht_only_online_ppl_matches_baseline():
    model_layer = ModelLayer()
    ids = _eval_ids(model_layer)
    ppl = evaluate_perplexity(
        model_layer,
        ids,
        TurboQuantCompressor(bitwidth=4, stage=TurboQuantStage.WHT_ONLY),
        stride=512,
    )
    baseline = evaluate_perplexity(model_layer, ids, IdentityCompressor(), stride=512)
    assert math.isfinite(ppl)
    assert abs(ppl - baseline) / baseline < 0.05


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="Model not downloaded")
def test_wht_quant_online_ppl_is_reasonable():
    model_layer = ModelLayer()
    ids = _eval_ids(model_layer)
    baseline = evaluate_perplexity(model_layer, ids, IdentityCompressor(), stride=512)
    ppl = evaluate_perplexity(
        model_layer,
        ids,
        TurboQuantCompressor(bitwidth=4, stage=TurboQuantStage.WHT_QUANT),
        stride=512,
    )
    assert math.isfinite(ppl)
    assert ppl < baseline * 2.0


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="Model not downloaded")
def test_full_turboquant_online_ppl_is_reasonable():
    model_layer = ModelLayer()
    ids = _eval_ids(model_layer)
    baseline = evaluate_perplexity(model_layer, ids, IdentityCompressor(), stride=512)
    ppl = evaluate_perplexity(
        model_layer,
        ids,
        TurboQuantCompressor(bitwidth=4, stage=TurboQuantStage.FULL),
        stride=512,
    )
    assert math.isfinite(ppl)
    assert ppl < baseline * 2.0


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="Model not downloaded")
def test_full_turboquant_attention_rmse_is_reasonable():
    from eval.attention_score_error import evaluate_attention_fidelity

    model_layer = ModelLayer()
    ids = _eval_ids(model_layer)
    metrics = evaluate_attention_fidelity(
        model_layer,
        ids,
        TurboQuantCompressor(bitwidth=4, stage=TurboQuantStage.FULL),
    )
    assert metrics.rmse < 10.0
    assert metrics.cosine_similarity > 0.8
