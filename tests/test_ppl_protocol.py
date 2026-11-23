"""Audit B1: perplexity must be measured with compressed KV driving decode."""

from pathlib import Path

import pytest
import torch

from eval.behavior.task_quality import (
    evaluate_perplexity_baseline_result,
    evaluate_perplexity_result,
    ppl_prefill_length,
)

GEMMA3 = Path(__file__).resolve().parent.parent / "models" / "gemma3_270m"

_TEXT = (
    "The history of the printing press is closely tied to the spread of literacy in Europe. "
    "Before movable type, books were copied by hand, which made them rare and expensive. "
    "Johannes Gutenberg introduced a system that let printers assemble pages from reusable "
    "metal letters, and within a few decades presses had appeared in hundreds of towns. "
    "Scholars could now compare identical editions, merchants could publish price lists, and "
    "reformers could circulate pamphlets faster than authorities were able to suppress them."
)


@pytest.fixture(autouse=True)
def _production_ppl_window(monkeypatch):
    """These tests assert on the production 64-token window; undo the suite-wide shrink."""
    monkeypatch.delenv("KV_PPL_DECODE_TOKENS", raising=False)


def test_ppl_prefill_length_scores_decode_window():
    assert ppl_prefill_length(128) == 64
    assert ppl_prefill_length(256) == 192
    assert ppl_prefill_length(512) == 448
    assert ppl_prefill_length(32) == 16  # short inputs: at most half is scored
    with pytest.raises(ValueError):
        ppl_prefill_length(1)


@pytest.fixture(scope="module")
def gemma_fp32():
    pytest.importorskip("transformers")
    if not (GEMMA3 / "config.json").exists():
        pytest.skip("Gemma3 checkpoint not present")
    from framework.model import ModelLayer

    layer = ModelLayer(model_path=GEMMA3, device=torch.device("cpu"), torch_dtype=torch.float32)
    ids = layer.tokenize(_TEXT)[:, :96]
    return layer, ids


def test_identity_matches_baseline_on_same_positions(gemma_fp32):
    from compressors.identity import IdentityCompressor

    layer, ids = gemma_fp32
    base = evaluate_perplexity_baseline_result(layer, ids)
    ident = evaluate_perplexity_result(layer, ids, IdentityCompressor())
    expected = ids.size(1) - ppl_prefill_length(ids.size(1))
    assert base.n_tokens == ident.n_tokens == expected
    assert abs(ident.perplexity - base.perplexity) / base.perplexity < 1e-3


def test_ppl_reflects_lossy_compression(gemma_fp32):
    """Regression: TurboQuant PPL used to equal identity at every bit width."""
    from compressors.identity import IdentityCompressor
    from compressors.registry import get_compressor

    layer, ids = gemma_fp32
    ident = evaluate_perplexity_result(layer, ids, IdentityCompressor()).perplexity
    b4 = evaluate_perplexity_result(layer, ids, get_compressor("turboquant", stage="full", bitwidth=4)).perplexity
    b2 = evaluate_perplexity_result(layer, ids, get_compressor("turboquant", stage="full", bitwidth=2)).perplexity
    assert b4 > ident * 1.001, (ident, b4)
    assert b2 > b4 * 1.5, (b4, b2)
