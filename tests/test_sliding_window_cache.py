"""Audit B4: sliding-window models must not re-quantize stored tokens, and rebuilt caches
must keep the logical length so masks stay correct once the window has filled."""

from pathlib import Path

import pytest
import torch

from compressors.identity import IdentityCompressor
from eval.behavior.task_quality import (
    evaluate_perplexity_baseline_result,
    evaluate_perplexity_result,
)

GEMMA3 = Path(__file__).resolve().parent.parent / "models" / "gemma3_270m"
WINDOW = 32  # far below the real 512 so the window fills within a fast CPU test

_TEXT = (
    "The history of the printing press is closely tied to the spread of literacy in Europe. "
    "Before movable type, books were copied by hand, which made them rare and expensive. "
    "Johannes Gutenberg introduced a system that let printers assemble pages from reusable "
    "metal letters, and within a few decades presses had appeared in hundreds of towns. "
) * 2


@pytest.fixture()
def small_window_layer():
    pytest.importorskip("transformers")
    if not (GEMMA3 / "config.json").exists():
        pytest.skip("Gemma3 checkpoint not present")
    from framework.model import ModelLayer

    layer = ModelLayer(model_path=GEMMA3, device=torch.device("cpu"), torch_dtype=torch.float32)
    layer.model.config.sliding_window = WINDOW
    for block in layer.model.model.layers:
        if getattr(block.self_attn, "sliding_window", None):
            block.self_attn.sliding_window = WINDOW
    return layer


class _Counting(IdentityCompressor):
    def __init__(self):
        super().__init__()
        self.tokens_compressed = 0

    def compress_kv(self, tensor, layer=0, mode="key"):
        self.tokens_compressed += tensor.shape[2]
        return super().compress_kv(tensor, layer, mode)


def test_identity_engine_matches_baseline_after_window_fills(small_window_layer):
    layer = small_window_layer
    ids = layer.tokenize(_TEXT)[:, :112]
    assert ids.shape[1] - 64 > WINDOW  # prefill already exceeds the window
    base = evaluate_perplexity_baseline_result(layer, ids).perplexity
    got = evaluate_perplexity_result(layer, ids, IdentityCompressor()).perplexity
    assert abs(got - base) / base < 1e-3, (base, got)


def test_stored_tokens_are_never_recompressed(small_window_layer):
    layer = small_window_layer
    ids = layer.tokenize(_TEXT)[:, :112]
    comp = _Counting()
    evaluate_perplexity_result(layer, ids, comp)
    num_layers = len(layer.model.model.layers)
    linear_bound = 2 * num_layers * ids.shape[1]  # every token once per layer, K and V
    assert comp.tokens_compressed <= linear_bound, (comp.tokens_compressed, linear_bound)
