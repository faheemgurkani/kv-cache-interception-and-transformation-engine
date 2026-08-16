"""BEHAVIOR / Reasoning: multi-step arithmetic accuracy under compressed KV.

Synthetic arithmetic chains (not scraped from a benchmark, so no license/contamination
concerns) exercise the model's ability to carry intermediate state through a compressed
KV cache across several autoregressive steps, rather than a single next-token guess.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

import torch

from compressors.base import KVCompressor
from framework.model import ModelLayer

_ANSWER_RE = re.compile(r"-?\d+")


@dataclass
class ReasoningMetrics:
    num_trials: int
    exact_match_accuracy: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _make_problem(rng: random.Random) -> tuple[str, int]:
    a, b, c = (rng.randint(2, 49) for _ in range(3))
    total = a + b - c
    prompt = (
        f"Q: Start with {a}. Add {b}. Subtract {c}. What is the final result?\n"
        "Show no work, answer with only the final integer.\nA:"
    )
    return prompt, total


@torch.no_grad()
def evaluate_reasoning(
    model_layer: ModelLayer,
    compressor: KVCompressor,
    num_trials: int = 10,
    max_new_tokens: int = 8,
    seed: int = 0,
) -> ReasoningMetrics:
    """Exact-match accuracy on synthetic add/subtract chains, generated via KVCacheEngine."""
    if hasattr(compressor, "reset_state"):
        compressor.reset_state()
    engine = model_layer.make_kv_engine(compressor)
    rng = random.Random(seed)

    correct = 0
    for _ in range(num_trials):
        prompt, expected = _make_problem(rng)
        input_ids = model_layer.tokenizer(prompt, return_tensors="pt").input_ids.to(model_layer.device)

        if hasattr(compressor, "reset_state"):
            compressor.reset_state()
        generated = engine.generate(input_ids, max_new_tokens=max_new_tokens)
        new_tokens = generated[:, input_ids.shape[1] :]
        completion = model_layer.tokenizer.decode(new_tokens[0], skip_special_tokens=True)

        match = _ANSWER_RE.search(completion)
        if match is not None and int(match.group()) == expected:
            correct += 1

    return ReasoningMetrics(
        num_trials=num_trials,
        exact_match_accuracy=correct / num_trials if num_trials else 0.0,
    )
