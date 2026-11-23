"""BEHAVIOR / Retrieval: needle-in-haystack recall under compressed KV.

Embeds a unique fact at a controlled depth inside a long filler context, then asks
the model to recall it through the KVCacheEngine (compressed KV in the autoregressive
loop, same code path as task_quality/throughput). This tests whether a compressor
preserves *specific* early-context information, not just aggregate perplexity.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import torch

from compressors.base import KVCompressor
from framework.model import ModelLayer

_FILLER = (
    "The quick brown fox jumps over the lazy dog near the riverbank. "
    "Weather patterns shift gradually across the continent each season. "
)
_CODE_DIGITS = "0123456789"
# 5 trials quantised accuracy to steps of 0.2; 20 gives 0.05 steps (still coarse: +/-~0.1 at 1 sigma).
DEFAULT_RETRIEVAL_TRIALS = 20


def default_retrieval_trials() -> int:
    """20 unless ``KV_RETRIEVAL_TRIALS`` is set (the test suite lowers it for CPU smoke runs)."""
    raw = os.environ.get("KV_RETRIEVAL_TRIALS", "").strip()
    return int(raw) if raw else DEFAULT_RETRIEVAL_TRIALS


@dataclass
class RetrievalMetrics:
    needle_depth_frac: float
    context_length: int
    num_trials: int
    exact_match_accuracy: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _build_prompt(tokenizer, context_length: int, depth_frac: float, code: str) -> tuple[str, int]:
    """Needle-in-haystack prompt that tokenizes to at most ``context_length`` tokens.

    The filler length is found by binary search so the prompt lands within a few tokens of
    ``context_length`` (it used to overshoot by up to one filler repeat: 129/251/491 tokens for
    128/256/512, and the later hard truncation could cut the question off; audit R1).
    """
    needle = f"The secret code is {code}. Remember this code. "
    question = "\nQuestion: What is the secret code?\nAnswer: The secret code is"

    def count(text: str) -> int:
        return len(tokenizer(text).input_ids)

    per_repeat = max(count(_FILLER), 1)
    budget = max(context_length - count(needle + question), 0)
    filler = _FILLER * (budget // per_repeat + 2)

    def assemble(n_chars: int) -> str:
        head = filler[:n_chars]
        split_at = int(len(head) * depth_frac)
        return head[:split_at] + needle + head[split_at:] + question

    lo, hi = 0, len(filler)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(assemble(mid)) <= context_length:
            lo = mid
        else:
            hi = mid - 1
    return assemble(lo), len(code)


@torch.no_grad()
def evaluate_retrieval(
    model_layer: ModelLayer,
    compressor: KVCompressor,
    context_length: int = 512,
    depth_frac: float = 0.5,
    num_trials: int | None = None,
    max_new_tokens: int = 6,
    seed: int = 0,
) -> RetrievalMetrics:
    """Exact-match accuracy for recalling a numeric code inserted at `depth_frac`."""
    num_trials = default_retrieval_trials() if num_trials is None else num_trials
    if hasattr(compressor, "reset_state"):
        compressor.reset_state()
    rng = random.Random(seed)

    correct = 0
    for _ in range(num_trials):
        code = "".join(rng.choice(_CODE_DIGITS) for _ in range(5))
        prompt, code_len = _build_prompt(model_layer.tokenizer, context_length, depth_frac, code)
        input_ids = model_layer.tokenizer(prompt, return_tensors="pt").input_ids.to(model_layer.device)

        if hasattr(compressor, "reset_state"):
            compressor.reset_state()
        # Fresh engine per trial so decode-prefix / HF cache state cannot accumulate.
        trial_engine = model_layer.make_kv_engine(compressor)
        generated = trial_engine.generate(input_ids, max_new_tokens=max_new_tokens)
        del trial_engine
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        new_tokens = generated[:, input_ids.shape[1] :]
        completion = model_layer.tokenizer.decode(new_tokens[0], skip_special_tokens=True)

        if code in completion:
            correct += 1

    return RetrievalMetrics(
        needle_depth_frac=depth_frac,
        context_length=context_length,
        num_trials=num_trials,
        exact_match_accuracy=correct / num_trials if num_trials else 0.0,
    )
