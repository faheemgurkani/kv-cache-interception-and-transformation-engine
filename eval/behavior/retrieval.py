"""BEHAVIOR / Retrieval: needle-in-haystack recall under compressed KV.

Embeds a unique fact at a controlled depth inside a long filler context, then asks
the model to recall it through the KVCacheEngine (compressed KV in the autoregressive
loop, same code path as task_quality/throughput). This tests whether a compressor
preserves *specific* early-context information, not just aggregate perplexity.
"""

from __future__ import annotations

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


@dataclass
class RetrievalMetrics:
    needle_depth_frac: float
    context_length: int
    num_trials: int
    exact_match_accuracy: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _build_prompt(tokenizer, context_length: int, depth_frac: float, code: str) -> tuple[str, int]:
    needle = f"The secret code is {code}. Remember this code. "
    question = "\nQuestion: What is the secret code?\nAnswer: The secret code is"

    filler_tokens_needed = context_length - len(tokenizer(needle + question).input_ids)
    filler_tokens_needed = max(filler_tokens_needed, 0)
    repeat = (filler_tokens_needed // max(len(tokenizer(_FILLER).input_ids), 1)) + 1
    filler = _FILLER * repeat

    split_at = int(len(filler) * depth_frac)
    prompt = filler[:split_at] + needle + filler[split_at:] + question
    return prompt, len(code)


@torch.no_grad()
def evaluate_retrieval(
    model_layer: ModelLayer,
    compressor: KVCompressor,
    context_length: int = 512,
    depth_frac: float = 0.5,
    num_trials: int = 5,
    max_new_tokens: int = 6,
    seed: int = 0,
) -> RetrievalMetrics:
    """Exact-match accuracy for recalling a numeric code inserted at `depth_frac`."""
    if hasattr(compressor, "reset_state"):
        compressor.reset_state()
    rng = random.Random(seed)

    correct = 0
    for _ in range(num_trials):
        code = "".join(rng.choice(_CODE_DIGITS) for _ in range(5))
        prompt, code_len = _build_prompt(model_layer.tokenizer, context_length, depth_frac, code)
        input_ids = model_layer.tokenizer(prompt, return_tensors="pt").input_ids.to(model_layer.device)
        if input_ids.shape[1] > context_length:
            input_ids = input_ids[:, :context_length]

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
