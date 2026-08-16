"""BEHAVIOR / Instruction following: format-compliance rate under compressed KV.

Each prompt gives an explicit output-format constraint (single word, from a fixed
set). Compliance is checked structurally (does the completion match the required
format), independent of whether the *content* is also correct — this isolates
whether compression degrades the model's ability to follow instructions at all,
which perplexity alone does not capture.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

import torch

from compressors.base import KVCompressor
from framework.model import ModelLayer

_CHOICES = ("yes", "no")
_WORD_RE = re.compile(r"[a-zA-Z]+")

_TEMPLATE = (
    "Answer the question with exactly one word, either \"yes\" or \"no\". "
    "Do not explain.\nQuestion: {question}\nAnswer:"
)
_QUESTIONS = (
    "Is water wet?",
    "Is the sky green?",
    "Do fish live in trees?",
    "Is ice cold?",
    "Can birds fly?",
    "Is fire cold?",
)


@dataclass
class InstructionFollowingMetrics:
    num_trials: int
    format_compliance_rate: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@torch.no_grad()
def evaluate_instruction_following(
    model_layer: ModelLayer,
    compressor: KVCompressor,
    num_trials: int = 6,
    max_new_tokens: int = 4,
    seed: int = 0,
) -> InstructionFollowingMetrics:
    """Fraction of completions that are a single word from the allowed choice set."""
    if hasattr(compressor, "reset_state"):
        compressor.reset_state()
    engine = model_layer.make_kv_engine(compressor)
    rng = random.Random(seed)
    questions = list(_QUESTIONS)
    rng.shuffle(questions)
    trials = questions[: min(num_trials, len(questions))] or questions[:num_trials]

    compliant = 0
    for question in trials:
        prompt = _TEMPLATE.format(question=question)
        input_ids = model_layer.tokenizer(prompt, return_tensors="pt").input_ids.to(model_layer.device)

        if hasattr(compressor, "reset_state"):
            compressor.reset_state()
        generated = engine.generate(input_ids, max_new_tokens=max_new_tokens)
        new_tokens = generated[:, input_ids.shape[1] :]
        completion = model_layer.tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()

        words = _WORD_RE.findall(completion)
        if len(words) >= 1 and words[0].lower() in _CHOICES:
            compliant += 1

    n = len(trials)
    return InstructionFollowingMetrics(
        num_trials=n,
        format_compliance_rate=compliant / n if n else 0.0,
    )
