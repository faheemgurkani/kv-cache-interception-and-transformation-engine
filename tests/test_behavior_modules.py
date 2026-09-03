"""Unit and module-isolation tests for eval/behavior/."""

from __future__ import annotations

import math
import random
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from compressors.identity import IdentityCompressor
from eval.behavior import BehaviorMetrics, evaluate_behavior
from eval.behavior.instruction_following import (
    InstructionFollowingMetrics,
    _CHOICES,
    _TEMPLATE,
    _WORD_RE,
    evaluate_instruction_following,
)
from eval.behavior.reasoning import ReasoningMetrics, _ANSWER_RE, _make_problem, evaluate_reasoning
from eval.behavior.retrieval import RetrievalMetrics, _build_prompt, evaluate_retrieval
from eval.behavior.task_quality import evaluate_perplexity, evaluate_perplexity_baseline
from framework.model import ModelLayer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_CANDIDATES = [
    PROJECT_ROOT / "models" / "legacy" / "qwen3_1.7b",
    PROJECT_ROOT / "models" / "olmo2_1b",
    PROJECT_ROOT / "models" / "qwen3_0.6b",
]


def _first_model_path() -> Path | None:
    for path in MODEL_CANDIDATES:
        if path.exists():
            return path
    return None


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --- Pure logic / math (no model) -------------------------------------------------


def test_reasoning_problem_arithmetic_is_consistent():
    rng = random.Random(0)
    for _ in range(50):
        prompt, expected = _make_problem(rng)
        nums = [int(x) for x in re.findall(r"\d+", prompt.split("\n")[0])]
        assert len(nums) == 3
        a, b, c = nums
        assert expected == a + b - c
        assert f"Start with {a}" in prompt
        assert f"Add {b}" in prompt
        assert f"Subtract {c}" in prompt


def test_reasoning_answer_parser():
    assert _ANSWER_RE.search("42").group() == "42"
    assert int(_ANSWER_RE.search("result is -7 today").group()) == -7
    assert _ANSWER_RE.search("no digits") is None


def test_reasoning_metrics_accuracy_formula():
    metrics = ReasoningMetrics(num_trials=8, exact_match_accuracy=3 / 8)
    assert metrics.exact_match_accuracy == pytest.approx(0.375)
    assert metrics.to_dict()["exact_match_accuracy"] == pytest.approx(0.375)


def test_retrieval_metrics_accuracy_formula():
    metrics = RetrievalMetrics(
        needle_depth_frac=0.5,
        context_length=128,
        num_trials=4,
        exact_match_accuracy=1.0,
    )
    assert metrics.exact_match_accuracy == 1.0
    payload = metrics.to_dict()
    assert payload["num_trials"] == 4
    assert payload["needle_depth_frac"] == 0.5


def test_instruction_following_compliance_logic():
    compliant_cases = ["yes", "Yes,.", "no!", "NO extra"]
    non_compliant = ["", "maybe", "perhaps not", "123"]

    for text in compliant_cases:
        words = _WORD_RE.findall(text.strip())
        assert len(words) >= 1
        assert words[0].lower() in _CHOICES

    for text in non_compliant:
        words = _WORD_RE.findall(text.strip())
        if not words:
            continue
        assert words[0].lower() not in _CHOICES


def test_instruction_following_metrics_rate_formula():
    metrics = InstructionFollowingMetrics(num_trials=6, format_compliance_rate=4 / 6)
    assert metrics.format_compliance_rate == pytest.approx(2 / 3)


def test_perplexity_formula():
    nll_sum = 10.0
    n_tokens = 5
    assert math.exp(nll_sum / n_tokens) == pytest.approx(7.38905609893065, rel=1e-6)


def test_build_prompt_embeds_needle_and_question():
    class _Tok:
        def __call__(self, text, return_tensors=None):
            ids = list(range(max(1, len(text.split()))))
            if return_tensors == "pt":
                return SimpleNamespace(input_ids=torch.tensor([ids]))
            return SimpleNamespace(input_ids=ids)

        def decode(self, ids, skip_special_tokens=True):
            return "12345"

    tok = _Tok()
    prompt, code_len = _build_prompt(tok, context_length=32, depth_frac=0.5, code="12345")
    assert "The secret code is 12345" in prompt
    assert "Question: What is the secret code?" in prompt
    assert code_len == 5


def test_evaluate_behavior_wiring_respects_flags(monkeypatch):
    model = MagicMock()
    compressor = MagicMock()
    input_ids = torch.tensor([[1, 2, 3]])

    from eval.behavior.task_quality import PerplexityResult

    monkeypatch.setattr(
        "eval.behavior.evaluate_perplexity_result",
        lambda *a, **k: PerplexityResult(perplexity=2.0, n_tokens=4, nll_sum=2.8, prefill_tokens=3),
    )
    monkeypatch.setattr("eval.behavior.evaluate_perplexity_baseline", lambda *a, **k: 2.1)
    monkeypatch.setattr(
        "eval.behavior.evaluate_retrieval",
        lambda *a, **k: RetrievalMetrics(0.5, 128, 1, 1.0),
    )
    monkeypatch.setattr(
        "eval.behavior.evaluate_instruction_following",
        lambda *a, **k: InstructionFollowingMetrics(1, 1.0),
    )
    monkeypatch.setattr(
        "eval.behavior.evaluate_reasoning",
        lambda *a, **k: ReasoningMetrics(1, 1.0),
    )

    full = evaluate_behavior(
        model,
        input_ids,
        compressor,
        run_task_quality=True,
        run_retrieval=True,
        run_instruction_following=True,
        run_reasoning=True,
        include_baseline=True,
    )
    assert full.perplexity == 2.0
    assert full.perplexity_baseline == 2.1
    assert full.n_tokens == 4
    assert full.retrieval is not None
    assert full.instruction_following is not None
    assert full.reasoning is not None

    ppl_only = evaluate_behavior(
        model,
        input_ids,
        compressor,
        run_retrieval=False,
        run_instruction_following=False,
        run_reasoning=False,
    )
    assert ppl_only.retrieval is None
    assert ppl_only.instruction_following is None
    assert ppl_only.reasoning is None


def test_behavior_metrics_to_dict_shape():
    metrics = BehaviorMetrics(
        perplexity=10.0,
        perplexity_baseline=10.1,
        retrieval=RetrievalMetrics(0.5, 256, 3, 2 / 3),
        instruction_following=InstructionFollowingMetrics(4, 0.75),
        reasoning=ReasoningMetrics(5, 0.2),
    )
    payload = metrics.to_dict()
    assert payload["task_quality"]["perplexity"] == 10.0
    assert "n_tokens" in payload["task_quality"]
    assert payload["retrieval"]["exact_match_accuracy"] == pytest.approx(2 / 3)
    assert payload["instruction_following"]["format_compliance_rate"] == 0.75
    assert payload["reasoning"]["exact_match_accuracy"] == 0.2


# --- Module integration (real model, minimal cost) --------------------------------

MODEL_PATH = _first_model_path()
pytestmark_model = pytest.mark.skipif(MODEL_PATH is None, reason="No shortlist/legacy model downloaded")


@pytest.fixture(scope="module")
def model_layer():
    return ModelLayer(model_path=MODEL_PATH, device=_device())


@pytestmark_model
def test_evaluate_retrieval_returns_valid_metrics(model_layer: ModelLayer):
    metrics = evaluate_retrieval(
        model_layer,
        IdentityCompressor(),
        context_length=64,
        depth_frac=0.5,
        num_trials=2,
        max_new_tokens=4,
        seed=0,
    )
    assert metrics.num_trials == 2
    assert 0.0 <= metrics.exact_match_accuracy <= 1.0
    assert metrics.exact_match_accuracy in (0.0, 0.5, 1.0)
    assert metrics.context_length == 64


@pytestmark_model
def test_evaluate_instruction_following_returns_valid_rate(model_layer: ModelLayer):
    metrics = evaluate_instruction_following(
        model_layer,
        IdentityCompressor(),
        num_trials=3,
        max_new_tokens=3,
        seed=0,
    )
    assert metrics.num_trials == 3
    assert 0.0 <= metrics.format_compliance_rate <= 1.0
    assert _TEMPLATE.format(question="Is water wet?").startswith("Answer the question")


@pytestmark_model
def test_evaluate_reasoning_returns_valid_accuracy(model_layer: ModelLayer):
    metrics = evaluate_reasoning(
        model_layer,
        IdentityCompressor(),
        num_trials=2,
        max_new_tokens=4,
        seed=0,
    )
    assert metrics.num_trials == 2
    assert 0.0 <= metrics.exact_match_accuracy <= 1.0


@pytestmark_model
def test_identity_perplexity_close_to_baseline(model_layer: ModelLayer):
    ids = model_layer.tokenize("Behavior module PPL isolation check.")[:, :48]
    ppl_compressed = evaluate_perplexity(model_layer, ids, IdentityCompressor(), stride=16)
    ppl_baseline = evaluate_perplexity_baseline(model_layer, ids, stride=16)
    assert ppl_compressed > 1.0
    assert abs(ppl_compressed - ppl_baseline) / ppl_baseline < 0.10
