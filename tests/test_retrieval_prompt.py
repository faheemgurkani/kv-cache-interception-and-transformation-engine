"""Audit R1: retrieval prompts hit the requested context length and keep the question."""

from pathlib import Path

import pytest

from eval.behavior.retrieval import DEFAULT_RETRIEVAL_TRIALS, _build_prompt

GEMMA3 = Path(__file__).resolve().parent.parent / "models" / "gemma3_270m"


def test_default_trials_raised_above_five():
    assert DEFAULT_RETRIEVAL_TRIALS >= 20


@pytest.mark.parametrize("ctx", [128, 256, 512])
def test_prompt_length_is_within_a_few_tokens_of_context(ctx):
    pytest.importorskip("transformers")
    if not (GEMMA3 / "config.json").exists():
        pytest.skip("Gemma3 checkpoint not present")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(GEMMA3)
    for code in ("12345", "90817"):
        prompt, code_len = _build_prompt(tok, ctx, 0.5, code)
        n = len(tok(prompt).input_ids)
        assert ctx - 6 <= n <= ctx, (ctx, n)  # used to be 129 / 251 / 491
        assert prompt.endswith("Answer: The secret code is")
        assert f"The secret code is {code}." in prompt and code_len == 5
