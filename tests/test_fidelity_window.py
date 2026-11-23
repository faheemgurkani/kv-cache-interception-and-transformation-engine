"""Audit F1: the attention-fidelity window is capped, recorded, and identical everywhere."""

from eval.fidelity import ATTENTION_FIDELITY_MAX_TOKENS, attention_fidelity_window


def test_window_is_capped_and_bounded_by_sequence():
    cfg = {"attention_fidelity_tokens": 512}
    assert attention_fidelity_window(cfg, 64) == 64
    assert attention_fidelity_window(cfg, 128) == 128
    assert attention_fidelity_window(cfg, 512) == ATTENTION_FIDELITY_MAX_TOKENS
    assert attention_fidelity_window({"attention_fidelity_tokens": 32}, 512) == 32
    assert attention_fidelity_window({}, 2048) == ATTENTION_FIDELITY_MAX_TOKENS
