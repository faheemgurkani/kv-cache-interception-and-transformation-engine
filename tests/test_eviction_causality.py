"""Audit B2: SnapKV / RocketKV must not leak future tokens through prefill eviction."""

from pathlib import Path

import pytest
import torch

from compressors.registry import get_compressor
from eval.behavior.task_quality import evaluate_perplexity_result

GEMMA3 = Path(__file__).resolve().parent.parent / "models" / "gemma3_270m"

_TEXT = (
    "The history of the printing press is closely tied to the spread of literacy in Europe. "
    "Before movable type, books were copied by hand, which made them rare and expensive. "
    "Johannes Gutenberg introduced a system that let printers assemble pages from reusable "
    "metal letters, and within a few decades presses had appeared in hundreds of towns. "
    "Scholars could now compare identical editions, merchants could publish price lists, and "
    "reformers could circulate pamphlets faster than authorities were able to suppress them. "
)

EVICTORS = {
    "snapkv": dict(max_capacity_prompt=24, window_size=8, kernel_size=5),
    "rocketkv": dict(token_budget=24, hsa_budget=24, window_size=8),
}
NOOP = {
    "snapkv": dict(max_capacity_prompt=10_000, window_size=8, kernel_size=5),
    "rocketkv": dict(token_budget=10_000, hsa_budget=10_000, window_size=8),
}


def _layer():
    pytest.importorskip("transformers")
    if not (GEMMA3 / "config.json").exists():
        pytest.skip("Gemma3 checkpoint not present")
    from framework.model import ModelLayer

    return ModelLayer(model_path=GEMMA3, device=torch.device("cpu"), torch_dtype=torch.float32)


def _ids(layer, n=96):
    return layer.tokenize(_TEXT * 2)[:, :n]


@pytest.mark.parametrize("name", sorted(EVICTORS))
def test_prefill_logits_are_causal(name):
    """Changing token t must not change logits at positions < t (even with eviction on)."""
    layer = _layer()
    ids = _ids(layer, 48)
    changed = ids.clone()
    changed[0, 40] = (changed[0, 40] + 7) % 200_000
    comp = get_compressor(name, **EVICTORS[name])
    outs = []
    for x in (ids, changed):
        comp.reset_state()
        engine = layer.make_kv_engine(comp)
        logits, _ = engine.step(
            x,
            attention_mask=torch.ones(1, x.shape[1], dtype=torch.long),
            compressed_cache=None,
            position_ids=torch.arange(x.shape[1]).unsqueeze(0),
        )
        outs.append(logits[0].float())
    delta = (outs[0] - outs[1]).abs().amax(dim=-1)
    assert delta[:40].max().item() == 0.0, delta[:40].max().item()
    assert delta[40:].max().item() > 0.0  # sanity: the perturbation is visible downstream


@pytest.mark.parametrize("name", sorted(EVICTORS))
def test_noop_budget_matches_identity_and_small_budget_hurts(name):
    layer = _layer()
    ids = _ids(layer)
    identity = evaluate_perplexity_result(layer, ids, get_compressor("identity")).perplexity

    noop_layer = _layer()
    noop = evaluate_perplexity_result(noop_layer, ids, get_compressor(name, **NOOP[name])).perplexity
    assert abs(noop - identity) / identity < 1e-3

    small_layer = _layer()
    small = evaluate_perplexity_result(small_layer, ids, get_compressor(name, **EVICTORS[name])).perplexity
    assert small > identity * 1.02, (identity, small)


@pytest.mark.parametrize("name", sorted(EVICTORS))
def test_decode_positions_continue_from_logical_length(name):
    """Audit B6: after eviction, implicit decode positions must equal explicit logical ones."""
    prefill = 48
    logits = {}
    for mode in ("implicit", "explicit"):
        layer = _layer()
        ids = _ids(layer, 64)
        comp = get_compressor(name, **EVICTORS[name])
        engine = layer.make_kv_engine(comp)
        _, cache = engine.step(
            ids[:, :prefill],
            attention_mask=torch.ones(1, prefill, dtype=torch.long),
            compressed_cache=None,
            position_ids=torch.arange(prefill).unsqueeze(0),
        )
        assert cache.seq_length < prefill  # eviction really happened
        assert cache.logical_seq_len == prefill
        outs = []
        for pos in range(prefill, prefill + 4):
            pid = torch.tensor([[pos]]) if mode == "explicit" else None
            step_logits, cache = engine.step(
                ids[:, pos : pos + 1],
                attention_mask=torch.ones(1, cache.seq_length + 1, dtype=torch.long),
                compressed_cache=cache,
                position_ids=pid,
            )
            outs.append(step_logits[0, -1].float())
        logits[mode] = torch.stack(outs)
    assert torch.equal(logits["implicit"], logits["explicit"])


@pytest.mark.parametrize("name", sorted(EVICTORS))
def test_long_prompt_eviction_then_multistep_decode_builds_matching_masks(name):
    """Regression: restoring logical length on sliding layers broke eviction decode
    (mask 320 vs keys 286 on Gemma3 in the reasoning prompt). Window (512) is not filled here."""
    layer = _layer()
    ids = layer.tokenize(_TEXT * 4)[:, :200]
    comp = get_compressor(name, **EVICTORS[name])
    engine = layer.make_kv_engine(comp)
    logits, cache = engine.step(ids, attention_mask=torch.ones(1, ids.shape[1], dtype=torch.long))
    assert cache.seq_length < ids.shape[1] and cache.logical_seq_len == ids.shape[1]
    for _ in range(5):
        nxt = logits[:, -1:].argmax(-1)
        logits, cache = engine.step(nxt, compressed_cache=cache)  # default mask / implicit positions
    assert cache.logical_seq_len == ids.shape[1] + 5
