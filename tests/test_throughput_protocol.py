"""Audit S2: throughput is measured after a warm-up, as the median of several runs."""

from eval.system.latency_throughput import ThroughputMetrics, _median_run


def _m(elapsed):
    return ThroughputMetrics(
        context_length=8,
        generated_tokens=4,
        elapsed_seconds=elapsed,
        tokens_per_second=4 / elapsed,
        latency_ms_per_token=elapsed / 4 * 1000,
    )


def test_median_run_picks_middle_elapsed_and_records_spread():
    chosen = _median_run([_m(3.0), _m(1.0), _m(2.0)])
    assert chosen.elapsed_seconds == 2.0
    assert chosen.repeats == 3
    assert sorted(chosen.elapsed_seconds_all) == [1.0, 2.0, 3.0]


def test_single_run_is_its_own_median():
    chosen = _median_run([_m(1.5)])
    assert chosen.elapsed_seconds == 1.5 and chosen.repeats == 1


def test_warmup_then_repeats_on_real_model():
    from pathlib import Path

    import pytest
    import torch

    root = Path(__file__).resolve().parent.parent / "models" / "gemma3_270m"
    if not (root / "config.json").exists():
        pytest.skip("Gemma3 checkpoint not present")
    from compressors.identity import IdentityCompressor
    from eval.system.latency_throughput import evaluate_throughput, evaluate_throughput_baseline
    from framework.model import ModelLayer

    layer = ModelLayer(model_path=root, device=torch.device("cpu"), torch_dtype=torch.float32)
    ids = layer.tokenize("Throughput protocol check sentence for warm-up and repeats.")[:, :12]
    comp = evaluate_throughput(layer, ids, IdentityCompressor(), num_new_tokens=3, warmup_tokens=1, repeats=3)
    base = evaluate_throughput_baseline(layer, ids, num_new_tokens=3, warmup_tokens=1, repeats=3)
    for m in (comp, base):
        assert m.repeats == 3 and len(m.elapsed_seconds_all) == 3
    assert comp.ttft_ms is not None and base.online_compressed_kv is False
