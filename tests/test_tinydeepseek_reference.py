"""TinyDeepSeek-0.5B MLA reference: expanded-KV adapter + full eval-branch verification."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from compressors.identity import IdentityCompressor
from compressors.registry import get_compressor
from eval.fidelity.memory import kv_cache_bytes
from eval.runner import EvaluationRunner
from framework.compatibility import (
    check_attention_gate,
    check_loader_state_gate,
    check_state_semantics_gate,
    evaluate_compatibility_gates,
)
from framework.kv_cache import get_cache_size_bytes, iter_layer_kv
from framework.model import ModelLayer
from framework.model_adapter import (
    load_attention_ops,
    project_attention_states,
    pre_attention_hidden,
    resolve_key_head_dim,
    resolve_value_head_dim,
)
from framework.model_capabilities import CAPABILITIES_BY_MODEL_TYPE, get_model_eval_metadata
from framework.rope import build_rope_context
from framework.state_interface import visible_state_bytes

TINYDEEPSEEK_PATH = Path(__file__).resolve().parent.parent / "models" / "tinydeepseek_0.5b"
QWEN3_PATH = Path(__file__).resolve().parent.parent / "models" / "qwen3_0.6b"

TINYDEEPSEEK_SPEC = dict(
    num_layers=26,
    num_q_heads=4,
    num_kv_heads=4,
    key_head_dim=64,
    value_head_dim=32,
    kv_lora_rank=256,
)


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture(scope="module")
def tinydeepseek_model():
    if not TINYDEEPSEEK_PATH.exists():
        pytest.skip("TinyDeepSeek-0.5B not downloaded")
    return ModelLayer(model_path=TINYDEEPSEEK_PATH, device=_device())


@pytest.fixture(scope="module")
def tinydeepseek_runner(tinydeepseek_model):
    return EvaluationRunner(model_layer=tinydeepseek_model, compressor=IdentityCompressor())


def test_mla_asymmetric_memory_formula():
    """Plan §17: M = T·b·(H_K·D_k + H_V·D_v) when D_k ≠ D_v."""
    seq_len = 64
    num_kv_heads = 4
    key_dim = 64
    value_dim = 32
    bytes_per_elt = 2
    symmetric = kv_cache_bytes(
        num_layers=1,
        seq_len=seq_len,
        num_kv_heads=num_kv_heads,
        head_dim=32,
        bytes_per_element=bytes_per_elt,
    )
    asymmetric = kv_cache_bytes(
        num_layers=1,
        seq_len=seq_len,
        num_kv_heads=num_kv_heads,
        head_dim=key_dim,
        value_head_dim=value_dim,
        bytes_per_element=bytes_per_elt,
    )
    assert asymmetric / symmetric == pytest.approx((key_dim + value_dim) / (2 * 32), rel=1e-9)


@pytest.mark.skipif(not TINYDEEPSEEK_PATH.exists(), reason="TinyDeepSeek-0.5B not downloaded")
def test_tinydeepseek_adapter_conformance(tinydeepseek_model: ModelLayer):
    """Plan §15–16: adapter + expanded-cache MLA path."""
    model_layer = tinydeepseek_model
    config = model_layer.config
    spec = TINYDEEPSEEK_SPEC
    caps = CAPABILITIES_BY_MODEL_TYPE["deepseek_v3"]

    assert model_layer.attn_implementation == "eager"
    assert load_attention_ops(config).model_type == "deepseek_v3"
    assert caps.adapter_registered is True

    input_ids = model_layer.tokenize("The quick brown fox jumps over the lazy dog")
    with torch.no_grad():
        outputs = model_layer.model(
            input_ids,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

    loader_gate = check_loader_state_gate(
        model_loaded=True,
        forward_ok=True,
        past_key_values=outputs.past_key_values,
    )
    assert loader_gate.passed is True
    assert check_attention_gate(config).passed is True

    semantics_gate = check_state_semantics_gate(config, outputs.past_key_values)
    assert semantics_gate.passed is False
    assert "latent" in semantics_gate.detail.lower()

    layers = list(iter_layer_kv(outputs.past_key_values))
    assert len(layers) == spec["num_layers"]

    key, value = layers[0]
    assert key.shape[1] == spec["num_kv_heads"]
    assert value.shape[1] == spec["num_kv_heads"]
    assert key.shape[3] == spec["key_head_dim"]
    assert value.shape[3] == spec["value_head_dim"]

    ops = load_attention_ops(config)
    rope_ctx = build_rope_context(
        model_layer.model,
        outputs.hidden_states[0],
        torch.arange(input_ids.shape[1], device=model_layer.device).unsqueeze(0),
        config=config,
    )
    layer = model_layer.model.model.layers[0]
    hidden = pre_attention_hidden(layer, outputs.hidden_states[0], ops)
    cos, sin = rope_ctx.get_rope(0)
    query_rope, _, _ = project_attention_states(
        layer.self_attn, hidden, ops, cos, sin, config=config
    )
    assert query_rope.shape[1] == spec["num_q_heads"]
    assert query_rope.shape[3] == spec["key_head_dim"]

    compressor = IdentityCompressor()
    key_hat, value_hat = compressor.decompress(compressor.compress(key, value, layer=0))
    assert torch.allclose(key, key_hat, atol=1e-5)
    assert torch.allclose(value, value_hat, atol=1e-5)


@pytest.mark.skipif(not TINYDEEPSEEK_PATH.exists(), reason="TinyDeepSeek-0.5B not downloaded")
def test_tinydeepseek_eval_metadata_matches_plan(tinydeepseek_model: ModelLayer):
    """Plan §17–18: MLA metadata + expanded_kv disclosure."""
    meta = get_model_eval_metadata(tinydeepseek_model.config, local_path=str(TINYDEEPSEEK_PATH))
    caps = CAPABILITIES_BY_MODEL_TYPE["deepseek_v3"]

    assert meta["model_type"] == "deepseek_v3"
    assert meta["attention_family"] == "mla"
    assert meta["num_q_heads"] == 4
    assert meta["num_kv_heads"] == 4
    assert meta["key_head_dim"] == 64
    assert meta["value_head_dim"] == 32
    assert meta["num_layers"] == 26
    assert meta["rope_mode"] == "split_nope_rope"
    assert meta["adapter"] == "deepseek_v3"
    assert meta["adapter_registered"] is True
    assert meta["cache_representation"] == "expanded_kv"
    assert meta["kv_lora_rank"] == TINYDEEPSEEK_SPEC["kv_lora_rank"]
    assert meta["qk_nope_head_dim"] == 32
    assert meta["qk_rope_head_dim"] == 32
    assert meta["qk_head_dim"] == 64
    assert meta["state_semantics_complete"] == caps.state_semantics_complete
    assert "expanded_kv_disclosure" in meta


@pytest.mark.skipif(not TINYDEEPSEEK_PATH.exists(), reason="TinyDeepSeek-0.5B not downloaded")
def test_tinydeepseek_memory_formula_alignment(tinydeepseek_model: ModelLayer):
    """M_KV = L × T × H_KV × (D_k + D_v) × b for asymmetric MLA expanded cache."""
    input_ids = tinydeepseek_model.tokenize("Hello world")
    with torch.no_grad():
        outputs = tinydeepseek_model.forward_with_cache(input_ids, use_cache=True)

    pkv = outputs.past_key_values
    seq_len = input_ids.shape[1]
    key_dim = resolve_key_head_dim(tinydeepseek_model.config)
    value_dim = resolve_value_head_dim(tinydeepseek_model.config)
    bytes_per_elt = next(iter_layer_kv(pkv))[0].element_size()

    measured = visible_state_bytes(pkv)
    analytical = kv_cache_bytes(
        num_layers=TINYDEEPSEEK_SPEC["num_layers"],
        seq_len=seq_len,
        num_kv_heads=TINYDEEPSEEK_SPEC["num_kv_heads"],
        head_dim=key_dim,
        value_head_dim=value_dim,
        bytes_per_element=bytes_per_elt,
    )
    assert measured == analytical
    assert measured == get_cache_size_bytes(pkv)


@pytest.mark.skipif(
    not TINYDEEPSEEK_PATH.exists() or not QWEN3_PATH.exists(),
    reason="Need TinyDeepSeek and Qwen3 for cross-model ratio check",
)
def test_tinydeepseek_vs_qwen3_measured_kv_ratio(tinydeepseek_model: ModelLayer):
    """Same T,b: ratio scales with L × H_KV × (D_k + D_v)."""
    text = "The quick brown fox jumps over the lazy dog"
    t_ids = tinydeepseek_model.tokenize(text)
    qwen3 = ModelLayer(model_path=QWEN3_PATH, device=_device())
    q_ids = qwen3.tokenize(text)

    with torch.no_grad():
        t_out = tinydeepseek_model.forward_with_cache(t_ids, use_cache=True)
        q_out = qwen3.forward_with_cache(q_ids, use_cache=True)

    ratio = visible_state_bytes(t_out.past_key_values) / visible_state_bytes(q_out.past_key_values)
    t_spec = TINYDEEPSEEK_SPEC
    expected = (t_spec["num_layers"] * t_spec["num_kv_heads"] * (t_spec["key_head_dim"] + t_spec["value_head_dim"])) / (
        28 * 8 * (128 + 128)
    )
    assert ratio == pytest.approx(expected, rel=1e-2)


def _assert_eval_branches_complete(result, compressor_name: str) -> None:
    assert result.fidelity is not None
    assert result.behavior is not None
    assert result.system is not None
    assert result.model_metadata is not None
    assert result.model_metadata["adapter"] == "deepseek_v3"
    assert result.model_metadata["attention_family"] == "mla"
    assert result.model_metadata["cache_representation"] == "expanded_kv"

    assert result.fidelity.memory.compression_ratio > 0
    assert result.behavior.perplexity is not None and result.behavior.perplexity > 0
    assert result.system.throughput is not None
    assert result.system.throughput.tokens_per_second > 0

    if compressor_name == "identity":
        assert result.fidelity.representation.key_cosine_similarity > 0.99
        assert result.fidelity.attention.cosine_similarity > 0.99
        assert result.fidelity.memory.compression_ratio == pytest.approx(1.0, rel=1e-3)
    elif compressor_name == "turboquant":
        assert result.fidelity.representation.key_cosine_similarity > 0.8
        assert result.fidelity.attention.cosine_similarity > 0.8
    elif compressor_name == "qjl":
        assert result.fidelity.representation.value_cosine_similarity > 0.99
        assert result.fidelity.attention.cosine_similarity > 0.5
    elif compressor_name == "rocketkv":
        assert result.fidelity.representation.key_rmse < 1e-4
        assert result.fidelity.attention.cosine_similarity > 0.5


@pytest.mark.parametrize(
    "compressor_name,compressor_kwargs",
    [
        ("identity", {}),
        ("turboquant", {"bitwidth": 4}),
        ("qjl", {}),
        ("rocketkv", {"token_budget": 256}),
    ],
)
@pytest.mark.skipif(not TINYDEEPSEEK_PATH.exists(), reason="TinyDeepSeek-0.5B not downloaded")
def test_tinydeepseek_all_eval_branches(
    tinydeepseek_model: ModelLayer, compressor_name: str, compressor_kwargs: dict
):
    """Plan §17: full FIDELITY/BEHAVIOR/SYSTEM on expanded MLA cache."""
    compressor = get_compressor(compressor_name, **compressor_kwargs)
    runner = EvaluationRunner(model_layer=tinydeepseek_model, compressor=compressor)
    result = runner.run(
        context_length=128,
        run_fidelity=True,
        run_behavior=True,
        run_perplexity=True,
        run_retrieval=True,
        run_reasoning=True,
        run_instruction_following=True,
        run_system=True,
        run_throughput=True,
        run_memory_bandwidth=True,
        run_kernel_cost=True,
        generated_tokens=8,
        perplexity_stride=64,
    )
    _assert_eval_branches_complete(result, compressor_name)


@pytest.mark.skipif(not TINYDEEPSEEK_PATH.exists(), reason="TinyDeepSeek-0.5B not downloaded")
def test_tinydeepseek_identity_acceptance(tinydeepseek_runner: EvaluationRunner):
    """Plan §17: identity Δ_logit ≈ 0 and attention cosine ≈ 1 (bf16 tolerance)."""
    result = tinydeepseek_runner.run(
        context_length=128,
        run_fidelity=True,
        run_behavior=True,
        run_perplexity=True,
        run_system=True,
        run_throughput=True,
        generated_tokens=8,
        perplexity_stride=64,
    )
    assert result.fidelity.attention.cosine_similarity > 0.99
    assert result.fidelity.representation.key_cosine_similarity > 0.99
    assert result.behavior.perplexity is not None and result.behavior.perplexity > 0
    assert result.model_metadata["cache_representation"] == "expanded_kv"
