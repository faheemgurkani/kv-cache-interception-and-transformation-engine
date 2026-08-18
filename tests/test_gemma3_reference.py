"""Gemma3-270M reference: adapter + per-layer RoPE + full eval-branch verification."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from compressors.identity import IdentityCompressor
from compressors.registry import get_compressor
from eval.fidelity.memory import kv_cache_bytes
from eval.runner import EvaluationRunner
from framework.compatibility import evaluate_compatibility_gates
from framework.kv_cache import get_cache_size_bytes, iter_layer_kv
from framework.model import ModelLayer
from framework.model_adapter import load_attention_ops, project_qkv, pre_attention_hidden, resolve_head_dim
from framework.model_capabilities import CAPABILITIES_BY_MODEL_TYPE, get_model_eval_metadata
from framework.rope import build_rope_context
from framework.state_interface import visible_state_bytes

GEMMA3_PATH = Path(__file__).resolve().parent.parent / "models" / "gemma3_270m"
OLMO2_PATH = Path(__file__).resolve().parent.parent / "models" / "olmo2_1b"

GEMMA3_SPEC = dict(
    num_layers=18,
    num_q_heads=4,
    num_kv_heads=1,
    head_dim=256,
    kv_to_q_ratio=0.25,
    sliding_window=512,
)


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture(scope="module")
def gemma3_model():
    if not GEMMA3_PATH.exists():
        pytest.skip("Gemma3-270M not downloaded")
    return ModelLayer(model_path=GEMMA3_PATH, device=_device())


@pytest.fixture(scope="module")
def gemma3_runner(gemma3_model):
    return EvaluationRunner(model_layer=gemma3_model, compressor=IdentityCompressor())


def test_gemma3_mqa_memory_head_ratio_is_quarter_of_mha():
    """Plan §7: H_KV/H_Q = 1/4 → per-layer KV-head bytes are 1/16 of OLMo2 MHA (1 KV vs 16 KV)."""
    seq_len = 64
    head_dim = 256
    bytes_per_elt = 2
    olmo2_layer = kv_cache_bytes(
        num_layers=1, seq_len=seq_len, num_kv_heads=16, head_dim=head_dim, bytes_per_element=bytes_per_elt
    )
    gemma3_layer = kv_cache_bytes(
        num_layers=1, seq_len=seq_len, num_kv_heads=1, head_dim=head_dim, bytes_per_element=bytes_per_elt
    )
    assert gemma3_layer / olmo2_layer == pytest.approx(1 / 16, rel=1e-9)


@pytest.mark.skipif(not GEMMA3_PATH.exists(), reason="Gemma3-270M not downloaded")
def test_gemma3_per_layer_rope_tables_differ(gemma3_model: ModelLayer):
    """Plan §9: sliding and full layers must not share one RoPE table."""
    input_ids = gemma3_model.tokenize("Hello world")
    with torch.no_grad():
        outputs = gemma3_model.model(
            input_ids,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
    position_ids = torch.arange(input_ids.shape[1], device=gemma3_model.device).unsqueeze(0)
    ctx = build_rope_context(
        gemma3_model.model,
        outputs.hidden_states[0],
        position_ids,
        config=gemma3_model.config,
    )
    layer_types = gemma3_model.config.layer_types
    sliding_idx = layer_types.index("sliding_attention")
    full_idx = layer_types.index("full_attention")
    cos_s, _ = ctx.get_rope(sliding_idx)
    cos_f, _ = ctx.get_rope(full_idx)
    assert not torch.equal(cos_s, cos_f)


@pytest.mark.skipif(not GEMMA3_PATH.exists(), reason="Gemma3-270M not downloaded")
def test_gemma3_wrong_rope_breaks_query_projection(gemma3_model: ModelLayer):
    """Using full RoPE on a sliding layer must not match the native query states."""
    input_ids = gemma3_model.tokenize("Hello world")
    with torch.no_grad():
        outputs = gemma3_model.model(
            input_ids,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
    position_ids = torch.arange(input_ids.shape[1], device=gemma3_model.device).unsqueeze(0)
    ctx = build_rope_context(
        gemma3_model.model,
        outputs.hidden_states[0],
        position_ids,
        config=gemma3_model.config,
    )
    layer_types = gemma3_model.config.layer_types
    sliding_idx = layer_types.index("sliding_attention")
    full_idx = layer_types.index("full_attention")

    layer = gemma3_model.model.model.layers[sliding_idx]
    ops = load_attention_ops(gemma3_model.config)
    hidden = pre_attention_hidden(layer, outputs.hidden_states[sliding_idx], ops)
    query, key, _ = project_qkv(layer.self_attn, hidden, ops)
    cos_ok, sin_ok = ctx.get_rope(sliding_idx)
    cos_bad, sin_bad = ctx.get_rope(full_idx)
    query_ok, _ = ops.apply_rotary_pos_emb(query, key, cos_ok, sin_ok)
    query_bad, _ = ops.apply_rotary_pos_emb(query, key, cos_bad, sin_bad)
    assert not torch.allclose(query_ok, query_bad, atol=1e-3)


@pytest.mark.skipif(not GEMMA3_PATH.exists(), reason="Gemma3-270M not downloaded")
def test_gemma3_adapter_conformance(gemma3_model: ModelLayer):
    """Plan §13: load → forward → cache → RoPE → identity round-trip."""
    model_layer = gemma3_model
    config = model_layer.config
    spec = GEMMA3_SPEC

    assert model_layer.attn_implementation == "eager"
    assert load_attention_ops(config).model_type == "gemma3_text"

    input_ids = model_layer.tokenize("The quick brown fox jumps over the lazy dog")
    with torch.no_grad():
        outputs = model_layer.model(
            input_ids,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

    layers = list(iter_layer_kv(outputs.past_key_values))
    assert len(layers) == spec["num_layers"]

    key, value = layers[0]
    assert key.shape[1] == spec["num_kv_heads"]
    assert value.shape[1] == spec["num_kv_heads"]
    assert key.shape[3] == spec["head_dim"]

    ops = load_attention_ops(config)
    rope_ctx = build_rope_context(
        model_layer.model,
        outputs.hidden_states[0],
        torch.arange(input_ids.shape[1], device=model_layer.device).unsqueeze(0),
        config=config,
    )
    for layer_idx in (0, spec["num_layers"] // 2, spec["num_layers"] - 1):
        layer = model_layer.model.model.layers[layer_idx]
        hidden = pre_attention_hidden(layer, outputs.hidden_states[layer_idx], ops)
        query, key_proj, _ = project_qkv(layer.self_attn, hidden, ops)
        cos, sin = rope_ctx.get_rope(layer_idx)
        query_rope, _ = ops.apply_rotary_pos_emb(query, key_proj, cos, sin)
        assert query_rope.shape[1] == spec["num_q_heads"]

    compressor = IdentityCompressor()
    key_hat, value_hat = compressor.decompress(compressor.compress(key, value, layer=0))
    assert torch.allclose(key, key_hat, atol=1e-5)
    assert torch.allclose(value, value_hat, atol=1e-5)

    gates = evaluate_compatibility_gates(
        config=config,
        model_loaded=True,
        forward_ok=True,
        past_key_values=outputs.past_key_values,
    )
    assert all(result.passed for result in gates.values())


@pytest.mark.skipif(not GEMMA3_PATH.exists(), reason="Gemma3-270M not downloaded")
def test_gemma3_eval_metadata_matches_plan(gemma3_model: ModelLayer):
    """Plan §12: per-layer attention metadata recorded with every run."""
    meta = get_model_eval_metadata(gemma3_model.config, local_path=str(GEMMA3_PATH))
    caps = CAPABILITIES_BY_MODEL_TYPE["gemma3_text"]

    assert meta["model_type"] == "gemma3_text"
    assert meta["attention_family"] == "mqa"
    assert meta["num_q_heads"] == 4
    assert meta["num_kv_heads"] == 1
    assert meta["head_dim"] == 256
    assert meta["num_layers"] == 18
    assert meta["rope_mode"] == "per_layer_type"
    assert meta["adapter"] == "gemma3_text"
    assert meta["adapter_registered"] is True
    assert meta["state_semantics_complete"] == caps.state_semantics_complete
    assert meta["sliding_window"] == GEMMA3_SPEC["sliding_window"]

    layer_meta = meta["layer_attention"]
    assert len(layer_meta) == 18
    assert layer_meta[0]["attention_type"] == "sliding_attention"
    assert layer_meta[5]["attention_type"] == "full_attention"
    assert layer_meta[0]["rope_type"] == "sliding_attention"
    assert layer_meta[5]["rope_type"] == "full_attention"
    assert layer_meta[0]["is_sliding"] is True
    assert layer_meta[5]["is_sliding"] is False


@pytest.mark.skipif(not GEMMA3_PATH.exists(), reason="Gemma3-270M not downloaded")
def test_gemma3_memory_formula_alignment(gemma3_model: ModelLayer):
    """M_KV = L × T × H_KV × (D_k + D_v) × b for MQA (H_KV=1)."""
    input_ids = gemma3_model.tokenize("Hello world")
    with torch.no_grad():
        outputs = gemma3_model.forward_with_cache(input_ids, use_cache=True)

    pkv = outputs.past_key_values
    seq_len = input_ids.shape[1]
    head_dim = resolve_head_dim(gemma3_model.config)
    bytes_per_elt = next(iter_layer_kv(pkv))[0].element_size()

    measured = visible_state_bytes(pkv)
    analytical = kv_cache_bytes(
        num_layers=GEMMA3_SPEC["num_layers"],
        seq_len=seq_len,
        num_kv_heads=GEMMA3_SPEC["num_kv_heads"],
        head_dim=head_dim,
        bytes_per_element=bytes_per_elt,
    )
    assert measured == analytical
    assert measured == get_cache_size_bytes(pkv)


@pytest.mark.skipif(
    not GEMMA3_PATH.exists() or not OLMO2_PATH.exists(),
    reason="Need Gemma3 and OLMo2 for cross-model ratio check",
)
def test_gemma3_vs_olmo2_measured_kv_ratio(gemma3_model: ModelLayer):
    """Empirical: same T,D,b → ratio = (L_gemma × H_kv_gemma) / (L_olmo × H_kv_olmo)."""
    from data.loader import build_long_context_ids, load_wikitext2

    dataset = load_wikitext2()
    seq_len = 64
    g_ids = build_long_context_ids(gemma3_model.tokenizer, dataset, seq_len).to(gemma3_model.device)
    olmo2 = ModelLayer(model_path=OLMO2_PATH, device=_device())
    o_ids = build_long_context_ids(olmo2.tokenizer, dataset, seq_len).to(olmo2.device)
    assert g_ids.shape[1] == o_ids.shape[1] == seq_len

    with torch.no_grad():
        g_out = gemma3_model.forward_with_cache(g_ids, use_cache=True)
        o_out = olmo2.forward_with_cache(o_ids, use_cache=True)

    ratio = visible_state_bytes(g_out.past_key_values) / visible_state_bytes(o_out.past_key_values)
    expected = (18 * 1) / (16 * 16)
    assert ratio == pytest.approx(expected, rel=1e-3)


def _assert_eval_branches_complete(result, compressor_name: str) -> None:
    assert result.fidelity is not None
    assert result.behavior is not None
    assert result.system is not None
    assert result.model_metadata is not None
    assert result.model_metadata["adapter"] == "gemma3_text"
    assert result.model_metadata["attention_family"] == "mqa"
    assert "layer_attention" in result.model_metadata

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
@pytest.mark.skipif(not GEMMA3_PATH.exists(), reason="Gemma3-270M not downloaded")
def test_gemma3_all_eval_branches(gemma3_model: ModelLayer, compressor_name: str, compressor_kwargs: dict):
    """Plan §13: identity first, then all production compressors."""
    compressor = get_compressor(compressor_name, **compressor_kwargs)
    runner = EvaluationRunner(model_layer=gemma3_model, compressor=compressor)
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


@pytest.mark.skipif(not GEMMA3_PATH.exists(), reason="Gemma3-270M not downloaded")
def test_gemma3_identity_acceptance(gemma3_runner: EvaluationRunner):
    """Plan §13: identity Δ_logit ≈ 0 and attention cosine ≈ 1 (bf16 tolerance)."""
    result = gemma3_runner.run(
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
