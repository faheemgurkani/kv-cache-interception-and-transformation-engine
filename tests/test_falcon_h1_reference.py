"""Falcon-H1-0.5B hybrid reference: adapter + dual-state memory + full eval-branch verification."""

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
from framework.kv_cache import (
    apply_compressor,
    decompress_to_legacy_cache,
    get_cache_size_bytes,
    iter_layer_kv,
)
from framework.kv_engine import KVCacheEngine
from framework.model import ModelLayer
from framework.model_adapter import load_attention_ops, project_qkv, pre_attention_hidden, resolve_head_dim
from framework.model_capabilities import CAPABILITIES_BY_MODEL_TYPE, get_model_eval_metadata
from framework.rope import build_rope_context
from framework.state_interface import (
    attention_kv_bytes,
    hybrid_layer_detected,
    iter_layer_states,
    recurrent_state_bytes,
    visible_state_bytes,
)

FALCON_PATH = Path(__file__).resolve().parent.parent / "models" / "falcon_h1_0.5b"
QWEN3_PATH = Path(__file__).resolve().parent.parent / "models" / "qwen3_0.6b"

FALCON_SPEC = dict(
    num_layers=36,
    num_q_heads=8,
    num_kv_heads=2,
    head_dim=64,
)


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture(scope="module")
def falcon_model():
    if not FALCON_PATH.exists():
        pytest.skip("Falcon-H1-0.5B not downloaded")
    return ModelLayer(model_path=FALCON_PATH, device=_device())


@pytest.fixture(scope="module")
def falcon_runner(falcon_model):
    return EvaluationRunner(model_layer=falcon_model, compressor=IdentityCompressor())


def test_gqa_memory_formula():
    """Plan §20: GQA with H_Q=8, H_KV=2 → KV bytes scale by H_KV/H_Q vs MHA at same head_dim."""
    seq_len = 64
    head_dim = 64
    bytes_per_elt = 2
    mha = kv_cache_bytes(num_layers=1, seq_len=seq_len, num_kv_heads=8, head_dim=head_dim, bytes_per_element=bytes_per_elt)
    gqa = kv_cache_bytes(num_layers=1, seq_len=seq_len, num_kv_heads=2, head_dim=head_dim, bytes_per_element=bytes_per_elt)
    assert gqa / mha == pytest.approx(2 / 8, rel=1e-9)


@pytest.mark.skipif(not FALCON_PATH.exists(), reason="Falcon-H1-0.5B not downloaded")
def test_falcon_h1_adapter_conformance(falcon_model: ModelLayer):
    """Plan §20–21: adapter + hybrid state discovery."""
    model_layer = falcon_model
    config = model_layer.config
    spec = FALCON_SPEC
    caps = CAPABILITIES_BY_MODEL_TYPE["falcon_h1"]

    assert model_layer.attn_implementation == "eager"
    ops = load_attention_ops(config)
    assert ops.model_type == "falcon_h1"
    assert ops.qk_norm_layout == "none"
    assert caps.adapter_registered is True

    input_ids = model_layer.tokenize("The quick brown fox jumps over the lazy dog")
    with torch.no_grad():
        outputs = model_layer.model(
            input_ids,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

    assert check_loader_state_gate(
        model_loaded=True, forward_ok=True, past_key_values=outputs.past_key_values
    ).passed
    assert check_attention_gate(config).passed is True
    assert check_state_semantics_gate(config, outputs.past_key_values).passed is True

    assert hybrid_layer_detected(outputs.past_key_values) is True
    layers = list(iter_layer_kv(outputs.past_key_values))
    assert len(layers) == spec["num_layers"]
    key, value = layers[0]
    assert key.shape[1] == spec["num_kv_heads"]
    assert key.shape[3] == spec["head_dim"]

    rope_ctx = build_rope_context(
        model_layer.model,
        outputs.hidden_states[0],
        torch.arange(input_ids.shape[1], device=model_layer.device).unsqueeze(0),
        config=config,
    )
    layer = model_layer.model.model.layers[0]
    hidden = pre_attention_hidden(layer, outputs.hidden_states[0], ops)
    cos, sin = rope_ctx.get_rope(0)
    query, key, _ = project_qkv(layer.self_attn, hidden, ops)
    query, _ = ops.apply_rotary_pos_emb(query, key, cos, sin)
    assert query.shape[1] == spec["num_q_heads"]
    assert query.shape[3] == spec["head_dim"]

    gates = evaluate_compatibility_gates(
        config=config,
        model_loaded=True,
        forward_ok=True,
        past_key_values=outputs.past_key_values,
    )
    assert all(g.passed for g in gates.values())


@pytest.mark.skipif(not FALCON_PATH.exists(), reason="Falcon-H1-0.5B not downloaded")
def test_falcon_h1_eval_metadata_matches_plan(falcon_model: ModelLayer):
    """Plan §24–28: hybrid metadata + compression policy disclosure."""
    meta = get_model_eval_metadata(falcon_model.config, local_path=str(FALCON_PATH))
    caps = CAPABILITIES_BY_MODEL_TYPE["falcon_h1"]

    assert meta["model_type"] == "falcon_h1"
    assert meta["attention_family"] == "gqa"
    assert meta["num_q_heads"] == 8
    assert meta["num_kv_heads"] == 2
    assert meta["head_dim"] == 64
    assert meta["num_layers"] == 36
    assert meta["rope_mode"] == "global"
    assert meta["state_type"] == "hybrid"
    assert meta["qk_norm_layout"] == "none"
    assert meta["adapter"] == "falcon_h1"
    assert meta["adapter_registered"] is True
    assert meta["state_semantics_complete"] == caps.state_semantics_complete
    assert meta["compression_policy"] == {"attention": "compressible", "recurrent": "passthrough"}
    assert "hybrid_state_disclosure" in meta


@pytest.mark.skipif(not FALCON_PATH.exists(), reason="Falcon-H1-0.5B not downloaded")
def test_falcon_h1_total_memory_includes_mamba(falcon_model: ModelLayer):
    """Plan §23: M_state = M_KV + M_Mamba; attention-only undercounts."""
    input_ids = falcon_model.tokenize("Hello world")
    with torch.no_grad():
        outputs = falcon_model.forward_with_cache(input_ids, use_cache=True)

    pkv = outputs.past_key_values
    attn = attention_kv_bytes(pkv)
    recurrent = recurrent_state_bytes(pkv)
    total = visible_state_bytes(pkv)

    assert recurrent > 0
    assert attn > 0
    assert total == attn + recurrent
    assert total > get_cache_size_bytes(pkv)
    assert get_cache_size_bytes(pkv) == attn


@pytest.mark.skipif(not FALCON_PATH.exists(), reason="Falcon-H1-0.5B not downloaded")
def test_falcon_h1_recurrent_preserved_through_compress_decompress(falcon_model: ModelLayer):
    """Plan §25–26: R'_t = R_t after K/V compression round-trip."""
    input_ids = falcon_model.tokenize("Recurrent preservation check")
    with torch.no_grad():
        outputs = falcon_model.forward_with_cache(input_ids, use_cache=True)

    pkv = outputs.past_key_values
    state0 = next(iter_layer_states(pkv))
    rs_before = state0.recurrent.recurrent_states.clone()
    cs_before = state0.recurrent.conv_states.clone()

    compressor = IdentityCompressor()
    compressed = apply_compressor(pkv, compressor)
    merged = decompress_to_legacy_cache(
        compressed,
        compressor,
        falcon_model.config,
        device=falcon_model.device,
        template_cache=pkv,
    )
    merged_state = next(iter_layer_states(merged))
    assert torch.equal(merged_state.recurrent.recurrent_states, rs_before)
    assert torch.equal(merged_state.recurrent.conv_states, cs_before)


@pytest.mark.skipif(not FALCON_PATH.exists(), reason="Falcon-H1-0.5B not downloaded")
def test_falcon_h1_online_engine_preserves_recurrent_evolution(falcon_model: ModelLayer):
    """Plan §26: online decode must not reset Mamba state between steps."""
    ids = falcon_model.tokenize("Online hybrid decode")[:, :4]
    engine = KVCacheEngine(falcon_model.model, IdentityCompressor())

    with torch.no_grad():
        _, cache = engine.step(ids[:, :2])
        rs_after_prefill = engine._last_full_cache.layers[0].recurrent_states.clone()
        _, cache = engine.step(ids[:, 2:3], compressed_cache=cache)
        rs_after_token = engine._last_full_cache.layers[0].recurrent_states.clone()

    assert not torch.equal(rs_after_prefill, rs_after_token)


@pytest.mark.skipif(
    not FALCON_PATH.exists() or not QWEN3_PATH.exists(),
    reason="Need Falcon and Qwen3 for cross-model KV ratio",
)
def test_falcon_vs_qwen3_attention_kv_ratio(falcon_model: ModelLayer):
    """Same T,b: attention-KV ratio scales with L × H_KV × D."""
    text = "The quick brown fox jumps over the lazy dog"
    t_ids = falcon_model.tokenize(text)
    qwen3 = ModelLayer(model_path=QWEN3_PATH, device=_device())
    q_ids = qwen3.tokenize(text)

    with torch.no_grad():
        t_out = falcon_model.forward_with_cache(t_ids, use_cache=True)
        q_out = qwen3.forward_with_cache(q_ids, use_cache=True)

    t_seq = t_out.past_key_values.layers[0].keys.shape[2]
    q_seq = q_out.past_key_values.layers[0].keys.shape[2]
    ratio = attention_kv_bytes(t_out.past_key_values) / get_cache_size_bytes(q_out.past_key_values)
    expected = (
        FALCON_SPEC["num_layers"] * FALCON_SPEC["num_kv_heads"] * FALCON_SPEC["head_dim"] * t_seq
    ) / (28 * 8 * 128 * q_seq)
    assert ratio == pytest.approx(expected, rel=1e-2)


def _assert_eval_branches_complete(result, compressor_name: str) -> None:
    assert result.fidelity is not None
    assert result.behavior is not None
    assert result.system is not None
    assert result.model_metadata is not None
    assert result.model_metadata["adapter"] == "falcon_h1"
    assert result.model_metadata["state_type"] == "hybrid"
    assert result.model_metadata["compression_policy"]["recurrent"] == "passthrough"

    mem = result.fidelity.memory
    assert mem.recurrent_state_bytes is not None and mem.recurrent_state_bytes > 0
    assert mem.attention_kv_bytes is not None and mem.attention_kv_bytes > 0
    assert mem.uncompressed_bytes == mem.attention_kv_bytes + mem.recurrent_state_bytes
    assert mem.kv_compression_ratio is not None

    assert result.behavior.perplexity is not None and result.behavior.perplexity > 0
    assert result.system.throughput is not None
    assert result.system.throughput.tokens_per_second > 0

    if compressor_name == "identity":
        assert result.fidelity.representation.key_cosine_similarity > 0.99
        assert result.fidelity.attention.cosine_similarity > 0.99
        assert mem.compression_ratio == pytest.approx(1.0, rel=1e-3)
        assert mem.kv_compression_ratio == pytest.approx(1.0, rel=1e-3)
    elif compressor_name == "turboquant":
        assert result.fidelity.representation.key_cosine_similarity > 0.8
        assert result.fidelity.attention.cosine_similarity > 0.8
        assert mem.kv_compression_ratio > mem.compression_ratio
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
@pytest.mark.skipif(not FALCON_PATH.exists(), reason="Falcon-H1-0.5B not downloaded")
def test_falcon_h1_all_eval_branches(
    falcon_model: ModelLayer, compressor_name: str, compressor_kwargs: dict
):
    """Plan §24–27: full FIDELITY/BEHAVIOR/SYSTEM on hybrid state."""
    compressor = get_compressor(compressor_name, **compressor_kwargs)
    runner = EvaluationRunner(model_layer=falcon_model, compressor=compressor)
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


@pytest.mark.skipif(not FALCON_PATH.exists(), reason="Falcon-H1-0.5B not downloaded")
def test_falcon_h1_identity_acceptance(falcon_runner: EvaluationRunner):
    """Plan §25: identity Δ_logit ≈ 0 and attention cosine ≈ 1."""
    result = falcon_runner.run(
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
    assert result.fidelity.memory.compression_ratio == pytest.approx(1.0, rel=1e-3)
    assert result.behavior.perplexity is not None and result.behavior.perplexity > 0
