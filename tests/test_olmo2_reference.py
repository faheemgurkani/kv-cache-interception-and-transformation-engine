"""OLMo2-1B reference model: conformance + full evaluation-branch verification."""

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

OLMO2_PATH = Path(__file__).resolve().parent.parent / "models" / "olmo2_1b"
OLMO2_SPEC = dict(
    num_layers=16,
    num_q_heads=16,
    num_kv_heads=16,
    head_dim=128,
    hidden_size=2048,
)


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture(scope="module")
def olmo2_model():
    if not OLMO2_PATH.exists():
        pytest.skip("OLMo2-1B not downloaded")
    return ModelLayer(model_path=OLMO2_PATH, device=_device())


@pytest.fixture(scope="module")
def olmo2_runner(olmo2_model):
    return EvaluationRunner(model_layer=olmo2_model, compressor=IdentityCompressor())


@pytest.mark.skipif(not OLMO2_PATH.exists(), reason="OLMo2-1B not downloaded")
def test_olmo2_adapter_conformance(olmo2_model: ModelLayer):
    """Plan §3: ten-point conformance checklist for the legacy OLMo2 path."""
    model_layer = olmo2_model
    config = model_layer.config
    spec = OLMO2_SPEC

    assert model_layer.attn_implementation == "eager"
    assert config.use_cache is True

    input_ids = model_layer.tokenize("The quick brown fox jumps over the lazy dog")
    with torch.no_grad():
        outputs = model_layer.model(
            input_ids,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

    assert outputs.past_key_values is not None
    layers = list(iter_layer_kv(outputs.past_key_values))
    assert len(layers) == spec["num_layers"]

    key, value = layers[0]
    assert key.shape[1] == spec["num_kv_heads"]
    assert value.shape[1] == spec["num_kv_heads"]
    assert key.shape[3] == spec["head_dim"]
    assert value.shape == key.shape
    assert key.shape[2] == input_ids.shape[1]

    ops = load_attention_ops(config)
    rope_ctx = build_rope_context(
        model_layer.model,
        outputs.hidden_states[0],
        torch.arange(input_ids.shape[1], device=model_layer.device).unsqueeze(0),
        config=config,
    )
    layer0 = model_layer.model.model.layers[0]
    hidden = pre_attention_hidden(layer0, outputs.hidden_states[0], ops)
    query, key_proj, _ = project_qkv(layer0.self_attn, hidden, ops)
    cos, sin = rope_ctx.get_rope(0)
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


@pytest.mark.skipif(not OLMO2_PATH.exists(), reason="OLMo2-1B not downloaded")
def test_olmo2_eval_metadata_matches_plan(olmo2_model: ModelLayer):
    """Plan §4: evaluation framework records OLMo2 reference metadata."""
    meta = get_model_eval_metadata(olmo2_model.config, local_path=str(OLMO2_PATH))
    caps = CAPABILITIES_BY_MODEL_TYPE["olmo2"]

    assert meta["model_type"] == "olmo2"
    assert meta["attention_family"] == "mha"
    assert meta["num_q_heads"] == 16
    assert meta["num_kv_heads"] == 16
    assert meta["head_dim"] == 128
    assert meta["num_layers"] == 16
    assert meta["rope_mode"] == "global"
    assert meta["state_type"] == "conventional_kv"
    assert meta["adapter"] == "olmo2"
    assert meta["adapter_registered"] is True
    assert meta["state_semantics_complete"] == caps.state_semantics_complete


@pytest.mark.skipif(not OLMO2_PATH.exists(), reason="OLMo2-1B not downloaded")
def test_olmo2_memory_formula_alignment(olmo2_model: ModelLayer):
    """M_KV = L × T × H_KV × (D_k + D_v) × b for symmetric MHA."""
    input_ids = olmo2_model.tokenize("Hello world")
    with torch.no_grad():
        outputs = olmo2_model.forward_with_cache(input_ids, use_cache=True)

    pkv = outputs.past_key_values
    seq_len = input_ids.shape[1]
    head_dim = resolve_head_dim(olmo2_model.config)
    bytes_per_elt = next(iter_layer_kv(pkv))[0].element_size()

    measured = visible_state_bytes(pkv)
    analytical = kv_cache_bytes(
        num_layers=OLMO2_SPEC["num_layers"],
        seq_len=seq_len,
        num_kv_heads=OLMO2_SPEC["num_kv_heads"],
        head_dim=head_dim,
        bytes_per_element=bytes_per_elt,
    )
    assert measured == analytical
    assert measured == get_cache_size_bytes(pkv)


def _assert_eval_branches_complete(result, compressor_name: str) -> None:
    assert result.fidelity is not None
    assert result.behavior is not None
    assert result.system is not None
    assert result.model_metadata is not None
    assert result.model_metadata["adapter"] == "olmo2"

    assert result.fidelity.memory.compression_ratio > 0
    assert result.behavior.perplexity is not None and result.behavior.perplexity > 0
    assert result.behavior.retrieval is not None
    assert result.behavior.reasoning is not None
    assert result.behavior.instruction_following is not None
    assert result.system.throughput is not None
    assert result.system.throughput.tokens_per_second > 0
    assert result.system.memory_bandwidth is not None
    assert result.system.kernel_cost is not None

    if compressor_name == "identity":
        assert result.fidelity.representation.key_cosine_similarity > 0.99
        assert result.fidelity.attention.cosine_similarity > 0.999
        assert result.fidelity.memory.compression_ratio == pytest.approx(1.0, rel=1e-3)
    elif compressor_name == "turboquant":
        assert result.fidelity.representation.key_cosine_similarity > 0.8
        assert result.fidelity.attention.cosine_similarity > 0.8
    elif compressor_name == "qjl":
        # QJL stores 1-bit key sketches; value passthrough stays exact.
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
@pytest.mark.skipif(not OLMO2_PATH.exists(), reason="OLMo2-1B not downloaded")
def test_olmo2_all_eval_branches(olmo2_model: ModelLayer, compressor_name: str, compressor_kwargs: dict):
    """Plan §4: FIDELITY + BEHAVIOR + SYSTEM for each production compressor."""
    compressor = get_compressor(compressor_name, **compressor_kwargs)
    runner = EvaluationRunner(model_layer=olmo2_model, compressor=compressor)
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


@pytest.mark.skipif(not OLMO2_PATH.exists(), reason="OLMo2-1B not downloaded")
def test_olmo2_identity_regression_baseline(olmo2_runner: EvaluationRunner):
    """Plan §4 acceptance: identity path stays numerically stable (within run noise)."""
    result = olmo2_runner.run(
        context_length=128,
        run_fidelity=True,
        run_behavior=True,
        run_perplexity=True,
        run_system=True,
        run_throughput=True,
        generated_tokens=8,
        perplexity_stride=64,
    )
    assert result.fidelity.attention.cosine_similarity > 0.999
    assert 5.0 < result.behavior.perplexity < 50.0
    assert result.system.throughput.tokens_per_second > 0.5
