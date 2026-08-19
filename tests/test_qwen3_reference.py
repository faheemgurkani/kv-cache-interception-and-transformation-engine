"""Qwen3 GQA reference: parameterized conformance + full evaluation-branch verification."""

from __future__ import annotations

from dataclasses import dataclass
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
from framework.model_adapter import load_attention_ops, project_qkv, pre_attention_hidden, resolve_head_dim, resolve_model_type
from framework.model_capabilities import CAPABILITIES_BY_MODEL_TYPE, get_model_eval_metadata
from framework.rope import build_rope_context
from framework.state_interface import visible_state_bytes

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QWEN3_06B_PATH = PROJECT_ROOT / "models" / "qwen3_0.6b"
QWEN3_17B_PATH = PROJECT_ROOT / "models" / "legacy" / "qwen3_1.7b"
OLMO2_PATH = PROJECT_ROOT / "models" / "olmo2_1b"

QWEN3_GQA_SPEC = dict(
    num_layers=28,
    num_q_heads=16,
    num_kv_heads=8,
    head_dim=128,
    kv_to_q_ratio=0.5,
)


@dataclass(frozen=True)
class Qwen3Checkpoint:
    name: str
    path: Path


QWEN3_CHECKPOINTS = [
    Qwen3Checkpoint("qwen3_0.6b", QWEN3_06B_PATH),
    Qwen3Checkpoint("qwen3_1.7b", QWEN3_17B_PATH),
]


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture(scope="module")
def qwen3_06b_model():
    if not QWEN3_06B_PATH.exists():
        pytest.skip("Qwen3-0.6B not downloaded")
    return ModelLayer(model_path=QWEN3_06B_PATH, device=_device())


@pytest.fixture(scope="module")
def qwen3_06b_runner(qwen3_06b_model):
    return EvaluationRunner(model_layer=qwen3_06b_model, compressor=IdentityCompressor())


@pytest.fixture(params=QWEN3_CHECKPOINTS, ids=lambda ckpt: ckpt.name)
def qwen3_checkpoint_model(request):
    ckpt: Qwen3Checkpoint = request.param
    if not ckpt.path.exists():
        pytest.skip(f"Model not downloaded: {ckpt.path}")
    return ckpt, ModelLayer(model_path=ckpt.path, device=_device())


def test_qwen3_gqa_memory_head_ratio_is_half_of_mha():
    """Plan §6: per-layer KV-head component M_Qwen/M_OLMo2 = H_KV/Qwen / H_KV/OLMo = 8/16."""
    seq_len = 128
    head_dim = 128
    bytes_per_elt = 2
    mha_bytes = kv_cache_bytes(
        num_layers=1,
        seq_len=seq_len,
        num_kv_heads=16,
        head_dim=head_dim,
        bytes_per_element=bytes_per_elt,
    )
    gqa_bytes = kv_cache_bytes(
        num_layers=1,
        seq_len=seq_len,
        num_kv_heads=8,
        head_dim=head_dim,
        bytes_per_element=bytes_per_elt,
    )
    assert gqa_bytes / mha_bytes == pytest.approx(0.5, rel=1e-9)


@pytest.mark.parametrize("checkpoint", QWEN3_CHECKPOINTS, ids=lambda ckpt: ckpt.name)
def test_qwen3_checkpoints_share_adapter_path(checkpoint: Qwen3Checkpoint):
    """Plan §5: architecture family, not checkpoint identity, selects the adapter."""
    if not checkpoint.path.exists():
        pytest.skip(f"Model not downloaded: {checkpoint.path}")
    model_layer = ModelLayer(model_path=checkpoint.path, device=_device())
    assert resolve_model_type(model_layer.config) == "qwen3"
    ops = load_attention_ops(model_layer.config)
    assert ops.model_type == "qwen3"
    assert ops.qk_norm_layout == "per_head"


def test_qwen3_adapter_conformance(qwen3_checkpoint_model):
    """Plan §5: parameterized Qwen3 conformance (0.6B + 1.7B share qwen3 adapter)."""
    _ckpt, model_layer = qwen3_checkpoint_model
    config = model_layer.config
    spec = QWEN3_GQA_SPEC

    assert model_layer.attn_implementation == "eager"
    assert resolve_model_type(config) == "qwen3"
    assert int(config.num_attention_heads) == spec["num_q_heads"]
    assert int(getattr(config, "num_key_value_heads", config.num_attention_heads)) == spec["num_kv_heads"]
    assert spec["num_kv_heads"] / spec["num_q_heads"] == spec["kv_to_q_ratio"]

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
    assert value.shape == key.shape

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


@pytest.mark.skipif(not QWEN3_06B_PATH.exists(), reason="Qwen3-0.6B not downloaded")
def test_qwen3_eval_metadata_matches_plan(qwen3_06b_model: ModelLayer):
    """Plan §6: evaluation records GQA architecture metadata."""
    meta = get_model_eval_metadata(qwen3_06b_model.config, local_path=str(QWEN3_06B_PATH))
    caps = CAPABILITIES_BY_MODEL_TYPE["qwen3"]

    assert meta["model_type"] == "qwen3"
    assert meta["attention_family"] == "gqa"
    assert meta["num_q_heads"] == 16
    assert meta["num_kv_heads"] == 8
    assert meta["head_dim"] == 128
    assert meta["num_layers"] == 28
    assert meta["rope_mode"] == "global"
    assert meta["qk_norm_layout"] == "per_head"
    assert meta["state_type"] == "conventional_kv"
    assert meta["adapter"] == "qwen3"
    assert meta["adapter_registered"] is True
    assert meta["state_semantics_complete"] == caps.state_semantics_complete


@pytest.mark.skipif(not QWEN3_06B_PATH.exists(), reason="Qwen3-0.6B not downloaded")
def test_qwen3_memory_formula_alignment(qwen3_06b_model: ModelLayer):
    """M_KV = L × T × H_KV × (D_k + D_v) × b for GQA (H_KV=8)."""
    input_ids = qwen3_06b_model.tokenize("Hello world")
    with torch.no_grad():
        outputs = qwen3_06b_model.forward_with_cache(input_ids, use_cache=True)

    pkv = outputs.past_key_values
    seq_len = input_ids.shape[1]
    head_dim = resolve_head_dim(qwen3_06b_model.config)
    bytes_per_elt = next(iter_layer_kv(pkv))[0].element_size()

    measured = visible_state_bytes(pkv)
    analytical = kv_cache_bytes(
        num_layers=QWEN3_GQA_SPEC["num_layers"],
        seq_len=seq_len,
        num_kv_heads=QWEN3_GQA_SPEC["num_kv_heads"],
        head_dim=head_dim,
        bytes_per_element=bytes_per_elt,
    )
    assert measured == analytical
    assert measured == get_cache_size_bytes(pkv)


@pytest.mark.skipif(
    not QWEN3_06B_PATH.exists() or not OLMO2_PATH.exists(),
    reason="Need both Qwen3-0.6B and OLMo2 for cross-model ratio check",
)
def test_qwen3_vs_olmo2_measured_kv_ratio(qwen3_06b_model: ModelLayer):
    """Empirical check: same T,D,b → ratio = (L_qwen × H_kv_qwen) / (L_olmo × H_kv_olmo)."""
    text = "The quick brown fox jumps over the lazy dog"
    q_ids = qwen3_06b_model.tokenize(text)
    olmo2 = ModelLayer(model_path=OLMO2_PATH, device=_device())
    o_ids = olmo2.tokenize(text)
    assert q_ids.shape[1] == o_ids.shape[1]

    with torch.no_grad():
        q_out = qwen3_06b_model.forward_with_cache(q_ids, use_cache=True)
        o_out = olmo2.forward_with_cache(o_ids, use_cache=True)

    q_bytes = visible_state_bytes(q_out.past_key_values)
    o_bytes = visible_state_bytes(o_out.past_key_values)
    expected_ratio = (28 * 8) / (16 * 16)
    assert q_bytes / o_bytes == pytest.approx(expected_ratio, rel=1e-3)


def _assert_eval_branches_complete(result, compressor_name: str) -> None:
    assert result.fidelity is not None
    assert result.behavior is not None
    assert result.system is not None
    assert result.model_metadata is not None
    assert result.model_metadata["adapter"] == "qwen3"
    assert result.model_metadata["attention_family"] == "gqa"
    assert result.model_metadata["num_kv_heads"] == 8

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
        assert result.fidelity.representation.value_cosine_similarity > 0.99
        assert result.fidelity.attention.cosine_similarity > 0.5
    elif compressor_name == "rocketkv":
        assert result.fidelity.representation.key_rmse < 1e-4
        assert result.fidelity.attention.cosine_similarity > 0.5
    elif compressor_name == "snapkv":
        assert result.fidelity.representation.key_rmse < 1e-4
        assert result.fidelity.memory.compression_ratio >= 1.0
    elif compressor_name == "palu":
        assert result.fidelity.representation.key_rmse >= 0.0
        assert result.fidelity.memory.compression_ratio > 0


@pytest.mark.parametrize(
    "compressor_name,compressor_kwargs",
    [
        ("identity", {}),
        ("turboquant", {"bitwidth": 4}),
        ("qjl", {}),
        ("rocketkv", {"token_budget": 256}),
        ("snapkv", {"max_capacity_prompt": 64, "window_size": 8}),
        ("palu", {"compression_rate": 0.5, "group_size": 4}),
    ],
)
@pytest.mark.skipif(not QWEN3_06B_PATH.exists(), reason="Qwen3-0.6B not downloaded")
def test_qwen3_all_eval_branches(qwen3_06b_model: ModelLayer, compressor_name: str, compressor_kwargs: dict):
    """Plan §6: FIDELITY + BEHAVIOR + SYSTEM for each production compressor."""
    compressor = get_compressor(compressor_name, **compressor_kwargs)
    runner = EvaluationRunner(model_layer=qwen3_06b_model, compressor=compressor)
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


@pytest.mark.skipif(not QWEN3_06B_PATH.exists(), reason="Qwen3-0.6B not downloaded")
def test_qwen3_identity_regression_baseline(qwen3_06b_runner: EvaluationRunner):
    """Identity path stable on GQA; memory should reflect 8 KV heads."""
    result = qwen3_06b_runner.run(
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
    assert result.behavior.perplexity is not None and result.behavior.perplexity > 0
    assert result.system.throughput.tokens_per_second > 0.5
    assert result.fidelity.memory.num_kv_elements > 0
