"""Unit tests for Phase 6–7 controlled experimental conditions."""

from __future__ import annotations

import os

import pytest
import torch

from compressors.identity import IdentityCompressor
from compressors.rocketkv import RocketKVCompressor
from compressors.snapkv import SnapKVCompressor
from compressors.turboquant import TurboQuantCompressor
from compressors.qjl import QJLCompressor
from quantizers.turboquant_pipeline import TurboQuantStage
from eval.controlled_conditions import (
    PHASE6_PRINCIPLE,
    PHASE7_PRINCIPLE,
    REQUIRED_FIXED_AXES,
    build_controlled_conditions,
    build_decoding_configuration,
    build_evaluation_metrics_profile,
    build_input_construction,
    detect_hardware,
    extract_compression_budget,
    validate_controlled_contract,
)


def _eval_config() -> dict:
    return {
        "batch_size": 1,
        "perplexity_stride": 512,
        "generated_tokens": 64,
        "attention_fidelity_tokens": 512,
        "wikitext": {
            "name": "Salesforce/wikitext",
            "config": "wikitext-2-raw-v1",
            "split": "test",
        },
    }


class _FakeTokenizer:
    name_or_path = "/models/qwen3"
    vocab_size = 151_936
    model_max_length = 32_768
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = None


def test_build_controlled_conditions_includes_all_phase7_fixed_axes():
    contract = build_controlled_conditions(
        model_metadata={"model_id": "Qwen/Qwen3-1.7B"},
        eval_config=_eval_config(),
        context_length=256,
        compressor=TurboQuantCompressor(bitwidth=4, stage=TurboQuantStage.FULL),
        tokenizer=_FakeTokenizer(),
        model_path="/models/qwen3",
        device=torch.device("cpu"),
    )

    assert contract.phase == "7"
    assert contract.principle == PHASE7_PRINCIPLE
    for axis in REQUIRED_FIXED_AXES:
        assert axis in contract.fixed

    assert contract.fixed["context_length"] == 256
    assert contract.fixed["generation_length"] == 64
    assert contract.fixed["batch_size"] == 1
    assert contract.fixed["tokenizer"]["name_or_path"] == "/models/qwen3"
    assert contract.fixed["tokenizer"]["vocab_size"] == 151_936
    assert contract.fixed["dataset"]["config"] == "wikitext-2-raw-v1"
    assert contract.fixed["input_construction"]["fidelity"]["target_length"] == 256
    assert contract.fixed["decoding_configuration"]["strategy"] == "greedy"
    assert contract.fixed["decoding_configuration"]["do_sample"] is False
    assert contract.fixed["decoding_configuration"]["selection"] == "argmax"
    assert contract.fixed["hardware"]["device_type"] == "cpu"
    assert contract.variable["compressor"] == "turboquant"
    assert contract.variable["compression_budget"]["bitwidth"] == 4
    assert contract.variable["compression_budget"]["stage"] == "full"

    validate_controlled_contract(contract)
    payload = contract.to_dict()
    assert payload["principle"] == PHASE7_PRINCIPLE
    assert "fidelity" in payload["evaluation_branches"]
    assert payload["variable"]["compression_budget"]["compressor"] == "turboquant"


def test_only_compressor_axis_varies_between_methods():
    base_kwargs = dict(
        model_metadata={"model_id": "Qwen/Qwen3-1.7B"},
        eval_config=_eval_config(),
        context_length=512,
        tokenizer=_FakeTokenizer(),
        device=torch.device("cpu"),
    )
    identity = build_controlled_conditions(compressor=IdentityCompressor(), **base_kwargs)
    rocket = build_controlled_conditions(
        compressor=RocketKVCompressor(token_budget=256, hsa_budget=256),
        **base_kwargs,
    )

    identity_fixed = {k: v for k, v in identity.fixed.items() if k != "model"}
    rocket_fixed = {k: v for k, v in rocket.fixed.items() if k != "model"}
    assert identity_fixed == rocket_fixed
    assert identity.variable["compressor"] == "identity"
    assert rocket.variable["compressor"] == "rocketkv"
    assert rocket.variable["compression_budget"]["token_budget"] == 256
    assert rocket.variable["compression_budget"]["hsa_budget"] == 256


@pytest.mark.parametrize(
    ("compressor", "expected_keys"),
    [
        (IdentityCompressor(), {"compressor": "identity", "bitwidth": 16}),
        (
            RocketKVCompressor(token_budget=512, hsa_budget=1024, window_size=32),
            {"token_budget": 512, "hsa_budget": 1024, "window_size": 32},
        ),
        (
            SnapKVCompressor(max_capacity_prompt=1024, window_size=32, kernel_size=5),
            {"max_capacity_prompt": 1024, "window_size": 32, "kernel_size": 5},
        ),
        (
            TurboQuantCompressor(bitwidth=2, stage=TurboQuantStage.WHT_ONLY),
            {"bitwidth": 2, "stage": "wht_only"},
        ),
    ],
)
def test_extract_compression_budget_method_specific(compressor, expected_keys):
    budget = extract_compression_budget(compressor)
    for key, value in expected_keys.items():
        assert budget[key] == value


def test_extract_compression_budget_includes_qjl_seed():
    budget = extract_compression_budget(QJLCompressor(seed=42))
    assert budget["seed"] == 42
    assert budget["compression_method"] == "qjl"


def test_build_controlled_conditions_exports_precision():
    contract = build_controlled_conditions(
        model_metadata={"model_id": "test"},
        eval_config=_eval_config(),
        context_length=128,
        compressor=IdentityCompressor(),
        tokenizer=_FakeTokenizer(),
        device=torch.device("cpu"),
        model_precision=torch.float16,
    )
    assert contract.fixed["precision"] == "float16"


def test_build_decoding_configuration_is_deterministic_greedy():
    cfg = build_decoding_configuration(generation_length=64, perplexity_stride=512)
    assert cfg["strategy"] == "greedy"
    assert cfg["do_sample"] is False
    assert cfg["temperature"] is None
    assert cfg["generation_length"] == 64
    assert cfg["perplexity_stride"] == 512


def test_build_input_construction_wikitext_and_synthetic_sources():
    construction = build_input_construction(
        dataset="wikitext-2-raw-v1",
        dataset_split="test",
        context_length=128,
        generation_length=32,
        run_retrieval=True,
        run_instruction_following=True,
        run_reasoning=False,
    )
    assert construction["fidelity"]["source"] == "wikitext2_concat"
    assert construction["behavior_perplexity"]["target_length"] == 128
    assert construction["behavior_retrieval"]["source"] == "synthetic_needle_in_haystack"
    assert construction["behavior_instruction_following"]["source"] == "synthetic_instruction_compliance"
    assert "behavior_reasoning" not in construction
    assert construction["system_throughput_prefix"]["generation_length"] == 32


def test_build_evaluation_metrics_profile_reflects_flags():
    profile = build_evaluation_metrics_profile(
        run_fidelity=True,
        run_behavior=True,
        run_perplexity=True,
        run_retrieval=False,
        run_reasoning=False,
        run_instruction_following=True,
        run_system=True,
        run_throughput=True,
        run_peak_memory=True,
        run_memory_bandwidth=False,
        run_kernel_cost=False,
        run_gpu_utilization=False,
        run_cost=True,
        attention_fidelity_tokens=512,
    )
    assert profile["branches"] == {
        "fidelity": True,
        "behavior": True,
        "system": True,
        "cost": True,
    }
    assert profile["behavior"] == ["perplexity", "instruction_following"]
    assert profile["system"] == ["ttft", "itl", "throughput", "latency", "peak_memory"]


def test_detect_hardware_cpu_and_env_override(monkeypatch):
    monkeypatch.delenv("KV_HARDWARE_PROFILE", raising=False)
    profile = detect_hardware(torch.device("cpu"))
    assert profile["device_type"] == "cpu"
    assert profile["device_name"] == "CPU"
    assert "NVIDIA A10G" in profile["documented_reference_gpu"]
    assert profile["single_gpu_policy"] is True
    assert profile["multi_gpu_matrix"] is False

    monkeypatch.setenv("KV_HARDWARE_PROFILE", "NVIDIA L4")
    profile = detect_hardware(torch.device("cpu"))
    assert profile["documented_reference_gpu"] == "NVIDIA L4"


def test_validate_controlled_contract_rejects_incomplete():
    from eval.controlled_conditions import ControlledInterceptionContract

    incomplete = ControlledInterceptionContract(fixed={"model": {}}, variable={"compressor": "identity"})
    with pytest.raises(ValueError, match="missing fixed axes"):
        validate_controlled_contract(incomplete)


def test_phase6_principle_still_documented():
    assert "same inference path" in PHASE6_PRINCIPLE.lower()
    assert "different kv transformation" in PHASE7_PRINCIPLE.lower()
