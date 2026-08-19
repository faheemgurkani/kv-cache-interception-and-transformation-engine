"""Tests for Phase 14 reproducibility harness (RESEARCH_REDESIGN_PLAN L1095–1179)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from compressors.identity import IdentityCompressor
from compressors.qjl import QJLCompressor
from compressors.turboquant import TurboQuantCompressor
from eval.cost.accounting import evaluate_cost
from eval.fidelity.memory import MemoryMetrics
from eval.reproducibility.manifest import (
    PHASE14_FIELDS,
    extract_phase14_manifest,
    validate_phase14_manifest,
)
from eval.runner import EvaluationRunner
from framework.config import PROJECT_ROOT, load_eval_config, load_model_config
from framework.model import ModelLayer
from quantizers.turboquant_pipeline import TurboQuantStage

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "legacy" / "qwen3_1.7b"
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "model_qwen3.yaml"


def _fake_result_payload(**overrides) -> dict:
    """Minimal EvaluationResult-shaped dict satisfying Phase 14 invariants."""
    base = {
        "compressor": "turboquant",
        "context_length": 256,
        "model": {"model_id": "Qwen/Qwen3-1.7B", "num_layers": 28},
        "controlled_conditions": {
            "fixed": {
                "model": {"model_id": "Qwen/Qwen3-1.7B", "num_layers": 28},
                "context_length": 256,
                "generation_length": 64,
                "batch_size": 1,
                "precision": "float16",
                "dataset": {
                    "name": "Salesforce/wikitext",
                    "config": "wikitext-2-raw-v1",
                    "split": "test",
                },
                "hardware": {"device_type": "cpu", "single_gpu_policy": True},
            },
            "variable": {
                "compressor": "turboquant",
                "compression_budget": {
                    "compressor": "turboquant",
                    "compression_method": "turboquant",
                    "bitwidth": 4,
                    "stage": "full",
                    "seed": 42,
                },
            },
        },
        "hardware": {"device_type": "cpu", "execution_platform": "local"},
        "fidelity": {
            "memory": {
                "uncompressed_bytes": 4000,
                "compressed_bytes": 1000,
                "compression_ratio": 4.0,
            }
        },
        "cost": {
            "compression": {
                "theoretical_compression_ratio": 4.0,
                "actual_compression_ratio": 4.0,
                "actual_memory_reduction_bytes": 3000,
                "uncompressed_bytes": 4000,
                "compressed_bytes": 1000,
            },
            "offline": {
                "calibration_required": True,
                "calibration_dataset": "gaussian_synthetic_1M",
                "calibration_time_ms": 12.5,
            },
        },
    }
    base.update(overrides)
    return base


def test_phase14_field_checklist_matches_plan():
    assert PHASE14_FIELDS == (
        "model",
        "context_length",
        "generation_length",
        "hardware",
        "batch_size",
        "compression_method",
        "compression_ratio",
        "calibration",
        "dataset",
        "seed",
        "precision",
    )


def test_extract_phase14_manifest_maps_documented_paths():
    manifest = extract_phase14_manifest(_fake_result_payload())
    assert manifest["model"]["model_id"] == "Qwen/Qwen3-1.7B"
    assert manifest["context_length"] == 256
    assert manifest["generation_length"] == 64
    assert manifest["batch_size"] == 1
    assert manifest["precision"] == "float16"
    assert manifest["compression_method"] == "turboquant"
    assert manifest["seed"] == 42
    assert manifest["dataset"]["config"] == "wikitext-2-raw-v1"
    assert manifest["hardware"]["device_type"] == "cpu"
    assert manifest["compression_ratio"]["measured"] == 4.0
    assert manifest["compression_ratio"]["theoretical"] == 4.0
    assert manifest["calibration"]["calibration_required"] is True


def test_validate_phase14_manifest_passes_consistent_payload():
    errors = validate_phase14_manifest(_fake_result_payload())
    assert errors == []


def test_validate_phase14_manifest_catches_compression_ratio_math_error():
    payload = _fake_result_payload()
    payload["fidelity"]["memory"]["compression_ratio"] = 2.0  # should be 4.0
    errors = validate_phase14_manifest(payload)
    assert any("uncompressed/compressed" in err for err in errors)


def test_validate_phase14_manifest_catches_cost_memory_reduction_mismatch():
    payload = _fake_result_payload()
    payload["cost"]["compression"]["actual_memory_reduction_bytes"] = 999
    errors = validate_phase14_manifest(payload)
    assert any("actual_memory_reduction_bytes" in err for err in errors)


def test_validate_phase14_manifest_catches_compressor_name_drift():
    payload = _fake_result_payload(compressor="qjl")
    errors = validate_phase14_manifest(payload)
    assert any("compression_method" in err for err in errors)


def test_cost_block_aligns_with_fidelity_memory_math():
    memory = MemoryMetrics(
        context_length=128,
        num_kv_elements=100,
        uncompressed_bytes=8000,
        compressed_bytes=2000,
        shared_metadata_bytes=128,
        compression_ratio=4.0,
        effective_bits_per_kv_element=16.0,
        process_memory_mb=50.0,
    )

    mem = memory

    class _FakeFidelity:
        memory = mem

    cost = evaluate_cost(TurboQuantCompressor(bitwidth=4), context_length=128, fidelity=_FakeFidelity())
    assert cost.compression.actual_compression_ratio == 4.0
    assert cost.compression.actual_memory_reduction_bytes == 6000
    assert cost.compression.uncompressed_bytes == 8000
    assert cost.compression.compressed_bytes == 2000

    payload = _fake_result_payload()
    payload["fidelity"]["memory"] = {
        "uncompressed_bytes": memory.uncompressed_bytes,
        "compressed_bytes": memory.compressed_bytes,
        "compression_ratio": memory.compression_ratio,
    }
    payload["cost"] = cost.to_dict()
    payload["controlled_conditions"]["variable"]["compressor"] = "turboquant"
    payload["compressor"] = "turboquant"
    assert validate_phase14_manifest(payload) == []


@pytest.mark.parametrize(
    ("compressor", "calibration_required", "seed"),
    [
        (IdentityCompressor(), False, None),
        (QJLCompressor(seed=42), False, 42),
        (TurboQuantCompressor(bitwidth=4, stage=TurboQuantStage.FULL, seed=42), True, 42),
    ],
)
def test_calibration_and_seed_logical_alignment(compressor, calibration_required, seed):
    from eval.controlled_conditions import build_controlled_conditions, extract_compression_budget

    eval_config = load_eval_config()
    contract = build_controlled_conditions(
        model_metadata={"model_id": "test"},
        eval_config=eval_config,
        context_length=512,
        compressor=compressor,
        model_precision=torch.float16,
    )
    budget = extract_compression_budget(compressor)
    offline = compressor.offline_cost_metadata()

    assert budget["compression_method"] == compressor.name
    assert contract.variable["compressor"] == compressor.name
    assert contract.fixed["batch_size"] == eval_config["batch_size"]
    assert contract.fixed["generation_length"] == eval_config["generated_tokens"]
    assert contract.fixed["precision"] == "float16"
    assert contract.fixed["dataset"]["config"] == eval_config["wikitext"]["config"]
    assert offline.calibration_required is calibration_required
    if seed is None:
        assert "seed" not in budget
    else:
        assert budget["seed"] == seed


def test_yaml_config_source_of_truth_matches_eval_export():
    eval_cfg = load_eval_config()
    model_cfg = load_model_config(CONFIG_PATH if CONFIG_PATH.exists() else PROJECT_ROOT / "configs" / "model.yaml")

    contract_fields = {
        "batch_size": eval_cfg["batch_size"],
        "generation_length": eval_cfg["generated_tokens"],
        "perplexity_stride": eval_cfg["perplexity_stride"],
        "dataset_config": eval_cfg["wikitext"]["config"],
        "context_lengths": model_cfg["context_lengths"],
    }

    assert contract_fields["batch_size"] == 1
    assert contract_fields["generation_length"] == 64
    assert contract_fields["perplexity_stride"] == 512
    assert contract_fields["dataset_config"] == "wikitext-2-raw-v1"
    assert contract_fields["context_lengths"] == [128, 256, 512]

    sweeps_path = PROJECT_ROOT / "configs" / "modal_sweeps.yaml"
    if sweeps_path.exists():
        sweeps = yaml.safe_load(sweeps_path.read_text())
        qjl_entries = sweeps.get("presets", {}).get("qjl", [])
        if qjl_entries:
            entry = qjl_entries[0]
            seed = entry.get("seed") or entry.get("compressor_kwargs", {}).get("seed")
            assert seed == 42


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="Model not downloaded")
def test_runner_to_dict_satisfies_phase14_harness():
    model_layer = ModelLayer(model_path=MODEL_DIR)
    yaml_cfg = load_model_config(CONFIG_PATH)
    runner = EvaluationRunner(
        model_layer=model_layer,
        compressor=TurboQuantCompressor(bitwidth=4, stage=TurboQuantStage.FULL, seed=42),
        model_config=yaml_cfg,
    )
    result = runner.run(
        context_length=128,
        run_fidelity=True,
        run_perplexity=False,
        run_retrieval=False,
        run_instruction_following=False,
        run_throughput=False,
        run_cost=True,
        generated_tokens=8,
    )
    payload = result.to_dict()
    manifest = extract_phase14_manifest(payload)

    for field in PHASE14_FIELDS:
        assert manifest[field] is not None, field

    assert manifest["compression_method"] == "turboquant"
    assert manifest["seed"] == 42
    assert manifest["precision"] is not None
    assert manifest["context_length"] == 128
    assert manifest["generation_length"] == 8
    assert manifest["calibration"]["calibration_required"] is True

    errors = validate_phase14_manifest(payload)
    assert errors == [], errors

    # Measured ratio must match byte accounting when fidelity memory ran.
    memory = payload["fidelity"]["memory"]
    assert memory["compression_ratio"] == pytest.approx(
        memory["uncompressed_bytes"] / memory["compressed_bytes"],
        rel=1e-5,
    )
