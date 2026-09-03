"""Dummy-local taxonomy smoke: compressor math + KPI collection wiring.

Does not load a language model. Exercises every *active* plug-in (not KIVI stub)
on synthetic tensors, builds the Modal job grid, synthesizes a complete
EvaluationResult-shaped payload, and validates schema / merge / Phase-14.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from compressors.registry import get_compressor
from compressors.taxonomy import (
    CompressionCategory,
    active_eval_methods,
    get_method_taxonomy,
    taxonomy_categories_covered,
)
from eval.cost.accounting import evaluate_cost
from eval.reproducibility.manifest import collect_git_sha
from eval.kpi_schema import (
    GATE_NAMES,
    OAKEN_LAYER_NAMES,
    validate_bundle,
    validate_payload_invariants,
)
from modal_app.job_spec import build_sweep_jobs, get_preset_options, get_sweep_configs
from modal_app.merge import flatten_result_payload, write_merged_reports
from quantizers.palu import truncated_svd_factors

TAXONOMY_SMOKE_PRESET = "taxonomy_smoke"
SMOKE_CONTEXT_LENGTH = 128


def _kv(batch: int = 1, heads: int = 4, seq: int = 32, dim: int = 64, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    key = torch.randn(batch, heads, seq, dim, generator=gen)
    value = torch.randn(batch, heads, seq, dim, generator=gen)
    return key, value


def _finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all().item())


def run_dummy_compressor_math() -> list[dict[str, Any]]:
    """Round-trip / invariant checks per active compressor. No model weights."""
    reports: list[dict[str, Any]] = []

    identity = get_compressor("identity")
    key, value = _kv(seed=1)
    compressed = identity.compress(key, value, layer=0)
    k2, v2 = identity.decompress(compressed)
    reports.append(
        {
            "compressor": "identity",
            "ok": _finite(k2) and _finite(v2) and torch.allclose(k2, key) and torch.allclose(v2, value),
            "detail": "exact FP round-trip",
        }
    )

    turboquant = get_compressor("turboquant", stage="full", bitwidth=4)
    key, value = _kv(seed=2)
    tq = turboquant.compress(key, value, layer=0)
    tk, tv = turboquant.decompress(tq)
    tq_err = turboquant.reconstruction_error(key, value, layer=0)
    reports.append(
        {
            "compressor": "turboquant",
            "ok": (
                _finite(tk)
                and _finite(tv)
                and tq_err["key_rmse"] > 0
                and math.isfinite(tq_err["key_rmse"])
                and (turboquant.theoretical_compression_ratio(context_length=SMOKE_CONTEXT_LENGTH) or 0) > 1
            ),
            "detail": f"key_rmse={tq_err['key_rmse']:.4f} value_rmse={tq_err['value_rmse']:.4f}",
        }
    )

    qjl = get_compressor("qjl", bitwidth=1, seed=42)
    key, value = _kv(seed=3)
    qk = qjl.compress_kv(key, layer=0, mode="key")
    qv = qjl.compress_kv(value, layer=0, mode="value")
    qk2 = qjl.decompress_kv(qk, mode="key")
    qv2 = qjl.decompress_kv(qv, mode="value")
    query = torch.randn_like(key)
    scores = qjl.estimate_attention_scores(query, qk, head_dim=key.shape[-1])
    reports.append(
        {
            "compressor": "qjl",
            "ok": _finite(qk2) and _finite(qv2) and torch.allclose(qv2, value) and _finite(scores),
            "detail": f"attention_estimator_shape={tuple(scores.shape)}",
        }
    )

    rocketkv = get_compressor("rocketkv", token_budget=16, hsa_budget=16, window_size=8)
    key, value = _kv(seq=48, seed=4)
    rk = rocketkv.compress_layer_from_kv(key, value, layer=0, original_seq_len=key.shape[2])
    kept = rk.keys.keys if hasattr(rk.keys, "keys") else None
    kept_len = int(kept.shape[2]) if kept is not None else -1
    reports.append(
        {
            "compressor": "rocketkv",
            "ok": kept is not None and 0 < kept_len <= 16 and _finite(kept),
            "detail": f"kept_seq={kept_len} budget=16 orig={key.shape[2]}",
        }
    )

    snapkv = get_compressor("snapkv", max_capacity_prompt=16, window_size=8, kernel_size=5)
    key, value = _kv(seq=48, seed=5)
    sk = snapkv.compress(key, value, layer=0, query_states=key)
    sk_len = int(sk.keys.keys.shape[2]) if hasattr(sk.keys, "keys") else int(sk.keys.shape[2])
    reports.append(
        {
            "compressor": "snapkv",
            "ok": sk_len == 16,
            "detail": f"evicted {key.shape[2]} → {sk_len}",
        }
    )

    palu = get_compressor("palu", compression_rate=0.5, group_size=1)
    key, value = _kv(heads=1, dim=64, seq=32, seed=6)
    palu_err = palu.reconstruction_error(key, value, layer=0)
    weight = torch.randn(64, 128)
    a, b = truncated_svd_factors(weight, rank=16)
    svd_rel = (weight - a @ b).norm() / weight.norm()
    reports.append(
        {
            "compressor": "palu",
            "ok": (
                math.isfinite(palu_err["key_rmse"])
                and palu_err["key_rmse"] >= 0
                and float(svd_rel) < 0.9
                and palu.theoretical_compression_ratio() == 2.0
            ),
            "detail": f"lowrank_key_rmse={palu_err['key_rmse']:.4f} svd_rel={float(svd_rel):.4f}",
        }
    )

    gemma_like = _fake_gemma_mqa_model()
    palu_g = get_compressor("palu", compression_rate=0.5, group_size=1)
    palu_g.bind_model(gemma_like)
    factors = palu_g.layer_factors(0)
    reports.append(
        {
            "compressor": "palu_gemma_geometry",
            "ok": (
                factors is not None
                and factors.num_kv_heads == 1
                and factors.head_dim == 256
            ),
            "detail": (
                f"num_kv_heads={getattr(factors, 'num_kv_heads', None)} "
                f"head_dim={getattr(factors, 'head_dim', None)}"
            ),
        }
    )

    return reports


def _fake_gemma_mqa_model():
    """Minimal stand-in for Gemma3-270M attention geometry (1 KV head, dim 256)."""
    import torch.nn as nn

    class _Cfg:
        num_key_value_heads = 1
        head_dim = 256

    class _Attn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head_dim = 256
            self.config = _Cfg()
            self.k_proj = nn.Linear(640, 256, bias=False)
            self.v_proj = nn.Linear(640, 256, bias=False)

    class _Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = _Attn()

    class _Inner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([_Layer()])

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = _Inner()

    return _Model()


def _dummy_oaken_layers(calibration_required: bool) -> list[dict[str, Any]]:
    return [
        {"layer": name, "description": name, "measured": True, "metrics": {}}
        for name in OAKEN_LAYER_NAMES
    ]


def build_dummy_payload(compressor_name: str, *, label: str, context_length: int = SMOKE_CONTEXT_LENGTH) -> dict[str, Any]:
    """Synthesize a complete EvaluationResult.to_dict() so collection/merge can be tested offline."""
    compressor = get_compressor(compressor_name)
    if compressor_name == "palu":
        compressor.bind_model(_fake_gemma_mqa_model())
    taxonomy = get_method_taxonomy(compressor_name)
    assert taxonomy is not None
    cost = evaluate_cost(compressor, context_length=context_length, fidelity=None, system=None)
    identity = compressor_name == "identity"
    git_sha = collect_git_sha() or "dummy-local"
    payload = {
        "label": label,
        "compressor": compressor_name,
        "bitwidth": getattr(compressor, "bitwidth", None),
        "stage": getattr(getattr(compressor, "stage", None), "value", None),
        "context_length": context_length,
        "model": {"model_type": "gemma3_text", "architecture_family": "mqa"},
        "compatibility_gates": {
            name: {"gate": name, "passed": True, "detail": "dummy"} for name in GATE_NAMES
        },
        "compatibility_manifest": {"architecture": {"family": "mqa"}},
        "fidelity": {
            "representation": {
                "key_rmse": 0.0 if identity else 0.12,
                "value_rmse": 0.0 if identity else 0.11,
                "key_relative_error": 0.0 if identity else 0.05,
                "value_relative_error": 0.0 if identity else 0.05,
                "key_cosine_similarity": 1.0 if identity else 0.97,
                "value_cosine_similarity": 1.0 if identity else 0.96,
                "reconstruction_degenerate": compressor_name == "palu",
                "reconstruction_degenerate_reason": (
                    "effective_rank >= min(seq, head_dim) at ctx=128" if compressor_name == "palu" else None
                ),
            },
            "attention": {
                "rmse": 0.0 if identity else 0.08,
                "cosine_similarity": 1.0 if identity else 0.95,
                "max_error": 0.0 if identity else 0.4,
            },
            "memory": {
                "uncompressed_bytes": 4096,
                "compressed_bytes": 4096 if identity else 2048,
                "compression_ratio": 1.0 if identity else 2.0,
                "effective_bits_per_kv_element": 16.0 if identity else 8.0,
                "shared_metadata_bytes": 0,
            },
            "recurrent": {
                "applicable": False,
                "layers_with_recurrent": 0,
                "exact_preservation": True,
                "max_abs_error": 0.0,
                "per_layer": [],
            },
        },
        "behavior": {
            "task_quality": {
                "perplexity": 14.2 if compressor_name != "snapkv" else 16.8,
                "perplexity_baseline": 14.2,
                "n_tokens": 96,
                "nll_sum": 450.0,
                "prefill_tokens": context_length,
            },
            "retrieval": {"needle_depth_frac": 0.5, "context_length": context_length, "num_trials": 5, "exact_match_accuracy": 1.0},
            "reasoning": {"num_trials": 10, "exact_match_accuracy": 0.5},
            "instruction_following": {"num_trials": 6, "format_compliance_rate": 0.8},
        },
        "system": {
            "latency_throughput": {
                "tokens_per_second": 40.0,
                "ttft_ms": 12.0,
                "itl_ms_mean": 8.0,
                "end_to_end_latency_ms": 80.0,
                "latency_ms_per_token": 25.0,
                "online_compressed_kv": True,
            },
            "peak_memory": {
                "peak_allocated_mb": 1024.0,
                "peak_reserved_mb": 1100.0,
                "kv_uncompressed_mb": 2.3,
                "kv_compressed_mb": 2.3 if identity else 1.15,
                "weight_dominated": True,
            },
            "memory_bandwidth": {"effective_bandwidth_gbps": 1.2},
            "kernel_cost": {
                "compress_time_ms": 0.1,
                "decompress_time_ms": 0.1,
                "compress_decompress_time_ms": 0.2,
                "attention_execution_time_ms": 1.0,
            },
            "gpu_utilization": {"mean_utilization_pct": 50.0, "max_utilization_pct": 80.0},
        },
        "cost": cost.to_dict(),
        "taxonomy": taxonomy.to_dict(),
        "controlled_conditions": {
            "phase": "7",
            "principle": "dummy",
            "fixed": {
                "model": {"model_type": "gemma3_text"},
                "tokenizer": {"name_or_path": "models/gemma3_270m", "vocab_size": 262144},
                "dataset": {"name": "Salesforce/wikitext", "split": "test"},
                "input_construction": {"context_length": context_length},
                "context_length": context_length,
                "generation_length": 8,
                "batch_size": 1,
                "precision": "bfloat16",
                "decode_loop": "KVCacheEngine",
                "decoding_configuration": {"strategy": "greedy"},
                "hardware": {"device_type": "cpu", "execution_platform": "local_dummy"},
                "evaluation_metrics": {"branches": {"fidelity": True, "behavior": True, "system": True}},
                "attention_implementation": "eager",
                "evaluation_orchestrator": "eval/runner.py",
                "kv_interception_engine": "framework/kv_engine.py",
                "git_sha": git_sha,
                "paper_contract": {
                    "generation_length": 64,
                    "precision": "float16",
                    "actual_generation_length": 8,
                    "actual_precision": "bfloat16",
                    "deviates": True,
                },
            },
            "variable": {
                "compressor": compressor_name,
                "compression_budget": {
                    "compressor": compressor_name,
                    "compression_method": compressor_name,
                    "bitwidth": getattr(compressor, "bitwidth", None),
                    "seed": 42,
                },
            },
            "evaluation_branches": ["fidelity", "behavior", "system"],
        },
        "hardware": {
            "device_type": "cpu",
            "execution_platform": "local_dummy",
            "single_gpu_policy": True,
            "multi_gpu_matrix": False,
            "configured_gpu": None,
        },
        "git_sha": git_sha,
        "status": "ok",
    }
    # Fill cost fields that dummy evaluate_cost left None so schema/math checks pass.
    payload["cost"]["compression"]["theoretical_compression_ratio"] = (
        compressor.theoretical_compression_ratio(context_length=context_length) or (1.0 if identity else 2.0)
    )
    payload["cost"]["compression"]["actual_compression_ratio"] = 1.0 if identity else 2.0
    payload["cost"]["online"]["end_to_end_decode_cost_ms"] = 80.0
    payload["cost"]["online"]["compression_time_ms"] = 0.1
    payload["cost"]["online"]["decompression_time_ms"] = 0.1
    payload["cost"]["online"]["attention_cost_ms"] = 1.0
    payload["cost"]["online"]["compress_decompress_time_ms"] = 0.2
    payload["cost"]["online"]["kernel_cost_measured"] = True
    payload["cost"]["oaken_layers"] = _dummy_oaken_layers(bool(payload["cost"]["offline"]["calibration_required"]))
    return payload


def run_dummy_collection(output_dir) -> dict[str, Any]:
    """Build jobs, dummy payloads, merge reports, and return a machine-readable summary."""
    from pathlib import Path

    jobs = build_sweep_jobs(context_lengths=[SMOKE_CONTEXT_LENGTH], preset=TAXONOMY_SMOKE_PRESET)
    options = get_preset_options(TAXONOMY_SMOKE_PRESET)
    configs = get_sweep_configs(TAXONOMY_SMOKE_PRESET)
    method_names = [job.compressor for job in jobs]
    math_reports = run_dummy_compressor_math()
    math_failed = [item for item in math_reports if not item["ok"]]

    payloads = [build_dummy_payload(job.compressor, label=job.label, context_length=job.context_length) for job in jobs]
    schema_errors = validate_bundle(
        payloads,
        require_smoke_extras=True,
        execution_platform="local_dummy",
        require_full_taxonomy=True,
    )
    # Dummy platform is local_dummy; per-payload checks used that. Coverage uses active methods.
    json_path, csv_path = write_merged_reports(payloads, Path(output_dir), "taxonomy_smoke_dummy")

    coverage = sorted(c.value for c in taxonomy_categories_covered(tuple(method_names)))
    summary = {
        "preset": TAXONOMY_SMOKE_PRESET,
        "job_count": len(jobs),
        "methods": method_names,
        "categories_covered": coverage,
        "preset_options": options,
        "sweep_config_count": len(configs),
        "math_reports": math_reports,
        "math_failed": math_failed,
        "schema_errors": schema_errors,
        "merged_json": str(json_path),
        "merged_csv": str(csv_path),
        "flatten_columns_ok": all(
            flatten_result_payload(item).get("taxonomy_primary") == get_method_taxonomy(item["compressor"]).primary.value  # type: ignore[union-attr]
            for item in payloads
        ),
        "active_eval_methods": list(active_eval_methods()),
        "ok": not math_failed and not schema_errors,
    }
    return summary
