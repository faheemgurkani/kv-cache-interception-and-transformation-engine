"""Taxonomy-coverage smoke: dummy math, KPI schema, Palu Gemma geometry, Modal job grid."""

from __future__ import annotations

from pathlib import Path

from framework.snapkv_online import _align_attention_mask
from compressors.palu import PaluCompressor
from compressors.taxonomy import CompressionCategory, active_eval_methods, taxonomy_categories_covered
from eval.kpi_schema import validate_bundle
from eval.taxonomy_smoke import (
    TAXONOMY_SMOKE_PRESET,
    _fake_gemma_mqa_model,
    run_dummy_collection,
    run_dummy_compressor_math,
)
from modal_app.job_spec import build_sweep_jobs
from modal_app.merge import CSV_FIELDNAMES


def test_active_eval_methods_cover_all_taxonomy_categories():
    methods = active_eval_methods()
    assert "kivi" not in methods
    assert set(methods) == {"identity", "turboquant", "qjl", "rocketkv", "snapkv", "palu"}
    assert taxonomy_categories_covered(methods) == set(CompressionCategory)


def test_taxonomy_smoke_preset_covers_active_methods():
    jobs = build_sweep_jobs(context_lengths=[128], preset=TAXONOMY_SMOKE_PRESET)
    names = [job.compressor for job in jobs]
    assert names == list(active_eval_methods())
    assert all(job.run_reasoning and job.run_kernel_cost and job.run_memory_bandwidth for job in jobs)
    assert all(job.generated_tokens == 8 for job in jobs)
    palu = next(job for job in jobs if job.compressor == "palu")
    assert palu.compressor_kwargs["group_size"] == 1


def test_dummy_compressor_math_and_collection(tmp_path: Path):
    math_reports = run_dummy_compressor_math()
    failed = [item["compressor"] for item in math_reports if not item["ok"]]
    assert not failed, failed
    summary = run_dummy_collection(tmp_path)
    assert summary["ok"] is True
    assert summary["job_count"] == 6
    errors = validate_bundle(
        # re-read via merge is enough; dummy already validated
        __import__("json").loads(Path(summary["merged_json"]).read_text())["results"],
        require_smoke_extras=True,
        execution_platform="local_dummy",
    )
    assert errors == []
    assert "taxonomy_primary" in CSV_FIELDNAMES
    assert "oaken_layers_measured" in CSV_FIELDNAMES
    assert "gate_loader_state" in CSV_FIELDNAMES


def test_snapkv_aligns_decode_attention_mask_to_evicted_kv_len():
    import torch

    mask = torch.zeros(1, 1, 1, 129)
    aligned = _align_attention_mask(mask, q_len=1, k_len=65)
    assert aligned is not None
    assert aligned.shape == (1, 1, 1, 65)


def test_palu_bind_uses_config_kv_heads_for_gemma_mqa():
    compressor = PaluCompressor(compression_rate=0.5, group_size=1)
    compressor.bind_model(_fake_gemma_mqa_model())
    factors = compressor.layer_factors(0)
    assert factors is not None
    assert factors.num_kv_heads == 1
    assert factors.head_dim == 256
