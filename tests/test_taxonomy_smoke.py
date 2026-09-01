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


def test_vanilla_attention_restores_after_patch():
    import types

    from framework.attention_patches import ensure_vanilla_attention

    class _Attn:
        def forward(self, x):
            return ("vanilla", x)

    class _Layer:
        def __init__(self):
            self.self_attn = _Attn()

    class _Inner:
        def __init__(self):
            self.layers = [_Layer(), _Layer()]

    class _Model:
        def __init__(self):
            self.model = _Inner()

    model = _Model()
    ensure_vanilla_attention(model)
    assert model.model.layers[0].self_attn.forward("x") == ("vanilla", "x")

    def patched(self, x):
        return ("patched", x)

    model.model.layers[0].self_attn.forward = types.MethodType(patched, model.model.layers[0].self_attn)
    assert model.model.layers[0].self_attn.forward("x") == ("patched", "x")
    model._qjl_online_enabled = True
    ensure_vanilla_attention(model)
    assert model.model.layers[0].self_attn.forward("x") == ("vanilla", "x")
    assert not hasattr(model, "_qjl_online_enabled")


def test_palu_representation_roundtrip_is_not_identity():
    import torch

    from eval.fidelity.representation import _layer_roundtrip

    compressor = PaluCompressor(compression_rate=0.25, group_size=2)
    key = torch.randn(1, 4, 16, 32)
    value = torch.randn(1, 4, 16, 32)
    k_hat, v_hat, key_ref, value_ref = _layer_roundtrip(key, value, compressor, 0)
    assert k_hat.shape == key.shape
    rmse = (k_hat.float() - key_ref.float()).pow(2).mean().sqrt().item()
    assert rmse > 1e-3


def test_attention_cosine_clamped_to_unit_interval():
    from eval.fidelity.attention import clamp_cosine

    assert clamp_cosine(1.0078125) == 1.0
    assert clamp_cosine(-1.01) == -1.0
    assert clamp_cosine(float("nan")) == 0.0


def test_paper_collection_flags_qjl_ppl_anomaly():
    from eval.paper_collection import paper_quality_anomalies

    identity = {
        "label": "identity_baseline",
        "compressor": "identity",
        "fidelity": {
            "representation": {"key_rmse": 0.0, "key_cosine_similarity": 1.0},
            "attention": {"rmse": 0.0, "cosine_similarity": 1.0},
            "memory": {"compression_ratio": 1.0},
        },
        "behavior": {"task_quality": {"perplexity": 100.0}},
        "system": {"latency_throughput": {"tokens_per_second": 10.0}},
        "cost": {
            "offline": {"calibration_required": False},
            "oaken_layers": [{"layer": "offline_evaluation", "measured": True}],
        },
    }
    qjl = {
        "label": "qjl_default",
        "compressor": "qjl",
        "fidelity": {
            "representation": {"key_rmse": 5.0, "key_cosine_similarity": 0.01},
            "attention": {"rmse": 5.0, "cosine_similarity": 0.6},
            "memory": {"compression_ratio": 1.5},
        },
        "behavior": {"task_quality": {"perplexity": 5000.0}},
        "system": {"latency_throughput": {"tokens_per_second": 1.0}},
        "cost": {
            "offline": {"calibration_required": False},
            "oaken_layers": [{"layer": "offline_evaluation", "measured": True}],
        },
    }
    anomalies = paper_quality_anomalies([identity, qjl])
    assert any("QJL PPL" in item for item in anomalies)


def test_export_bundle_documentation(tmp_path: Path):
    from reporting.documentation import export_bundle_documentation

    jobs = tmp_path / "jobs"
    jobs.mkdir()
    (jobs / "identity.json").write_text(
        '{"compressor": "identity", "fidelity": {"attention": {"per_layer": []}}}'
    )
    (tmp_path / "run.log").write_text("ok\n")
    dest = export_bundle_documentation(tmp_path, tmp_path / "RESULTS_COMPLETE.md", model_name="test")
    text = dest.read_text()
    assert "identity.json" in text
    assert "run.log" in text
