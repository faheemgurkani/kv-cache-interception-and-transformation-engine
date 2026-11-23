"""Audit S4: KPI gating is explicit, uniform across methods, and recorded."""

from modal_app.job_spec import EvalJobSpec, resolve_kpi_plan

HW = {"hardware": {"collect_peak_memory": True, "collect_gpu_utilization": True}}


def _spec(ctx, **kw):
    return EvalJobSpec(compressor="identity", context_length=ctx, run_reasoning=True, **kw)


def test_no_gating_by_default_runs_everything_at_every_ctx():
    for ctx in (128, 256, 512):
        plan = resolve_kpi_plan(_spec(ctx), {**HW, "kpi_gating": {"skip_extra_passes_ctx_ge": None}}, True)
        assert plan["gated"] is False and plan["skipped"] == []
        assert plan["run_reasoning"] and plan["run_peak_memory"] and plan["run_gpu_utilization"]
        assert plan["include_baselines"]


def test_missing_gating_section_means_no_gating():
    plan = resolve_kpi_plan(_spec(512), HW, True)
    assert plan["gated"] is False and plan["skipped"] == []


def test_explicit_gate_skips_listed_passes_only_at_or_above_ctx():
    cfg = {**HW, "kpi_gating": {"skip_extra_passes_ctx_ge": 256}}
    low = resolve_kpi_plan(_spec(128), cfg, True)
    high = resolve_kpi_plan(_spec(256), cfg, True)
    assert low["skipped"] == []
    assert high["gated"] is True
    assert set(high["skipped"]) == {"reasoning", "peak_memory", "gpu_utilization", "baselines"}
    assert not high["run_reasoning"] and not high["include_baselines"]


def test_reasoning_not_reported_skipped_when_never_requested():
    spec = EvalJobSpec(compressor="identity", context_length=512, run_reasoning=False)
    plan = resolve_kpi_plan(spec, {**HW, "kpi_gating": {"skip_extra_passes_ctx_ge": 256}}, True)
    assert "reasoning" not in plan["skipped"]
