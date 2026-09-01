"""Real local EvaluationRunner smoke: Gemma3-270M × every active compressor.

Unlike dummy tensor checks, this loads the checkpoint and runs the same
FIDELITY / BEHAVIOR / SYSTEM / COST path used by ``scripts/run_eval.py`` and
Modal ``eval_worker``.
"""

from __future__ import annotations

import json
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from compressors.registry import get_compressor
from eval.kpi_schema import validate_bundle
from eval.taxonomy_smoke import TAXONOMY_SMOKE_PRESET
from framework.attention_patches import ensure_vanilla_attention
from framework.config import PROJECT_ROOT, load_model_config
from framework.model import ModelLayer
from modal_app.job_spec import build_sweep_jobs
from modal_app.merge import write_merged_reports
from reporting.documentation import export_bundle_documentation
from reporting.reporter import ResultReporter

LOCAL_LIVE_CONTEXT = 64
LOCAL_LIVE_GENERATED_TOKENS = 4


def run_local_live_collection(
    output_dir: Path,
    *,
    model_config_path: str | Path | None = None,
    context_length: int = LOCAL_LIVE_CONTEXT,
    generated_tokens: int = LOCAL_LIVE_GENERATED_TOKENS,
) -> dict[str, Any]:
    """Run the full eval runner locally for the taxonomy_smoke job grid."""
    output_dir = Path(output_dir)
    jobs_dir = output_dir / "jobs"
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "taxonomy_smoke_local_live.log"

    model_cfg = load_model_config(model_config_path)
    model_path = PROJECT_ROOT / model_cfg["local_path"]
    if not (model_path / "config.json").exists():
        raise FileNotFoundError(f"Local checkpoint missing: {model_path}")

    _log(log_path, f"Loading {model_cfg.get('model_name')} from {model_path}")
    model_layer = ModelLayer(model_path=model_path)
    jobs = build_sweep_jobs(context_lengths=[context_length], preset=TAXONOMY_SMOKE_PRESET)

    from eval.runner import EvaluationRunner

    results = []
    payloads: list[dict[str, Any]] = []
    job_errors: list[dict[str, str]] = []

    for job in jobs:
        try:
            ensure_vanilla_attention(model_layer.model)
        except AttributeError:
            pass
        compressor = get_compressor(job.compressor, **job.get_compressor_kwargs())
        runner = EvaluationRunner(
            model_layer=model_layer,
            compressor=compressor,
            model_config=model_cfg,
        )
        _log(log_path, f"START {job.label} compressor={job.compressor} ctx={context_length}")
        try:
            result = runner.run(
                context_length,
                run_reasoning=True,
                run_kernel_cost=True,
                run_memory_bandwidth=True,
                generated_tokens=generated_tokens,
                include_baselines=True,
            )
        except Exception as exc:  # noqa: BLE001 — surface per-method failures
            err = {"label": job.label, "compressor": job.compressor, "error": str(exc)}
            job_errors.append(err)
            (jobs_dir / f"{job.result_stem}.error.json").write_text(
                json.dumps({**err, "traceback": traceback.format_exc()}, indent=2)
            )
            _log(log_path, f"ERROR {job.label}: {exc}")
            continue
        payload = result.to_dict()
        payload["label"] = job.label
        payload["status"] = "ok"
        payload["finished_at"] = datetime.now(UTC).isoformat()
        (jobs_dir / f"{job.result_stem}.json").write_text(json.dumps(payload, indent=2))
        results.append(result)
        payloads.append(payload)
        mem = (payload.get("fidelity") or {}).get("memory") or {}
        ppl = ((payload.get("behavior") or {}).get("task_quality") or {}).get("perplexity")
        _log(
            log_path,
            f"OK {job.label} ratio={mem.get('compression_ratio')} ppl={ppl}",
        )

    reporter = ResultReporter(output_dir)
    if results:
        reporter.save_json(results, "taxonomy_smoke_local_live")
        reporter.save_summary_csv(results, "taxonomy_smoke_local_live")
        reporter.print_summary(results)
        try:
            reporter.save_pareto(
                results,
                name="taxonomy_smoke_local_live_pareto",
                context_length=context_length,
                write_plot=False,
            )
        except Exception as exc:  # noqa: BLE001
            job_errors.append({"label": "pareto", "compressor": "n/a", "error": str(exc)})
        try:
            reporter.save_cross_dim(
                results,
                name="taxonomy_smoke_local_live_cross_dim",
                context_length=context_length,
                write_plot=False,
            )
        except Exception as exc:  # noqa: BLE001
            job_errors.append({"label": "cross_dim", "compressor": "n/a", "error": str(exc)})

    merged_json = merged_csv = None
    if payloads:
        merged_json, merged_csv = write_merged_reports(
            payloads, output_dir, "taxonomy_smoke_local_live_merged"
        )

    schema_errors = validate_bundle(
        payloads,
        require_smoke_extras=False,
        execution_platform=None,
        require_full_taxonomy=True,
    )
    for payload in payloads:
        system = payload.get("system") or {}
        behavior = payload.get("behavior") or {}
        if behavior.get("reasoning") is None:
            schema_errors.append(f"{payload.get('label')}: missing behavior.reasoning")
        if system.get("kernel_cost") is None:
            schema_errors.append(f"{payload.get('label')}: missing system.kernel_cost")
        if system.get("memory_bandwidth") is None:
            schema_errors.append(f"{payload.get('label')}: missing system.memory_bandwidth")

    report_md = output_dir / "TAXONOMY_SMOKE_LOCAL_LIVE.md"
    _write_markdown_report(report_md, payloads, schema_errors, job_errors, model_cfg)

    docs_md = output_dir / "RESULTS_COMPLETE.md"
    try:
        export_bundle_documentation(
            output_dir,
            docs_md,
            title="Local live taxonomy smoke — complete results",
            model_name=str(model_cfg.get("model_name") or "unknown"),
        )
    except Exception as exc:  # noqa: BLE001
        job_errors.append({"label": "documentation", "compressor": "n/a", "error": str(exc)})

    try:
        import csv

        from eval.cost.benchmark_dimensions import benchmark_dimensions_from_dict

        dim_rows = []
        for record in payloads:
            dims = benchmark_dimensions_from_dict(record)
            if dims is None:
                continue
            dim_rows.append(
                {
                    "compressor": record.get("compressor"),
                    "context_length": record.get("context_length"),
                    **dims.to_dict(),
                }
            )
        if dim_rows:
            dim_csv = output_dir / "taxonomy_smoke_local_live_benchmark_dimensions.csv"
            with dim_csv.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(dim_rows[0].keys()))
                writer.writeheader()
                writer.writerows(dim_rows)
    except Exception as exc:  # noqa: BLE001
        job_errors.append({"label": "benchmark_dimensions", "compressor": "n/a", "error": str(exc)})

    summary = {
        "model": model_cfg.get("model_name"),
        "model_path": str(model_path),
        "context_length": context_length,
        "generated_tokens": generated_tokens,
        "job_count": len(jobs),
        "ok_count": len(payloads),
        "methods_ok": [item.get("compressor") for item in payloads],
        "job_errors": job_errors,
        "schema_errors": schema_errors,
        "merged_json": str(merged_json) if merged_json else None,
        "merged_csv": str(merged_csv) if merged_csv else None,
        "report_md": str(report_md),
        "docs_md": str(docs_md),
        "log": str(log_path),
        "ok": not job_errors and not schema_errors and len(payloads) == len(jobs),
    }
    (output_dir / "taxonomy_smoke_local_live_summary.json").write_text(json.dumps(summary, indent=2))
    _log(log_path, f"DONE ok={summary['ok']} jobs={summary['ok_count']}/{summary['job_count']}")
    return summary


def _log(path: Path, message: str) -> None:
    line = f"{datetime.now(UTC).isoformat()} {message}"
    print(line)
    with path.open("a") as handle:
        handle.write(line + "\n")


def _write_markdown_report(
    path: Path,
    payloads: list[dict[str, Any]],
    schema_errors: list[str],
    job_errors: list[dict[str, str]],
    model_cfg: dict[str, Any],
) -> None:
    lines = [
        "# Local live taxonomy smoke",
        "",
        f"Model: **{model_cfg.get('model_name')}** (`{model_cfg.get('local_path')}`).",
        "",
        "This report is generated from a real `EvaluationRunner` pass (not dummy tensors).",
        "",
        "## Jobs",
        "",
        "| compressor | taxonomy | ratio | key RMSE | attn RMSE | PPL | tok/s | retrieval | IF | reasoning |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for payload in payloads:
        fid = payload.get("fidelity") or {}
        beh = payload.get("behavior") or {}
        sys = payload.get("system") or {}
        tax = payload.get("taxonomy") or {}
        mem = fid.get("memory") or {}
        rep = fid.get("representation") or {}
        attn = fid.get("attention") or {}
        tq = beh.get("task_quality") or {}
        lt = sys.get("latency_throughput") or {}
        retr = (beh.get("retrieval") or {}).get("exact_match_accuracy")
        instr = (beh.get("instruction_following") or {}).get("format_compliance_rate")
        reason = (beh.get("reasoning") or {}).get("exact_match_accuracy")
        lines.append(
            f"| {payload.get('compressor')} | {tax.get('primary')} | "
            f"{mem.get('compression_ratio')} | {rep.get('key_rmse')} | {attn.get('rmse')} | "
            f"{tq.get('perplexity')} | {lt.get('tokens_per_second')} | {retr} | {instr} | {reason} |"
        )
    lines.extend(["", "## Schema / math errors", ""])
    lines.extend([f"- {err}" for err in schema_errors] or ["_(none)_"])
    lines.extend(["", "## Job exceptions", ""])
    if job_errors:
        for item in job_errors:
            lines.append(f"- `{item['label']}` ({item['compressor']}): {item['error']}")
    else:
        lines.append("_(none)_")
    path.write_text("\n".join(lines) + "\n")
