"""
Launch parallel TurboQuant evaluation sweeps on Modal NVIDIA GPUs.

Usage:
  # One-time model download to Modal Volume (~3.2 GB)
  modal run modal_app/worker.py::ensure_model

  # Full parallel sweep (30 jobs: 5 configs × 6 context lengths)
  modal run --detach modal_app/sweep.py

  # Subset sweep
  modal run --detach modal_app/sweep.py --context-lengths 128,512 --labels tq_full_b4

  # Sync run — waits and writes merged JSON/CSV locally
  modal run modal_app/sweep.py --sync

  # Merge payloads already downloaded from the results volume
  modal run modal_app/sweep.py::merge_local --input-dir results/modal_volume
"""

from __future__ import annotations

from pathlib import Path

from modal_app.job_spec import build_sweep_jobs, filter_existing_jobs
from modal_app.merge import load_payloads_from_directory, write_merged_reports
from modal_app.worker import app, eval_worker, list_completed_jobs


@app.local_entrypoint()
def main(
    context_lengths: str = "128,512,4096,8192,16384,32768",
    labels: str = "",
    sync: bool = False,
    resume: bool = True,
    skip_perplexity: bool = False,
    skip_throughput: bool = False,
    output: str = "phase5_modal_sweep",
):
    lengths = [int(item.strip()) for item in context_lengths.split(",") if item.strip()]
    label_filter = [item.strip() for item in labels.split(",") if item.strip()] or None

    jobs = build_sweep_jobs(
        context_lengths=lengths,
        labels=label_filter,
        skip_perplexity=skip_perplexity,
        skip_throughput=skip_throughput,
    )

    if resume:
        completed = set(list_completed_jobs.remote())
        jobs = filter_existing_jobs(jobs, completed)
        if not jobs:
            print("All jobs already complete on Modal results volume.")
            return

    job_dicts = [job.to_dict() for job in jobs]
    print(f"Submitting {len(job_dicts)} eval jobs to Modal ({'sync' if sync else 'spawn'} mode).")

    if sync:
        results = list(eval_worker.map(job_dicts))
        ok = [item for item in results if item.get("status") == "ok"]
        errors = [item for item in results if item.get("status") == "error"]
        json_path, csv_path = write_merged_reports(ok, Path("results"), output)
        print(f"Merged {len(ok)} ok / {len(errors)} error jobs.")
        print(f"JSON: {json_path}")
        print(f"CSV:  {csv_path}")
        if errors:
            for item in errors:
                print(f"  ERROR {item.get('label')} ctx={item['job']['context_length']}: {item.get('error')}")
        return

    eval_worker.spawn_map(job_dicts)
    print(f"Spawned {len(job_dicts)} jobs on Modal GPUs.")
    print("Results persist to volume: kv-engine-results")
    print("Fetch locally: bash scripts/modal_fetch_results.sh")
    print("Or re-run with --sync after jobs finish to merge locally.")


@app.local_entrypoint()
def merge_local(
    input_dir: str = "results/modal_volume",
    output: str = "phase5_modal_sweep",
):
    directory = Path(input_dir)
    payloads = load_payloads_from_directory(directory)
    if not payloads:
        raise SystemExit(f"No job JSON payloads found under {directory}")
    json_path, csv_path = write_merged_reports(payloads, Path("results"), output)
    print(f"Merged {len(payloads)} payloads.")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
