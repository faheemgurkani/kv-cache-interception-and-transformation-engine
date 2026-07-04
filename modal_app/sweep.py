"""Launch parallel TurboQuant evaluation sweeps on Modal."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import modal

from modal_app.job_spec import build_sweep_jobs
from modal_app.worker import app, eval_worker


@app.local_entrypoint()
def main(
    context_lengths: str = "128,512,4096,8192,16384,32768",
    detach: bool = False,
    output: str = "phase5_modal_sweep",
):
    lengths = [int(x.strip()) for x in context_lengths.split(",") if x.strip()]
    jobs = [job.to_dict() for job in build_sweep_jobs(lengths)]

    if detach:
        handles = list(eval_worker.spawn_map(jobs))
        print(f"Spawned {len(handles)} jobs.")
        for handle in handles:
            print(handle.object_id)
        return

    results = list(eval_worker.map(jobs))
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"timestamp": stamp, "results": results}
    json_path = out_dir / f"{output}_{stamp}.json"
    json_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote {json_path} ({len(results)} jobs)")
