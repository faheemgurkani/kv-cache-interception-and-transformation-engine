# Modal GPU Evaluation

CUDA sweeps for the **KV-Cache Interception and Transformation Engine** on [Modal](https://modal.com). Same eval code as local; only device and orchestration differ.

Overview: [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md) · Results: [PHASE5_EVAL_RESULTS.md](PHASE5_EVAL_RESULTS.md) · Reproducibility: [REPRODUCIBILITY.md](REPRODUCIBILITY.md)

## Summary

| Item | Value |
|---|---|
| GPU | A10G (24 GB) per job; fallbacks L4, any |
| Parallelism | One job = one GPU via `eval_worker.spawn_map()` (up to ~30) |
| Within-job PPL | Sequential (by design) |
| Model volume | `kv-engine-qwen3` → `/models/qwen3_1.7b/` |
| Results volume | `kv-engine-results` → `/results/{stem}.json` |
| Secret | `huggingface-secret` (`HF_TOKEN`) |
| Timeout | 4 h/job (`configs/modal.yaml`) |

```text
Local: modal run --detach modal_app/sweep.py::main
         │
         ▼
   spawn_map → N × eval_worker @ A10G
         │
         ▼
   eval/runner.py (same as local) → JSON on volume
```

**Not in image:** `fast-hadamard-transform` — scipy WHT fallback on CUDA.

## Sweep presets

`configs/modal_sweeps.yaml` — select with `--preset`:

| Preset | Jobs (× ctx 128, 256, 512) |
|---|---|
| `baseline` | 3 |
| `turboquant` | 12 |
| `qjl` | 3 |
| `rocketkv` | 9 |

Identity baseline runs once under `baseline`; method sweeps do not re-run identity.

Result stems: TurboQuant/QJL `{label}_ctx{len}_b{bw}_{stage}.json` · RocketKV `{label}_ctx{len}_b{budget}_hsa{hsa}_ws{win}.json` (e.g. `rocketkv_r256_ctx512_b256_hsa256_ws32.json`)

## Runbook

```bash
pip install modal
bash scripts/modal_setup_model.sh

bash scripts/modal_run_sweep_baseline.sh
bash scripts/modal_run_sweep.sh              # turboquant
bash scripts/modal_run_sweep_qjl.sh
bash scripts/modal_run_sweep_rocketkv.sh

bash scripts/modal_smoke_eval.sh qjl         # single job @ ctx=128
bash scripts/modal_fetch_results.sh
python scripts/restructure_modal_results.py   # bundles under results/phase5_modal_*/

# Or merge one preset manually:
modal run modal_app/sweep.py::merge_local \
  --input-dir results/modal_volume \
  --output phase5_modal_qjl \
  --label-prefixes qjl_default
```

Resume: re-run with default `--resume` (skips successful `.json` on volume). Fresh run: `NO_RESUME=1` or `--no-resume` after code changes.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for seeds, config files, verification checklist, and sanity-check baselines.

Wall-clock: ~15–90 min per preset (longest job dominates).

## Layout

```text
modal_app/
  image.py, settings.py, worker.py, sweep.py, job_spec.py, merge.py
scripts/
  modal_setup_model.sh, modal_run_sweep*.sh, modal_smoke_eval.sh, modal_fetch_results.sh
configs/modal.yaml, configs/modal_sweeps.yaml
```

Workers mount repo at `/root/kv-cache-engine`; configs resolved via `KV_PROJECT_ROOT` in `modal_app/settings.py`.

## Config highlights

`configs/modal.yaml` — GPU, volumes, secrets, timeout.

`configs/eval.yaml` — `perplexity_stride: 512`, `attention_fidelity_tokens: 512` (Section A window for long ctx).

## Limits

- Online PPL must stay sequential — batched forwards would change the metric.
- Section A uses windowed QK fidelity (512 tokens) to avoid OOM at long ctx on A10G.
- Eager attention required (same as local).

## References

- [Modal GPU](https://modal.com/docs/guide/gpu) · [spawn_map](https://modal.com/docs/guide/scale) · [Volumes](https://modal.com/docs/guide/volumes)
