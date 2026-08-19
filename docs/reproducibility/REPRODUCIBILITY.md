# Reproducibility Guide

How to reproduce **KV Cache Interception and Transformation Engine** evaluations from scratch — locally (smoke / dev) or on Modal (full Phase 5 sweeps).

Related: [SYSTEM_DESIGN.md](../architecture/SYSTEM_DESIGN.md) · [Qwen3-1.7B PHASE5_EVAL_RESULTS.md](../results/qwen3_1.7b/PHASE5_EVAL_RESULTS.md) · [OLMo2-1B PHASE5_EVAL_RESULTS.md](../results/olmo2_1b/PHASE5_EVAL_RESULTS.md)

---

## 1. What is fixed vs. variable

| Fixed (same for every method) | Variable (per compressor) |
|---|---|
| Model: Qwen3-1.7B, FP16, eager attention | `compressors/*` plug-in |
| Dataset: WikiText-2 test (`wikitext-2-raw-v1`) | Preset in `configs/modal_sweeps.yaml` |
| Eval code: `eval/runner.py`, FIDELITY + BEHAVIOR + SYSTEM | Bitwidth, stage, token budget, seed |
| Context lengths: 128, 256, 512 | |
| PPL stride: 512; FIDELITY attention window: 512 tokens | |
| Throughput: 64 generated tokens per run | Opt-in BEHAVIOR/SYSTEM sub-metrics (`--reasoning`, `--kernel-cost`, etc.); retrieval + instruction-following on by default |

Every method job runs **the same** `EvaluationRunner` path. Only the compressor and its kwargs change.

---

## 2. Environment

### Requirements

| Item | Value |
|---|---|
| Python | **3.11** (tested) |
| OS (local dev) | macOS (MPS) or Linux |
| GPU (full sweeps) | CUDA via [Modal](https://modal.com) A10G |
| Disk | ~6 GB (model + deps) |
| Secrets | Hugging Face read token (`HF_TOKEN`) |

### Setup

```bash
git clone https://github.com/faheemgurkani/kv-cache-compression-benchmark.git
cd kv-cache-compression-benchmark

python3.11 -m venv .venv && source .venv/bin/activate
pip install torch torchvision torchaudio
pip install -r requirements.txt

cp .env.example .env   # set HF_TOKEN=...
python scripts/download_model.py
python scripts/verify_kv_cache.py
pytest tests/ -q
```

**Record the git commit** when publishing numbers:

```bash
git rev-parse HEAD
```

Pin that SHA in papers / issue reports so others can check out the same code.

### Platform notes

| Platform | Limitation |
|---|---|
| macOS | `fast-hadamard-transform` often fails to build — skip it; TurboQuant uses scipy WHT fallback |
| Local CPU/MPS | Good for smoke tests; PPL/throughput differ from Modal CUDA |
| Modal | Recommended for Phase 5 numbers in [Qwen3-1.7B PHASE5_EVAL_RESULTS.md](../results/qwen3_1.7b/PHASE5_EVAL_RESULTS.md) |

---

## 3. Configuration (source of truth)

All sweep parameters live in version-controlled YAML — not CLI flags scattered across scripts.

| File | Controls |
|---|---|
| `configs/model.yaml` | Model path, `context_lengths`, TurboQuant defaults |
| `configs/eval.yaml` | Dataset, `perplexity_stride: 512`, `attention_fidelity_tokens: 512`, `generated_tokens: 64` |
| `configs/modal_sweeps.yaml` | Preset grid: labels, bitwidths, RocketKV budgets, QJL `seed: 42` |
| `configs/modal.yaml` | GPU type (A10G), volumes, timeout |

Edit these files to change the experimental grid; re-run sweeps with `--no-resume` (Modal) to avoid stale volume cache.

---

## 4. Determinism and seeds

| Component | Seed / determinism |
|---|---|
| QJL projection matrix | `seed=42` (+ `head_dim` offset) in `quantizers/qjl.py` |
| TurboQuant Lloyd-Max centroids | `seed=42` in `quantizers/lloyd_max.py` |
| WikiText-2 sample | Deterministic concatenation in `data/loader.py` (fixed split, fixed target length) |
| Perplexity | Sequential sliding-window — **not** batched (batching would change the metric) |
| Attention | `attn_implementation="eager"` required |

**Expect small run-to-run variance** on GPU (floating-point order, CUDA kernels). PPL and throughput may differ slightly between machines; trends and order-of-magnitude should match.

Modal uses **scipy WHT** (no `fast-hadamard-transform` in the image). Local CUDA with FHT may differ slightly from Modal TurboQuant numbers.

---

## 5. Evaluation protocol

Three branches instead of an offline/online split — FIDELITY always runs; BEHAVIOR defaults to perplexity + retrieval + instruction following; SYSTEM defaults to latency/throughput. Reasoning and extra SYSTEM metrics are opt-in (`--reasoning`, `--peak-memory`, `--memory-bandwidth`, `--kernel-cost`, `--gpu-utilization`). Disable default BEHAVIOR tasks with `--skip-retrieval` / `--skip-instruction-following`.

### FIDELITY — did the transformation preserve the KV representation and attention behavior?

Single forward pass on a fixed WikiText-2 window:

- Key / value tensor RMSE, relative reconstruction error, cosine similarity (after compress → decompress)
- Attention score MSE/RMSE/cosine/max-error (method-specific: QJL uses estimator; RocketKV uses post-selection kept tokens), plus attention-output RMSE and attention-distribution KL divergence where the compressor exposes raw quantized scores
- Memory: uncompressed vs compressed bytes, compression ratio, effective bits/KV, metadata overhead

Window capped at `attention_fidelity_tokens` (512) for long contexts to avoid OOM.

### BEHAVIOR — does the model still behave correctly after KV transformation?

Autoregressive loop through `KVCacheEngine`:

1. Baseline PPL runs **first** (before RocketKV / QJL attention patches)
2. Compressed KV updated incrementally each decode step
3. Sliding-window perplexity (`perplexity_stride: 512`) — on by default
4. Retrieval (needle-in-haystack) and instruction-following (format compliance) — **on by default**; reasoning (synthetic arithmetic) — opt-in (`--reasoning`)

Modal jobs set `include_baselines=True` so each JSON carries `perplexity_baseline` for that context length. Phase 5 tables use the **shared identity baseline** from preset `baseline` (run once); per-job baselines should match within noise.

### SYSTEM — does the compression actually make inference better?

Also through `KVCacheEngine`:

1. TTFT, inter-token latency (mean/p50/p99), decode/end-to-end latency, tokens/sec — on by default (64 new tokens through compressed path)
2. Peak VRAM, memory bandwidth (analytical GB/s), kernel cost (compress/decompress vs. rest of forward pass), GPU utilization (CUDA + `pynvml` only) — opt-in

A higher FIDELITY compression ratio can still lose on SYSTEM if it adds enough per-step compute — this is why SYSTEM is a separate branch rather than folded into FIDELITY.

---

## 6. Local reproduction (single job)

Run from repo root with venv active.

```bash
# Identity baseline @ ctx=512
python scripts/run_eval.py --compressor identity --context-length 512

# TurboQuant 4-bit full pipeline
python scripts/run_eval.py --compressor turboquant --stage full --bitwidth 4 --context-length 512

# QJL (seed 42 via compressor default)
python scripts/run_eval.py --compressor qjl --context-length 512

# RocketKV (defaults: token_budget=512; match modal_sweeps.yaml for full grid)
python scripts/run_eval.py --compressor rocketkv --context-length 512

# Opt-in BEHAVIOR/SYSTEM sub-metrics (each adds its own generate() pass)
python scripts/run_eval.py --compressor turboquant --reasoning
python scripts/run_eval.py --compressor turboquant --skip-retrieval --skip-instruction-following
python scripts/run_eval.py --compressor turboquant --peak-memory --memory-bandwidth --kernel-cost --gpu-utilization
```

For non-default RocketKV budgets (`r256`, `r1024`), kwargs are not exposed on `run_eval.py` CLI — use the Modal preset or Python:

```python
from compressors.registry import get_compressor
from eval.runner import EvaluationRunner

compressor = get_compressor("rocketkv", token_budget=256, hsa_budget=256, window_size=32)
runner = EvaluationRunner(compressor=compressor)
result = runner.run(context_length=512)
print(result.perplexity)
```

Outputs: `results/eval_results.json` and `results/eval_results.csv` (stem from `--output`).

### Sanity-check baselines (Modal reference)

| ctx | identity PPL | tok/s (ref) |
|---:|---:|---:|
| 128 | ~14.21 | ~23.7 |
| 256 | ~17.66 | ~17.7 |
| 512 | ~14.11 | ~13.9 |

If local identity PPL is orders of magnitude off, check model download, eager attention, and that you run from repo root.

---

## 7. Modal reproduction (full Phase 5 sweeps)

### One-time setup

```bash
pip install modal
modal token new                    # authenticate
cp .env.example .env && # set HF_TOKEN
bash scripts/modal_setup_model.sh  # Qwen3-1.7B → Modal volume kv-engine-qwen3
```

Create Modal secret `huggingface-secret` with key `HF_TOKEN` (see the Modal summary table in §11 below).

### Full sweep order

Run **baseline once**, then method presets. Use `--no-resume` for a clean re-sweep.

```bash
# 1. Shared identity baseline (3 jobs)
bash scripts/modal_run_sweep_baseline.sh

# 2. Method sweeps (27 jobs total)
bash scripts/modal_run_sweep.sh              # turboquant: 12
bash scripts/modal_run_sweep_qjl.sh          # qjl: 3
bash scripts/modal_run_sweep_rocketkv.sh     # rocketkv: 9
```

Equivalent explicit commands:

```bash
NO_RESUME=1 bash scripts/modal_run_sweep_qjl.sh
NO_RESUME=1 bash scripts/modal_run_sweep_rocketkv.sh
```

Detached launches return immediately; workers run on Modal GPUs (~15–90 min per preset, longest job dominates).

### Smoke test before a full sweep

```bash
bash scripts/modal_smoke_eval.sh qjl          # 1 job @ ctx=128
bash scripts/modal_smoke_eval.sh rocketkv     # rocketkv_r512 @ ctx=128
bash scripts/modal_smoke_eval.sh turboquant   # tq_full_b4 @ ctx=128
```

### Fetch and merge results

```bash
bash scripts/modal_fetch_results.sh           # → results/modal_volume/

# Merge into versioned bundles (recommended)
python scripts/restructure_modal_results.py
```

Or merge a single preset manually:

```bash
modal run modal_app/sweep.py::merge_local \
  --input-dir results/modal_volume \
  --output phase5_modal_qjl \
  --label-prefixes qjl_default
```

### Result file naming

| Preset | JSON stem pattern |
|---|---|
| baseline / turboquant / qjl | `{label}_ctx{len}_b{bw}_{stage}` |
| rocketkv | `{label}_ctx{len}_b{budget}_hsa{hsa}_ws{window}` |

Example: `rocketkv_r256_ctx512_b256_hsa256_ws32.json`

### Resume vs fresh run

| Flag | Behavior |
|---|---|
| default (`--resume`) | Skips jobs whose `.json` already exists on Modal volume |
| `--no-resume` | Submits all grid jobs again (overwrites on completion) |

After implementation fixes, always use `--no-resume` so stale volume entries are replaced.

### Sync mode (wait + local merge)

```bash
modal run modal_app/sweep.py::main \
  --preset qjl \
  --context-lengths 128,256,512 \
  --no-resume \
  --sync \
  --output phase5_modal_qjl
```

Blocks until all jobs finish; writes merged CSV/JSON under `results/`.

---

## 8. Result artifacts

```
results/
  phase5_modal_baseline/          # shared identity (3 jobs)
  phase5_modal_sweep_128_256_512/ # turboquant (12 jobs)
  phase5_modal_qjl/               # qjl (3 jobs)
  phase5_modal_rocketkv/          # rocketkv (9 jobs)
    jobs/                         # per-job JSON payloads
    manifest.json                 # sweep metadata
    phase5_modal_*_{timestamp}.csv
```

Each job JSON includes:

- `fidelity` — representation (RMSE/relative-error/cosine), attention (score + output RMSE, KL divergence), memory
- `behavior` — `task_quality.perplexity`, `task_quality.perplexity_baseline`, and (if enabled) `retrieval`/`instruction_following`/`reasoning`
- `system` — `latency_throughput` (TTFT, ITL, tok/s, end-to-end latency), and (if enabled) `peak_memory`/`memory_bandwidth`/`kernel_cost`/`gpu_utilization`
- `job` — full compressor kwargs (bitwidth, budgets, seed)
- `started_at` / `finished_at` — UTC timestamps

Older bundles (before the FIDELITY/BEHAVIOR/SYSTEM redesign) instead use `section_a_fidelity` / `section_b_inference` keys; `modal_app/merge.py` and `scripts/export_results_documentation.py` read both shapes so historical archives still merge and regenerate docs correctly.

Published tables: [Qwen3-1.7B PHASE5_EVAL_RESULTS.md](../results/qwen3_1.7b/PHASE5_EVAL_RESULTS.md) · [OLMo2-1B PHASE5_EVAL_RESULTS.md](../results/olmo2_1b/PHASE5_EVAL_RESULTS.md). Raw bundles are gitignored; regenerate via Modal steps above.

---

## 9. Verification checklist

Before trusting new numbers:

```bash
# Unit tests (compressor + online paths)
pytest tests/test_qjl.py tests/test_qjl_online.py -q
pytest tests/test_rocketkv.py tests/test_rocketkv_online.py -q

# Modal smoke (1 GPU job)
bash scripts/modal_smoke_eval.sh qjl
bash scripts/modal_smoke_eval.sh rocketkv

# Full sweep + merge
NO_RESUME=1 bash scripts/modal_run_sweep_qjl.sh
bash scripts/modal_fetch_results.sh
python scripts/restructure_modal_results.py
```

Compare merged CSV to the relevant `PHASE5_EVAL_RESULTS.md`. Identity baseline PPL at each context length should match within ~1%.

---

## 10. Known non-reproducibility sources

Documented in [CURRENT_STATE.md](../methodology/CURRENT_STATE.md):

- Single model family per published sweep / dataset / ctx ≤512 per run — not a multi-benchmark study yet
- TurboQuant online speed ~0.08 tok/s @ ctx=512 (implementation overhead)
- QJL / RocketKV PPL catastrophic under this pipeline on some configs — faithful implementation, not paper-matched quality
- FIDELITY metrics do not always predict BEHAVIOR/PPL (by design)

---

## 11. Modal GPU evaluation (infrastructure reference)

CUDA sweeps on [Modal](https://modal.com). Same eval code as local; only device and orchestration differ. This section was previously its own `MODAL_GPU_EVAL_DESIGN.md` file — merged here since its runbook was pure duplication of §7 above; only the infra-specific reference material is kept.

### Summary

| Item | Value |
|---|---|
| GPU | A10G (24 GB) per job; fallbacks L4, any |
| Parallelism | One job = one GPU via `eval_worker.spawn_map()` (up to ~30) |
| Within-job PPL | Sequential (by design) |
| Model volume | `kv-engine-qwen3` → `/models/qwen3_1.7b/` (per-model volumes for other model configs, see `configs/modal_qwen3.yaml`) |
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

### Layout

```text
modal_app/
  image.py, settings.py, worker.py, sweep.py, job_spec.py, merge.py
scripts/
  modal_setup_model.sh, modal_run_sweep*.sh, modal_smoke_eval.sh, modal_fetch_results.sh
configs/modal.yaml, configs/modal_sweeps.yaml
```

Workers mount repo at `/root/kv-cache-engine`; configs resolved via `KV_PROJECT_ROOT` in `modal_app/settings.py`.

### Config highlights

`configs/modal.yaml` — GPU, volumes, secrets, timeout.

`configs/eval.yaml` — `perplexity_stride: 512`, `attention_fidelity_tokens: 512` (FIDELITY attention window for long ctx).

### Limits

- Online PPL must stay sequential — batched forwards would change the metric.
- FIDELITY uses windowed QK fidelity (512 tokens) to avoid OOM at long ctx on A10G.
- Eager attention required (same as local).

### References

- [Modal GPU](https://modal.com/docs/guide/gpu) · [spawn_map](https://modal.com/docs/guide/scale) · [Volumes](https://modal.com/docs/guide/volumes)

---

## 12. Documentation index

| Document | Purpose |
|---|---|
| [METHODOLOGY.md](../methodology/METHODOLOGY.md) | System design + compression + eval protocol |
| [MATHEMATICS_AND_ALGORITHMS.md](../methodology/MATHEMATICS_AND_ALGORITHMS.md) | Equations and pseudocode |
| [Qwen3-1.7B RESULTS_COMPLETE.md](../results/qwen3_1.7b/RESULTS_COMPLETE.md) | Every Phase 5 metric, per-layer stats, logs |
| [Qwen3-1.7B PHASE5_EVAL_RESULTS.md](../results/qwen3_1.7b/PHASE5_EVAL_RESULTS.md) | Summary tables for papers/README |
| [OLMo2-1B RESULTS_COMPLETE.md](../results/olmo2_1b/RESULTS_COMPLETE.md) | Every Phase 5 metric for the OLMo2 replication |
| [OLMo2-1B PHASE5_EVAL_RESULTS.md](../results/olmo2_1b/PHASE5_EVAL_RESULTS.md) | OLMo2 summary tables |
| [shortlist_5model_eval/](../results/shortlist_5model_eval/) | Evaluation-framework correspondence for the 5-model architecture-matrix shortlist (MHA/GQA/MQA/MLA/Hybrid) |

Regenerate complete results: `python scripts/export_results_documentation.py`
