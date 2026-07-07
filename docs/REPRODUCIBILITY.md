# Reproducibility Guide

How to reproduce the **KV-Cache Interception and Transformation Engine** evaluations from scratch — locally (smoke / dev) or on Modal (full Phase 5 sweeps).

Related: [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md) · [PHASE5_EVAL_RESULTS.md](PHASE5_EVAL_RESULTS.md) · [MODAL_GPU_EVAL_DESIGN.md](MODAL_GPU_EVAL_DESIGN.md)

---

## 1. What is fixed vs. variable

| Fixed (same for every method) | Variable (per compressor) |
|---|---|
| Model: Qwen3-1.7B, FP16, eager attention | `compressors/*` plug-in |
| Dataset: WikiText-2 test (`wikitext-2-raw-v1`) | Preset in `configs/modal_sweeps.yaml` |
| Eval code: `eval/runner.py`, Section A + B | Bitwidth, stage, token budget, seed |
| Context lengths: 128, 256, 512 | |
| PPL stride: 512; Section A window: 512 tokens | |
| Throughput: 64 generated tokens per run | |

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
| Modal | Recommended for Phase 5 numbers in [PHASE5_EVAL_RESULTS.md](PHASE5_EVAL_RESULTS.md) |

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

### Section A — offline fidelity

Single forward pass on a fixed WikiText-2 window:

- Key / value tensor RMSE (after compress → decompress)
- Attention score error (method-specific: QJL uses estimator; RocketKV uses post-selection kept tokens)
- Memory: uncompressed vs compressed bytes, compression ratio, effective bits/KV

Window capped at `attention_fidelity_tokens` (512) for long contexts to avoid OOM.

### Section B — online inference

Autoregressive loop through `KVCacheEngine`:

1. Baseline PPL runs **first** (before RocketKV / QJL attention patches)
2. Compressed KV updated incrementally each decode step
3. Sliding-window perplexity (`perplexity_stride: 512`)
4. Throughput: 64 new tokens through compressed path

Modal jobs set `include_baselines=True` so each JSON carries `perplexity_baseline` for that context length. Phase 5 tables use the **shared identity baseline** from preset `baseline` (run once); per-job baselines should match within noise.

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

Create Modal secret `huggingface-secret` with key `HF_TOKEN` (see [MODAL_GPU_EVAL_DESIGN.md](MODAL_GPU_EVAL_DESIGN.md)).

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

- `section_a_fidelity` — RMSE, attention, memory
- `section_b_inference` — `perplexity`, `perplexity_baseline`, throughput
- `job` — full compressor kwargs (bitwidth, budgets, seed)
- `started_at` / `finished_at` — UTC timestamps

Published tables: [PHASE5_EVAL_RESULTS.md](PHASE5_EVAL_RESULTS.md). Raw bundles are gitignored; regenerate via Modal steps above.

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

Compare merged CSV to [PHASE5_EVAL_RESULTS.md](PHASE5_EVAL_RESULTS.md). Identity baseline PPL at each context length should match within ~1%.

---

## 10. Known non-reproducibility sources

Documented in [CURRENT_STATE.md](CURRENT_STATE.md):

- Single model / dataset / ctx ≤512 — not a multi-benchmark study yet
- TurboQuant online speed ~0.08 tok/s @ ctx=512 (implementation overhead)
- QJL / RocketKV PPL catastrophic on Qwen3-1.7B under this pipeline — faithful implementation, not paper-matched quality
- Section A metrics do not always predict Section B PPL (by design)

## 10. Documentation index

| Document | Purpose |
|---|---|
| [METHODOLOGY.md](METHODOLOGY.md) | System design + compression + eval protocol |
| [MATHEMATICS_AND_ALGORITHMS.md](MATHEMATICS_AND_ALGORITHMS.md) | Equations and pseudocode |
| [RESULTS_COMPLETE.md](RESULTS_COMPLETE.md) | Every Phase 5 metric, per-layer stats, logs |
| [PHASE5_EVAL_RESULTS.md](PHASE5_EVAL_RESULTS.md) | Summary tables for papers/README |

Regenerate complete results: `python scripts/export_results_documentation.py`
