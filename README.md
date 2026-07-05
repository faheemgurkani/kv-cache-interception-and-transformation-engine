# KV-Cache Interception + Transformation Engine

A **KV-cache interception + transformation engine** inside an LLM forward pass, with a fixed evaluation stack for comparing compression methods.

Built on **Qwen3-1.7B** ([Hugging Face](https://huggingface.co/Qwen/Qwen3-1.7B)) with PyTorch and HuggingFace Transformers, optimized for Apple Silicon (MPS).

---

## Overview

This repository is **not** a TurboQuant script or a single-paper reimplementation. It provides a **KV-cache interception + transformation engine** that sits inside the model forward loop, with pluggable compressors and a shared evaluation pipeline.

```text
Tokenizer → Model Forward → KV Cache → (intercept here) → Attention → Next tokens
                                              │
                                    KVCompressor (plug-in)
                                              │
                         TurboQuant | KIVI | QJL | RocketKV | identity
```

| Component                | Role                                           | Changes per paper? |
| ------------------------ | ---------------------------------------------- | ------------------ |
| `framework/kv_engine.py` | Intercepts `past_key_values` between steps     | No                 |
| `compressors/*`          | `KVCompressor` plug-ins (transform KV tensors) | **Yes**            |
| `eval/` + `reporting/`   | Fixed metrics (memory, speed, perplexity)      | No                 |

**TurboQuant** is one implementation of `KVCompressor`, not the project itself. KIVI, QJL, and RocketKV plug into the same engine and eval pipeline without touching model or metric code.

See [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md) for the full architecture.

### Documentation

| Document | Description |
|---|---|
| [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md) | Architecture, plug-in model, compressor design |
| [docs/MODAL_GPU_EVAL_DESIGN.md](docs/MODAL_GPU_EVAL_DESIGN.md) | Modal GPU runtime, parallelism, runbook |
| [docs/PHASE5_EVAL_RESULTS.md](docs/PHASE5_EVAL_RESULTS.md) | Phase 5 sweep results and findings |
| [docs/CURRENT_STATE.md](docs/CURRENT_STATE.md) | Compressor readiness, setup snapshot, known limits |

Raw job JSON and CSV exports live under `results/` (gitignored). Fetch with `bash scripts/modal_fetch_results.sh`.

**Methods supported (compression layer):** TurboQuant → KIVI → QJL → RocketKV

---

## Repository Contents

The repository ships **code, configs, and scripts only**. Large artifacts are downloaded locally and are **gitignored**:

| Artifact            | Location                       | Approx. size | How to obtain                      |
| ------------------- | ------------------------------ | -----------: | ---------------------------------- |
| Model weights       | `models/qwen3_1.7b/`           |      ~3.2 GB | `python scripts/download_model.py` |
| WikiText-2 cache    | `.cache/huggingface/datasets/` |       ~10 MB | auto on first eval / test run      |
| Virtual environment | `.venv/`                       |        ~2 GB | `pip install -r requirements.txt`  |
| Experiment results  | `results/`, `plots/`           |       varies | produced by eval scripts           |

Do not commit `.env` (contains HuggingFace tokens).

---

## Requirements

| Requirement         | Version / notes                        |
| ------------------- | -------------------------------------- |
| Python              | 3.11                                   |
| OS                  | macOS (Apple Silicon MPS) or Linux/CPU |
| Disk space          | ~6 GB free (venv + model + cache)      |
| HuggingFace account | Token with model read access           |

> **Apple Silicon note:** `fast-hadamard-transform` requires CUDA/nvcc and does **not** build on MPS. Core eval works without it; TurboQuant Hadamard steps need a CUDA machine or a fallback (Phase 1).

---

## Quick Start

Follow these steps in order on a fresh clone.

### 1. Clone and create the virtual environment

```bash
git clone https://github.com/<your-username>/<repository>.git
cd <repository>

python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
```

### 2. Install dependencies

Install PyTorch first (required for some native extensions), then the rest:

```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

If `fast-hadamard-transform` fails on Mac, skip it for now — the identity baseline and evaluation framework still work:

```bash
pip install -r requirements.txt --ignore-installed fast-hadamard-transform 2>/dev/null || \
  grep -v fast-hadamard-transform requirements.txt | pip install -r /dev/stdin
```

**Conda alternative:**

```bash
conda env create -f environment.yml
conda activate kv-cache-engine
```

### 3. Configure HuggingFace authentication

Copy the example env file and add a token ([create one here](https://huggingface.co/settings/tokens)):

```bash
cp .env.example .env
# Edit .env:
#   HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

`HF_TOKEN` is read automatically by `huggingface_hub` and Transformers. Do not commit `.env`.

### 4. Download the model

```bash
python scripts/download_model.py
```

Saves `Qwen/Qwen3-1.7B` to `models/qwen3_1.7b/` (~3.2 GB).

### 5. Verify KV-cache access

```bash
python scripts/verify_kv_cache.py
```

Expected output (shapes vary slightly by batch/seq):

```text
Using device: mps   # or cpu
Key shape:   (1, 8, 2, 128)
Value shape: (1, 8, 2, 128)
KV cache access verified.
```

### 6. Run tests

```bash
pytest tests/ -v
```

All 43 tests should pass (memory accounting, attention fidelity, online inference, incremental KV cache, compressor, TurboQuant quality, QJL, RocketKV, WikiText-2 loader, KV-cache shapes, eval runner).

### 7. Run an evaluation

```bash
# Quick smoke test (~512 tokens)
python scripts/run_eval.py --compressor identity --context-length 512

# Single baseline with JSON output
python scripts/run_baseline.py --baseline identity --context-length 512

# Full context-length sweep (128, 256, 512)
python scripts/run_eval.py --compressor identity --all-context-lengths
```

Results are written to `results/` as `.json` and `.csv`.

---

## Configuration Reference

### `configs/model.yaml`

| Key               | Description                  | Default             |
| ----------------- | ---------------------------- | ------------------- |
| `model_name`      | HuggingFace model ID         | `Qwen/Qwen3-1.7B`   |
| `local_path`      | Where weights are saved      | `models/qwen3_1.7b` |
| `context_lengths` | Eval sweep (local + Modal)   | `128, 256, 512`     |
| `bitwidths`       | Target compression bitwidths | `2, 3, 4`           |

### `configs/eval.yaml`

| Key                      | Description                | Default               |
| ------------------------ | -------------------------- | --------------------- |
| `wikitext.name`          | HF dataset ID              | `Salesforce/wikitext` |
| `wikitext.config`        | Dataset config             | `wikitext-2-raw-v1`   |
| `wikitext.split`         | Eval split                 | `test`                |
| `perplexity_stride`      | Sliding-window stride      | `512`                 |
| `generated_tokens`       | Tokens for throughput test | `64`                  |
| `default_context_length` | Default `--context-length` | `512`                 |

WikiText-2 documents are short; the framework **concatenates samples** until the target context length is reached (standard practice for long-context KV-cache eval).

---

## Architecture

Four fixed layers plus one pluggable compression layer. **Only `compressors/` changes between papers.**

```text
Tokenizer
    │
Model Forward                    ← framework/model.py (fixed)
    │
KV Cache (past_key_values)
    │
KVCacheEngine                    ← framework/kv_engine.py (fixed) — intercept here
    │
KVCompressor (plug-in)           ← compressors/ (variable)
    │
Attention → Next tokens
    │
Evaluation + Reporting           ← eval/ + reporting/ (fixed)
```

**Full system design:** [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md) (eager attention rationale, interception flow, TurboQuant math, execution order)

**QJL + RocketKV implementation:** [docs/QJL_AND_ROCKETKV.md](docs/QJL_AND_ROCKETKV.md)

| Layer                 | Directory                                         |
| --------------------- | ------------------------------------------------- |
| Model                 | `framework/model.py`                              |
| KV interception       | `framework/kv_engine.py`, `framework/kv_cache.py` |
| Compression (plug-in) | `compressors/`, `quantizers/`                     |
| Evaluation            | `eval/`                                           |
| Reporting             | `reporting/`                                      |

### Compressors

All methods implement `KVCompressor`:

```python
class KVCompressor:
    def compress_kv(self, tensor, layer, mode): ...  # single K or V
    def compress(self, key, value, layer): ...       # full layer
    def decompress(self, compressed): ...
```

| Name         | Status     | Paper pipeline                             |
| ------------ | ---------- | ------------------------------------------ |
| `identity`   | ✅ working | no compression (baseline)                  |
| `turboquant` | ✅ Phase 1 | WHT → Lloyd-Max → QJL residual             |
| `kivi`       | 🔜 Phase 2 | asymmetric INT2                            |
| `qjl`        | ✅ Phase 3 | random projection → 1-bit sign (keys only) |
| `rocketkv`   | ✅ Phase 4 | token selection → eviction                 |

### TurboQuant (summary)

The compression layer is **distinct** from model and eval — see [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md) for math and design rationale.

```text
quantizers/                 # WHT, Lloyd-Max, QJL primitives
compressors/turboquant.py   # TurboQuantCompressor plug-in
framework/kv_engine.py      # KVCacheEngine interception
```

Pipeline: `pad → WHT → ÷√D → Lloyd-Max → residual → QJL → store`

```bash
python scripts/validate_turboquant.py --phase stages
python scripts/validate_turboquant.py --phase intercept
python scripts/run_eval.py --compressor turboquant --stage full --context-length 512
```

The model loads with `attn_implementation="eager"` — required because FlashAttention hides KV internals (see system design doc).

### QJL (summary)

Standalone QJL compresses **keys only** via random Gaussian projection and 1-bit sign quantization. Values are stored uncompressed. The goal is inner-product preservation, not vector reconstruction.

```text
quantizers/qjl_pipeline.py   → QJLPipeline, QJLTensorPayload
compressors/qjl.py           → QJLCompressor plug-in
```

Pipeline: `k → sign(S @ k) + ||k||` — no WHT, Lloyd-Max, or centroids.

Attention uses the asymmetric estimator `estimate_attention_scores()` (see [docs/QJL_AND_ROCKETKV.md](docs/QJL_AND_ROCKETKV.md)). The engine still calls `decompress_kv()` for online inference, which returns an approximate key reconstruction.

```bash
pytest tests/test_qjl.py -v
python scripts/run_eval.py --compressor qjl --context-length 512
```

### RocketKV (summary)

RocketKV **drops tokens** instead of quantizing vectors. Stage 1 permanently filters tokens via a SnapKV-inspired `TokenSelector`; Stage 2 selects dynamic top-k tokens via `HybridSparseAttention` at decode time.

```text
quantizers/rocketkv.py       → TokenSelector, HybridSparseAttention
compressors/rocketkv.py      → RocketKVCompressor plug-in
```

Output per layer: `{selected_indices, kept_K, kept_V}` — no quantization codes.

```bash
pytest tests/test_rocketkv.py -v
python scripts/run_eval.py --compressor rocketkv --context-length 512
```

Full implementation details, configuration, and known limitations: [docs/QJL_AND_ROCKETKV.md](docs/QJL_AND_ROCKETKV.md).

### Evaluation metrics

**Section A — Compression Fidelity (offline)**

| Metric                                 | Module                          |
| -------------------------------------- | ------------------------------- |
| Tensor RMSE (K/V)                      | `eval/fidelity.py`              |
| Attention RMSE / cosine / max (`QK^T`) | `eval/attention_score_error.py` |
| Memory / compression ratio             | `eval/memory.py`                |

**Section B — Inference Impact (online, compressed KV in loop)**

| Metric     | Module                                            |
| ---------- | ------------------------------------------------- |
| Perplexity | `eval/perplexity.py` → `KVCacheEngine.step()`     |
| Speed      | `eval/throughput.py` → `KVCacheEngine.generate()` |

---

## Project Structure

```text
.
├── docs/               # SYSTEM_DESIGN.md, QJL_AND_ROCKETKV.md
├── configs/            # model.yaml, eval.yaml
├── framework/          # model layer, device, kv_cache utilities
├── compressors/        # KVCompressor plug-ins (TurboQuant, QJL, RocketKV, …)
├── quantizers/         # Compression primitives (WHT, Lloyd-Max, QJL, RocketKV)
├── baselines/          # re-exports of compressor implementations
├── eval/               # perplexity, memory, throughput, runner
├── data/               # WikiText-2 loader + long-context builder
├── datasets/           # placeholder dirs (cache lives in .cache/)
├── reporting/          # JSON/CSV export
├── modal_app/          # Modal CUDA workers + parallel sweep orchestrator
├── models/             # downloaded weights (gitignored)
├── results/            # experiment outputs (gitignored)
├── plots/              # figures (gitignored)
├── notebooks/          # exploratory analysis
├── scripts/            # download, verify, run eval
└── tests/              # pytest suite
```

---

## Modal GPU Evaluation (NVIDIA)

Full design: [docs/MODAL_GPU_EVAL_DESIGN.md](docs/MODAL_GPU_EVAL_DESIGN.md) — implemented NVIDIA runtime, job-level parallelism, storage, runbook.

Local Mac (MPS) stays for development and smoke tests. **Full Phase 5 sweeps run on Modal** — one **A10G GPU per job**, parallelized via `spawn_map()`. Sweep presets live in `configs/modal_sweeps.yaml`:

| Preset                 | Configs | Jobs (× ctx 128, 256, 512) |
| ---------------------- | ------- | -------------------------- |
| `baseline`             | 1       | 3 (shared identity — run once, reuse for all methods) |
| `turboquant` (default) | 4       | 12                         |
| `qjl`                  | 1       | 3                          |
| `rocketkv`             | 3       | 9                          |

**Results layout:** shared baseline lives in `results/phase5_modal_baseline/`; method sweeps reference it (TurboQuant: `results/phase5_modal_sweep_128_256_512/`, RocketKV: `results/phase5_modal_rocketkv/`).

**Prerequisites:** [Modal account](https://modal.com), `pip install modal`, and the existing secret `huggingface-secret` (`HF_TOKEN`).

```bash
# 1. One-time: download Qwen3-1.7B into Modal Volume (~3.2 GB)
bash scripts/modal_setup_model.sh

# 2. Launch detached parallel sweep
bash scripts/modal_run_sweep_baseline.sh                     # shared identity baseline (3 jobs, once)
bash scripts/modal_run_sweep.sh                              # turboquant (12 jobs)
bash scripts/modal_run_sweep_qjl.sh                          # qjl (3 jobs)
bash scripts/modal_run_sweep_rocketkv.sh                     # rocketkv (9 jobs)
# or: PRESET=qjl OUTPUT=phase5_modal_qjl bash scripts/modal_run_sweep.sh

# Subset example
PRESET=rocketkv CONTEXT_LENGTHS=128,512 LABELS=rocketkv_r50 bash scripts/modal_run_sweep.sh

# 3. Fetch per-job JSON from Modal Volume
bash scripts/modal_fetch_results.sh

# 4. Merge into local CSV/JSON
modal run modal_app/sweep.py::merge_local --input-dir results/modal_volume --output phase5_modal_rocketkv
```

| Modal artifact | Name                 | Purpose                                                                                                                         |
| -------------- | -------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Model volume   | `kv-engine-qwen3`    | Persist Qwen3-1.7B weights                                                                                                      |
| Results volume | `kv-engine-results`  | Per-job JSON (TurboQuant/QJL: `{label}_ctx{len}_b{bw}_{stage}.json`; RocketKV: `{label}_ctx{len}_r{keep}_ws{win}_k{topk}.json`) |
| Secret         | `huggingface-secret` | HF_TOKEN for first-time model download                                                                                          |

Config: `configs/modal.yaml` (GPU type, timeouts, volume names). Sweep grids: `configs/modal_sweeps.yaml`.

---

## Scripts Reference

| Script                                | Purpose                                                                              |
| ------------------------------------- | ------------------------------------------------------------------------------------ |
| `scripts/download_model.py`           | Download Qwen3-1.7B from HuggingFace                                                 |
| `scripts/verify_kv_cache.py`          | Confirm `past_key_values` access                                                     |
| `scripts/run_eval.py`                 | Full eval runner (memory + speed + perplexity)                                       |
| `scripts/run_baseline.py`             | Single-baseline eval with JSON output                                                |
| `scripts/validate_turboquant.py`      | TurboQuant stage ablation + KV intercept smoke test                                  |
| `scripts/modal_setup_model.sh`        | One-time Qwen3 download to Modal Volume                                              |
| `scripts/modal_run_sweep_baseline.sh` | Shared identity baseline sweep (3 jobs) |
| `scripts/modal_run_sweep.sh`          | Detached parallel eval sweep on Modal A10G GPUs (`PRESET=baseline\|turboquant\|qjl\|rocketkv`) |
| `scripts/modal_run_sweep_qjl.sh`      | QJL preset sweep (3 jobs)                                                            |
| `scripts/modal_run_sweep_rocketkv.sh` | RocketKV preset sweep (9 jobs)                                                       |
| `scripts/restructure_modal_results.py`| Split baseline vs method bundles from `results/modal_volume/`                        |
| `scripts/modal_smoke_eval.sh`         | One-job Modal smoke (`qjl`, `rocketkv`, `baseline`, `turboquant`) @ ctx=128          |
| `scripts/modal_fetch_results.sh`      | Pull job JSON from `kv-engine-results` volume                                        |

### `run_eval.py` flags

```bash
python scripts/run_eval.py \
  --compressor identity \       # identity | turboquant | kivi | qjl | rocketkv
  --bitwidth 4 \                # optional, method-specific
  --context-length 512 \        # single length
  --all-context-lengths \       # sweep configs/model.yaml lengths
  --skip-perplexity \           # skip slow perplexity pass
  --output my_run               # results/my_run.json + .csv
```

---

## Datasets

**Phase 1:** WikiText-2 only (`Salesforce/wikitext`, config `wikitext-2-raw-v1`).

```python
from data.loader import load_wikitext2, build_long_context_ids

dataset = load_wikitext2()
token_ids = build_long_context_ids(tokenizer, dataset, target_length=4096)
```

**Phase 2:** small C4 subset (config stub in `configs/eval.yaml`).

Cache directory: `.cache/huggingface/datasets/` (gitignored).

---

## Troubleshooting

| Issue                                  | Fix                                                                            |
| -------------------------------------- | ------------------------------------------------------------------------------ |
| `ModuleNotFoundError: compressors`     | Run scripts from repo root: `python scripts/run_eval.py`                       |
| `Model not found at models/qwen3_1.7b` | Run `python scripts/download_model.py`                                         |
| WikiText load fails with `HfUriError`  | Use `Salesforce/wikitext` (already set in `configs/eval.yaml`)                 |
| `fast-hadamard-transform` build error  | Expected on Mac; skip for identity/eval smoke tests                            |
| Slow eval on long contexts             | Start with `--context-length 512`; use `--skip-perplexity` for speed-only runs |
| MPS OOM at 32K context                 | Reduce `--context-length` or run on CPU                                        |
| Full sweep too slow locally            | Use Modal: `bash scripts/modal_run_sweep.sh`                                   |

---

## Roadmap

1. **Phase 0** — Repository setup, model download, KV cache verification ✅
2. **Phase 0.5** — Generic evaluation framework, WikiText-2 loader, compressor interface ✅
3. **Phase 1** — TurboQuant implementation ✅
4. **Phase 2** — KIVI baseline
5. **Phase 3** — QJL baseline ✅
6. **Phase 4** — RocketKV baseline ✅
7. **Phase 5** — Full evaluation sweep across all methods (Modal parallel sweep)

See [docs/SYSTEM_DESIGN.md §11](docs/SYSTEM_DESIGN.md#11-design-verification-checklist) for verification against common KV-compression mistakes (granularity, attention, QJL storage, Hadamard scaling, paper reusability).

---

## License

See individual paper implementations for respective licenses.
