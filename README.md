# KV-Cache Compression Benchmark

Reproduction and benchmarking framework for KV-cache compression methods:

**TurboQuant → KIVI → QJL → RocketKV**

Built on **Qwen3-1.7B** ([Hugging Face](https://huggingface.co/Qwen/Qwen3-1.7B)) with PyTorch and HuggingFace Transformers, optimized for Apple Silicon (MPS).

---

## What Is (and Is Not) in This Repo

This repository contains **code, configs, and scripts only**. Large artifacts are downloaded locally and are **gitignored**:

| Artifact | Location | Approx. size | How to obtain |
|---|---|---:|---|
| Model weights | `models/qwen3_1.7b/` | ~3.2 GB | `python scripts/download_model.py` |
| WikiText-2 cache | `.cache/huggingface/datasets/` | ~10 MB | auto on first eval / test run |
| Virtual environment | `.venv/` | ~2 GB | `pip install -r requirements.txt` |
| Experiment results | `results/`, `plots/` | varies | produced by eval scripts |

Never commit `.env` (contains your HuggingFace token).

---

## Requirements

| Requirement | Version / notes |
|---|---|
| Python | 3.11 |
| OS | macOS (Apple Silicon MPS) or Linux/CPU |
| Disk space | ~6 GB free (venv + model + cache) |
| HuggingFace account | Token with model read access |

> **Apple Silicon note:** `fast-hadamard-transform` requires CUDA/nvcc and does **not** build on MPS. Core eval works without it; TurboQuant Hadamard steps need a CUDA machine or a fallback (Phase 1).

---

## Reproduction Guide

Follow these steps in order on a fresh clone.

### 1. Clone and create the virtual environment

```bash
git clone https://github.com/<your-username>/kv-cache-compression-benchmark.git
cd kv-cache-compression-benchmark

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
conda activate kv-cache-compression-benchmark
```

### 3. Configure HuggingFace authentication

Copy the example env file and add your token ([create one here](https://huggingface.co/settings/tokens)):

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

All 5 tests should pass (compressor roundtrip, WikiText-2 loader, KV-cache shapes, eval smoke test).

### 7. Run a benchmark

```bash
# Quick smoke test (~512 tokens)
python scripts/run_eval.py --compressor identity --context-length 512

# Single baseline with JSON output
python scripts/run_baseline.py --baseline identity --context-length 512

# Full context-length sweep (4096 → 32768; slow on MPS)
python scripts/run_eval.py --compressor identity --all-context-lengths
```

Results are written to `results/` as `.json` and `.csv`.

---

## Configuration Reference

### `configs/model.yaml`

| Key | Description | Default |
|---|---|---|
| `model_name` | HuggingFace model ID | `Qwen/Qwen3-1.7B` |
| `local_path` | Where weights are saved | `models/qwen3_1.7b` |
| `context_lengths` | Long-context eval sweep | `4096, 8192, 16384, 32768` |
| `bitwidths` | Target compression bitwidths | `2, 3, 4` |

### `configs/eval.yaml`

| Key | Description | Default |
|---|---|---|
| `wikitext.name` | HF dataset ID | `Salesforce/wikitext` |
| `wikitext.config` | Dataset config | `wikitext-2-raw-v1` |
| `wikitext.split` | Eval split | `test` |
| `perplexity_stride` | Sliding-window stride | `512` |
| `generated_tokens` | Tokens for throughput test | `64` |
| `default_context_length` | Default `--context-length` | `512` |

WikiText-2 documents are short; the framework **concatenates samples** until the target context length is reached (standard practice for long-context KV-cache eval).

---

## Architecture

Four-layer research framework — only the compression layer changes per paper:

```text
Model Layer          → framework/model.py
KV Compression Layer → compressors/
Evaluation Layer     → eval/
Reporting Layer      → reporting/
```

### Compressors

All methods implement `KVCompressor`:

```python
class KVCompressor:
    def compress(self, key, value): ...
    def decompress(self, compressed): ...
```

| Name | Status | Paper pipeline |
|---|---|---|
| `identity` | ✅ working | no compression (baseline) |
| `turboquant` | ✅ Phase 1 | WHT → Lloyd-Max → QJL residual |

### TurboQuant KV compression layer

The compression layer is **distinct** from the model and eval layers:

```text
quantizers/          # math primitives (WHT, Lloyd-Max, QJL)
compressors/turboquant.py   # TurboQuantCompressor plug-in
framework/kv_engine.py      # KVCacheEngine interception
```

Pipeline per K/V tensor:

```text
pad(D→2^k) → WHT → normalize(÷√D) → Lloyd-Max → residual → QJL → store
```

Step-by-step ablation (recommended order):

```bash
python scripts/validate_turboquant.py --phase stages
python scripts/validate_turboquant.py --phase intercept

python scripts/run_eval.py --compressor turboquant --stage wht_only --context-length 512
python scripts/run_eval.py --compressor turboquant --stage wht_quant --context-length 512
python scripts/run_eval.py --compressor turboquant --stage full --context-length 512
```

Model layer loads with `attn_implementation="eager"` (required for KV hooks; no FlashAttention).
| `kivi` | 🔜 Phase 2 | asymmetric INT2 |
| `qjl` | 🔜 Phase 3 | random projection → 1-bit |
| `rocketkv` | 🔜 Phase 4 | token selection → eviction |

### Evaluation metrics (paper-independent)

| Metric | Module | Output |
|---|---|---|
| Memory | `eval/memory.py` | KV cache bytes, compression ratio |
| Speed | `eval/throughput.py` | tokens/sec, latency ms/token |
| Quality | `eval/perplexity.py` | perplexity (WikiText-2, sliding window) |

---

## Project Structure

```text
kv-cache-compression-benchmark/
├── configs/            # model.yaml, eval.yaml
├── framework/          # model layer, device, kv_cache utilities
├── compressors/        # KVCompressor interface + method stubs
├── quantizers/         # TurboQuant building blocks (WHT, Lloyd-Max)
├── baselines/          # re-exports of compressor implementations
├── eval/               # perplexity, memory, throughput, runner
├── data/               # WikiText-2 loader + long-context builder
├── datasets/           # placeholder dirs (cache lives in .cache/)
├── reporting/          # JSON/CSV export
├── models/             # downloaded weights (gitignored)
├── results/            # experiment outputs (gitignored)
├── plots/              # figures (gitignored)
├── notebooks/          # exploratory analysis
├── scripts/            # download, verify, run eval
└── tests/              # pytest suite
```

---

## Scripts Reference

| Script | Purpose |
|---|---|
| `scripts/download_model.py` | Download Qwen3-1.7B from HuggingFace |
| `scripts/verify_kv_cache.py` | Confirm `past_key_values` access |
| `scripts/run_eval.py` | Full eval runner (memory + speed + perplexity) |
| `scripts/run_baseline.py` | Single-baseline eval with JSON output |
| `scripts/validate_turboquant.py` | TurboQuant stage ablation + KV intercept smoke test |

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

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: compressors` | Run scripts from repo root: `python scripts/run_eval.py` |
| `Model not found at models/qwen3_1.7b` | Run `python scripts/download_model.py` |
| WikiText load fails with `HfUriError` | Use `Salesforce/wikitext` (already set in `configs/eval.yaml`) |
| `fast-hadamard-transform` build error | Expected on Mac; skip for identity/eval smoke tests |
| Slow eval on long contexts | Start with `--context-length 512`; use `--skip-perplexity` for speed-only runs |
| MPS OOM at 32K context | Reduce `--context-length` or run on CPU |

---

## Roadmap

1. **Phase 0** — Repository setup, model download, KV cache verification ✅
2. **Phase 0.5** — Generic evaluation framework, WikiText-2 loader, compressor interface ✅
3. **Phase 1** — TurboQuant implementation
4. **Phase 2** — KIVI baseline
5. **Phase 3** — QJL baseline
6. **Phase 4** — RocketKV baseline

---

## License

See individual paper implementations for respective licenses.
