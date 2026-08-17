# KV Cache Interception and Transformation Engine

**A unified interception engine and three-dimensional evaluation stack for KV-cache compression in small language models (SLMs).**

This repository implements the **KV Cache Interception and Transformation Engine**: it intercepts K/V tensors at the decode boundary, transforms them through interchangeable compression plug-ins, and measures fidelity, generation behavior, and system-level inference cost under incremental decode. The reproducible benchmarking protocol and case-study results are published under the name **[KVBench](docs/research_paper_writeup/conference_101719.tex)** (research manuscript in preparation).

```text
Model (fixed) → KVCacheEngine (fixed) → KVCompressor (variable) → eval/ (fixed)
```

The engine is **not** a single-algorithm reproduction. It fixes the model, incremental decode loop, and metrics while exposing KV-cache compressors as interchangeable plug-ins, and always reports three independent evaluation dimensions instead of a coarse offline/online split:

- **FIDELITY** *(did the transformation preserve the KV representation and attention behavior?)* — tensor RMSE, relative reconstruction error, cosine similarity, attention-score/output RMSE, attention-distribution KL divergence, compression ratio, actual memory reduction, metadata overhead.
- **BEHAVIOR** *(does the model still behave correctly after KV transformation?)* — sliding-window perplexity, plus opt-in needle-in-haystack retrieval, instruction-following compliance, and synthetic reasoning accuracy, all measured through real compressed-KV decoding, not a single forward pass.
- **SYSTEM** *(does the compression actually make inference better?)* — TTFT, inter-token latency, decode/end-to-end latency, tokens/sec, peak VRAM, actual KV memory, compress/decompress time, and (best-effort) memory bandwidth and GPU utilization. A method with a higher compression ratio can still lose here if it adds enough per-step compute.

FIDELITY does not reliably predict BEHAVIOR — that gap is the central finding this framework is built to surface — and SYSTEM exists as its own branch because a compression ratio win on paper can be a runtime loss in practice.

**License:** [Apache-2.0](LICENSE) · **Status:** research manuscript in preparation · **Contributions:** welcome ([CONTRIBUTING.md](CONTRIBUTING.md)) · **Roadmap:** [ROADMAP.md](ROADMAP.md)

> We welcome contributions that improve the engine, expand evaluations, or add new KV-compression methods. Significant research contributions may be considered for co-authorship in accordance with academic authorship standards.

## Why this project?

Published KV-compression methods are hard to compare: protocols differ, and offline tensor/attention metrics often fail to predict online decode quality. The interception-and-transformation engine makes that gap measurable under one incremental loop on resource-constrained SLMs (~1B-scale), where full factorial sweeps are feasible on a single GPU. **KVBench** is the name we use for this standardized evaluation framework and its empirical study.

## Main results @ context length 512

### Qwen3-1.7B (GQA) · WikiText-2 · Modal A10G

| Method | Config | PPL | vs baseline | Memory | tok/s |
|---|---|---:|---:|---:|---:|
| Identity | — | 14.11 | 1.0× | 1.0× | 13.85 |
| TurboQuant | `tq_full_b4` | 18.6 | **1.3×** | **3.1×** | 0.08 |
| QJL | `qjl_default` | ~2.2e5 | ≫1 | 1.9× | 0.36 |
| RocketKV | `rocketkv_r256` | ~6.8e6 | ≫1 | 2.0× | 9.25 |

### OLMo 2 1B (MHA) · same protocol

| Method | Config | PPL | vs baseline | Memory | tok/s |
|---|---|---:|---:|---:|---:|
| Identity | — | 8.31 | 1.0× | 1.0× | 22.49 |
| TurboQuant | `tq_full_b4` | 8.51 | **1.02×** | **3.12×** | 0.14 |
| QJL | `qjl_default` | 359 | 43× | 1.85× | 0.58 |
| RocketKV | `rocketkv_r256` | 8.77 | **1.05×** | **1.99×** | 14.76 |

**Takeaway:** offline fidelity does **not** predict online quality, and rankings can flip across attention layouts (GQA vs MHA). Full tables: [docs/results/qwen3_1.7b/PHASE5_EVAL_RESULTS.md](docs/results/qwen3_1.7b/PHASE5_EVAL_RESULTS.md), [docs/results/olmo2_1b/RESULTS_COMPLETE.md](docs/results/olmo2_1b/RESULTS_COMPLETE.md).

## Compressors (plug-ins)

| Name | Status | Role |
|---|---|---|
| `identity` | ready | Shared uncompressed baseline |
| `turboquant` | ready | Vector quantization (WHT + Lloyd-Max ± residual) |
| `qjl` | ready | Sketch / 1-bit key signs |
| `rocketkv` | ready | Token eviction + sparse HSA |
| `kivi` | stub | Planned — see [ROADMAP.md](ROADMAP.md) |

## Prerequisites

- Python 3.11+, Hugging Face token, disk for model weights  
- Local: macOS MPS or Linux/CPU for smoke tests  
- Full CUDA sweeps: [Modal](https://modal.com) account (recommended)

## Quick start

```bash
git clone https://github.com/faheemgurkani/kv-cache-compression-benchmark-.git
cd kv-cache-compression-benchmark-

python3.11 -m venv .venv && source .venv/bin/activate
pip install torch torchvision torchaudio
pip install -r requirements.txt

cp .env.example .env   # set HF_TOKEN
python scripts/download_model.py
python scripts/verify_kv_cache.py
pytest tests/ -q

python scripts/run_eval.py --compressor identity --context-length 512
```

> `fast-hadamard-transform` may fail on macOS — skip it; Modal uses a scipy WHT fallback.

## Usage

**Local**

```bash
python scripts/run_eval.py --compressor turboquant --stage full --context-length 512
python scripts/run_eval.py --compressor qjl --context-length 512
python scripts/run_eval.py --compressor rocketkv --context-length 512

# Opt-in BEHAVIOR / SYSTEM sub-metrics (each adds its own generate() pass)
python scripts/run_eval.py --compressor turboquant --retrieval --instruction-following
python scripts/run_eval.py --compressor turboquant --peak-memory --memory-bandwidth --kernel-cost
```

FIDELITY always runs; BEHAVIOR/task_quality (perplexity) and SYSTEM/latency_throughput run by default. Retrieval, instruction-following, reasoning, peak VRAM, memory bandwidth, kernel cost, and GPU utilization are opt-in flags — see `python scripts/run_eval.py --help`.

**Modal sweeps** — see [docs/reproducibility/REPRODUCIBILITY.md §11](docs/reproducibility/REPRODUCIBILITY.md)

```bash
pip install modal
bash scripts/modal_setup_model.sh
bash scripts/modal_run_sweep_baseline.sh
bash scripts/modal_run_sweep.sh
bash scripts/modal_run_sweep_qjl.sh
bash scripts/modal_run_sweep_rocketkv.sh
bash scripts/modal_fetch_results.sh
```

Configs: `configs/model.yaml`, `configs/eval.yaml`, `configs/modal.yaml`, `configs/modal_sweeps.yaml`

## Reproducibility

Step-by-step: **[docs/reproducibility/REPRODUCIBILITY.md](docs/reproducibility/REPRODUCIBILITY.md)**

| Step | Command |
|---|---|
| Verify install | `pytest tests/ -q` |
| Local smoke | `python scripts/run_eval.py --compressor identity --context-length 512` |
| Modal smoke | `bash scripts/modal_smoke_eval.sh qjl` |
| Fetch + bundle (Qwen3) | `bash scripts/modal_fetch_results.sh && python scripts/restructure_modal_results.py` |
| Fetch + bundle (OLMo 2) | `bash scripts/modal_fetch_results.sh && python scripts/restructure_olmo2_modal_results.py` |

Record `git rev-parse HEAD` when citing results. Use `--no-resume` on Modal after code changes that affect metrics.

## Documentation

| Doc | Contents |
|---|---|
| [SYSTEM_DESIGN.md](docs/architecture/SYSTEM_DESIGN.md) | Architecture (high-level) |
| [ENGINE_INTERNALS.md](docs/architecture/ENGINE_INTERNALS.md) | Complete implementation walkthrough + generalizing to other model architectures |
| [MODEL_ARCHITECTURE_MATRIX.md](docs/architecture/MODEL_ARCHITECTURE_MATRIX.md) | Current 5-model architecture-matrix shortlist (MHA/GQA/MQA/MLA/Hybrid), engine-support status, adoption plan |
| [SLM_COMPATIBILITY.md](docs/architecture/SLM_COMPATIBILITY.md) | Historical: original 6-candidate SLM compatibility probe |
| [METHODOLOGY.md](docs/methodology/METHODOLOGY.md) | Experimental methodology |
| [MATHEMATICS_AND_ALGORITHMS.md](docs/methodology/MATHEMATICS_AND_ALGORITHMS.md) | Equations and pseudocode |
| [PHASE5_EVAL_RESULTS.md](docs/results/qwen3_1.7b/PHASE5_EVAL_RESULTS.md) | Qwen3 Phase-5 summary |
| [RESULTS_COMPLETE.md](docs/results/qwen3_1.7b/RESULTS_COMPLETE.md) | Qwen3 full metrics |
| [OLMo2 RESULTS_COMPLETE.md](docs/results/olmo2_1b/RESULTS_COMPLETE.md) | OLMo 2 full metrics + logs |
| [shortlist_5model_eval/](docs/results/shortlist_5model_eval/) | Live eval-framework run against the 5-model shortlist |
| [REPRODUCIBILITY.md](docs/reproducibility/REPRODUCIBILITY.md) | Reproduction guide (local + Modal) |
| [ROADMAP.md](ROADMAP.md) | Planned work |
| [research paper write-up](docs/research_paper_writeup/conference_101719.tex) | KVBench manuscript (title unchanged) |

## Citation

If you use this engine or the KVBench evaluation study in research, please cite the repository (see [CITATION.cff](CITATION.cff)):

```bibtex
@software{kvbench2026,
  author = {Faheem, Muhammad},
  title  = {{KVBench}: Bridging Offline Fidelity and Online Inference Evaluation for {KV} Cache Compression in Small Language Models},
  year   = {2026},
  url    = {https://github.com/faheemgurkani/kv-cache-compression-benchmark-},
  note   = {Implementation: KV Cache Interception and Transformation Engine. Research manuscript in preparation.}
}
```

## Contributing

See **[CONTRIBUTING.md](CONTRIBUTING.md)**. We especially welcome:

- Additional SLMs  
- Additional KV-compression algorithms  
- Additional evaluation datasets  
- Additional inference backends / providers  
- Reproducibility improvements  
- Documentation improvements  

Significant research contributions may be considered for co-authorship in accordance with academic authorship standards. Authorship is **not** guaranteed by opening a pull request alone.

## License

Licensed under the [Apache License 2.0](LICENSE). Copyright © 2026 Muhammad Faheem. Upstream model and dataset licenses still apply when you download those assets.
