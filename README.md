# KVBench

**A unified online and offline evaluation framework for KV-cache compression in small language models (SLMs).**

```text
Model (fixed) → KVCacheEngine (fixed) → KVCompressor (variable) → eval/ (fixed)
```

KVBench is a reproducible **benchmarking and evaluation framework**, not a single-algorithm reproduction. It fixes the model, incremental decode loop, and metrics while exposing KV-cache compressors as interchangeable plug-ins, and always reports:

- **Section A (offline fidelity):** tensor RMSE, attention-score preservation, memory accounting  
- **Section B (online inference):** sliding-window perplexity and throughput with compressed KV in the autoregressive loop  

**License:** [Apache-2.0](LICENSE) · **Status:** research manuscript in preparation · **Contributions:** welcome ([CONTRIBUTING.md](CONTRIBUTING.md)) · **Roadmap:** [ROADMAP.md](ROADMAP.md)

> Research manuscript in preparation. We welcome contributions that improve the framework, expand evaluations, or add new KV-compression methods. Significant research contributions may be considered for co-authorship in accordance with academic authorship standards.

## Why KVBench?

Published KV-compression methods are hard to compare: protocols differ, and offline tensor/attention metrics often fail to predict online decode quality. KVBench makes that gap measurable under one incremental engine on resource-constrained SLMs (~1B-scale), where full factorial sweeps are feasible on a single GPU.

## Main results @ context length 512

### Qwen3-1.7B (GQA) · WikiText-2 · Modal A10G

| Method | Config | PPL | vs baseline | Memory | tok/s |
|---|---|---:|---:|---:|---:|
| Identity | — | 14.11 | 1.0× | 1.0× | 13.85 |
| TurboQuant | `tq_full_b4` | 18.6 | **1.3×** | **3.1×** | 0.08 |
| QJL | `qjl_default` | ~1e8 | ≫1 | 1.9× | 0.27 |
| RocketKV | `rocketkv_r256` | ~6.8e6 | ≫1 | 2.0× | 9.25 |

### OLMo 2 1B (MHA) · same protocol

| Method | Config | PPL | vs baseline | Memory | tok/s |
|---|---|---:|---:|---:|---:|
| Identity | — | 8.31 | 1.0× | 1.0× | 22.49 |
| TurboQuant | `tq_full_b4` | 8.51 | **1.02×** | **3.12×** | 0.14 |
| QJL | `qjl_default` | 873 | 105× | 1.85× | 0.36 |
| RocketKV | `rocketkv_r256` | 8.77 | **1.05×** | **1.99×** | 14.76 |

**Takeaway:** offline fidelity does **not** predict online quality, and rankings can flip across attention layouts (GQA vs MHA). Full tables: [docs/PHASE5_EVAL_RESULTS.md](docs/PHASE5_EVAL_RESULTS.md), [docs/OLMO2_RESULTS_COMPLETE.md](docs/OLMO2_RESULTS_COMPLETE.md).

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
```

**Modal sweeps** — see [docs/MODAL_GPU_EVAL_DESIGN.md](docs/MODAL_GPU_EVAL_DESIGN.md)

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

Step-by-step: **[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)**

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
| [SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md) | Architecture |
| [METHODOLOGY.md](docs/METHODOLOGY.md) | Experimental methodology |
| [MATHEMATICS_AND_ALGORITHMS.md](docs/MATHEMATICS_AND_ALGORITHMS.md) | Equations and pseudocode |
| [PHASE5_EVAL_RESULTS.md](docs/PHASE5_EVAL_RESULTS.md) | Qwen3 Phase-5 summary |
| [RESULTS_COMPLETE.md](docs/RESULTS_COMPLETE.md) | Qwen3 full metrics |
| [OLMO2_RESULTS_COMPLETE.md](docs/OLMO2_RESULTS_COMPLETE.md) | OLMo 2 full metrics + logs |
| [OLMO2_ARCHITECTURE_PROBE.md](docs/OLMO2_ARCHITECTURE_PROBE.md) | OLMo 2 coupling notes |
| [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | Reproduction guide |
| [MODAL_GPU_EVAL_DESIGN.md](docs/MODAL_GPU_EVAL_DESIGN.md) | Modal runbook |
| [ROADMAP.md](ROADMAP.md) | Planned work |

## Citation

If you use KVBench in research, please cite this repository (see [CITATION.cff](CITATION.cff)):

```bibtex
@software{kvbench2026,
  author = {Faheem, Muhammad},
  title  = {{KVBench}: Bridging Offline Fidelity and Online Inference Evaluation for {KV} Cache Compression in Small Language Models},
  year   = {2026},
  url    = {https://github.com/faheemgurkani/kv-cache-compression-benchmark-},
  note   = {Research manuscript in preparation}
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
