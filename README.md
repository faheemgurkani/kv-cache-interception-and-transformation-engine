# KV-Cache Interception and Transformation Engine

A **modular framework for KV-cache compression analysis and benchmarking** — not a single-algorithm reproduction. One fixed interception engine, plug-in compressors, and a unified online + offline evaluation pipeline. **Qwen3-1.7B** is the reference model; **TurboQuant**, **QJL**, and **RocketKV** are case studies under a common eval stack.

```text
Model (fixed) → KVCacheEngine (fixed) → KVCompressor (variable) → eval/ (fixed)
```

Architecture: [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md) · Methodology: [docs/METHODOLOGY.md](docs/METHODOLOGY.md) · Math: [docs/MATHEMATICS_AND_ALGORITHMS.md](docs/MATHEMATICS_AND_ALGORITHMS.md) · Results: [docs/PHASE5_EVAL_RESULTS.md](docs/PHASE5_EVAL_RESULTS.md) ([complete tables](docs/RESULTS_COMPLETE.md)) · Reproducibility: [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) · Modal: [docs/MODAL_GPU_EVAL_DESIGN.md](docs/MODAL_GPU_EVAL_DESIGN.md)

## Research positioning

**Contribution:** a reproducible **KV-cache compression evaluation framework** with faithful online inference (perplexity, throughput) and offline fidelity (tensor RMSE, attention score error, memory) — not “we invented TurboQuant.”

| Question | Verdict |
|---|---|
| Enough for arXiv? | **Yes** — clean, reproducible framework + multi-method case studies |
| Workshop paper? | **Likely** |
| Top conference (NeurIPS / ICLR / ICML / ACL)? | **Not yet** — needs broader empirical study |
| Worth continuing? | **Yes** |

**Strongest framing:** *A unified online evaluation framework for KV-cache compression in LLMs*, with TurboQuant, QJL, and RocketKV demonstrating how different strategies behave under one pipeline.

**To strengthen further:** more models (Phi-3 Mini, Llama-3.2 3B), contexts (2K–8K), algorithms (SnapKV, KIVI, AdaKV), and benchmarks (LongBench, RULER, needle-in-a-haystack).

## Main results @ ctx=512

Qwen3-1.7B · WikiText-2 · Modal A10G · shared identity baseline PPL ≈ 14.11

| Method | Config | PPL | vs baseline | Memory | tok/s |
|---|---|---:|---:|---:|---:|
| Identity | — | 14.11 | 1.0× | 1.0× | 13.85 |
| TurboQuant | `tq_full_b4` | 18.6 | **1.3×** | **3.1×** | 0.08 |
| QJL | `qjl_default` | 101M | ~7.2M× | 1.9× | 0.27 |
| RocketKV | `rocketkv_r256` | 6.76M | ~479k× | 2.0× | 9.25 |

Post-fix sweeps (July 2026): QJL uses asymmetric estimator online; RocketKV uses token budgets `r256`/`r512`/`r1024`. Full tables: [docs/PHASE5_EVAL_RESULTS.md](docs/PHASE5_EVAL_RESULTS.md).

**Takeaway:** **TurboQuant 4-bit @ ctx≥256** stays near baseline PPL with ~3× memory savings. **QJL** (faithful online estimator) still shows catastrophic PPL under 1-bit keys. **RocketKV** keeps near-baseline speed but PPL remains in the millions even with largest budgets.

## Prerequisites

- Python 3.11, Hugging Face token, ~6 GB disk
- Local: macOS MPS or Linux/CPU for dev/smoke
- Full CUDA sweeps: [Modal](https://modal.com) account (recommended)

## Quick start

```bash
git clone https://github.com/faheemgurkani/kv-cache-compression-benchmark.git
cd kv-cache-compression-benchmark

python3.11 -m venv .venv && source .venv/bin/activate
pip install torch torchvision torchaudio && pip install -r requirements.txt

cp .env.example .env   # set HF_TOKEN
python scripts/download_model.py
python scripts/verify_kv_cache.py
pytest tests/ -q

python scripts/run_eval.py --compressor identity --context-length 512
```

> `fast-hadamard-transform` fails on Mac — skip it; Modal uses scipy WHT fallback.

## Usage

**Local**

```bash
python scripts/run_eval.py --compressor turboquant --stage full --context-length 512
python scripts/run_eval.py --compressor qjl --context-length 512
python scripts/run_eval.py --compressor rocketkv --context-length 512
```

**Modal sweeps** — see [docs/MODAL_GPU_EVAL_DESIGN.md § Runbook](docs/MODAL_GPU_EVAL_DESIGN.md#runbook)

```bash
pip install modal
bash scripts/modal_setup_model.sh
bash scripts/modal_run_sweep_baseline.sh    # identity (3 jobs, once)
bash scripts/modal_run_sweep.sh             # turboquant (12)
bash scripts/modal_run_sweep_qjl.sh         # qjl (3)
bash scripts/modal_run_sweep_rocketkv.sh    # rocketkv (9)
bash scripts/modal_fetch_results.sh
```

Sweep presets: `configs/modal_sweeps.yaml` · Config: `configs/model.yaml`, `configs/eval.yaml`, `configs/modal.yaml`

## Reproducibility

Full step-by-step guide: **[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)**

| Step | Command |
|---|---|
| Verify install | `pytest tests/ -q` |
| Local smoke | `python scripts/run_eval.py --compressor identity --context-length 512` |
| Modal smoke | `bash scripts/modal_smoke_eval.sh qjl` |
| Full Phase 5 sweep | `bash scripts/modal_run_sweep_baseline.sh` then method scripts (see guide) |
| Fetch + bundle | `bash scripts/modal_fetch_results.sh && python scripts/restructure_modal_results.py` |

Record `git rev-parse HEAD` when citing results. Config YAML files are the experimental source of truth; use `--no-resume` on Modal for clean re-sweeps after code changes.

## Documentation

| Doc | Contents |
|---|---|
| [SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md) | Architecture overview |
| [METHODOLOGY.md](docs/METHODOLOGY.md) | Full experimental + algorithm methodology |
| [MATHEMATICS_AND_ALGORITHMS.md](docs/MATHEMATICS_AND_ALGORITHMS.md) | Equations, notation, pseudocode |
| [PHASE5_EVAL_RESULTS.md](docs/PHASE5_EVAL_RESULTS.md) | Summary results tables |
| [RESULTS_COMPLETE.md](docs/RESULTS_COMPLETE.md) | **Every metric, per-layer stats, logs** (auto-generated) |
| [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | How to reproduce sweeps |

Regenerate complete results doc: `python scripts/export_results_documentation.py`

## Compressors (plug-ins)

| Name | Status | Role in benchmark |
|---|---|---|
| `identity` | ✅ | Shared uncompressed baseline |
| `turboquant` | ✅ | Quantization case study (WHT + Lloyd-Max + optional QJL residual) |
| `qjl` | ✅ | Sketch / sign case study |
| `rocketkv` | ✅ | Token eviction + sparse attention case study |
| `kivi` | stub | Planned extension |

## Troubleshooting

| Issue | Fix |
|---|---|
| Model not found | `python scripts/download_model.py` |
| Import errors | Run from repo root |
| `fast-hadamard-transform` on Mac | Skip; use Modal for TurboQuant CUDA |
| Slow / OOM locally | `--context-length 512`; use Modal for full grid |

## License

MIT — see [LICENSE](LICENSE) (Copyright © 2026 Muhammad Faheem). Qwen3-1.7B: Apache 2.0. Respect upstream paper licenses when extending compressors.
