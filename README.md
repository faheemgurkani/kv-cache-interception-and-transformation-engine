# KV-Cache Compression Benchmark

Pluggable **KV-cache interception engine** on **Qwen3-1.7B** with a fixed eval pipeline for **TurboQuant**, **QJL**, and **RocketKV**.

```text
Model (fixed) → KVCacheEngine (fixed) → KVCompressor (variable) → eval/ (fixed)
```

Architecture: [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md) · Results: [docs/PHASE5_EVAL_RESULTS.md](docs/PHASE5_EVAL_RESULTS.md) · Modal: [docs/MODAL_GPU_EVAL_DESIGN.md](docs/MODAL_GPU_EVAL_DESIGN.md)

## Main results @ ctx=512

Qwen3-1.7B · WikiText-2 · Modal A10G · shared identity baseline PPL ≈ 14.11

| Method | Config | PPL | vs baseline | Memory | tok/s |
|---|---|---:|---:|---:|---:|
| Identity | — | 14.11 | 1.0× | 1.0× | 13.85 |
| TurboQuant | `tq_full_b4` | 18.6 | **1.3×** | **3.1×** | 0.08 |
| QJL | `qjl_default` | 533k | ~38k× | 1.9× | 0.35 |
| RocketKV | `rocketkv_r75` | 11.8M | ~838k× | 1.3× | 9.22 |

**Takeaway:** Only **TurboQuant 4-bit @ ctx≥256** stays near baseline PPL with ~3× memory savings. QJL and RocketKV degrade badly online under current paths. Full tables: [docs/PHASE5_EVAL_RESULTS.md](docs/PHASE5_EVAL_RESULTS.md).

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

## Compressors

| Name | Status | Notes |
|---|---|---|
| `identity` | ✅ | Shared baseline |
| `turboquant` | ✅ | WHT + Lloyd-Max + optional QJL residual |
| `qjl` | ✅ | 1-bit key signs; Section B uses reconstruct, not estimator |
| `rocketkv` | ✅ | Token eviction + online HSA patch |
| `kivi` | stub | Not implemented |

## Troubleshooting

| Issue | Fix |
|---|---|
| Model not found | `python scripts/download_model.py` |
| Import errors | Run from repo root |
| `fast-hadamard-transform` on Mac | Skip; use Modal for TurboQuant CUDA |
| Slow / OOM locally | `--context-length 512`; use Modal for full grid |

## License

MIT — see [LICENSE](LICENSE) (Copyright © 2026 Muhammad Faheem). Qwen3-1.7B: Apache 2.0. Respect upstream paper licenses when extending compressors.
