# Phase 5 Evaluation Results

Case-study numbers for the **KV-Cache Interception and Transformation Engine** on Qwen3-1.7B. Methods are compared under one pipeline — not as standalone paper reproductions.

Qwen3-1.7B · WikiText-2 test · ctx **128 / 256 / 512** · Modal A10G · July 2026 (post-fix sweeps).

Raw JSON/CSV: `results/` (gitignored). To reproduce: [REPRODUCIBILITY.md](REPRODUCIBILITY.md) · fetch with `bash scripts/modal_fetch_results.sh` · bundle with `python scripts/restructure_modal_results.py`.

## Identity baseline

| ctx | PPL | tok/s |
|---:|---:|---:|
| 128 | 14.21 | 23.68 |
| 256 | 17.66 | 17.68 |
| 512 | 14.11 | 13.85 |

## Cross-method @ ctx=512

| Method | Config | PPL | vs baseline | Memory | tok/s |
|---|---|---:|---:|---:|---:|
| Identity | — | 14.11 | 1.0× | 1.0× | 13.85 |
| TurboQuant | `tq_full_b4` | 18.6 | 1.3× | 3.1× | 0.08 |
| QJL | `qjl_default` | 101M | ~7.2M× | 1.9× | 0.27 |
| RocketKV | `rocketkv_r256` | 6.76M | ~479k× | 2.0× | 9.25 |

## TurboQuant — perplexity (Section B)

| config | ctx=128 | ctx=256 | ctx=512 |
|---|---:|---:|---:|
| tq_full_b2 | 19,414 | 86,998 | 50,875 |
| tq_full_b3 | 96.5 | 454.4 | 214.8 |
| tq_full_b4 | 18.1 | 25.3 | **18.6** |
| tq_mse_b4 | 21.7 | 25.3 | 18.8 |

Baseline PPL at each ctx: 14.21 / 17.66 / 14.11.

**ctx=512 fidelity:** `tq_full_b4` — 3.12× compression, key RMSE 0.36, attn cosine 0.602, 0.08 tok/s.

## QJL (`qjl_default`)

Post-fix: asymmetric QJL attention estimator online + per-query-head GQA scoring.

| ctx | compress | attn cos | PPL | tok/s |
|---:|---:|---:|---:|---:|
| 128 | 1.84× | 0.628 | 42.5M | 0.85 |
| 256 | 1.85× | 0.546 | 72.9M | 0.65 |
| 512 | 1.85× | 0.411 | 101M | 0.27 |

Key RMSE ~6.7–6.9; attn RMSE ~44–50. High PPL reflects 1-bit key signs on Qwen3-1.7B under this pipeline, not a decompress-vs-estimator mismatch.

Source: `results/phase5_modal_qjl/phase5_modal_qjl_20260705T202532Z.csv`

## RocketKV

Post-fix: token budgets `r256` / `r512` / `r1024` + stage-1 lock + online HSA (`framework/rocketkv_online.py`).

| config | ctx=128 PPL | ctx=256 PPL | ctx=512 PPL | ctx=512 compress | ctx=512 tok/s |
|---|---:|---:|---:|---:|---:|
| r256 | 11.1M | 8.75M | **6.76M** | 1.98× | 9.25 |
| r512 | 11.1M | 8.75M | 7.20M | 0.99× | 7.46 |
| r1024 | 11.1M | 8.74M | 7.21M | 0.99× | 18.1 |

Section A now reports post-selection fidelity (e.g. `r256` @ ctx=512: attn cosine 0.44, attn RMSE 29.8). At ctx≤256, token budget ≥ sequence length → compression ratio ≈1.0 and near-identity Section A.

Source: `results/phase5_modal_rocketkv/phase5_modal_rocketkv_20260705T202549Z.csv`

## Findings (framework lens)

- **TurboQuant 4-bit** is the only method with paper-plausible quality (~1.3× baseline PPL, ~3× memory) at ctx≥256 in this pipeline; online inference is very slow.
- **QJL** saves ~1.9× memory with faithful online estimator; attn cosine 0.41–0.63 but PPL remains catastrophic vs baseline.
- **RocketKV** preserves near-baseline throughput but PPL stays in the millions even with largest budgets; smallest budget (`r256`) yields best PPL at ctx=512.
- These results illustrate **what the framework exposes**, not final claims about each upstream paper.

Shared identity baseline from preset `baseline` in `configs/modal_sweeps.yaml` — not re-run per method.
