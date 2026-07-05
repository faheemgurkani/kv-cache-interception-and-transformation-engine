# Phase 5 Evaluation Results

Case-study numbers for the **KV-Cache Interception and Transformation Engine** on Qwen3-1.7B. Methods are compared under one pipeline — not as standalone paper reproductions.

Qwen3-1.7B · WikiText-2 test · ctx **128 / 256 / 512** · Modal A10G · July 2026.

Raw JSON/CSV: `results/` (gitignored). Fetch with `bash scripts/modal_fetch_results.sh`.

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
| QJL | `qjl_default` | 532,967 | ~38k× | 1.9× | 0.35 |
| RocketKV | `rocketkv_r75` | 11,829,014 | ~838k× | 1.3× | 9.22 |

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

| ctx | compress | PPL | tok/s |
|---:|---:|---:|---:|
| 128 | 1.84× | 238,983 | 1.61 |
| 256 | 1.85× | 352,909 | 0.65 |
| 512 | 1.85× | 532,967 | 0.35 |

Key RMSE ~6.7; attn RMSE ~44–50 @ ctx=512.

## RocketKV

| config | ctx=512 PPL | compress | tok/s |
|---|---:|---:|---:|
| r25 | 51,534,737 | 3.34× | 9.30 |
| r50 | 36,697,567 | 1.87× | 8.21 |
| r75 | 11,829,014 | 1.30× | 9.22 |

Section A reports 0 RMSE (full-precision kept tokens); Section B fails because tokens are evicted online.

## Findings (framework lens)

- **TurboQuant 4-bit** is the only method with paper-plausible quality (~1.3× baseline PPL, ~3× memory) at ctx≥256 in this pipeline; online inference is very slow.
- **QJL** saves ~1.9× memory but Section B PPL explodes — online uses key reconstruct, not the QJL attention estimator.
- **RocketKV** (historical `r25`/`r50`/`r75` presets) kept near-baseline speed but catastrophic PPL; presets are now **token budgets** (`r256`/`r512`/`r1024`) with improved online fidelity hooks.
- These results illustrate **what the framework exposes**, not final claims about each upstream paper.

Shared identity baseline from preset `baseline` in `configs/modal_sweeps.yaml` — not re-run per method.
