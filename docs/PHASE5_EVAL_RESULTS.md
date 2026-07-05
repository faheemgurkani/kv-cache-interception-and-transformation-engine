# Phase 5 Modal Evaluation Results

Comparative sweep on **Qwen3-1.7B**, **WikiText-2**, context lengths **128 / 256 / 512**, **Modal A10G**.

> **Note:** Raw job JSON, CSV exports, and logs live under `results/` (gitignored). This file is the **version-controlled** record of those runs.

| Method | Jobs | Status | Modal app | Local bundle (gitignored) |
|---|---|---|---|---|
| Identity (shared baseline) | 3 | ✅ complete | `ap-ek9dIxujlrECcfFaOa3ok3` | `results/phase5_modal_baseline/` |
| TurboQuant | 12 | ✅ complete | `ap-ek9dIxujlrECcfFaOa3ok3` | `results/phase5_modal_sweep_128_256_512/` |
| QJL | 3 | ✅ complete | `ap-Pck6cN9lPU80IfFCb4waT2` | `results/phase5_modal_qjl/` |
| RocketKV | 9 | ✅ complete | `ap-ZCFcYJgwGzBb7ZpLWBViLV` | `results/phase5_modal_rocketkv/` |

All methods share the same uncompressed baseline perplexity (reported per job as `perplexity_baseline` from the shared identity sweep).

---

## 1. Shared baseline (identity)

| ctx | PPL | tok/s | latency ms/tok |
|---:|---:|---:|---:|
| 128 | 14.21 | 23.68 | 42.2 |
| 256 | 17.66 | 17.68 | 56.6 |
| 512 | 14.11 | 13.85 | 72.2 |

Merged CSV: `phase5_modal_baseline_20260705T130825Z.csv` (in local bundle).

---

## 2. TurboQuant

Configs: `tq_full_b2`, `tq_full_b3`, `tq_full_b4`, `tq_mse_b4` (4-bit WHT-only variant).

### Section B — perplexity (ratio vs baseline ≈ 1.0 is ideal)

| config | ctx | PPL compressed | PPL baseline | ratio vs baseline |
|---|---:|---:|---:|---:|
| tq_full_b2 | 128 | 19,414 | 14.21 | 1,366× |
| tq_full_b2 | 256 | 86,998 | 17.66 | 4,926× |
| tq_full_b2 | 512 | 50,875 | 14.11 | 3,605× |
| tq_full_b3 | 128 | 96.5 | 14.21 | 6.8× |
| tq_full_b3 | 256 | 454.4 | 17.66 | 25.7× |
| tq_full_b3 | 512 | 214.8 | 14.11 | 15.2× |
| tq_full_b4 | 128 | 18.1 | 14.21 | 1.3× |
| tq_full_b4 | 256 | 25.3 | 17.66 | 1.4× |
| tq_full_b4 | 512 | **18.6** | 14.11 | **1.3×** |
| tq_mse_b4 | 128 | 21.7 | 14.21 | 1.5× |
| tq_mse_b4 | 256 | 25.3 | 17.66 | 1.4× |
| tq_mse_b4 | 512 | 18.8 | 14.11 | 1.3× |

### Section A — compression & fidelity (ctx=512)

| config | compress ratio | eff. bits/KV | key RMSE | attn cosine |
|---|---:|---:|---:|---:|
| tq_full_b2 | 5.12× | 3.13 | 1.78 | 0.575 |
| tq_full_b3 | 3.88× | 4.13 | 0.74 | 0.598 |
| tq_full_b4 | 3.12× | 5.13 | 0.36 | 0.602 |
| tq_mse_b4 | 3.56× | 4.50 | 0.36 | 0.602 |

### Section B — throughput (ctx=512, tok/s)

| config | tok/s | latency ms/tok |
|---|---:|---:|
| tq_full_b2 | 0.073 | 13,744 |
| tq_full_b3 | 0.083 | 12,039 |
| tq_full_b4 | 0.081 | 12,360 |
| tq_mse_b4 | 0.096 | 10,401 |
| identity (ref) | 13.85 | 72.2 |

Merged CSV: `phase5_modal_sweep_128_256_512_20260705T130825Z.csv` (in local bundle).

---

## 3. QJL

Config: `qjl_default` (1-bit key signs, values FP16).

| ctx | compress ratio | eff. bits/KV | key RMSE | attn RMSE | attn cosine | PPL compressed | PPL baseline | tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1.84× | 8.70 | 6.73 | 49.8 | 0.584 | 238,983 | 14.21 | 1.61 |
| 256 | 1.85× | 8.66 | 6.79 | 47.0 | 0.497 | 352,909 | 17.66 | 0.65 |
| 512 | 1.85× | 8.64 | 6.87 | 44.1 | 0.365 | 532,967 | 14.11 | 0.35 |

Merged CSV: `phase5_modal_qjl_20260705T155114Z.csv` (in local bundle).

---

## 4. RocketKV

Configs: `rocketkv_r25`, `rocketkv_r50`, `rocketkv_r75` (keep ratios 0.25 / 0.50 / 0.75).

### Section B — perplexity

| config | ctx | PPL compressed | PPL baseline | compress ratio | tok/s |
|---|---:|---:|---:|---:|---:|
| r25 | 128 | 10,414,510 | 14.21 | 2.27× | 17.64 |
| r25 | 256 | 35,453,467 | 17.66 | 2.89× | 14.51 |
| r25 | 512 | 51,534,737 | 14.11 | 3.34× | 9.30 |
| r50 | 128 | 8,741,889 | 14.21 | 1.59× | 17.62 |
| r50 | 256 | 14,526,022 | 17.66 | 1.76× | 12.34 |
| r50 | 512 | 36,697,567 | 14.11 | 1.87× | 8.21 |
| r75 | 128 | 9,448,166 | 14.21 | 1.22× | 17.60 |
| r75 | 256 | 9,226,797 | 17.66 | 1.27× | 14.12 |
| r75 | 512 | 11,829,014 | 14.11 | 1.30× | 9.22 |

### Section A — offline fidelity

All RocketKV runs report **key RMSE = 0**, **attention RMSE = 0**, and **attention cosine identical to identity** at each context length. Values stay at full FP16 precision; savings come from **fewer tokens**, not vector quantization.

Merged CSV: `phase5_modal_rocketkv_20260705T155401Z.csv` (in local bundle).

---

## 5. Cross-method summary @ ctx=512

| Method | Representative config | PPL | vs baseline | Memory savings | tok/s |
|---|---|---:|---:|---:|---:|
| Identity | — | 14.11 | 1.0× | 1.0× | 13.85 |
| TurboQuant | tq_full_b4 | 18.64 | 1.3× | 3.1× | 0.08 |
| QJL | qjl_default | 532,967 | ~37,800× | 1.9× | 0.35 |
| RocketKV | r75 | 11,829,014 | ~838,000× | 1.3× | 9.22 |

---

## 6. Evaluation protocol

- **Section A (offline):** single forward pass; tensor RMSE, attention score error, memory from full-layer compress.
- **Section B (online):** incremental compressed KV in autoregressive loop; sliding-window perplexity (stride 512).
- **RocketKV online:** stage-1 token filter + stage-2 HSA via `framework/rocketkv_online.py` (Qwen3 eager attention patch).
- **QJL online:** approximate key **reconstruction** in the forward pass; Section A attention uses the QJL **estimator** (more faithful than online decompress).
- **Baseline:** single shared identity run; not re-run per method. See `configs/modal_sweeps.yaml` preset `baseline`.

---

## 7. Findings (plain language)

### TurboQuant — best quality–compression tradeoff

**4-bit full TurboQuant (`tq_full_b4`)** is the stand-out: at ctx=512, perplexity stays close to baseline (~18.6 vs ~14.1, about 1.3× worse) with ~3× memory savings. Trade-off: online inference is very slow (~0.08 tok/s vs ~14 for identity) due to per-step compress/decompress.

**Anomaly:** 2-bit TurboQuant at short context (128) gives catastrophic PPL (~19k); 4-bit at ctx=128 is near baseline (~18). Quality stabilizes at ctx=512.

### QJL — moderate memory, severe online quality loss

~1.9× memory savings. Section A shows clear key error (RMSE ~6.7). Section B perplexity explodes (hundreds of thousands vs ~14 baseline) because online eval uses reconstructed keys, not the QJL attention estimator used in Section A.

### RocketKV — fastest compressed method, worst perplexity

Throughput is near baseline (~8–18 tok/s) — unique among compressed methods. Memory savings are modest (~1.2–3.3×) because tokens are dropped, not quantized. Perplexity is catastrophic (millions vs ~14).

**Unique split:** Section A looks perfect (0 RMSE, same attention cosine as identity) while Section B fails badly. Offline metrics measure full-precision vectors; the cost is **missing tokens**, which Section B captures but Section A does not.

### Comparative takeaway

Only **TurboQuant 4-bit at ctx≥256** lands in a paper-plausible range for quality-aware comparison (~1.3–1.5× baseline PPL with ~3× compression). QJL and RocketKV show large online perplexity degradation in this benchmark under the current online paths.

---

## 8. Reproducing locally

```bash
# Fetch raw JSON from Modal volume (gitignored output)
bash scripts/modal_fetch_results.sh

# Re-merge a method subset
modal run modal_app/sweep.py::merge_local \
  --input-dir results/modal_volume/qjl \
  --output phase5_modal_qjl \
  --label-prefixes qjl_default
```

See [CURRENT_STATE.md](CURRENT_STATE.md) and [MODAL_GPU_EVAL_DESIGN.md](MODAL_GPU_EVAL_DESIGN.md) for full setup.
