# OLMo 2 1B — Complete Phase-5 Results, Metrics & Logs

**Generated (UTC):** 2026-08-02T19:37:51.206658+00:00
**Model:** `allenai/OLMo-2-0425-1B` (`Olmo2ForCausalLM`, MHA 16/16, 16 layers, head_dim 128, FP16, eager attn)
**Hardware:** Modal NVIDIA A10G
**Dataset:** WikiText-2 test, batch=1, contexts `{128,256,512}`
**Grid:** Identity×3 + TurboQuant×12 + QJL×3 + RocketKV×9 = **27/27 OK, 0 errors**
**Volumes:** `kv-engine-olmo2` (weights), `kv-engine-results-olmo2` (job JSON)
**Local raw:** `results/modal_volume_olmo2/`
**Bundles:** `results/olmo2_phase5_{baseline,turboquant,qjl,rocketkv}/`
**Inventory CSV:** `results/olmo2_phase5_inventory.csv`
**Paper:** `docs/research_paper_writeup/conference_101719.tex` §\ref{sec:olmo2}

## Completeness checklist

| Check | Status |
|---|---|
| Remote Modal result files | 27 ok / 0 error |
| Local fetched JSON | 27 |
| Expected filename grid | 27/27 |
| Section A (tensor/attn/memory) present | PASS |
| Section B (PPL + throughput) present | PASS |
| Per-layer attention (16 layers) | PASS |
| Timestamps + model_name stamped | PASS |

## Metric catalog (every job)

### Section A — offline fidelity
- `key_rmse`, `value_rmse` (tensor reconstruction)
- Attention: `mse`, `rmse`, `cosine_similarity`, `max_error`, `per_layer[16]`
- Memory: `uncompressed_bytes`, `compressed_bytes`, `shared_metadata_bytes`, `compression_ratio`, `effective_bits_per_kv_element`, `process_memory_mb`

### Section B — online inference
- `perplexity`, `perplexity_baseline` (sliding-window WikiText-2)
- Throughput: `tokens_per_second`, `latency_ms_per_token`, `generated_tokens` (=64), `elapsed_seconds`, `online_compressed_kv`

### Job metadata / logs
- `label`, `compressor`, `bitwidth`, `stage`, `context_length`, `job` kwargs
- `model_name`, `model_path`, `started_at`, `finished_at`, `status`

## Master results table (all 27 jobs)

| Label | T | K RMSE | V RMSE | Attn cos | Mem× | Bits/KV | PPL | Base PPL | PPL× | tok/s | ms/tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| identity_baseline | 128 | 0.000 | 0.000 | 1.000 | 1.00 | 16.00 | 10.99 | 10.99 | 1.00 | 38.605 | 25.9 |
| identity_baseline | 256 | 0.000 | 0.000 | 0.999 | 1.00 | 16.00 | 12.22 | 12.23 | 1.00 | 32.399 | 30.9 |
| identity_baseline | 512 | 0.000 | 0.000 | 0.994 | 1.00 | 16.00 | 8.31 | 8.31 | 1.00 | 22.486 | 44.5 |
| qjl_default | 128 | 2.156 | 0.000 | 0.696 | 1.84 | 8.69 | 716.72 | 10.99 | 65.19 | 1.531 | 653.2 |
| qjl_default | 256 | 2.159 | 0.000 | 0.696 | 1.85 | 8.66 | 841.44 | 12.23 | 68.81 | 1.124 | 889.8 |
| qjl_default | 512 | 2.157 | 0.000 | 0.729 | 1.85 | 8.64 | 873.44 | 8.31 | 105.06 | 0.361 | 2,770.94 |
| rocketkv_r1024 | 128 | 0.000 | 0.000 | 1.000 | 1.00 | 16.06 | 10.99 | 10.99 | 1.00 | 31.349 | 31.9 |
| rocketkv_r1024 | 256 | 0.000 | 0.000 | 0.999 | 1.00 | 16.06 | 12.22 | 12.23 | 1.00 | 28.518 | 35.1 |
| rocketkv_r1024 | 512 | 0.000 | 0.000 | 0.994 | 1.00 | 16.06 | 8.31 | 8.31 | 1.00 | 23.839 | 41.9 |
| rocketkv_r256 | 128 | 0.000 | 0.000 | 1.000 | 1.00 | 16.06 | 10.99 | 10.99 | 1.00 | 30.874 | 32.4 |
| rocketkv_r256 | 256 | 0.000 | 0.000 | 0.999 | 1.00 | 16.06 | 12.22 | 12.23 | 1.00 | 17.006 | 58.8 |
| rocketkv_r256 | 512 | 0.000 | 0.000 | 0.759 | 1.99 | 8.03 | 8.77 | 8.31 | 1.05 | 14.761 | 67.7 |
| rocketkv_r512 | 128 | 0.000 | 0.000 | 1.000 | 1.00 | 16.06 | 10.99 | 10.99 | 1.00 | 19.908 | 50.2 |
| rocketkv_r512 | 256 | 0.000 | 0.000 | 0.999 | 1.00 | 16.06 | 12.22 | 12.22 | 1.00 | 37.972 | 26.3 |
| rocketkv_r512 | 512 | 0.000 | 0.000 | 0.994 | 1.00 | 16.06 | 8.31 | 8.31 | 1.00 | 13.893 | 72.0 |
| tq_full_b2 | 128 | 0.654 | 0.431 | 0.948 | 5.12 | 3.13 | 34.21 | 10.99 | 3.11 | 0.498 | 2,009.30 |
| tq_full_b2 | 256 | 0.659 | 0.408 | 0.950 | 5.12 | 3.13 | 36.62 | 12.23 | 2.99 | 0.263 | 3,797.15 |
| tq_full_b2 | 512 | 0.662 | 0.391 | 0.955 | 5.12 | 3.13 | 26.56 | 8.31 | 3.20 | 0.146 | 6,865.71 |
| tq_full_b3 | 128 | 0.268 | 0.175 | 0.991 | 3.88 | 4.13 | 12.44 | 10.99 | 1.13 | 0.495 | 2,022.10 |
| tq_full_b3 | 256 | 0.270 | 0.165 | 0.991 | 3.88 | 4.13 | 13.50 | 12.23 | 1.10 | 0.263 | 3,797.86 |
| tq_full_b3 | 512 | 0.271 | 0.158 | 0.988 | 3.88 | 4.13 | 9.33 | 8.31 | 1.12 | 0.096 | 10,424.80 |
| tq_full_b4 | 128 | 0.122 | 0.079 | 0.998 | 3.12 | 5.13 | 11.16 | 10.99 | 1.01 | 0.487 | 2,051.84 |
| tq_full_b4 | 256 | 0.123 | 0.074 | 0.998 | 3.12 | 5.13 | 12.31 | 12.23 | 1.01 | 0.231 | 4,338.28 |
| tq_full_b4 | 512 | 0.123 | 0.071 | 0.993 | 3.12 | 5.13 | 8.51 | 8.31 | 1.02 | 0.141 | 7,106.43 |
| tq_mse_b4 | 128 | 0.122 | 0.050 | 0.998 | 3.55 | 4.50 | 11.10 | 10.99 | 1.01 | 0.386 | 2,591.50 |
| tq_mse_b4 | 256 | 0.123 | 0.047 | 0.998 | 3.56 | 4.50 | 12.24 | 12.23 | 1.00 | 0.309 | 3,232.14 |
| tq_mse_b4 | 512 | 0.123 | 0.045 | 0.993 | 3.56 | 4.50 | 8.50 | 8.31 | 1.02 | 0.166 | 6,016.29 |

## Identity baseline — run log

| File | Started (UTC) | Finished (UTC) | Status |
|---|---|---|---|
| `identity_baseline_ctx128_bna_na.json` | 2026-07-31T21:36:41.695318+00:00 | 2026-07-31T21:36:55.596941+00:00 | ok |
| `identity_baseline_ctx256_bna_na.json` | 2026-07-31T21:36:42.919687+00:00 | 2026-07-31T21:37:05.135270+00:00 | ok |
| `identity_baseline_ctx512_bna_na.json` | 2026-07-31T21:36:42.492536+00:00 | 2026-07-31T21:37:15.937382+00:00 | ok |

## TurboQuant — run log

| File | Started (UTC) | Finished (UTC) | Status |
|---|---|---|---|
| `tq_full_b2_ctx128_b2_full.json` | 2026-07-31T21:41:00.881209+00:00 | 2026-07-31T21:45:08.941775+00:00 | ok |
| `tq_full_b2_ctx256_b2_full.json` | 2026-07-31T21:41:03.444041+00:00 | 2026-07-31T21:52:47.058859+00:00 | ok |
| `tq_full_b2_ctx512_b2_full.json` | 2026-07-31T21:45:14.595259+00:00 | 2026-07-31T22:20:10.666303+00:00 | ok |
| `tq_full_b3_ctx128_b3_full.json` | 2026-07-31T21:41:00.490749+00:00 | 2026-07-31T21:45:13.996830+00:00 | ok |
| `tq_full_b3_ctx256_b3_full.json` | 2026-07-31T21:41:05.563792+00:00 | 2026-07-31T21:52:41.553058+00:00 | ok |
| `tq_full_b3_ctx512_b3_full.json` | 2026-07-31T21:47:02.697889+00:00 | 2026-07-31T22:40:19.962206+00:00 | ok |
| `tq_full_b4_ctx128_b4_full.json` | 2026-07-31T21:42:55.442536+00:00 | 2026-07-31T21:47:14.903797+00:00 | ok |
| `tq_full_b4_ctx256_b4_full.json` | 2026-07-31T21:43:47.476281+00:00 | 2026-07-31T21:56:56.763149+00:00 | ok |
| `tq_full_b4_ctx512_b4_full.json` | 2026-07-31T21:41:02.578196+00:00 | 2026-07-31T22:17:32.707705+00:00 | ok |
| `tq_mse_b4_ctx128_b4_wht_quant.json` | 2026-07-31T21:41:29.714987+00:00 | 2026-07-31T21:47:02.098429+00:00 | ok |
| `tq_mse_b4_ctx256_b4_wht_quant.json` | 2026-07-31T21:41:01.547400+00:00 | 2026-07-31T21:51:15.203894+00:00 | ok |
| `tq_mse_b4_ctx512_b4_wht_quant.json` | 2026-07-31T21:45:09.802101+00:00 | 2026-07-31T22:16:14.153665+00:00 | ok |

## QJL — run log

| File | Started (UTC) | Finished (UTC) | Status |
|---|---|---|---|
| `qjl_default_ctx128_b1_na.json` | 2026-07-31T21:39:08.363049+00:00 | 2026-07-31T21:40:36.538974+00:00 | ok |
| `qjl_default_ctx256_b1_na.json` | 2026-07-31T21:38:35.789690+00:00 | 2026-07-31T21:41:30.384385+00:00 | ok |
| `qjl_default_ctx512_b1_na.json` | 2026-07-31T21:38:46.592513+00:00 | 2026-07-31T21:52:44.078074+00:00 | ok |

## RocketKV — run log

| File | Started (UTC) | Finished (UTC) | Status |
|---|---|---|---|
| `rocketkv_r1024_ctx128_b1024_hsa1024_ws32.json` | 2026-07-31T21:39:12.036418+00:00 | 2026-07-31T21:39:23.939515+00:00 | ok |
| `rocketkv_r1024_ctx256_b1024_hsa1024_ws32.json` | 2026-07-31T21:39:08.698617+00:00 | 2026-07-31T21:39:26.713118+00:00 | ok |
| `rocketkv_r1024_ctx512_b1024_hsa1024_ws32.json` | 2026-07-31T21:39:07.103599+00:00 | 2026-07-31T21:39:34.416500+00:00 | ok |
| `rocketkv_r256_ctx128_b256_hsa256_ws32.json` | 2026-07-31T21:38:36.578837+00:00 | 2026-07-31T21:38:50.377736+00:00 | ok |
| `rocketkv_r256_ctx256_b256_hsa256_ws32.json` | 2026-07-31T21:38:52.669598+00:00 | 2026-07-31T21:39:10.302873+00:00 | ok |
| `rocketkv_r256_ctx512_b256_hsa256_ws32.json` | 2026-07-31T21:39:16.136722+00:00 | 2026-07-31T21:39:59.783416+00:00 | ok |
| `rocketkv_r512_ctx128_b512_hsa512_ws32.json` | 2026-07-31T21:39:11.844806+00:00 | 2026-07-31T21:39:30.162388+00:00 | ok |
| `rocketkv_r512_ctx256_b512_hsa512_ws32.json` | 2026-07-31T21:39:06.056209+00:00 | 2026-07-31T21:39:36.860753+00:00 | ok |
| `rocketkv_r512_ctx512_b512_hsa512_ws32.json` | 2026-07-31T21:39:05.946164+00:00 | 2026-07-31T21:39:47.018182+00:00 | ok |

## Cross-method @ T=512

| Method | PPL | PPL× | Mem× | tok/s | Attn cos |
|---|---:|---:|---:|---:|---:|
| Identity | 8.31 | 1.00 | 1.00 | 22.49 | 0.994 |
| TurboQuant 4-bit full | 8.51 | 1.02 | 3.12 | 0.14 | 0.993 |
| TurboQuant 4-bit WHT | 8.50 | 1.02 | 3.56 | 0.17 | 0.993 |
| QJL | 873.44 | 105.06 | 1.85 | 0.36 | 0.729 |
| RocketKV B=256 | 8.77 | 1.05 | 1.99 | 14.76 | 0.759 |
| RocketKV B=512 | 8.31 | 1.00 | 1.00 | 13.89 | 0.994 |
| RocketKV B=1024 | 8.31 | 1.00 | 1.00 | 23.84 | 0.994 |

## Artifact index

| Path | Contents |
|---|---|
| `results/modal_volume_olmo2/*.json` | Per-job raw Modal outputs (27) |
| `results/olmo2_phase5_baseline/` | Identity merged CSV/JSON + jobs/ + manifest |
| `results/olmo2_phase5_turboquant/` | TurboQuant merged CSV/JSON + jobs/ + manifest |
| `results/olmo2_phase5_qjl/` | QJL merged CSV/JSON + jobs/ + manifest |
| `results/olmo2_phase5_rocketkv/` | RocketKV merged CSV/JSON + jobs/ + manifest |
| `results/olmo2_phase5_summary.json` | Compact numeric summary for paper sync |
| `results/olmo2_phase5_inventory.csv` | Flat KPI inventory (all metrics) |
| `docs/OLMO2_ARCHITECTURE_PROBE.md` | Architecture coupling notes |
| `docs/OLMO2_PHASE5_EVAL_RESULTS.md` | Condensed tables |
| `docs/research_paper_writeup/conference_101719.tex` | IEEE tables §OLMo~2 |

