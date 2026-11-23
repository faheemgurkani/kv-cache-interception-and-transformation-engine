# Known Limits

Scope and caveats for the **KV Cache Interception and Transformation Engine** (compression analysis / benchmark). The evaluation protocol and paper use the name **KVBench**. Setup: [README](../README.md). Architecture: [SYSTEM_DESIGN.md](../architecture/SYSTEM_DESIGN.md).

## Research scope

> **Status source of truth:** [`RESEARCH_REDESIGN_PLAN.md` → Part 0](../RESEARCH_REDESIGN_PLAN.md#part-0--reframe-adoption-and-master-status-2026-11-24). Research framing (adopted 2026-11-24): [`RESEARCH_REFRAME_AND_PRIOR_ART.md`](../RESEARCH_REFRAME_AND_PRIOR_ART.md). This page lists known limits only.

| Status | Item |
|---|---|
| ✅ | Unified KV interception + plug-in compressors + FIDELITY/BEHAVIOR/SYSTEM eval (supporting artifact under the reframe) |
| ✅ | Phase 6–7 controlled conditions + Phase 9 Pareto (`eval/pareto/`, `scripts/analyze_pareto.py`) |
| ✅ | Compression taxonomy (Phase 4) + SnapKV + Palu plug-ins |
| ✅ | Engine support for 5 attention classes: `olmo2_1b` (MHA), `qwen3_0.6b` (GQA), `gemma3_270m` (MQA + sliding/global), `tinydeepseek_0.5b` (MLA, expanded KV; **Gate C fails by design**), `falcon_h1_0.5b` (Mamba-2 hybrid; FIDELITY includes a **recurrent** sub-metric). "Supported" = gates + reference tests pass, **not** valid results. See [MODEL_ARCHITECTURE_MATRIX.md](../architecture/MODEL_ARCHITECTURE_MATRIX.md) |
| ⚠️ | **No valid measured results yet.** The Oct-2026 audit invalidated the legacy Qwen3-1.7B / OLMo2-1B Phase-5 BEHAVIOR/SYSTEM numbers (except TurboQuant key metrics and the `wht_quant` ablation) and the Gemma3 18-job bundle. Fixes committed (#201–#204); Modal verification pending (R2). See [GEMMA3_POST_OVERHAUL_AUDIT.md](../audits/GEMMA3_POST_OVERHAUL_AUDIT.md) |
| ❌ | Context ≥ 2–4K (configs cap at 512) — required by the reframe (R5 / E2) |
| ❌ | GDN hybrid model (e.g. Qwen3.5 sub-2B) — required by the reframe (R6 / E3) |
| ❌ | Native MLA latent (TinyDeepSeek Gate C) — required for hypothesis H3 (R7 / E4) |
| ❌ | Rank statistics with CIs in `eval/cross_dim/` (Pearson only today) — R8 / E5 |
| 🔜 | More algorithms (KIVI, AdaKV); external benchmarks beyond a RULER/NIAH-style task |
| 📝 | Paper rewrite to the reframe (R16) — after the clean grid |

## Implementation limits

| Topic | Limit |
|---|---|
| **KIVI** | Stub only (`NotImplementedError`) |
| **SnapKV** | Prefill-only eviction; standard attention on retained FP16 K/V; no decode-time eviction |
| **Palu** | G-LRD latent cache via truncated SVD; full fused RoPE Triton kernel not yet ported — reconstructs K/V at decompress |
| **QJL BEHAVIOR** | Literature ProdQJL online (`framework/qjl_online.py`): float `S q`, sign only on keys; FIDELITY uses the same estimator via `attention_fidelity`. The August 2026 re-sweep bundles (`results/*_qjl_prodqjl/`, `phase5_modal_qjl`, `olmo2_phase5_qjl`) are **invalid** (audit B3 double-append, B8 decode transpose); historical only. |
| **RocketKV** | Token budgets `r256`/`r512`/`r1024` + online HSA. Legacy PPL (~7–11M @ ctx=512) and `generate()`-based metrics predate audit fixes B2/B6 (causality, decode positions) — not trustworthy |
| **TurboQuant `full` stage** | All legacy `tq_full_b*` value metrics and PPL are invalid (audit B8, residual decode bug); `wht_quant` and key metrics unaffected. The old "2-bit @ ctx=128 anomaly" note predates B8 and should be re-checked |
| **TurboQuant online speed** | ~0.08 tok/s @ ctx=512 (per-step compress/decompress) |
| **Modal WHT** | Scipy fallback; no CUDA `fast-hadamard-transform` |
| **Attention** | `attn_implementation="eager"` required — FlashAttention breaks KV intercept |
| **Baseline eval order** | Baseline PPL runs before RocketKV attention patch (`eval/runner.py`) |
| **FIDELITY vs BEHAVIOR** | Whether FIDELITY predicts BEHAVIOR is an open, **conditional** question (hypothesis H4); published work reports both outcomes (KV-CoRE, KVDiagnosis vs Pitfalls, *Does Accuracy Equal Evidence?*). Do not claim a blanket "does not predict" |
| **BEHAVIOR retrieval/reasoning/instruction-following** | Synthetic, in-repo generators (needle-in-haystack, add/subtract chains, yes/no format compliance) — not scraped benchmarks, so no license/contamination risk, but also not LongBench/RULER/MMLU-scale coverage yet |
| **SYSTEM peak VRAM / GPU utilization** | **CUDA:** allocator peaks via `torch.cuda.max_memory_*`. **MPS:** `torch.mps.current_allocated_memory` / `driver_allocated_memory` (polled peak). **CPU:** process RSS peak via `psutil`. GPU util uses NVML on CUDA; **MPS/CPU fall back to process CPU %** during inference (`eval/system/device_metrics.py`). |
| **SYSTEM kernel_cost for RocketKV** | Timed wrapper now includes `compress_layer_from_kv` and layer `decompress` (no longer reads as pure attention time). |

Raw job JSON: `results/` (gitignored). Version-controlled numbers: [Qwen3-1.7B](../results/qwen3_1.7b/PHASE5_EVAL_RESULTS.md) · [OLMo2-1B](../results/olmo2_1b/PHASE5_EVAL_RESULTS.md) · [5-model shortlist](../results/shortlist_5model_eval/). Methodology: [METHODOLOGY.md](METHODOLOGY.md).
