# Known Limits

Scope and caveats for the **KV Cache Interception and Transformation Engine** (compression analysis / benchmark). The evaluation protocol and paper use the name **KVBench**. Setup: [README](../README.md). Architecture: [SYSTEM_DESIGN.md](../architecture/SYSTEM_DESIGN.md).

## Research scope

| Status | Item |
|---|---|
| ✅ | Unified KV interception + plug-in compressors + FIDELITY/BEHAVIOR/SYSTEM eval |
| ✅ | Case studies: TurboQuant, QJL, RocketKV on Qwen3-1.7B, replicated on OLMo2-1B |
| ✅ | Compression taxonomy (Phase 4) + SnapKV + Palu plug-ins |
| ⚠️ | WikiText-2, ctx ≤512 per run — sufficient for arXiv framework paper, not top-tier conference breadth |
| 🔜 | More contexts, algorithms (KIVI, AdaKV), benchmarks (LongBench, RULER) |
| ⚠️ | All **5 of 5** shortlist models are fully wired for eval (`olmo2_1b`, `qwen3_0.6b`, `gemma3_270m`, `tinydeepseek_0.5b`, `falcon_h1_0.5b`). **TinyDeepSeek Gate C fails by design** (expanded KV ≠ native latent); all other gates pass. FIDELITY includes a **recurrent** sub-metric for hybrid models (§25). Legacy `qwen3_1.7b`/`olmo1b` unchanged. See [ENGINE_AND_EVALUATION_FRAMEWORKS_REDESIGN_PLAN.md § audit](../ENGINE_AND_EVALUATION_FRAMEWORKS_REDESIGN_PLAN.md#implementation-verification-audit-2026-08-19-hardened), [MODEL_ARCHITECTURE_MATRIX.md](../architecture/MODEL_ARCHITECTURE_MATRIX.md), [results/shortlist_5model_eval/](../results/shortlist_5model_eval/) |

## Implementation limits

| Topic | Limit |
|---|---|
| **KIVI** | Stub only (`NotImplementedError`) |
| **QJL BEHAVIOR** | Literature ProdQJL online (`framework/qjl_online.py`): float `S q`, sign only on keys; FIDELITY uses the same estimator via `attention_fidelity`. QJL re-swept August 2026 (Modal `i220485`); bundles `results/*_qjl_prodqjl/` and canonical `phase5_modal_qjl` / `olmo2_phase5_qjl`. |
| **RocketKV** | Token budgets `r256`/`r512`/`r1024` + online HSA; post-fix PPL still ~7–11M @ ctx=512 |
| **TurboQuant 2-bit @ ctx=128** | Anomalously bad PPL; use ctx≥256 for comparisons |
| **TurboQuant online speed** | ~0.08 tok/s @ ctx=512 (per-step compress/decompress) |
| **Modal WHT** | Scipy fallback; no CUDA `fast-hadamard-transform` |
| **Attention** | `attn_implementation="eager"` required — FlashAttention breaks KV intercept |
| **Baseline eval order** | Baseline PPL runs before RocketKV attention patch (`eval/runner.py`) |
| **FIDELITY vs BEHAVIOR** | FIDELITY metrics do not always predict BEHAVIOR/PPL (by design — framework surfaces the gap) |
| **BEHAVIOR retrieval/reasoning/instruction-following** | Synthetic, in-repo generators (needle-in-haystack, add/subtract chains, yes/no format compliance) — not scraped benchmarks, so no license/contamination risk, but also not LongBench/RULER/MMLU-scale coverage yet |
| **SYSTEM peak VRAM / GPU utilization** | **CUDA:** allocator peaks via `torch.cuda.max_memory_*`. **MPS:** `torch.mps.current_allocated_memory` / `driver_allocated_memory` (polled peak). **CPU:** process RSS peak via `psutil`. GPU util uses NVML on CUDA; **MPS/CPU fall back to process CPU %** during inference (`eval/system/device_metrics.py`). |
| **SYSTEM kernel_cost for RocketKV** | Timed wrapper now includes `compress_layer_from_kv` and layer `decompress` (no longer reads as pure attention time). |

Raw job JSON: `results/` (gitignored). Version-controlled numbers: [Qwen3-1.7B](../results/qwen3_1.7b/PHASE5_EVAL_RESULTS.md) · [OLMo2-1B](../results/olmo2_1b/PHASE5_EVAL_RESULTS.md) · [5-model shortlist](../results/shortlist_5model_eval/). Methodology: [METHODOLOGY.md](METHODOLOGY.md).
