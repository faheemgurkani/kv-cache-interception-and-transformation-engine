# Known Limits

Scope and caveats for the **KV-Cache Interception and Transformation Engine** (compression analysis / benchmark). Setup: [README](../README.md). Architecture: [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md).

## Research scope

| Status | Item |
|---|---|
| ✅ | Unified KV interception + plug-in compressors + Section A/B eval |
| ✅ | Case studies: TurboQuant, QJL, RocketKV on Qwen3-1.7B |
| ⚠️ | Single model, WikiText-2, ctx ≤512 — sufficient for arXiv framework paper, not top-tier conference breadth |
| 🔜 | More models, contexts, algorithms (SnapKV, KIVI, AdaKV), benchmarks (LongBench, RULER) |

## Implementation limits

| Topic | Limit |
|---|---|
| **KIVI** | Stub only (`NotImplementedError`) |
| **QJL Section B** | Uses asymmetric QJL attention estimator online (`framework/qjl_online.py`); Section A uses same estimator via `attention_fidelity` |
| **RocketKV** | Token budgets `r256`/`r512`/`r1024` + online HSA; post-fix PPL still ~7–11M @ ctx=512 |
| **TurboQuant 2-bit @ ctx=128** | Anomalously bad PPL; use ctx≥256 for comparisons |
| **TurboQuant online speed** | ~0.08 tok/s @ ctx=512 (per-step compress/decompress) |
| **Modal WHT** | Scipy fallback; no CUDA `fast-hadamard-transform` |
| **Attention** | `attn_implementation="eager"` required — FlashAttention breaks KV intercept |
| **Baseline eval order** | Baseline PPL runs before RocketKV attention patch (`eval/runner.py`) |
| **Section A vs B** | Offline metrics do not always predict online PPL (by design — framework surfaces the gap) |

Raw job JSON: `results/` (gitignored). Version-controlled numbers: [PHASE5_EVAL_RESULTS.md](PHASE5_EVAL_RESULTS.md).
