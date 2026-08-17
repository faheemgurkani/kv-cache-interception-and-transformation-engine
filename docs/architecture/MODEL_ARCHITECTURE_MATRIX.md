# Model Architecture Matrix

Which models this engine has been pointed at, what their KV-cache architecture actually is, and what it took (or would take) to get each one working under FIDELITY/BEHAVIOR/SYSTEM. This is the index/summary doc — the deep, live-probed evidence lives in the two documents this page ties together:

- **[`models/ARCHITECTURE_REPORT.md`](../../models/ARCHITECTURE_REPORT.md)** — per-model deep probe: measured params/dtypes, real attention-module/RoPE-module internals, live cache-layer class and tensor shapes, and whether `framework/model_adapter.py`/`framework/kv_cache.py` actually accept each model.
- **[`../results/shortlist_5model_eval/EVAL_FRAMEWORK_CORRESPONDENCE.md`](../results/shortlist_5model_eval/EVAL_FRAMEWORK_CORRESPONDENCE.md)** — the live companion: what actually happens when `scripts/run_eval.py` is run against each shortlisted model, with real numbers for the ones that work and exact tracebacks for the ones that don't.
- **[`ENGINE_INTERNALS.md §8`](ENGINE_INTERNALS.md#8-diversifying-to-other-architecture-families)** — the general tier-based engineering framework (Tier 0–3) for adding a new architecture family, with the current shortlist as the worked examples.
- **[`SLM_COMPATIBILITY.md`](SLM_COMPATIBILITY.md)** — the original 6-candidate probe (Qwen3-1.7B, OLMo2-1B, Granite, Gemma3, MiniCPM4, Qwen3.5) that preceded the current shortlist decision; historical record, not the current model set.

## The two live model families

**`models/legacy/`** — the two models existing published FIDELITY/BEHAVIOR/SYSTEM results (TurboQuant/QJL/RocketKV) actually ran against:

| Model | Architecture | Role |
|---|---|---|
| `qwen3_1.7b` | GQA (16Q/8KV) | Primary/default model, `configs/model.yaml`. All Phase 5 published numbers: [`../results/qwen3_1.7b/`](../results/qwen3_1.7b/). |
| `olmo1b` | MHA (16Q/16KV), pre-norm, tied embeddings, 2024 generation | Was the shortlist's MHA slot until the 2026-08-20 swap in favor of the newer `olmo2_1b`; kept here for reproducibility. |

**`models/` (top level)** — the 5-model architecture-matrix shortlist, one representative per KV-cache family:

| # | Model | Family | Status |
|---|---|---|:---:|
| 1 | `olmo2_1b` | MHA (16Q/16KV) | ✅ fully supported (existing `olmo2` adapter branch) |
| 2 | `qwen3_0.6b` | GQA (16Q/8KV) | ✅ fully supported (existing `qwen3` adapter branch) |
| 3 | `gemma3_270m` | MQA (4Q/1KV) + alternating local/global attention | ❌ blocked — adapter gate + per-layer RoPE needed |
| 4 | `falcon_h1_0.5b` | Hybrid: attention (8Q/2KV GQA) + Mamba2, combined per layer | ❌ blocked — adapter gate + hybrid-cache abstraction needed for correctness |
| 5 | `tinydeepseek_0.5b` | MLA (native `deepseek_v3`, `kv_lora_rank=256`) | ❌ blocked — adapter gate only (loads fine; see correction in `models/ARCHITECTURE_REPORT.md`) |

Both `olmo2_1b` and `qwen3_0.6b` were confirmed end to end via a real `scripts/run_eval.py --compressor identity --context-length 128` run (FIDELITY + BEHAVIOR/task_quality + SYSTEM/latency_throughput all completed) — see the eval-framework correspondence doc linked above for the actual numbers.

## Why OLMo2 (and OLMo-1) work with zero adapter changes — implementation history

This is the historical record of what it originally took to wire up the `olmo2` family (kept for context — this is exactly the kind of patch a *new* family, e.g. `gemma3_text`/`falcon_h1`/`deepseek_v3`, would also need):

| File | Change |
|---|---|
| `framework/model_adapter.py` | New `model_type == "olmo2"` branch in `load_attention_ops`, importing `apply_rotary_pos_emb`/`eager_attention_forward`/`ALL_ATTENTION_FUNCTIONS` from `transformers.models.olmo2.modeling_olmo2`; `qk_norm_layout="flat"` (OLMo2's Q/K-norm is applied to the whole projection before reshaping into heads, unlike Qwen3's per-head layout); `has_input_layernorm=False` (OLMo2 is post-norm — no `input_layernorm` before attention). |
| `framework/qjl_online.py` / `framework/rocketkv_online.py` | No OLMo2-specific code — both already go through `load_attention_ops`/`project_qkv`, so adding the adapter branch was sufficient for QJL/RocketKV online too. |
| `eval/fidelity/attention.py` | No structural change needed — OLMo2 uses one global RoPE table like Qwen3, so the existing single `rotary_emb(...)` call is correct (unlike Gemma3, which needs per-layer RoPE selection — §8.4 in `ENGINE_INTERNALS.md`). |
| `configs/model.yaml` / a dedicated `configs/model_olmo2_1b.yaml` | Model path + name pointing at the OLMo2 checkpoint. |

**Algorithm coupling (historical, both offline/FIDELITY and online/BEHAVIOR+SYSTEM confirmed working):**

| Algorithm | Offline (FIDELITY) | Online (BEHAVIOR/SYSTEM) | Notes |
|---|:---:|:---:|---|
| Identity | ✓ | ✓ | Stock eager path |
| TurboQuant | ✓ | ✓ | `d=128` pow2; decompress → stock attn |
| QJL | ✓ | ✓ | Adapter: flat QK-norm + `modeling_olmo2` RoPE |
| RocketKV | ✓ | ✓ | Same adapter; MHA (`n_rep=1`); no sliding-window kwargs |
| KIVI | Stub | N/A | Not implemented |

**Modal bring-up (historical commands used):**

```bash
bash scripts/modal_setup_model.sh
bash scripts/modal_run_sweep_baseline.sh
PRESET=turboquant OUTPUT=olmo2_phase5_turboquant bash scripts/modal_run_sweep.sh
bash scripts/modal_run_sweep_qjl.sh
bash scripts/modal_run_sweep_rocketkv.sh
bash scripts/modal_fetch_results.sh
python scripts/restructure_olmo2_modal_results.py
```

Also touched: `configs/modal.yaml` (volumes `kv-engine-olmo2`/`kv-engine-results-olmo2`), `modal_app/worker.py` (dynamic `_model_dir()` from `local_path` basename).

**Phase-5 status (historical):** 27/27 Modal jobs OK (Identity×3 + TurboQuant×12 + QJL×3 + RocketKV×9) — full results in [`../results/olmo2_1b/`](../results/olmo2_1b/).

## Engine adoption and transformation plan (ranked)

Full technical detail per item: `ENGINE_INTERNALS.md §8` (code-level proposals) and `models/ARCHITECTURE_REPORT.md` (live evidence per model). Summary, ranked by how much of the shortlist each change unblocks:

1. **`load_attention_ops` branches for `gemma3_text`, `falcon_h1`, `deepseek_v3`.** Confirmed live as the actual point of failure for all 3 blocked models — each fails inside `eval/fidelity/attention.py::evaluate_attention_fidelity` with the same `NotImplementedError`. Falcon-H1 additionally needs a new `qk_norm_layout="none"` (no Q/K-norm exists on `FalconH1Attention`). Because FIDELITY always runs first and aborts the whole evaluation on failure, this single change is also what's blocking BEHAVIOR/SYSTEM from ever being reached for any of the three — even though `iter_layer_kv` already succeeds standalone for all of them.
2. **Per-layer RoPE selection**, needed in addition to (1) for Gemma3 specifically — `Gemma3RotaryEmbedding` genuinely holds two independent buffer sets (`sliding_attention_*`/`full_attention_*`), confirmed live; the engine's single global `rotary_emb(...)` call needs to become a per-`layer_type` selection.
3. **A real hybrid-cache abstraction in `framework/kv_cache.py`**, needed in addition to (1) for Falcon-H1 to be *correct*, not just non-crashing — its cache layer silently exposes only the attention half of a dual attention+Mamba state today, so naive compression would quietly under-count memory rather than fail loudly.
4. **A latent-KV (MLA) state type**, same abstraction direction as (3), needed for TinyDeepSeek to benchmark its actual compressed-latent representation rather than the already-expanded per-head K/V that HF's native `DeepseekV3Attention` currently materializes into the cache.
5. **A `--skip-fidelity`-style flag on `scripts/run_eval.py`** — smaller, independent, newly motivated by actually running the CLI: it would let BEHAVIOR/SYSTEM numbers be collected for the 3 blocked models *before* item 1 lands, useful for triage.

None of items 1–4 are "just add another `if model_type == ...:` branch" in the long run — they match the `ModelAdapterRegistry` / `StateAdapter` / typed-per-layer-state direction already sketched in `ENGINE_INTERNALS.md §8` as the durable shape for this engine as more architecture families are added.
