# Model Architecture Matrix

Which models this engine has been pointed at, what their KV-cache architecture actually is, and what it took (or would take) to get each one working under FIDELITY/BEHAVIOR/SYSTEM. This is the index/summary doc — the deep, live-probed evidence lives in the two documents this page ties together:

- **[`../ENGINE_AND_EVALUATION_FRAMEWORKS_REDESIGN_PLAN.md` § audit](../ENGINE_AND_EVALUATION_FRAMEWORKS_REDESIGN_PLAN.md#implementation-verification-audit-2026-08-18)** — code-grounded verification of §1–2 infrastructure (three pillars, live gates, verified commits 79–94).
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
| 1 | `olmo2_1b` | MHA (16Q/16KV) | ✅ fully supported (all three gates pass) |
| 2 | `qwen3_0.6b` | GQA (16Q/8KV) | ✅ fully supported (all three gates pass) |
| 3 | `gemma3_270m` | MQA (4Q/1KV) + alternating local/global attention | ✅ fully supported (`gemma3_text` adapter + per-layer RoPE; 86th commit) |
| 4 | `falcon_h1_0.5b` | Hybrid: attention (8Q/2KV GQA) + Mamba2, combined per layer | ✅ fully supported (`falcon_h1` adapter + hybrid memory; all gates pass) |
| 5 | `tinydeepseek_0.5b` | MLA (native `deepseek_v3`, `kv_lora_rank=256`) | ⚠️ Gate B pass; Gate C fails (expanded KV, not native latent; 94th commit) |

All five shortlist models were confirmed end to end via reference tests and live gate evaluation (2026-08-19).

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

1. **`load_attention_ops` branches.** ✅ **Done for all five families** (`gemma3_text`, `deepseek_v3`, `falcon_h1`).
2. **Per-layer RoPE selection** — ✅ **done for Gemma3** (`framework/rope.py::build_rope_context().get_rope(layer_idx)` in FIDELITY/attention).
3. **Hybrid state interface** — ✅ **done (WP1):** `iter_layer_states()` + `visible_state_bytes()`. Falcon Gate C passes; Mamba compression remains passthrough by policy.
4. **Latent-KV (MLA) state type** — still open for TinyDeepSeek (Gate C fails until native `kv_lora_rank` interception).
5. **`--skip-fidelity` flag** — ✅ **implemented (WP1).**

None of items 1–4 are "just add another `if model_type == ...:` branch" in the long run — they match the `ModelAdapterRegistry` / `StateAdapter` / typed-per-layer-state direction already sketched in `ENGINE_INTERNALS.md §8` as the durable shape for this engine as more architecture families are added.
