# Architecture Matrix — Downloaded Model Report

Deep-probed 2026-08-21 by loading each checkpoint with `AutoModelForCausalLM.from_pretrained(..., attn_implementation="eager")`, running a real forward pass (`"The quick brown fox jumps"`, `use_cache=True`), and inspecting: measured parameter count + dtype breakdown, the actual decoder-layer/attention-module class and attributes (not just `config.json` field names), the `rotary_emb` module's real buffers, the live `past_key_values` cache-layer class and tensor shapes, **and** whether the repo's own `framework/model_adapter.py::load_attention_ops` / `framework/kv_cache.py::iter_layer_kv` actually accept the model. `transformers==5.8.1` (this repo's `.venv`).

**Code-grounded status (2026-08-18):** 3 of 5 models pass all compatibility gates (`olmo2_1b`, `qwen3_0.6b`, `gemma3_270m`). Full audit: [`docs/ENGINE_AND_EVALUATION_FRAMEWORKS_REDESIGN_PLAN.md` § audit](../docs/ENGINE_AND_EVALUATION_FRAMEWORKS_REDESIGN_PLAN.md#implementation-verification-audit-2026-08-18).

## Directory layout

- `models/legacy/` — `qwen3_1.7b` (primary/default, `configs/model.yaml`) and `olmo1b` (`allenai/OLMo-1B-hf`, 2024 OLMo-1 generation) — the models the existing published FIDELITY/BEHAVIOR/SYSTEM results were run against, kept for reproducibility. Not covered in the tables below (unchanged since the OLMo slot swap; conformance detail was recorded in the previous version of this report and in `docs/methodology/CURRENT_STATE.md`/`docs/architecture/SLM_COMPATIBILITY.md`).
- `models/` (top level) — the 5-model architecture-matrix shortlist, deep-probed below: `olmo2_1b` (MHA), `qwen3_0.6b` (GQA), `gemma3_270m` (MQA + local/global), `tinydeepseek_0.5b` (MLA), `falcon_h1_0.5b` (Hybrid attention + Mamba2).
- Deleted entirely: `granite_4.0_350m`, `qwen3.5_0.8b`, `minicpm4_0.5b` — superseded candidate-probe downloads; findings remain only in `docs/architecture/SLM_COMPATIBILITY.md`/`docs/methodology/CURRENT_STATE.md` as historical record, not present on disk.

---

## 1. `olmo2_1b` — `allenai/OLMo-2-0425-1B` — MHA

**Measured:** 1,484,916,736 params (100% float32) · `Olmo2ForCausalLM` · tokenizer `GPT2Tokenizer` (BPE), vocab 100278 · 16 layers, `hidden_size=2048`.

**Decoder layer (`Olmo2DecoderLayer`):** no `input_layernorm` before attention — normalization is `post_attention_layernorm` / `post_feedforward_layernorm`, i.e. genuine **post-norm** (RMSNorm applied *after* the attention/MLP sublayer output, not before). This is the one substantive architectural difference from OLMo-1/most Llama-family models.

**Attention (`Olmo2Attention`):** `head_dim=128`, has both `q_norm` **and** `k_norm` — but layout is **flat**: RMSNorm is applied to the *entire* `q_proj`/`k_proj` output (shape `[..., hidden]`) before reshaping into heads, not per-head after reshaping (contrast with Qwen3/Gemma3 below).

**RoPE (`Olmo2RotaryEmbedding`):** single global `inv_freq`/`attention_scaling` pair — one RoPE table for the whole model, no per-layer-type split.

**Live cache:** 16 `DynamicLayer`s, every layer keys/values shape `(1, 16, 5, 128)` — 16 K-heads == 16 V-heads == `num_attention_heads`, confirming **true MHA**.

**Engine correspondence — ✅ FULLY SUPPORTED, zero changes needed:**
- `load_attention_ops(config)` → **succeeds** (`model_type="olmo2"` branch already exists, `qk_norm_layout="flat"`, `has_input_layernorm=False` — correctly encodes the post-norm layout above).
- `iter_layer_kv` → succeeds (standard `DynamicLayer` cache).
- FIDELITY/attention's single global `rotary_emb(...)` call is correct here (one RoPE table).
- `enable_qjl_online`/`enable_rocketkv_online` will work (both just call `load_attention_ops`).
- This is a second **zero-engine-modification control**, alongside Qwen3.

---

## 2. `qwen3_0.6b` — `Qwen/Qwen3-0.6B` — GQA

**Measured:** 596,049,920 params (100% bfloat16) · `Qwen3ForCausalLM` · tokenizer `Qwen2Tokenizer` (BPE), vocab 151643 · 28 layers, `head_dim=128`.

**Decoder layer (`Qwen3DecoderLayer`):** has `input_layernorm` → **pre-norm**, standard Llama-style.

**Attention (`Qwen3Attention`):** `q_norm`/`k_norm` present, layout **per_head**: projection is reshaped into `[B, H, T, D]` first, then RMSNorm is applied per-head over the `head_dim` axis — the opposite order from OLMo2's flat layout.

**RoPE (`Qwen3RotaryEmbedding`):** single global `inv_freq`/`attention_scaling`. `config.layer_types` is present but degenerate — all 28 entries are `"full_attention"` (no sliding window active in this checkpoint despite the field existing).

**Live cache:** 28 `DynamicLayer`s, every layer keys/values shape `(1, 8, 5, 128)` — 8 KV heads vs. presumably-16 Q heads (config confirms 16Q/8KV) → real 2:1 **GQA** grouping.

**Engine correspondence — ✅ FULLY SUPPORTED, zero changes needed:**
- `load_attention_ops` → succeeds (`model_type="qwen3"`, `qk_norm_layout="per_head"`, `has_input_layernorm=True`, `passes_sliding_window=True` — correctly anticipates a sliding-window field even though this checkpoint doesn't use it).
- `iter_layer_kv` → succeeds.
- Already the engine's **primary reference family** — same code path as the legacy `qwen3_1.7b`, just a smaller checkpoint. This is the shortlist's zero-engine-modification GQA control.

---

## 3. `tinydeepseek_0.5b` — `FreedomIntelligence/TinyDeepSeek-0.5B-base` — MLA

**Correction (2026-08-21, superseding the finding below from the original probe):** loading with `trust_remote_code=True` (as this section's original probe did, to inspect the vendored code directly) fails:

```
ImportError: cannot import name 'is_torch_fx_available' from 'transformers.utils.import_utils'
  at .../transformers_modules/tinydeepseek_0_dot_5b/.../modeling_tinydeepseek.py:56
```

The checkpoint ships its own `modeling_tinydeepseek.py`/`configuration_tinydeepseek.py`, and that vendored file imports a symbol `transformers==5.8.1` no longer exposes. There is also a self-reported type mismatch warning: *"You are using a model of type `deepseek_v3` to instantiate a model of type `tinydeepseek_v3`"* — the config's declared `model_type` doesn't match the vendored class's own self-identification.

**But the engine's actual load path (`framework/model.py::ModelLayer.__init__`) never passes `trust_remote_code`** — and run through that real path, `AutoModelForCausalLM.from_pretrained` **loads successfully** in ~29s, resolving to transformers' **native** `DeepseekV3ForCausalLM` (since `model_type="deepseek_v3"` matches a real, already-supported transformers architecture) rather than the broken vendored code. The import error above is never reached by the engine. Confirmed live via `ModelLayer()`:

- `type(model)` → `DeepseekV3ForCausalLM`; `type(layer0.self_attn)` → `DeepseekV3Attention`, exposing genuine MLA submodules: `kv_a_layernorm`, `kv_a_proj_with_mqa`, `kv_b_proj`, `kv_lora_rank=256`, `q_lora_rank`, `qk_head_dim`, `qk_nope_head_dim=32`, `qk_rope_head_dim=32`, `v_head_dim=32`.
- Measured params: 529,788,416.
- Real forward pass succeeds; cache is 26 plain `DynamicLayer`s. Layer 0 keys shape `(1, 4, 5, 64)`, values shape `(1, 4, 5, 32)` — note the **asymmetric key/value last dimension** (64 = `qk_nope_head_dim + qk_rope_head_dim`, 32 = `v_head_dim`), a structural difference from every other model in this shortlist, where key and value share one `head_dim`.
- `iter_layer_kv` **succeeds** (26 layers iterated) — the generic engine path already extracts this model's cache without error.

Full live-run evidence (exact commands, output) is in [`docs/results/shortlist_5model_eval/EVAL_FRAMEWORK_CORRESPONDENCE.md`](../docs/results/shortlist_5model_eval/EVAL_FRAMEWORK_CORRESPONDENCE.md).

**Engine correspondence — ✅ FULLY SUPPORTED on expanded-cache path (94th commit); Gate C fails by design (native latent not exposed):**
- `load_attention_ops` → **succeeds** (`model_type="deepseek_v3"`, `qk_norm_layout="mla"`). FIDELITY/attention uses `project_attention_states()` / `project_mla_qkv()` for split nope/RoPE with asymmetric K/V (`D_k=64`, `D_v=32`).
- `iter_layer_kv` → **succeeds** — FIDELITY/representation+memory, BEHAVIOR, and SYSTEM all run end-to-end. Verified: identity, TurboQuant, QJL, RocketKV (`tests/test_tinydeepseek_reference.py`).
- **Scientific caveat (unchanged):** the cache being iterated is HF's already-*expanded* per-head K/V (post `kv_b_proj`), not the native compressed-latent (`kv_lora_rank=256`) representation. Reports disclose `cache_representation="expanded_kv"`. Gate C fails until MLA-native interception lands.
- The vendored `modeling_tinydeepseek.py`'s import error is real but **irrelevant to this engine** unless something explicitly sets `trust_remote_code=True` in the future — worth remembering if that ever changes.

---

## 4. `falcon_h1_0.5b` — `tiiuae/Falcon-H1-0.5B-Base` — Hybrid Attention + Mamba2

**Measured:** 521,411,104 params (100% bfloat16) · `FalconH1ForCausalLM` · tokenizer is a raw `TokenizersBackend` (not a `transformers`-native tokenizer class — Falcon-H1 ships its own fast-tokenizer JSON), vocab 32768 · 36 layers.

**Decoder layer (`FalconH1DecoderLayer`):** distinct from every other model here — has **both** `self_attn`-style attention (`attention_in_multiplier`, `attn_out_multiplier`, `channels_attn` gating scalars — Falcon-H1-specific mixing weights) **and** a separate `mamba` submodule (`FalconH1Mixer`) on the *same* layer object. `config.layer_types` is `["hybrid"] * 36` — every layer is simultaneously attention **and** Mamba, run in parallel and combined (not alternating attention-layers/Mamba-layers as in some other hybrid designs).

**Attention (`FalconH1Attention`):** `head_dim=64`, **no `q_norm`/`k_norm` at all** — a third QK-norm layout the engine's `AttentionOps.qk_norm_layout` doesn't have a name for (only `"per_head"`/`"flat"` exist). Has its own `key_multiplier` scalar (Falcon-H1's per-head attention/Mamba output mixing scheme).

**RoPE (`FalconH1RotaryEmbedding`):** single global `inv_freq`/`attention_scaling` — despite the hybrid architecture, the attention half uses one shared RoPE table (unlike Gemma3's per-layer-type split), so no per-layer RoPE selection is needed for the attention component specifically.

**Live cache:** 36 layers, but **every layer's cache object is `LinearAttentionAndFullAttentionLayer`**, not a plain `DynamicLayer` — a combined type that (per its name) can carry both a linear/recurrent state and standard attention K/V. In this checkpoint's forward pass it exposed `.keys`/`.values` of shape `(1, 2, 6, 64)` (2 == `num_key_value_heads`, confirming the attention half runs a 4:2 GQA-style ratio — `num_attention_heads=8`, `num_key_value_heads=2`), so `iter_layer_kv`'s generic `layer.keys, layer.values` fallback works for the attention half. **Mamba/SSM state lives on the same cache layer** (`.recurrent_states`/`.conv_states`) and is **invisible to `iter_layer_kv`**, but is now visible to `iter_layer_states()` and counted by `visible_state_bytes()` (WP1, 79th commit).

**Engine correspondence — hybrid state accounted; adapter gate still blocks FIDELITY/attention:**
- `load_attention_ops` → **fails**: `NotImplementedError("... model_type='falcon_h1' ...")`. `project_qkv` already supports `qk_norm_layout="none"`; the builder branch is not yet registered.
- `iter_layer_states` / `visible_state_bytes()` → **counts attention K/V and Mamba state** (WP1). `get_cache_size_bytes()` / `iter_layer_kv()` remain attention-K/V-only for backward compatibility.
- Gate A (loader/state) → **passes**. Gate C (state semantics) → **passes**. Gate B (attention adapter) → **fails**.
- Compression policy remains attention-K/V-only (Mamba passthrough).

---

## 5. `gemma3_270m` — `google/gemma-3-270m` — MQA + local/global attention

**Measured:** 268,098,176 params (100% bfloat16) · `Gemma3ForCausalLM` · tokenizer `GemmaTokenizer`, vocab 262144 (by far the largest vocab in the shortlist) · 18 layers.

**Decoder layer (`Gemma3DecoderLayer`):** has `input_layernorm`, plus *both* `pre_feedforward_layernorm` and `post_feedforward_layernorm` — more norm points than any other model here (a "sandwich" norm pattern around the MLP specifically, on top of standard pre-norm attention).

**Attention (`Gemma3Attention`):** `head_dim=256` (unusually large for a 4-head model — `4 * 256 = 1024` ≠ `hidden_size=640`, i.e. Gemma3's attention inner dimension is decoupled from `hidden_size`, unlike every other model probed here where `heads * head_dim == hidden_size`). Has `q_norm`/`k_norm` (per-head layout, same family as Qwen3), plus Gemma3-specific `attn_logit_softcapping` and a `layer_type`/`is_sliding` attribute **on the attention module itself** — each layer knows its own type directly, not just via a config lookup.

**RoPE (`Gemma3RotaryEmbedding`):** **confirmed dual-table** — the module holds *two independent* buffer sets: `sliding_attention_inv_freq`/`sliding_attention_attention_scaling` and `full_attention_inv_freq`/`full_attention_attention_scaling`. This directly confirms the crash mechanism already documented in `docs/methodology/CURRENT_STATE.md`: `Gemma3RotaryEmbedding.forward` does `getattr(self, f"{layer_type}_inv_freq")`, so calling it without a `layer_type` (as the engine's single global call does) looks up a nonexistent `None_inv_freq` attribute.

`config.layer_types`: alternating `sliding_attention` × 5 then `full_attention` × 1, repeating — period-6 local/global split, 18 layers total (3 full-attention layers, 15 sliding).

**Live cache:** confirms the split is real at the object level, not just a config label — layer 0 (`sliding_attention`) is a `DynamicSlidingWindowLayer`, layer 17 (`full_attention`) is a plain `DynamicLayer`. Both report keys/values shape `(1, 1, 6, 256)` — 1 KV head vs. 4 Q heads confirms **true MQA** (the most extreme KV-sharing ratio in the shortlist).

**Engine correspondence — ✅ FULLY SUPPORTED (86th commit):**
- `load_attention_ops` → **succeeds** (`model_type="gemma3_text"`, `qk_norm_layout="per_head"`, `passes_sliding_window=True`).
- **Per-layer RoPE** → `framework/rope.py::build_rope_context(...).get_rope(layer_idx)` selects `sliding_attention` vs `full_attention` tables per layer; used in `eval/fidelity/attention.py`. QJL/RocketKV online receive correct `(cos, sin)` from the model's native forward pass.
- `iter_layer_kv` → succeeds (both `DynamicSlidingWindowLayer` and `DynamicLayer` expose `.keys`/`.values`).
- All three compatibility gates pass. Full FIDELITY/BEHAVIOR/SYSTEM verified via `tests/test_gemma3_reference.py` (identity, TurboQuant, QJL, RocketKV).
- `get_model_eval_metadata` records per-layer `layer_attention` metadata (attention type, rope type, window size).

---

## Cross-model correspondence summary

| Model | Family | Loads + real forward pass? | `load_attention_ops` | `iter_layer_kv` | FIDELITY/attention | BEHAVIOR+SYSTEM (identity/turboquant) | BEHAVIOR+SYSTEM (qjl/rocketkv) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| olmo2_1b | MHA | ✅ | ✅ (`olmo2`) | ✅ | ✅ | ✅ | ✅ |
| qwen3_0.6b | GQA | ✅ | ✅ (`qwen3`) | ✅ | ✅ | ✅ | ✅ |
| gemma3_270m | MQA + local/global | ✅ | ✅ (`gemma3_text`, 86th commit) | ✅ | ✅ (per-layer RoPE) | ✅ | ✅ |
| falcon_h1_0.5b | Hybrid Attn+Mamba2 | ✅ | ❌ no `falcon_h1` branch | ✅ (attn K/V only) | ❌ (Gate B) | ⚠️ `--skip-fidelity` runs; memory uses full visible bytes | ❌ (Gate B) |
| tinydeepseek_0.5b | MLA | ✅ (native `deepseek_v3`) | ✅ (`deepseek_v3`, 94th commit) | ✅ | ✅ (expanded KV; Gate C fails) | ✅ | ✅ |

**4 of 5 fully work today** (`olmo2_1b`, `qwen3_0.6b`, `gemma3_270m`, `tinydeepseek_0.5b` — confirmed via reference tests and live gate evaluation, 2026-08-19). **`falcon_h1_0.5b`** remains blocked at Gate B only. Falcon-H1 dual-state **accounting** is fixed (Gate C passes). TinyDeepSeek fails Gate C by design until MLA-native latent interception lands (`cache_representation="expanded_kv"` disclosure in reports).

## What would actually need to change in the engine (ranked by how much of the shortlist it unblocks)

1. **`framework/model_adapter.py::load_attention_ops`** — add `model_type == "falcon_h1"` branch. **Done for Gemma3** (`gemma3_text`, 86th commit) **and TinyDeepSeek** (`deepseek_v3`, 94th commit). Falcon-H1 additionally needs registering the existing `qk_norm_layout="none"` path in `project_qkv`.
2. **Per-layer RoPE selection** — **done for Gemma3** (`framework/rope.py::build_rope_context().get_rope(layer_idx)` in FIDELITY/attention; QJL/RocketKV receive per-layer embeddings from the native forward pass).
3. **Hybrid state interface** — **done (WP1):** `iter_layer_states()` + `visible_state_bytes()` count Falcon Mamba state; Gate C passes. Compression remains attention-K/V-only by policy.
4. **A latent-KV (MLA) state type** — still open for TinyDeepSeek: Gate C fails until native `kv_lora_rank` interception lands. Today's cache is HF's expanded per-head K/V reconstruction.
5. **`--skip-fidelity` on `scripts/run_eval.py`** — **implemented (WP1).** Lets BEHAVIOR/SYSTEM run on models where Gate B fails.

Live-run evidence (real tracebacks, successful-run numbers for `olmo2_1b`/`qwen3_0.6b`/`gemma3_270m`) behind every claim above: [`docs/results/shortlist_5model_eval/EVAL_FRAMEWORK_CORRESPONDENCE.md`](../docs/results/shortlist_5model_eval/EVAL_FRAMEWORK_CORRESPONDENCE.md).

The remaining structural gaps (Falcon-H1 adapter; optional MLA-native latent state for TinyDeepSeek Gate C) match the `ModelAdapterRegistry` / typed-state direction in `docs/ENGINE_AND_EVALUATION_FRAMEWORKS_REDESIGN_PLAN.md`.
