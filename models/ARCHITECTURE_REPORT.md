# Architecture Matrix — Downloaded Model Report

Deep-probed 2026-08-21 by loading each checkpoint with `AutoModelForCausalLM.from_pretrained(..., attn_implementation="eager")`, running a real forward pass (`"The quick brown fox jumps"`, `use_cache=True`), and inspecting: measured parameter count + dtype breakdown, the actual decoder-layer/attention-module class and attributes (not just `config.json` field names), the `rotary_emb` module's real buffers, the live `past_key_values` cache-layer class and tensor shapes, **and** whether the repo's own `framework/model_adapter.py::load_attention_ops` / `framework/kv_cache.py::iter_layer_kv` actually accept the model. `transformers==5.8.1` (this repo's `.venv`).

## Directory layout

- `models/legacy/` — `qwen3_1.7b` (primary/default, `configs/model.yaml`) and `olmo1b` (`allenai/OLMo-1B-hf`, 2024 OLMo-1 generation) — the models the existing published FIDELITY/BEHAVIOR/SYSTEM results were run against, kept for reproducibility. Not covered in the tables below (unchanged since the OLMo slot swap; conformance detail was recorded in the previous version of this report and in `docs/CURRENT_STATE.md`/`docs/SLM_COMPATIBILITY.md`).
- `models/` (top level) — the 5-model architecture-matrix shortlist, deep-probed below: `olmo2_1b` (MHA), `qwen3_0.6b` (GQA), `gemma3_270m` (MQA + local/global), `tinydeepseek_0.5b` (MLA), `falcon_h1_0.5b` (Hybrid attention + Mamba2).
- Deleted entirely: `granite_4.0_350m`, `qwen3.5_0.8b`, `minicpm4_0.5b` — superseded candidate-probe downloads; findings remain only in `docs/SLM_COMPATIBILITY.md`/`docs/CURRENT_STATE.md` as historical record, not present on disk.

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

**Cannot be measured or probed at all** — model load fails before any weights are materialized:

```
ImportError: cannot import name 'is_torch_fx_available' from 'transformers.utils.import_utils'
  at .../transformers_modules/tinydeepseek_0_dot_5b/.../modeling_tinydeepseek.py:56
```

The checkpoint ships its own `modeling_tinydeepseek.py`/`configuration_tinydeepseek.py` (loaded via `trust_remote_code=True`), and that vendored file imports a symbol `transformers==5.8.1` no longer exposes from `transformers.utils.import_utils`. There is also a self-reported type mismatch warning on every load attempt: *"You are using a model of type `deepseek_v3` to instantiate a model of type `tinydeepseek_v3`"* — the config's declared `model_type` (`deepseek_v3`, for `AutoConfig` resolution) doesn't match the vendored class's own self-identification (`tinydeepseek_v3`), a sign the checkpoint's remote code hasn't been kept in sync with its own config.

**From `config.json` only (unverified against real modules/weights):** `model_type=deepseek_v3`, `hidden_size=1024`, 26 layers, `num_attention_heads=4`, `num_key_value_heads=4`, `kv_lora_rank=256`, `qk_nope_head_dim=32`, `qk_rope_head_dim=32` — genuine DeepSeek-V3-style MLA fields (a compressed latent KV rank plus split RoPE/non-RoPE query-key head dims), but **none of this can be confirmed live** — no forward pass has ever succeeded, so the actual cache object, attention module structure, and MLA projection behavior remain unknown until this import is fixed.

**Engine correspondence — ❌ BLOCKED before any adapter question is reachable:**
- Model doesn't load at all — `load_attention_ops`/`iter_layer_kv` were never reached.
- **Fix options, in order of effort:** (1) patch the vendored `modeling_tinydeepseek.py` locally to drop/replace the dead import (smallest fix, but a local fork of third-party code needing re-application if the checkpoint is ever re-downloaded), (2) pin a `transformers` version old enough to still export `is_torch_fx_available` (risks breaking the other 4 shortlisted models' newer-transformers-dependent code paths — Falcon-H1 and Gemma3's per-layer-type RoPE support are both fairly recent transformers features), (3) wait for FreedomIntelligence to update the repo.
- **Even after it loads**, this is the single biggest engine gap in the whole shortlist: the current `KVCompressor` interface (`compress_kv(key, value)`) assumes plain per-head K/V tensors. MLA's native representation is a **compressed latent vector** (`kv_lora_rank=256`-dim) plus a small decoupled RoPE component — not a `[B, H, T, D]` K/V pair at all. Forcing MLA back into ordinary K/V (by decompressing to the "virtual" per-head K/V DeepSeek-V3 can reconstruct) would work mechanically with zero engine changes, but scientifically defeats the purpose of including an MLA model — it would benchmark compression of a *reconstruction*, not the model's actual native cache representation. A real MLA integration needs a new state type in the interception layer (see the cross-cutting section below), not just an `AttentionOps` branch.

---

## 4. `falcon_h1_0.5b` — `tiiuae/Falcon-H1-0.5B-Base` — Hybrid Attention + Mamba2

**Measured:** 521,411,104 params (100% bfloat16) · `FalconH1ForCausalLM` · tokenizer is a raw `TokenizersBackend` (not a `transformers`-native tokenizer class — Falcon-H1 ships its own fast-tokenizer JSON), vocab 32768 · 36 layers.

**Decoder layer (`FalconH1DecoderLayer`):** distinct from every other model here — has **both** `self_attn`-style attention (`attention_in_multiplier`, `attn_out_multiplier`, `channels_attn` gating scalars — Falcon-H1-specific mixing weights) **and** a separate `mamba` submodule (`FalconH1Mixer`) on the *same* layer object. `config.layer_types` is `["hybrid"] * 36` — every layer is simultaneously attention **and** Mamba, run in parallel and combined (not alternating attention-layers/Mamba-layers as in some other hybrid designs).

**Attention (`FalconH1Attention`):** `head_dim=64`, **no `q_norm`/`k_norm` at all** — a third QK-norm layout the engine's `AttentionOps.qk_norm_layout` doesn't have a name for (only `"per_head"`/`"flat"` exist). Has its own `key_multiplier` scalar (Falcon-H1's per-head attention/Mamba output mixing scheme).

**RoPE (`FalconH1RotaryEmbedding`):** single global `inv_freq`/`attention_scaling` — despite the hybrid architecture, the attention half uses one shared RoPE table (unlike Gemma3's per-layer-type split), so no per-layer RoPE selection is needed for the attention component specifically.

**Live cache:** 36 layers, but **every layer's cache object is `LinearAttentionAndFullAttentionLayer`**, not a plain `DynamicLayer` — a combined type that (per its name) can carry both a linear/recurrent state and standard attention K/V. In this checkpoint's forward pass it exposed `.keys`/`.values` of shape `(1, 2, 6, 64)` (2 == `num_key_value_heads`, confirming the attention half runs a 4:2 GQA-style ratio — `num_attention_heads=8`, `num_key_value_heads=2`), so `iter_layer_kv`'s generic `layer.keys, layer.values` fallback happened to work for the attention half — but this is incidental, not evidence the Mamba recurrent state is being seen or accounted for at all. **The Mamba/SSM state is invisible to `iter_layer_kv`** — it lives elsewhere on the cache/layer object, not in `.keys`/`.values`, so any compression run today would silently only ever touch the attention K/V half and leave 24 Mamba heads' worth of recurrent state completely untouched and unmeasured (neither compressed nor counted in FIDELITY/memory accounting).

**Engine correspondence — ❌ Cache-model gap, not just an adapter gap:**
- `load_attention_ops` → **fails**: `NotImplementedError("... model_type='falcon_h1' ... Supported: qwen3, olmo2.")`.
- `iter_layer_kv` → **"succeeds" but is misleading** — it silently only extracts the attention half of a genuinely dual-state layer; the Mamba recurrent state (24 heads' worth per layer, most of the model's actual state) is never touched, so today's FIDELITY/memory (compression ratio, bytes accounted) and any compressor's `compress()` call would report numbers that don't reflect the layer's true memory footprint.
- **This confirms the finding from the earlier probe**: Falcon-H1 needs the hybrid-cache abstraction described in `docs/SLM_COMPATIBILITY.md`/`docs/ENGINE_INTERNALS.md`'s Qwen3.5 discussion — `iter_layer_kv` (and everything built on it: `get_cache_size_bytes`, `count_kv_elements`, `apply_compressor`, `decompress_to_legacy_cache`) needs to explicitly recognize `LinearAttentionAndFullAttentionLayer` and either (a) extract *both* the attention K/V and the Mamba state as separate typed objects, treating Mamba state as compressor-passthrough (preserve exactly, don't compress) per the project's stated non-goal of inventing recurrent-state compression, or (b) fail loudly with a clear "hybrid cache unsupported" error instead of silently under-accounting memory, which is strictly worse than an explicit `NotImplementedError` because it produces plausible-looking but wrong numbers.
- Also needs: a third `qk_norm_layout` value (`"none"`, no q_norm/k_norm calls) in `project_qkv`, and a `model_type == "falcon_h1"` branch in `load_attention_ops` importing Falcon-H1's RoPE/eager-attention symbols — necessary but *not sufficient*, since those alone would still leave the Mamba-state blind spot above.

---

## 5. `gemma3_270m` — `google/gemma-3-270m` — MQA + local/global attention

**Measured:** 268,098,176 params (100% bfloat16) · `Gemma3ForCausalLM` · tokenizer `GemmaTokenizer`, vocab 262144 (by far the largest vocab in the shortlist) · 18 layers.

**Decoder layer (`Gemma3DecoderLayer`):** has `input_layernorm`, plus *both* `pre_feedforward_layernorm` and `post_feedforward_layernorm` — more norm points than any other model here (a "sandwich" norm pattern around the MLP specifically, on top of standard pre-norm attention).

**Attention (`Gemma3Attention`):** `head_dim=256` (unusually large for a 4-head model — `4 * 256 = 1024` ≠ `hidden_size=640`, i.e. Gemma3's attention inner dimension is decoupled from `hidden_size`, unlike every other model probed here where `heads * head_dim == hidden_size`). Has `q_norm`/`k_norm` (per-head layout, same family as Qwen3), plus Gemma3-specific `attn_logit_softcapping` and a `layer_type`/`is_sliding` attribute **on the attention module itself** — each layer knows its own type directly, not just via a config lookup.

**RoPE (`Gemma3RotaryEmbedding`):** **confirmed dual-table** — the module holds *two independent* buffer sets: `sliding_attention_inv_freq`/`sliding_attention_attention_scaling` and `full_attention_inv_freq`/`full_attention_attention_scaling`. This directly confirms the crash mechanism already documented in `docs/CURRENT_STATE.md`: `Gemma3RotaryEmbedding.forward` does `getattr(self, f"{layer_type}_inv_freq")`, so calling it without a `layer_type` (as the engine's single global call does) looks up a nonexistent `None_inv_freq` attribute.

`config.layer_types`: alternating `sliding_attention` × 5 then `full_attention` × 1, repeating — period-6 local/global split, 18 layers total (3 full-attention layers, 15 sliding).

**Live cache:** confirms the split is real at the object level, not just a config label — layer 0 (`sliding_attention`) is a `DynamicSlidingWindowLayer`, layer 17 (`full_attention`) is a plain `DynamicLayer`. Both report keys/values shape `(1, 1, 6, 256)` — 1 KV head vs. 4 Q heads confirms **true MQA** (the most extreme KV-sharing ratio in the shortlist).

**Engine correspondence — ❌ Two independent gaps, both previously documented, now directly confirmed against the live module:**
- `load_attention_ops` → **fails**: `NotImplementedError("... model_type='gemma3_text' ... Supported: qwen3, olmo2.")`.
- **FIDELITY/attention would additionally crash even with an adapter added**, because `eval/fidelity/attention.py::evaluate_attention_fidelity` (line 151) calls `model.model.rotary_emb(hidden_states[0], position_ids)` exactly once for the whole model — this needs to become two calls (one per `layer_type`, using the confirmed `sliding_attention_*`/`full_attention_*` buffer pairs above), with per-layer selection via `config.layer_types[layer_idx]` (or the module-level `attn.layer_type`/`attn.is_sliding`, which is even more direct — no config lookup needed).
- `iter_layer_kv` → succeeds today (generic path only reads `.keys`/`.values`, and both `DynamicSlidingWindowLayer`/`DynamicLayer` expose those) — so BEHAVIOR/SYSTEM under `identity`/`turboquant` already work; only FIDELITY/attention and the QJL/RocketKV `enable_*_online` paths (which also call `load_attention_ops` directly) are blocked.
- Needed: (1) `model_type == "gemma3_text"` branch in `load_attention_ops`, importing Gemma3's RoPE/eager-attention symbols; (2) extend `AttentionOps`/`project_qkv` or `_compute_layer_queries` to select the correct RoPE table per layer instead of assuming one global `(cos, sin)`; (3) `qk_norm_layout="per_head"` reuses the existing Qwen3 code path (confirmed by the probe — same per-head-then-norm layout), so no third layout value is needed for Gemma3 specifically (unlike Falcon-H1).

---

## Cross-model correspondence summary

| Model | Family | Loads + real forward pass? | `load_attention_ops` | `iter_layer_kv` | FIDELITY/attention | BEHAVIOR+SYSTEM (identity/turboquant) | BEHAVIOR+SYSTEM (qjl/rocketkv) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| olmo2_1b | MHA | ✅ | ✅ (`olmo2`, existing) | ✅ | ✅ | ✅ | ✅ |
| qwen3_0.6b | GQA | ✅ | ✅ (`qwen3`, existing) | ✅ | ✅ | ✅ | ✅ |
| gemma3_270m | MQA + local/global | ✅ | ❌ no `gemma3_text` branch | ✅ | ❌ single-RoPE-table crash (root cause now confirmed at buffer level) | ✅ | ❌ (adapter gate) |
| falcon_h1_0.5b | Hybrid Attn+Mamba2 | ✅ | ❌ no `falcon_h1` branch + no `"none"` qk_norm_layout | ⚠️ "succeeds" but silently drops Mamba state | ❌ (adapter gate) | ⚠️ runs, but memory/compression numbers exclude Mamba state entirely (misleading, not just missing) | ❌ (adapter gate) |
| tinydeepseek_0.5b | MLA | ❌ import error, never reaches model code | — (unreachable) | — (unreachable) | ❌ | ❌ | ❌ |

**2 of 5 fully work today** (`olmo2_1b`, `qwen3_0.6b` — both zero-engine-modification controls). **1 of 5** (`gemma3_270m`) is a well-scoped, previously-documented fix (adapter branch + per-layer RoPE). **1 of 5** (`falcon_h1_0.5b`) is architecturally the hardest of the two "clean" incompatibilities — it doesn't just need a new adapter, it needs the engine's cache-extraction layer taught that a layer can carry two independent state types simultaneously, and today it *silently* mis-measures rather than cleanly failing, which is worse than the other gaps. **1 of 5** (`tinydeepseek_0.5b`) can't even be assessed yet — an environment-level import failure blocks it before any engine question is reachable, and even once fixed, MLA needs the same order of interface generalization as Falcon-H1's Mamba state, just for a latent-KV representation instead of a recurrent one.

## What would actually need to change in the engine (ranked by how much of the shortlist it unblocks)

1. **`framework/model_adapter.py::load_attention_ops`** — add `model_type == "gemma3_text"` and `model_type == "falcon_h1"` branches (2 of the 3 remaining models gated here). Straightforward for Gemma3 (reuses the existing `"per_head"` qk_norm_layout); Falcon-H1 additionally needs a new `qk_norm_layout = "none"` value in `project_qkv` (skip q_norm/k_norm entirely — confirmed neither exists on `FalconH1Attention`).
2. **Per-layer RoPE selection** (`eval/fidelity/attention.py::evaluate_attention_fidelity` + `_compute_layer_queries`, and the `apply_rotary_pos_emb` calls in `framework/qjl_online.py`/`framework/rocketkv_online.py`) — currently hardcoded to one global `rotary_emb(...)` call. Needs to compute both `sliding_attention`/`full_attention` `(cos, sin)` pairs once and select per layer via `config.layer_types[layer_idx]` (confirmed available both on config and directly on each `Gemma3Attention.layer_type`/`.is_sliding`). Unblocks Gemma3's FIDELITY/attention and QJL/RocketKV online paths.
3. **A real hybrid-cache abstraction in `framework/kv_cache.py`** — `iter_layer_kv` and everything built on it currently assume one `(keys, values)` pair per layer. Falcon-H1's `LinearAttentionAndFullAttentionLayer` (and Qwen3.5's `LinearAttentionLayer`, documented earlier in `docs/SLM_COMPATIBILITY.md`) need the iteration contract to expose a typed union — attention K/V *and* recurrent/Mamba state as separate, independently-handled objects per layer — with an explicit policy (compress attention state, pass through Mamba state unmodified, per the project's documented non-goal of inventing recurrent-state compression) rather than the current behavior of silently reading only `.keys`/`.values` and ignoring the rest. This is the single highest-value fix for correctness (today's Falcon-H1 numbers are quietly wrong, not just absent), even though it unblocks "only" one model directly — the same abstraction is required for Qwen3.5 and any future hybrid model.
4. **A latent-KV (MLA) state type**, same abstraction layer as (3) — needed once TinyDeepSeek actually loads. `compressors/base.py::KVCompressor`'s `compress_kv(key, value)` signature has no notion of a compressed latent vector + decoupled-RoPE split; naively decompressing MLA to virtual per-head K/V before feeding it through the existing pipeline would "work" but would defeat the scientific purpose of benchmarking MLA's native representation.
5. **Unblock `tinydeepseek_0.5b`'s import error** — independent of all the above and the actual current bottleneck for that model: either patch the vendored `modeling_tinydeepseek.py`, or resolve the `transformers` version conflict. Nothing else about MLA support can even be probed until this is fixed.

None of the three remaining gaps (Gemma3's RoPE, Falcon-H1's hybrid cache, TinyDeepSeek's MLA state) are solved by "add another `if model_type == ...` branch" alone — items 2–4 are structural, matching the `ModelAdapterRegistry` / `StateAdapter` / typed-state-per-layer direction already sketched as the recommended long-term shape for this engine.
