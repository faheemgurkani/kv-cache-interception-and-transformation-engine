# SLM Compatibility with the FIDELITY / BEHAVIOR / SYSTEM Evaluation Framework

Which of the 6 locally available SLMs actually work with the redesigned evaluation framework (`eval/fidelity/`, `eval/behavior/`, `eval/system/`), which metrics run for each, and exactly what's missing for the ones that don't. Findings below are empirically verified against this repo's code (`framework/model_adapter.py`, `framework/kv_cache.py`, `framework/rocketkv_online.py`, `framework/qjl_online.py`, `eval/fidelity/attention.py`) — either via a live run (OLMo 2, Qwen3, and the Gemma3 crash) or via cheap `AutoModelForCausalLM.from_config(...)` structural probes (random weights, no multi-GB download) for the others, so the failure points are exact, not guessed.

Models: `scripts/download_candidate_models.py` (5-model shortlist: OLMo 2 1B, MiniCPM4-0.5B, Granite 4.0 350M, Gemma3-270M, Qwen3.5-0.8B) plus Qwen3-1.7B, the primary model configured in `configs/model.yaml`. All 6 are downloaded under `models/`.

## How compatibility is actually gated

Three independent gates, not one. A model can pass one and fail another:

1. **Generic engine compatibility** — does `AutoModelForCausalLM.from_pretrained(..., attn_implementation="eager")` load and forward, and does the returned `past_key_values` expose one `(key, value)` tensor pair per layer in a shape `framework/kv_cache.py::iter_layer_kv` can walk? This gate covers **FIDELITY/representation**, **FIDELITY/memory**, and **BEHAVIOR/SYSTEM under `identity`/`turboquant`** (these compressors need no model-family-specific code — `KVCacheEngine` only special-cases `qjl`/`rocketkv` by name).
2. **`framework/model_adapter.py::load_attention_ops` gate** — hardcoded to `{"qwen3", "qwen2", "olmo2"}`. Required for **FIDELITY/attention** (recomputes RoPE'd queries for every compressor, always) and for **BEHAVIOR/SYSTEM under `qjl`/`rocketkv`** (`enable_qjl_online`/`enable_rocketkv_online` both call `load_attention_ops(model.config)` directly).
3. **Per-model quirks inside gate 2** even when `model_type` matches — e.g. a model with a fine-grained RoPE scheme that a single shared `rotary_emb(hidden_states, position_ids)` call can't serve.

---

## 1. Qwen3-1.7B (`qwen3_1.7b`)

**1.72B params** (measured from safetensors) · `Qwen3ForCausalLM`, GQA 16Q/8KV heads, head_dim 128, 28 layers, per-head Q/K-norm, `attn_implementation="eager"` supported natively.

| Branch | Metric | Status |
|---|---|---|
| FIDELITY | representation, memory | ✅ |
| FIDELITY | attention (+ output-RMSE, KL) | ✅ — `model_type="qwen3"` is gate-1 supported |
| BEHAVIOR | task_quality, retrieval, instruction_following, reasoning | ✅ (all compressors, including QJL/RocketKV online) |
| SYSTEM | latency_throughput, vram, memory_bandwidth, kernel_cost, gpu_utilization | ✅ |

**Nothing missing.** This is the primary/default model — the README's published TurboQuant/QJL/RocketKV numbers are all on this model.

---

## 2. OLMo 2 1B (`olmo2_1b`)

**1.48B params** (measured; larger than the "1B" name suggests — untied embeddings + 100k vocab) · `Olmo2ForCausalLM`, MHA 16Q/16KV heads, 16 layers, flat (whole-projection) Q/K-norm, post-norm (`has_input_layernorm=False` in the adapter — OLMo2 applies norm after attention, not before).

| Branch | Metric | Status |
|---|---|---|
| FIDELITY | representation, memory | ✅ |
| FIDELITY | attention (+ output-RMSE, KL) | ✅ — `model_type="olmo2"` is gate-1 supported |
| BEHAVIOR | task_quality, retrieval, instruction_following, reasoning | ✅ (all compressors) |
| SYSTEM | all sub-metrics | ✅ |

**Verified live this session:** ran FIDELITY + BEHAVIOR + SYSTEM end to end with `identity` and `turboquant`; TurboQuant's compression ratio (3.12x) matched the README's published ~3.1x figure for Qwen3, a strong cross-check that the metric plumbing is correct across model families, not just tuned to one. **Nothing missing.**

---

## 3. Granite 4.0 350M (`granite_4.0_350m`)

**352M params** (measured) · registered as `GraniteMoeHybridForCausalLM`, but **this specific checkpoint's `layer_types` are 100% `"attention"`** (`num_experts_per_tok=0`, `num_local_experts=0` — no Mamba layers, no MoE routing actually active). Structurally: 16Q/4KV heads (GQA), head_dim 64, 28 layers, pre-norm (`input_layernorm` present), **no Q/K-norm at all** — a third QK-norm layout the adapter doesn't have a name for (only knows `"per_head"` and `"flat"`).

Verified via `AutoModelForCausalLM.from_config(cfg, attn_implementation="eager")` (random weights, structural probe only):
- Loads and forwards cleanly with `attn_implementation="eager"`.
- `past_key_values` is a standard `DynamicCache` with `.layers[i].keys`/`.values` — `iter_layer_kv` handles it with no changes.
- `model.model.rotary_emb` is a single shared module (no per-layer-type split, unlike Gemma3) — safe to call once and reuse across layers.
- `attn` module has `q_proj`/`k_proj`/`v_proj`/`head_dim` but **no `q_norm`/`k_norm`**.

| Branch | Metric | Status |
|---|---|---|
| FIDELITY | representation, memory | ✅ (compressor-only, no model_adapter needed) |
| FIDELITY | attention (+ output-RMSE, KL) | ❌ — `model_type="granitemoehybrid"` not in `load_attention_ops`'s supported set → `NotImplementedError` |
| BEHAVIOR | task_quality, retrieval, instruction_following, reasoning (identity/turboquant) | ✅ (generic engine path, no adapter needed) |
| BEHAVIOR | same, under `qjl`/`rocketkv` | ❌ — `enable_qjl_online`/`enable_rocketkv_online` both call `load_attention_ops` → same `NotImplementedError` |
| SYSTEM | latency_throughput, vram, memory_bandwidth, kernel_cost, gpu_utilization (identity/turboquant) | ✅ |
| SYSTEM | same, under `qjl`/`rocketkv` | ❌ — same adapter gate |

**What's missing, and where:** the closest of the four incompatible models to being fixed — it's architecturally a plain GQA transformer for this checkpoint, just not registered.

1. `framework/model_adapter.py::load_attention_ops` — add a `model_type == "granitemoehybrid"` branch importing `apply_rotary_pos_emb`/`eager_attention_forward`/`ALL_ATTENTION_FUNCTIONS` from `transformers.models.granitemoehybrid.modeling_granitemoehybrid` (verify these are exported with the same signatures as Qwen3/OLMo2's before wiring).
2. `framework/model_adapter.py::project_qkv` — add a third `qk_norm_layout` value (e.g. `"none"`) that skips the `q_norm`/`k_norm` calls entirely, since this checkpoint has neither.
3. Once (1)+(2) land, FIDELITY/attention and QJL/RocketKV's `enable_*_online` should work unmodified — they only depend on `load_attention_ops`/`project_qkv`, not on anything Granite-specific beyond that.
4. Caveat: `GraniteMoeHybridForCausalLM` as a *class* supports real Mamba2+MoE hybrid layers in other checkpoints of this family. If a future Granite checkpoint has `layer_types` containing `"mamba"` or non-zero `num_local_experts`, none of the above applies — that would need genuinely new engine work (see Qwen3.5 below, same category of problem).

---

## 4. Gemma3-270M (`gemma3_270m`)

**268M params** (measured) · `Gemma3ForCausalLM`, GQA 4Q/1KV heads, head_dim 256, 18 layers, per-head Q/K-norm (same layout family as Qwen3), pre-norm. Uses **per-layer-type RoPE**: `layer_types` alternates `"sliding_attention"` (local, most layers) and `"full_attention"` (global, every 6th layer), each with its **own** `inv_freq`/`attention_scaling` buffers on a *single shared* `Gemma3RotaryEmbedding` module.

**Confirmed by direct crash this session** (real weights, live FIDELITY run):
```
AttributeError: 'Gemma3RotaryEmbedding' object has no attribute 'None_inv_freq'
```
Root cause, traced to the exact line in `transformers.models.gemma3.modeling_gemma3.Gemma3RotaryEmbedding.forward`:
```python
def forward(self, x, position_ids, layer_type=None):
    inv_freq = getattr(self, f"{layer_type}_inv_freq")   # looks up e.g. "sliding_attention_inv_freq"
```
`eval/fidelity/attention.py::_compute_layer_queries` calls `model.model.rotary_emb(hidden_states[0], position_ids)` **once, with no `layer_type`**, expecting to reuse one `(cos, sin)` pair for every layer — true for Qwen3/OLMo2 (single global RoPE config) but false for Gemma3, where `layer_type` defaults to `None` and the lookup becomes `getattr(self, "None_inv_freq")`, which doesn't exist.

| Branch | Metric | Status |
|---|---|---|
| FIDELITY | representation, memory | ✅ (compressor-only forward, doesn't touch `rotary_emb` directly) |
| FIDELITY | attention (+ output-RMSE, KL) | ❌ — crashes as above, and `model_type="gemma3_text"` isn't in `load_attention_ops` either (would also need a new branch even after the RoPE fix) |
| BEHAVIOR | task_quality, retrieval, instruction_following, reasoning (identity/turboquant) | ✅ — generic engine path, `iter_layer_kv`/`decompress_to_legacy_cache` don't touch `rotary_emb` |
| BEHAVIOR | same, under `qjl`/`rocketkv` | ❌ — `load_attention_ops` gate (model_type not registered) |
| SYSTEM | latency_throughput, vram, memory_bandwidth, kernel_cost, gpu_utilization (identity/turboquant) | ✅ |
| SYSTEM | same, under `qjl`/`rocketkv` | ❌ — same gate |

**What's missing, and where:**

1. `eval/fidelity/attention.py::_compute_layer_queries` (and the single `position_embeddings = model.model.rotary_emb(...)` call in `evaluate_attention_fidelity`) assumes one RoPE table for the whole model. For Gemma3 this needs to become **per-layer**: compute `(cos, sin)` separately for `"sliding_attention"` vs `"full_attention"` layer types (two calls to `rotary_emb`, cached once, then selected per layer via `config.layer_types[layer_idx]`) instead of one shared call.
2. `framework/model_adapter.py::load_attention_ops` — add a `model_type == "gemma3_text"` branch (import RoPE/eager-attention symbols from `transformers.models.gemma3.modeling_gemma3`), and extend `AttentionOps` (or `project_qkv`) so callers know which RoPE table to use per layer — same underlying fix as (1), needed again for `framework/qjl_online.py`/`framework/rocketkv_online.py`'s `apply_rotary_pos_emb(query_states, key_states, cos, sin)` calls, which currently also assume one shared `(cos, sin)` pair for the whole model.
3. This is real, contained work (Gemma3's dual local/global RoPE is a well-defined, documented HF pattern) — not a fundamentally incompatible architecture like Qwen3.5 below.

---

## 5. MiniCPM4-0.5B (`minicpm4_0.5b`)

**434M params** (measured, BF16) · `MiniCPMForCausalLM` via `auto_map` custom modeling code (`configuration_minicpm.py`, `modeling_minicpm.py` shipped in the model directory), GQA 16Q/2KV heads, 24 layers, `longrope` RoPE scaling (a third RoPE variant beyond standard/NTK).

**Confirmed fails to load at all**, empirically:
```python
>>> AutoConfig.from_pretrained('models/minicpm4_0.5b')
ValueError: The repository models/minicpm4_0.5b contains custom code which must be
executed to correctly load the model. ... pass trust_remote_code=True
```
The tokenizer alone loads fine (falls back to plain `LlamaTokenizer`, no custom code needed there) — the failure is specifically at config/model resolution. `framework/model.py::ModelLayer.__init__` calls `AutoModelForCausalLM.from_pretrained(self.model_path, dtype=torch_dtype, attn_implementation=attn_implementation)` with **no `trust_remote_code` argument**, so this raises before any evaluation code runs.

| Branch | Metric | Status |
|---|---|---|
| FIDELITY | representation, attention, memory | ❌ — model never loads |
| BEHAVIOR | all sub-metrics, any compressor | ❌ — model never loads |
| SYSTEM | all sub-metrics, any compressor | ❌ — model never loads |

**What's missing, and where:**

1. `framework/model.py::ModelLayer.__init__` — both the `AutoTokenizer.from_pretrained(...)` and `AutoModelForCausalLM.from_pretrained(...)` calls need `trust_remote_code=True` (at minimum gated behind a config flag or model-specific override, since blanket `trust_remote_code=True` executes arbitrary code from the model repo — worth an explicit opt-in rather than a silent default).
2. Once it loads: `resolve_model_type(config)` would return whatever `MiniCPMConfig.model_type` is (need to check the custom config class — not present as a `model_type` key in `config.json` itself, so it's likely set as a class attribute in `configuration_minicpm.py`). Almost certainly not `"qwen3"`/`"olmo2"`, so `load_attention_ops` would still raise — same FIDELITY/attention and QJL/RocketKV gaps as Granite/Gemma3, needing a new adapter branch built against MiniCPM's custom attention module (which may not even match the `self_attn.q_proj/k_proj/v_proj` shape the adapter assumes — unverified, since loading is blocked upstream of that check).
3. FIDELITY/representation, FIDELITY/memory, and BEHAVIOR/SYSTEM under identity/turboquant would likely work once (1) is resolved, *if* MiniCPM's custom model still returns a standard per-layer K/V cache — this is the one open question that can't be answered without actually enabling `trust_remote_code`.

---

## 6. Qwen3.5-0.8B (`qwen3.5_0.8b`)

**873M params** (measured; mixed F32/BF16 — some parameters, likely embeddings/norms, stored in F32) · registered as `Qwen3_5ForConditionalGeneration` (full omni vision+video+text) with a plain-text `Qwen3_5ForCausalLM` variant that `AutoModelForCausalLM` resolves to. This is **not a standard transformer** — it's a **hybrid linear-attention / full-attention architecture**: of 24 layers, **18 use `linear_attn`** (a recurrent linear-attention module, not KV-cache attention) and only **6 use `self_attn`** (standard attention, at indices 3, 7, 11, 15, 19, 23).

**Confirmed via structural probe** (`AutoModelForCausalLM.from_config`, random weights, real forward pass):
```python
>>> type(model.model.layers[0])
Qwen3_5DecoderLayer   # layer_type = "linear_attention", has .linear_attn, no .self_attn
>>> type(model.model.layers[3])
Qwen3_5DecoderLayer   # layer_type = "full_attention", has .self_attn
>>> out = model(ids, use_cache=True); pkv = out.past_key_values
>>> type(pkv.layers[0])
LinearAttentionLayer   # NO .keys / .values attribute at all
>>> type(pkv.layers[3])
DynamicLayer            # standard .keys / .values tensors, shape [1, 2, T, head_dim]
```

This breaks the engine at the most basic level, **before** the `model_adapter.py` gate is even reached: `framework/kv_cache.py::iter_layer_kv`'s `.layers` branch does `yield layer.keys, layer.values` unconditionally for every layer — the moment it reaches layer 0 (a `LinearAttentionLayer`), it raises `AttributeError: 'LinearAttentionLayer' object has no attribute 'keys'`. This is **gate 1** (generic engine compatibility), not gate 2 — so unlike Granite/Gemma3/MiniCPM, no compressor or metric is spared, including `identity`.

| Branch | Metric | Status |
|---|---|---|
| FIDELITY | representation, memory | ❌ — `iter_layer_kv` crashes on the first linear-attention layer |
| FIDELITY | attention | ❌ — same crash, plus would also hit the gate-2 `model_type` issue if it got further |
| BEHAVIOR | all sub-metrics, any compressor | ❌ — `KVCacheEngine` decompress/compress path walks `iter_layer_kv` every step |
| SYSTEM | all sub-metrics, any compressor | ❌ — same |

**What's missing, and where:** this is categorically different from the other three — not a missing adapter registration, but a missing **hybrid-cache abstraction** the engine doesn't have at all.

1. `framework/kv_cache.py::iter_layer_kv` (and every function built on it — `get_cache_size_bytes`, `count_kv_elements`, `apply_compressor`, `decompress_to_legacy_cache`, …) would need to either (a) skip linear-attention layers entirely and only intercept/compress the 6 real-attention layers, treating the other 18 as opaque/uncompressed passthrough state, or (b) grow a separate compression path for linear-attention recurrent state (a fundamentally different object — a fixed-size state matrix per layer, not a growing K/V sequence — compression semantics don't obviously transfer).
2. `compressors/base.py::KVCompressor` and every compressor implementation assume `[B, H, T, D]` K/V tensors; none of them have any notion of linear-attention recurrent state.
3. Given the framework's explicit design purpose ("intercepts K/V tensors at the decode boundary") and non-goals list in `docs/SYSTEM_DESIGN.md` (no multi-GPU layer split, no exotic backends), a hybrid hard `NotImplementedError`/allowlist check at `ModelLayer` construction time (reject models with any non-`"full_attention"` `layer_types` entry, or a `linear_attn` module present) is the pragmatic near-term fix — cleanly reporting *why* it's unsupported — rather than attempting option (1)/(2) above, which is a substantial new-architecture-class research effort in its own right (comparable in scope to adding a new baseline family), not a config/adapter gap.

---

## Summary matrix

| Model | Params | FIDELITY/repr+mem | FIDELITY/attention | BEHAVIOR+SYSTEM (identity/turboquant) | BEHAVIOR+SYSTEM (qjl/rocketkv) |
|---|---:|:---:|:---:|:---:|:---:|
| Qwen3-1.7B | 1.72B | ✅ | ✅ | ✅ | ✅ |
| OLMo 2 1B | 1.48B | ✅ | ✅ | ✅ | ✅ |
| Granite 4.0 350M | 352M | ✅ | ❌ (adapter) | ✅ | ❌ (adapter) |
| Gemma3-270M | 268M | ✅ | ❌ (per-layer RoPE + adapter) | ✅ | ❌ (per-layer RoPE + adapter) |
| MiniCPM4-0.5B | 434M | ❌ (won't load) | ❌ (won't load) | ❌ (won't load) | ❌ (won't load) |
| Qwen3.5-0.8B | 873M | ❌ (hybrid cache) | ❌ (hybrid cache) | ❌ (hybrid cache) | ❌ (hybrid cache) |

## Recommended fix priority

1. **Granite 4.0 350M** — smallest, most contained fix: one `load_attention_ops` branch + one new `qk_norm_layout` value. No RoPE complications, no loader changes. Best next model to add.
2. **Gemma3-270M** — well-defined, documented HF pattern (per-layer-type RoPE); requires touching `eval/fidelity/attention.py`'s single-RoPE-call assumption in addition to the adapter, so more surface area than Granite but still contained.
3. **MiniCPM4-0.5B** — gated behind a `trust_remote_code` decision (security/trust tradeoff, not a technical blocker) before its actual compatibility can even be assessed.
4. **Qwen3.5-0.8B** — not a near-term fix. Requires new hybrid-cache handling in the core engine, or an explicit unsupported-architecture guard at load time so it fails clearly instead of raising a raw `AttributeError` deep in `iter_layer_kv`.
