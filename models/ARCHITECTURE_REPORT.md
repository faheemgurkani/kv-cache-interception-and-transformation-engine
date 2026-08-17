# Architecture Matrix — Downloaded Model Report

Conformance verified 2026-08-18 by loading each checkpoint with `AutoModelForCausalLM.from_pretrained(..., attn_implementation="eager")`, running a real forward pass (`"The quick brown fox"`, `use_cache=True`), and inspecting the actual returned `past_key_values` object/shapes — not just `config.json` fields. `transformers==5.8.1` (this repo's `.venv`).

## allenai/OLMo-1B-hf → `olmo1b` (MHA) — ✅ CONFORMS

- `model_type`: olmo · `hidden_size`: 2048 · `num_hidden_layers`: 16 · `num_attention_heads`: 16 · `num_key_value_heads`: 16
- weight files: `model.safetensors` · other artifacts: `generation_config.json`, `special_tokens_map.json`, `tokenizer.json`, `tokenizer_config.json`
- **Live probe:** loads as `OlmoForCausalLM`, forward pass succeeds. Cache: 16 `DynamicLayer`s, layer0 keys/values shape `(1, 16, 5, 128)` — 16 K-heads == 16 V-heads == `num_attention_heads`, confirming **true MHA** (no KV head sharing, matching the 1:1 Q:KV ratio the family claim requires).

## Qwen/Qwen3-0.6B → `qwen3_0.6b` (GQA) — ✅ CONFORMS

- `model_type`: qwen3 · `hidden_size`: 1024 · `num_hidden_layers`: 28 · `num_attention_heads`: 16 · `num_key_value_heads`: 8
- weight files: `model.safetensors` · other artifacts: `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`
- **Live probe:** loads as `Qwen3ForCausalLM`, forward pass succeeds. Cache: 28 `DynamicLayer`s, layer0 keys/values shape `(1, 8, 4, 128)` — 8 cached KV heads vs. 16 query heads confirms a real **2:1 GQA** group ratio, not degenerate MHA/MQA.

## FreedomIntelligence/TinyDeepSeek-0.5B-base → `tinydeepseek_0.5b` (MLA) — ❌ DOES NOT LOAD (unverified)

- `model_type`: deepseek_v3 (config also self-identifies as `tinydeepseek_v3` — mismatch, see below) · `hidden_size`: 1024 · `num_hidden_layers`: 26 · `num_attention_heads`: 4 · `num_key_value_heads`: 4 · `kv_lora_rank`: 256 · `qk_nope_head_dim`: 32 · `qk_rope_head_dim`: 32
- weight files: `model.safetensors` · other artifacts: `configuration_tinydeepseek.py`, `modeling_tinydeepseek.py`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`
- **Live probe FAILED:** `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` raises `ImportError: cannot import name 'is_torch_fx_available' from transformers.utils.import_utils`. The repo's shipped `modeling_tinydeepseek.py` (line 56) imports a symbol that `transformers==5.8.1` removed — the custom code is stale relative to this venv's transformers version. Also throws a mismatch warning: *"You are using a model of type `deepseek_v3` to instantiate a model of type `tinydeepseek_v3`"*.
- **Architecture claim (MLA) is plausible from config alone** (`kv_lora_rank`/`qk_nope_head_dim`/`qk_rope_head_dim` are genuine DeepSeek-V3-style MLA fields) but **cannot be confirmed by an actual forward pass / cache-shape inspection until this import error is fixed** — either patch `modeling_tinydeepseek.py` locally, pin an older `transformers`, or wait for an upstream fix.

## tiiuae/Falcon-H1-0.5B-Base → `falcon_h1_0.5b` (Hybrid Attention + Mamba2) — ✅ CONFORMS

- `model_type`: falcon_h1 · `hidden_size`: 1024 · `num_hidden_layers`: 36 · `num_attention_heads`: 8 · `num_key_value_heads`: 2 · `mamba_n_heads`: 24
- weight files: `model.safetensors` · other artifacts: `generation_config.json`, `special_tokens_map.json`, `tokenizer.json`, `tokenizer_config.json`
- **Live probe:** loads as `FalconH1ForCausalLM`, forward pass succeeds (falls back to the naive Mamba path — no `causal-conv1d`/`mamba-ssm` CUDA kernels installed, expected on macOS/no-CUDA). Cache layer0 is a `LinearAttentionAndFullAttentionLayer` — a combined per-layer object carrying **both** attention KV state and Mamba recurrent state, not a plain `DynamicLayer`. Attention keys/values shape `(1, 2, 4, 64)` — 2 == `num_key_value_heads`, confirming the attention half runs GQA within the hybrid block, matching the family claim. This confirms the engine's current `iter_layer_kv` (`yield layer.keys, layer.values`) **will not work unmodified** here — exactly the hybrid-cache gap already documented for Qwen3.5 in `docs/CURRENT_STATE.md`/`docs/SLM_COMPATIBILITY.md`.

## google/gemma-3-270m → `gemma3_270m` (MQA + local/global attention) — ✅ CONFORMS

- `model_type`: gemma3_text · `hidden_size`: 640 · `num_hidden_layers`: 18 · `num_attention_heads`: 4 · `num_key_value_heads`: 1
- `layer_types`: alternating `sliding_attention` (local) / `full_attention` (global), period 6, 18 layers total
- weight files: `model.safetensors` · other artifacts: `added_tokens.json`, `generation_config.json`, `special_tokens_map.json`, `tokenizer.json`, `tokenizer_config.json`
- **Live probe:** loads as `Gemma3ForCausalLM`, forward pass succeeds. Cache layer0 is a `DynamicSlidingWindowLayer` (not a plain `DynamicLayer` — confirms the sliding-window/full-attention split is real at the cache level, not just a config label), keys/values shape `(1, 1, 5, 256)` — 1 cached KV head vs. 4 query heads confirms **true MQA** (max KV-sharing ratio). Already known-broken in this engine for FIDELITY/attention (`docs/CURRENT_STATE.md` — per-layer RoPE lookup crash), unaffected here since this probe only exercises the generic forward path.

## Summary

| Model | Family | Loads + forwards? | Cache shape confirms architecture claim? |
|---|---|:---:|:---:|
| olmo1b | MHA | ✅ | ✅ 16 KV heads == 16 Q heads |
| qwen3_0.6b | GQA | ✅ | ✅ 8 KV heads : 16 Q heads (2:1) |
| tinydeepseek_0.5b | MLA | ❌ import error | ⚠️ unverified — config fields look right, forward pass blocked |
| falcon_h1_0.5b | Hybrid Attn+Mamba2 | ✅ | ✅ combined attention+Mamba layer object, GQA attention half (2 KV heads) |
| gemma3_270m | MQA + local/global | ✅ | ✅ 1 KV head (MQA) + sliding/full layer-type split reflected in cache class |

4 of 5 fully confirm their claimed architecture end-to-end (weights load, forward pass runs, cache tensor shapes match). TinyDeepSeek's config fields are consistent with MLA but the checkpoint's bundled custom code doesn't import cleanly against this venv's `transformers==5.8.1`, so its cache shape/MLA behavior is not yet verified.
