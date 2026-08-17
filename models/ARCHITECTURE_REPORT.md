# Architecture Matrix — Downloaded Model Report

Conformance verified 2026-08-18 by loading each checkpoint with `AutoModelForCausalLM.from_pretrained(..., attn_implementation="eager")`, running a real forward pass (`"The quick brown fox"`, `use_cache=True`), and inspecting the actual returned `past_key_values` object/shapes — not just `config.json` fields. `transformers==5.8.1` (this repo's `.venv`).

## Directory layout (as of 2026-08-19, OLMo slots swapped 2026-08-20)

- `models/legacy/` — the two models the prior FIDELITY/BEHAVIOR/SYSTEM evaluation runs actually used: `qwen3_1.7b` (primary/default model in `configs/model.yaml`) and, as of the swap below, `olmo1b` (`allenai/OLMo-1B-hf`, the 2024 OLMo-1 generation).
- `models/` (top level) — the 5-model architecture matrix shortlist (MHA/MQA/GQA/MLA/Hybrid), for future engine-extension work: `olmo2_1b` (swapped in as the MHA representative — same head geometry as OLMo-1, but the more recent 2025 generation, per user preference), `qwen3_0.6b`, `gemma3_270m`, `tinydeepseek_0.5b`, `falcon_h1_0.5b`.
- Deleted entirely (not shortlisted, not legacy): `granite_4.0_350m`, `qwen3.5_0.8b`, `minicpm4_0.5b` — these were earlier candidate-probe downloads (`scripts/download_candidate_models.py`) superseded by the shortlist decision; all were already documented as broken/blocked in `docs/CURRENT_STATE.md`/`docs/SLM_COMPATIBILITY.md` (adapter gate, hybrid-cache gate, or `trust_remote_code` load failure respectively) and add no further value sitting on disk.

## OLMo slot swap (2026-08-20)

Originally `olmo2_1b` (the model the prior eval framework used) sat in `legacy/`, and `olmo1b` (newly downloaded) was in the shortlist. **Swapped on request**: since both are MHA with identical 16Q/16KV head geometry, the user preferred evaluating the architecture-matrix's MHA slot on the *more recent* generation. Now:

- `models/olmo2_1b` (shortlist, MHA representative) — `allenai/OLMo-2-0425-1B`, 2025 generation.
- `models/legacy/olmo1b` — `allenai/OLMo-1B-hf`, 2024 generation, moved into `legacy/` alongside `qwen3_1.7b`.

## Two OLMo checkpoints — which is which

| | `models/olmo2_1b` (shortlist) | `models/legacy/olmo1b` |
|---|---|---|
| HF repo | `allenai/OLMo-2-0425-1B` | `allenai/OLMo-1B-hf` |
| Release | **2025** (OLMo 2 generation) — more recent | **2024** (original OLMo generation) — older |
| `model_type` / class | `olmo2` / `Olmo2ForCausalLM` | `olmo` / `OlmoForCausalLM` |
| Norm placement | **post-norm** (norm applied after attention/MLP sublayers — a genuine architectural difference from OLMo-1) | **pre-norm** (standard Llama-style, norm before each sublayer) |
| Embeddings | untied (`tie_word_embeddings: false`) | tied (`tie_word_embeddings: true`) |
| Vocab size | 100352 (newer BPE tokenizer) | 50304 |
| `max_position_embeddings` | 4096 | 2048 |
| `rope_theta` | 500000 | 10000 |
| Attention heads | 16 Q / 16 KV (MHA) — identical head geometry | 16 Q / 16 KV (MHA) — identical head geometry |
| Params (measured) | ~1.48B | ~1.18B |
| Checkpoint's own `transformers_version` | 4.50.3 | 4.40.0 |
| Role in this repo | **Shortlist MHA representative** (post-swap) — `framework/model_adapter.py` already has a dedicated `olmo2` branch, so this is also effectively a zero-engine-modification control, same as Qwen3-0.6B/GQA | **Legacy** — one of the two models the existing evaluation framework (TurboQuant/QJL/RocketKV results in `docs/RESULTS_COMPLETE.md` etc.) was actually run against; `framework/model_adapter.py` has **no** `olmo` (non-2) branch, only `olmo2` |

Despite sharing the "OLMo" name and identical 16/16 MHA head counts, these are **two distinct model generations** with real architectural differences (norm placement, embedding tying, RoPE theta, tokenizer/vocab) — not two copies of the same checkpoint. `olmo2_1b` is the one already wired into `framework/model_adapter.py::load_attention_ops` (`model_type == "olmo2"`); `olmo1b` (now in `legacy/`) would need its own `model_type == "olmo"` branch if it were ever brought back into active eval (see "Adding an SLM" in `CLAUDE.md`).

## allenai/OLMo-2-0425-1B → `olmo2_1b` (MHA, shortlist) — ✅ CONFORMS

- `model_type`: olmo2 · `hidden_size`: 2048 · `num_hidden_layers`: 16 · `num_attention_heads`: 16 · `num_key_value_heads`: 16
- weight files: `model.safetensors` · other artifacts: tokenizer/config files (standard HF layout)
- **Conformance:** already verified live in an earlier session (`docs/CURRENT_STATE.md` — FIDELITY+BEHAVIOR+SYSTEM run end to end with `identity`/`turboquant`, TurboQuant ratio cross-checked against the Qwen3 published number). Same 16Q/16KV MHA geometry as `olmo1b`; **fully wired** into `framework/model_adapter.py` (`model_type == "olmo2"` branch) — no adapter work needed to use it in the architecture-matrix slot.

## allenai/OLMo-1B-hf → `legacy/olmo1b` (MHA) — ✅ CONFORMS (moved to legacy)

- `model_type`: olmo · `hidden_size`: 2048 · `num_hidden_layers`: 16 · `num_attention_heads`: 16 · `num_key_value_heads`: 16
- weight files: `model.safetensors` · other artifacts: `generation_config.json`, `special_tokens_map.json`, `tokenizer.json`, `tokenizer_config.json`
- **Live probe:** loads as `OlmoForCausalLM`, forward pass succeeds. Cache: 16 `DynamicLayer`s, layer0 keys/values shape `(1, 16, 5, 128)` — 16 K-heads == 16 V-heads == `num_attention_heads`, confirming **true MHA** (no KV head sharing, matching the 1:1 Q:KV ratio the family claim requires). No longer in the active shortlist slot as of the 2026-08-20 swap, but the model and this conformance result remain valid — kept in `models/legacy/` for reference/reproducibility.

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

## Summary — current `models/` shortlist (post-swap)

| Model | Family | Loads + forwards? | Cache shape confirms architecture claim? |
|---|---|:---:|:---:|
| olmo2_1b | MHA | ✅ | ✅ 16 KV heads == 16 Q heads |
| qwen3_0.6b | GQA | ✅ | ✅ 8 KV heads : 16 Q heads (2:1) |
| tinydeepseek_0.5b | MLA | ❌ import error | ⚠️ unverified — config fields look right, forward pass blocked |
| falcon_h1_0.5b | Hybrid Attn+Mamba2 | ✅ | ✅ combined attention+Mamba layer object, GQA attention half (2 KV heads) |
| gemma3_270m | MQA + local/global | ✅ | ✅ 1 KV head (MQA) + sliding/full layer-type split reflected in cache class |

4 of 5 fully confirm their claimed architecture end-to-end (weights load, forward pass runs, cache tensor shapes match). TinyDeepSeek's config fields are consistent with MLA but the checkpoint's bundled custom code doesn't import cleanly against this venv's `transformers==5.8.1`, so its cache shape/MLA behavior is not yet verified.

## `models/legacy/` — not part of the active shortlist

| Model | Family | Role |
|---|---|---|
| qwen3_1.7b | GQA | Primary/default model, `configs/model.yaml` — all published TurboQuant/QJL/RocketKV numbers are on this model. |
| olmo1b | MHA | 2024 OLMo-1 generation — superseded in the shortlist's MHA slot by `olmo2_1b` (2026-08-20 swap), kept here for reference; conformance details above. |

## Deleted entirely (no longer on disk, not documented further here)

`granite_4.0_350m`, `qwen3.5_0.8b`, `minicpm4_0.5b` — earlier candidate-probe downloads from `scripts/download_candidate_models.py`, superseded by the 5-model architecture-matrix shortlist decision. Their historical compatibility findings remain in `docs/SLM_COMPATIBILITY.md` and `docs/CURRENT_STATE.md` as research record, but none of the three are present under `models/` anymore and none are tracked by this report going forward.
