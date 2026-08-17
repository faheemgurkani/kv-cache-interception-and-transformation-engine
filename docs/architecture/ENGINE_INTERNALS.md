# Engine Internals — Complete Implementation, Architecture, Flow, and Integration

A from-first-principles walkthrough of the KV Cache Interception and Transformation Engine: every component, how they connect, and the exact execution flow from `scripts/run_eval.py` down to a single compressed tensor. Closes with a concrete engineering analysis of what it would take to support model architectures beyond dense, decoder-only, uniform-K/V-cache transformers.

This document is the "how it actually works, wire by wire" reference. For narrower views: [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md) (high-level architecture), [METHODOLOGY.md](../methodology/METHODOLOGY.md) (experimental protocol + per-compressor math), [MATHEMATICS_AND_ALGORITHMS.md](../methodology/MATHEMATICS_AND_ALGORITHMS.md) (equations), [SLM_COMPATIBILITY.md](SLM_COMPATIBILITY.md) (which models work today and why), [MODEL_ARCHITECTURE_MATRIX.md](MODEL_ARCHITECTURE_MATRIX.md) (deep per-model probes + engine-correspondence gap analysis for the current 5-model shortlist).

---

## 1. What the engine actually does, in one sentence

It sits between a HuggingFace causal LM's decoder layers and its own KV cache, replacing the raw `(key, value)` tensors with a compressed representation between every autoregressive step, so that any compression algorithm can be measured under **real decoding** — not just a single offline forward pass — without changing a single line of model code.

```text
Tokenizer → Model forward → past_key_values → KVCacheEngine → KVCompressor → Attention (next step) → logits
                                                        │
                                     identity | turboquant | qjl | rocketkv (swappable)
```

Everything left of `KVCacheEngine` and right of `KVCompressor` is fixed. Only the box in the middle changes per method under test.

---

## 2. Directory-to-responsibility map

| Directory | Responsibility | Swappable? |
|---|---|---|
| `framework/` | Model loading, KV interception, per-model-family adapters, storage accounting | No (except `model_adapter.py`'s per-family branches, which are additive) |
| `compressors/` | The `KVCompressor` plug-in interface + concrete methods | **Yes** — this is the experimental variable |
| `quantizers/` | Low-level numeric building blocks (WHT, Lloyd-Max, QJL sketching, RocketKV selection) used by compressors | No — orchestrated by `compressors/`, not swapped independently |
| `eval/` | FIDELITY / BEHAVIOR / SYSTEM evaluation branches | No |
| `data/`, `datasets/` | WikiText-2 / C4 loading for perplexity | No |
| `modal_app/` | Cloud GPU orchestration (Modal) | No |
| `reporting/` | JSON/CSV serialization of results | No |
| `scripts/` | CLI entry points | No |
| `configs/` | YAML source of truth for every experimental parameter | No |
| `baselines/` | Reference/paper implementations, kept separate from the production `compressors/`+`quantizers/` path | N/A |
| `tests/` | Unit + integration tests, one file per component | N/A |

---

## 3. `framework/` — the fixed engine

### 3.1 `framework/model.py` — `ModelLayer`

Thin wrapper around a HuggingFace causal LM. Constructor:

```python
ModelLayer(model_path=None, device=None, torch_dtype=torch.float16, attn_implementation="eager")
```

- `model_path` defaults to `configs/model.yaml`'s `local_path` (currently `models/qwen3_1.7b`).
- `device` defaults to `framework.device.get_eval_device()`: `KV_EVAL_DEVICE=cuda` env var forces CUDA (Modal), otherwise MPS-if-available-else-CPU (local dev).
- **`attn_implementation="eager"` is load-bearing, not a default choice.** FlashAttention/fused SDPA never materialize `past_key_values` as explicit per-layer tensors the way eager attention does — compression hooks need to physically see and replace K/V, so any fused kernel is a non-starter for this design.
- `AutoTokenizer.from_pretrained` and `AutoModelForCausalLM.from_pretrained` are called with **no `trust_remote_code`** — this is why MiniCPM4 (custom modeling code) fails to load at all (see §8.5).
- `model.config.use_cache = True` is forced; `model.eval()` is called.

Exposes: `.config` (proxies `model.config`), `.tokenize(text)`, `.forward_with_cache(...)` (thin wrapper on `model(...)`), `.generate(...)` (uncompressed HF baseline, used only by BEHAVIOR/SYSTEM baseline paths), `.make_kv_engine(compressor)` (constructs a `KVCacheEngine` bound to this model).

### 3.2 `framework/model_adapter.py` — the per-model-family seam

This is the single place model-family differences are supposed to live. It exports:

- `resolve_model_type(config) -> str` — lowercased `config.model_type`.
- `resolve_head_dim(config, attn_module=None) -> int` — prefers `attn_module.head_dim`, falls back to `config.head_dim`, falls back to `hidden_size // num_attention_heads`. Handles OLMo2 (which omits `head_dim` in its config) automatically.
- `load_attention_ops(config) -> AttentionOps` — **the hard gate.** A literal `if model_type in {"qwen3","qwen2"}: ... elif model_type == "olmo2": ... else: raise NotImplementedError`. Each branch imports that family's `apply_rotary_pos_emb`, `eager_attention_forward`, and `ALL_ATTENTION_FUNCTIONS` directly from `transformers.models.<family>.modeling_<family>`, and returns an `AttentionOps` dataclass describing:
  - `qk_norm_layout`: `"per_head"` (Qwen3 — RMSNorm after reshaping to per-head, i.e. `q_norm(q_proj(x).view(...,H,D))`) or `"flat"` (OLMo2 — RMSNorm over the whole flat projection *before* reshaping, i.e. `q_norm(q_proj(x))` then `.view(...)`).
  - `has_input_layernorm`: whether to apply `layer.input_layernorm` before projecting Q/K/V (pre-norm families like Qwen3) or not (OLMo2 applies norm post-attention).
  - `passes_sliding_window`: whether the attention call needs a `sliding_window` kwarg forwarded.
- `project_qkv(attn, hidden_states, ops)` — dispatches Q/K/V projection + optional norm + reshape based on `ops.qk_norm_layout`. **This is a two-branch function today** (`"flat"` vs. everything-else-assumed-`"per_head"`) — there is no third branch for "no Q/K-norm at all" (Granite's actual layout), so a model with that layout would silently take the `"per_head"` path and crash calling `attn.q_norm(...)` on a module that doesn't exist.
- `pre_attention_hidden(layer, hidden_states, ops)` — applies `layer.input_layernorm` if `ops.has_input_layernorm`, else passthrough.
- `resolve_attention_interface(attn, config, ops)` / `attention_call_kwargs(attn, ops)` — resolve the actual eager-attention callable and its kwargs (dropout, scaling, optional sliding window).

**Who calls `load_attention_ops`, and why each caller needs it:**

| Caller | Why |
|---|---|
| `eval/fidelity/attention.py::evaluate_attention_fidelity` | Recomputes RoPE'd query states fresh (independent of whatever the compressor under test does) to get a ground-truth `QKᵀ` for comparison — needs `project_qkv`/`apply_rotary_pos_emb` for the model's actual layout |
| `framework/qjl_online.py::enable_qjl_online` | Monkey-patches every layer's `self_attn.forward` to run the QJL estimator instead of real attention — needs the same projection/RoPE code to reproduce the model's real Q/K/V pipeline before substituting the attention math |
| `framework/rocketkv_online.py::enable_rocketkv_online` | Same monkey-patch pattern, for RocketKV's sparse selection |

Note the asymmetry this creates: **FIDELITY/attention always needs this gate, for every compressor.** **BEHAVIOR/SYSTEM only need it for `qjl`/`rocketkv`** — `identity`/`turboquant` never call `load_attention_ops` at all, because `KVCacheEngine` only special-cases those two compressor names (§3.4). This is why Granite and Gemma3 can run FIDELITY/memory and BEHAVIOR/SYSTEM-under-identity-or-turboquant today, but not FIDELITY/attention or anything QJL/RocketKV.

### 3.3 `framework/kv_cache.py` — cache representation and utilities

Defines the `CompressedKV` consumption contract and every cache-manipulation primitive `KVCacheEngine` builds on:

- **`iter_layer_kv(past_key_values)`** — the universal entry point for reading a model's cache. Three-way dispatch: `.layers` attribute (modern HF `Cache` classes, e.g. `DynamicCache`) → `.key_cache`/`.value_cache` attributes (older HF cache API) → plain tuple-of-tuples (legacy format). **This function has a hard assumption baked into its first branch: every element of `.layers` has a `.keys`/`.values` attribute.** That assumption is exactly what breaks on Qwen3.5's hybrid cache (§8.6) — `LinearAttentionLayer` objects in that `.layers` list have neither attribute.
- **`extract_layer_kv`**, **`get_cache_size_bytes`**, **`count_kv_elements`** — read-only helpers built on `iter_layer_kv`.
- **`compress_token_slice(key, value, token_idx, layer_idx, compressor)`** — slices out exactly one token position `[:, :, token_idx:token_idx+1, :]` and compresses it. This is the unit of work for incremental (per-token) compression.
- **`build_incremental_layer(...)`** — wraps a *list* of per-token payloads into one `CompressedKV` (list-valued `.keys`/`.values` fields, `nbytes` summed from `payload_list_bytes`).
- **`decompress_to_legacy_cache(compressed_layers, compressor, model_config, device)`** — the inverse operation used before every forward pass: decompresses every layer, builds a plain tuple-of-`(key, value)` pairs, then wraps it in `transformers.cache_utils.DynamicCache(ddp_cache_data=legacy, config=model_config)` (falling back to the raw tuple if that constructor signature isn't available in the installed `transformers` version).
- **`trim_compressed_cache(compressed_cache, drop_tokens, compressor)`** — drops the oldest `drop_tokens` per-token payloads from every layer's incremental list. Used by BEHAVIOR/`task_quality.py` to keep the sliding-window perplexity cache bounded at `max_length`.
- **`decompress_compressed_layer`** — dispatches to `compressor.decompress_incremental_layer` if the compressor defines one (RocketKV does — see §5.3), else concatenates per-token decompressions along the sequence dim, else falls back to `compressor.decompress(compressed)` for whole-layer (non-incremental) payloads.

### 3.4 `framework/kv_engine.py` — `KVCacheEngine`, the interception point

```python
KVCacheEngine(model, compressor)
```

At construction: if `compressor.name == "rocketkv"`, calls `enable_rocketkv_online(model, compressor)`; if `"qjl"`, calls `enable_qjl_online(model, compressor)`. Both permanently monkey-patch every decoder layer's `self_attn.forward` on the `model` object — **this mutates the shared model in place**, which is why `EvaluationRunner` must run any uncompressed baseline (BEHAVIOR's `perplexity_baseline`, SYSTEM's `throughput_baseline`) *before* constructing a `KVCacheEngine` for RocketKV/QJL, documented as an explicit ordering constraint in `eval/runner.py` and `eval/behavior/__init__.py`/`eval/system/__init__.py`.

**`engine.step(input_ids, attention_mask, compressed_cache, position_ids)`** — the core primitive every BEHAVIOR/SYSTEM metric is built from:

1. If a prior `CompressedCache` exists: decompress it back to a `DynamicCache` via `decompress_to_legacy_cache`. For RocketKV, first restore per-layer selection state (`restore_state_from_payload`) since the compressor is stateful across calls; for QJL, resync the online key-payload cache (`sync_key_payloads_from_cache`).
2. Run `self.model(input_ids, attention_mask=..., past_key_values=past_kv, position_ids=..., use_cache=True)`.
3. Compress the **newly produced** token positions only (never the ones already compressed on a prior step — see §3.4.1) and return `(logits, new_CompressedCache)`.

**`engine.generate(input_ids, max_new_tokens, attention_mask)`** — a manual greedy loop calling `step` once per token, argmax-selecting the next token from `logits[:, -1, :]`. This is what every BEHAVIOR sub-metric (`retrieval`, `reasoning`, `instruction_following`) and the compressed baseline of SYSTEM's `latency_throughput` measurement ultimately call.

**`engine.compress_existing_cache(past_key_values)`** — one-shot compression of a full snapshot (used by FIDELITY, which only needs one forward pass, not an autoregressive loop).

#### 3.4.1 Three execution modes, because compressors are structurally different

| Mode | Compressors | Mechanism |
|---|---|---|
| **Default: per-token incremental append** | `identity`, `turboquant` | `_compress_new_tokens` compresses each new token's K/V independently via `compress_token_slice`, appends to a growing Python list per layer (`build_incremental_layer`). A token's payload, once written, is **never recomputed** — an explicit design decision (re-compressing the whole cache on every step previously caused NaN perplexity, per `docs/SYSTEM_DESIGN.md`). |
| **Physical eviction** | `rocketkv` | Doesn't fit the append model — RocketKV *drops* tokens (stage-1 permanent filter + stage-2 dynamic top-k), so the cache can shrink, not just grow. `KVCacheEngine.step` special-cases `compressor.name == "rocketkv"`: after the forward pass, calls `compressor.compress_layer_from_kv(key, value, layer_idx, original_seq_len, prior_payload)` per layer instead of the generic per-token path, carrying a `RocketKVLayerPayload` (selected global indices, kept K/V, lock state) across steps. |
| **Online estimator** | `qjl` | Values are stored uncompressed; keys are never reconstructed for attention at all — `enable_qjl_online`'s patched `self_attn.forward` calls `compressor.estimate_attention_scores` directly on sign-quantized key payloads (the literature ProdQJL asymmetric estimator: float query projection, signed key projection). The per-token append path still runs (for FIDELITY/BEHAVIOR bookkeeping of the *stored* payload), but the *attention computation itself* bypasses decompression entirely. |

### 3.5 `framework/qjl_online.py` / `framework/rocketkv_online.py` — the monkey-patch mechanism

Both follow an identical pattern, illustrating exactly what "supporting a new model family" requires at the attention level:

1. `ops = load_attention_ops(model.config)` — resolve the family's projection/RoPE/attention-interface code (§3.2).
2. For every `layer in model.model.layers`: build a per-layer `forward` closure over `layer_idx`, the original `attn` module, and `ops`.
3. Inside the closure: `project_qkv(attn, hidden_states, ops)` → `apply_rotary_pos_emb(query, key, cos, sin)` → merge with `past_key_values.update(...)` if present → **method-specific substitution** (RocketKV: `apply_online_kv_sparsity` then real attention on the sparse K/V; QJL: `qjl_eager_attention_forward` using the sign-estimator instead of real attention) → `attn.o_proj(attn_output)`.
4. `attn.forward = forward` — permanent replacement, guarded by an `model._rocketkv_online_enabled` / `_qjl_online_enabled` flag so re-construction is idempotent.

Everything in step 3 after "project_qkv" is method-specific; everything before it is the same `model_adapter.py` machinery FIDELITY/attention also depends on. **This confirms the earlier claim precisely: a new model family needs exactly one new thing (a `load_attention_ops` branch) to unlock FIDELITY/attention *and* both online compressors simultaneously** — they all bottleneck through the same adapter call.

### 3.6 `framework/storage_accounting.py`, `framework/config.py`, `framework/device.py`

- `storage_accounting.py` — pure bit-counting helpers (`index_storage_bits`, `sign_storage_bits`, `float32_storage_bits`, `effective_bits_per_element`). No model dependency; used by `compressors/*.py`'s `nbytes` computation and `eval/fidelity/memory.py`.
- `config.py` — two functions, `load_model_config()`/`load_eval_config()`, both just `yaml.safe_load` against `configs/model.yaml` / `configs/eval.yaml`. `PROJECT_ROOT` is derived once from `Path(__file__).resolve().parent.parent`.
- `device.py` — `get_eval_device()`: `KV_EVAL_DEVICE` env var (`cuda`/`cpu`/`mps`) overrides; unset defaults to MPS-if-available-else-CPU. This is how Modal (`KV_EVAL_DEVICE=cuda`) and local dev share the same code path.

---

## 4. `compressors/` — the plug-in interface

### 4.1 `compressors/base.py` — the contract

```python
class KVCompressor(ABC):
    name: str = "base"
    bitwidth: int = 16

    @abstractmethod
    def compress_kv(self, tensor, layer=0, mode="key") -> object: ...
    @abstractmethod
    def decompress_kv(self, payload, mode="key") -> Tensor: ...

    def compress(self, key, value, layer=0) -> CompressedKV: ...       # provided: calls compress_kv twice
    def decompress(self, compressed) -> tuple[Tensor, Tensor]: ...     # provided: calls decompress_kv (or concatenates a list)
    def shared_storage_bytes(self) -> int: return 0                    # override for shared tables (centroids, projections)
    def compression_ratio(self, key, value) -> float: ...              # provided, convenience
```

`CompressedKV` is a `@dataclass`: `keys`, `values` (either a single payload object, or a `list[payload]` for incremental per-token storage), `original_shape`, `nbytes`, `bitwidth`, `layer`.

**Optional hooks**, checked via `hasattr` at call sites rather than declared abstract (so compressors that don't need them stay simple):

| Hook | Used by | Purpose |
|---|---|---|
| `reconstruction_error(key, value, layer)` | `eval/fidelity/representation.py` | Compressor-specific RMSE semantics (e.g. RocketKV measures error on the post-selection kept subset, not a naive full round trip) |
| `attention_fidelity(query, key, value, head_dim, num_q_heads, num_kv_heads, layer)` | `eval/fidelity/attention.py` | Distortion-only estimator path (QJL, RocketKV) — returns `(mse, rmse, cosine, max_error)` without exposing raw quantized scores |
| `estimate_attention_scores(query, key_payload, head_dim)` | `eval/fidelity/attention.py`, `framework/qjl_online.py` | Raw quantized-score estimator (QJL) — score tensor *is* exposed, so FIDELITY can additionally compute attention-output RMSE / KL divergence from it |
| `reset_state()` | `eval/fidelity/__init__.py`, `eval/behavior/*`, `eval/system/*`, `framework/kv_engine.py` | Clear stateful online bookkeeping (QJL's per-token payload cache, RocketKV's per-layer selection state) between independent runs |
| `shared_storage_bytes()` | `eval/fidelity/memory.py` | Amortized shared cost (TurboQuant's Lloyd-Max centroid table, QJL's regenerated Gaussian projections) — counted once per model run, not per token |
| `decompress_incremental_layer(compressed)` | `framework/kv_cache.py::decompress_compressed_layer` | Compressor-specific reassembly of a list of per-token payloads (RocketKV re-runs stage-1 selection on the concatenated cache rather than a naive `torch.cat`) |
| `trim_layer(compressed, drop_tokens)` | `eval/behavior/task_quality.py` | RocketKV-specific cache trimming (its payload isn't a plain per-token list, so the generic `trim_compressed_cache` doesn't apply) |
| `compress_layer_from_kv(key, value, layer, original_seq_len, prior_payload)` | `framework/kv_engine.py` (RocketKV branch) | Physical-eviction compression entry point, bypassing the per-token append path entirely |

### 4.2 The four concrete compressors

| Compressor | File | Core idea | Values compressed? | Stateful across steps? |
|---|---|---|---|---|
| `identity` | `compressors/identity.py` | Passthrough (`tensor.detach().clone()`) — validates the pipeline end to end with zero distortion | Yes (trivially, uncompressed) | No |
| `turboquant` | `compressors/turboquant.py` → `quantizers/turboquant_pipeline.py` | Pad to power-of-two → unit-normalize → orthonormal Walsh-Hadamard transform (`quantizers/hadamard.py`) → Lloyd-Max vector quantization on a fixed Gaussian-fitted codebook (`quantizers/lloyd_max.py`, seed 42) → optional QJL-sign residual encoding on the FULL stage | Yes (same pipeline; residual QJL only on `stage="full"`) | No |
| `qjl` | `compressors/qjl.py` → `quantizers/qjl_pipeline.py` | Keys: `sign(S·k)` with a fixed Gaussian projection `S` (seed 42 + head_dim), plus stored `‖k‖₂`; attention uses the literature asymmetric ProdQJL estimator (float query projection × signed key projection), never reconstructing keys | No (FP16 passthrough) | Yes — `_online_key_payloads` dict, cleared by `reset_state()` |
| `rocketkv` | `compressors/rocketkv.py` → `quantizers/rocketkv.py` | Token eviction, not quantization. Stage 1 (`TokenSelector`): SnapKV-style permanent prefix filter, locked once the cache reaches `token_budget`. Stage 2 (`HybridSparseAttention`): per-decode-step dynamic top-k selection unioned with the permanent set, capped at `hsa_budget` | Kept tokens stay FP16 (no per-element quantization — the compression *is* the eviction) | Yes — `_layer_state: dict[int, RocketKVLayerState]` (locked flag, permanent/current global index sets), cleared by `reset_state()` |

Every compressor's `nbytes`/`shared_storage_bytes` feeds `eval/fidelity/memory.py`'s compression-ratio accounting; every compressor's `compress_kv`/`decompress_kv` pair feeds `eval/fidelity/representation.py`'s relative-error/cosine-similarity computation *regardless* of whether it also defines the optional `reconstruction_error` hook (documented explicitly in that module — see §5.1 below).

New compressors register in `compressors/registry.py`'s `COMPRESSORS: dict[str, type[KVCompressor]]`; `get_compressor(name, bitwidth=None, **kwargs)` is the factory every entry point (`scripts/run_eval.py`, `modal_app/worker.py`, `eval/runner.py`'s default) goes through.

---

## 5. `eval/` — FIDELITY / BEHAVIOR / SYSTEM

Full protocol tables already live in [METHODOLOGY.md §6](../methodology/METHODOLOGY.md#6-evaluation-methodology); this section covers the *mechanics* — how the three branches are wired together and what each one actually touches.

### 5.1 `eval/fidelity/` — single offline forward pass

`evaluate_fidelity(model_layer, input_ids, compressor)`:
1. One `model(input_ids, use_cache=True, output_hidden_states=True, return_dict=True)` call.
2. `evaluate_representation(past_key_values, compressor)` — for every layer, always computes `k_hat = compressor.decompress_kv(compressor.compress_kv(key, ...))` (the abstract-interface round trip every compressor must support) to get relative-error and cosine-similarity; RMSE uses `compressor.reconstruction_error(...)` if defined, else the same round trip.
3. `evaluate_attention_fidelity(...)` — needs `model_adapter.load_attention_ops` (§3.2) to recompute fresh RoPE'd queries; branches three ways per compressor (`attention_fidelity` hook → `estimate_attention_scores` hook → generic compress/decompress + matmul), and only the latter two branches expose a raw score tensor, hence attention-output-RMSE/KL-divergence being `None` for RocketKV/QJL's distortion-only path (documented in the module itself).
4. `evaluate_memory_from_cache(...)` — uncompressed vs. compressed bytes, ratio, effective bits/element, `shared_storage_bytes()`. No model-family dependency at all.
5. `compressor.reset_state()` if defined, so BEHAVIOR/SYSTEM start clean.

### 5.2 `eval/behavior/` — through `KVCacheEngine`

`evaluate_behavior(...)` orchestrates four independent sub-evaluations, each its own `engine.generate()` pass (so BEHAVIOR's cost scales with how many are enabled):

- `task_quality.py::evaluate_perplexity` — single persistent `CompressedCache` across sliding-window strides, token-by-token `engine.step` with explicit `attention_mask`/`position_ids`, `_maybe_trim_cache` keeping it bounded via `trim_compressed_cache` (or RocketKV's `trim_layer`).
- `retrieval.py::evaluate_retrieval` — synthetic needle-in-haystack: builds a prompt of filler text + an embedded 5-digit code at a controlled depth fraction, calls `engine.generate`, checks the code string appears in the decoded completion.
- `reasoning.py::evaluate_reasoning` — synthetic add/subtract chains, exact-match on the parsed final integer from `engine.generate`'s output.
- `instruction_following.py::evaluate_instruction_following` — yes/no format-constrained prompts, checks the first decoded word is in the allowed set (structural compliance, independent of content correctness).

All four call `compressor.reset_state()` before each trial (stateful compressors like QJL/RocketKV must not leak state between independent generations).

### 5.3 `eval/system/` — also through `KVCacheEngine`

- `latency_throughput.py::evaluate_throughput` — a **manual** `engine.step()` loop (not a single `engine.generate()` call) specifically so TTFT (first step) and ITL (every subsequent step, individually timed) can be split apart; `end_to_end_latency_ms` and `tokens_per_second` are derived from the same loop.
- `vram.py::evaluate_peak_vram` — `torch.cuda.reset_peak_memory_stats` / `max_memory_allocated` bracketing `engine.generate()`; reports `cuda_available=False` gracefully off-CUDA rather than raising.
- `memory_bandwidth.py::evaluate_memory_bandwidth` — analytical, not profiled: sums `2 × cache.nbytes` per step (decompress-read + recompress-write) over a manual step loop, divides by wall time.
- `kernel_cost.py::evaluate_kernel_cost` — monkey-patches `compressor.compress_kv`/`decompress_kv` (via a `contextmanager`, restored afterward) to accumulate wall time separately from total step time; the remainder is reported as `attention_execution_time_ms` (a proxy, not a real kernel trace). Documented caveat: RocketKV's online path calls `compress_layer_from_kv` directly, bypassing this wrapper, so its real per-step cost reads as "attention" time.
- `gpu_utilization.py::evaluate_gpu_utilization` — best-effort background-thread NVML polling during `engine.generate()`; requires CUDA + `pynvml`, else returns `available=False` rather than raising.

`eval/runner.py::EvaluationRunner.run()` threads `fidelity.memory.compressed_bytes` into `SystemMetrics.actual_kv_memory_bytes` so a SYSTEM-only consumer doesn't need to cross-reference FIDELITY output separately.

---

## 6. `data/`, `modal_app/`, `reporting/`, `scripts/`, `configs/`

- **`data/loader.py`** — `load_wikitext2()` (HF `datasets` load, cached under `.cache/huggingface/datasets`), `build_long_context_ids(tokenizer, dataset, target_length)` (concatenates WikiText-2 samples with a separator until the tokenized length reaches `target_length`, truncating exactly). Deterministic given a fixed dataset split.
- **`modal_app/`** — `worker.py` defines the Modal `app`, `ensure_model` (idempotent model download into a persistent Modal Volume), and `eval_worker` (one `(compressor, context_length)` job — constructs `EvaluationRunner` exactly like local execution, so **Modal and local runs share the identical evaluation code path**, differing only in device and volume mounting). `sweep.py` is the CLI entry point (`modal run modal_app/sweep.py::main --preset ...`) that builds a job grid from `configs/modal_sweeps.yaml` presets and either `spawn_map`s them (async) or `.map`s + merges synchronously. `merge.py` flattens result JSON (supporting both the current `fidelity`/`behavior`/`system` shape and legacy `section_a_fidelity`/`section_b_inference` archives) into CSV. `job_spec.py`/`settings.py` define job parameters and GPU/volume/secret configuration.
- **`reporting/reporter.py`** — `ResultReporter.save_json`/`save_summary_csv`/`print_summary`, all operating on `EvaluationResult.to_dict()`.
- **`scripts/`** — `run_eval.py` (primary local CLI, all FIDELITY/BEHAVIOR/SYSTEM flags), `run_baseline.py`, `run_turboquant_sweep.py` (TurboQuant ablation grid), `download_model.py`/`download_candidate_models.py`, verification scripts (`verify_kv_cache.py`, `verify_phase1_model.py`), Modal shell wrappers, result post-processing (`restructure_*_results.py`, `export_results_documentation.py`).
- **`configs/`** — `model.yaml` (path, context lengths, TurboQuant defaults), `eval.yaml` (dataset config, stride, `attention_fidelity_tokens`, `generated_tokens`), `modal.yaml`/`modal_sweeps.yaml` (GPU spec, sweep presets), plus per-model variants (`model_qwen3.yaml`, `modal_qwen3.yaml`).

---

## 7. End-to-end execution flow (one full run)

```text
scripts/run_eval.py --compressor turboquant --context-length 512
   │
   ├─ get_compressor("turboquant", ...)              compressors/registry.py
   ├─ EvaluationRunner(compressor=...)                eval/runner.py
   │    ├─ ModelLayer()                               framework/model.py
   │    │    └─ AutoModelForCausalLM.from_pretrained(attn_implementation="eager")
   │    └─ load_wikitext2()                           data/loader.py
   │
   └─ runner.run(context_length=512, ...)
        ├─ build_context(512)                         data/loader.py::build_long_context_ids
        │
        ├─ evaluate_fidelity(model_layer, input_ids, compressor)     [ALWAYS]
        │    ├─ model(input_ids, use_cache=True, output_hidden_states=True)   ← ONE forward pass
        │    ├─ evaluate_representation(past_key_values, compressor)
        │    ├─ evaluate_attention_fidelity(...)       needs model_adapter.load_attention_ops
        │    └─ evaluate_memory_from_cache(...)
        │
        ├─ evaluate_behavior(model_layer, input_ids, compressor, ...)  [if run_behavior]
        │    ├─ evaluate_perplexity(...)                              engine.step() × N tokens
        │    ├─ evaluate_retrieval(...)          [opt-in]             engine.generate()
        │    ├─ evaluate_reasoning(...)          [opt-in]             engine.generate() × trials
        │    └─ evaluate_instruction_following(...) [opt-in]          engine.generate() × trials
        │
        ├─ evaluate_system(model_layer, input_ids, compressor, ...)   [if run_system]
        │    ├─ evaluate_throughput(...)                              manual engine.step() loop
        │    ├─ evaluate_peak_vram(...)          [opt-in]
        │    ├─ evaluate_memory_bandwidth(...)   [opt-in]
        │    ├─ evaluate_kernel_cost(...)        [opt-in]
        │    └─ evaluate_gpu_utilization(...)    [opt-in]
        │
        └─ EvaluationResult(fidelity=..., behavior=..., system=...)
             │
             └─ ResultReporter.save_json / save_summary_csv / print_summary
```

Every `engine.step()`/`engine.generate()` call above internally does: decompress prior `CompressedCache` → `model.forward(past_key_values=...)` → compress new tokens (mode depends on compressor, §3.4.1) → return new `CompressedCache`. For `qjl`/`rocketkv`, the model's `self_attn.forward` has already been permanently replaced (§3.5) before any of this runs, so "forward" for those two doesn't mean standard attention at all — it means the monkey-patched estimator/sparsity path.

---

## 8. Diversifying to other architecture families

Building directly on [SLM_COMPATIBILITY.md](SLM_COMPATIBILITY.md)'s finding — the engine supports **dense, decoder-only transformers with a uniform per-layer K/V cache**, and within that class needs one more thing (a `model_adapter.py` branch) for full FIDELITY/attention + QJL/RocketKV support — this section is the concrete engineering plan for closing that gap, ordered by how structurally deep the change is. The tier framework below is general and durable; the specific example model at each tier has been updated to the current 5-model architecture-matrix shortlist (`olmo2_1b`/`qwen3_0.6b`/`gemma3_270m`/`falcon_h1_0.5b`/`tinydeepseek_0.5b` — see [MODEL_ARCHITECTURE_MATRIX.md](MODEL_ARCHITECTURE_MATRIX.md) for the live-probed detail behind each). The originally-named examples (Granite 4.0 350M, MiniCPM4-0.5B, Qwen3.5-0.8B) were deleted from `models/` once the shortlist was finalized — the mechanisms below still apply to their successors in the current shortlist.

### 8.1 Tier 0 (no code change needed): FIDELITY/representation, FIDELITY/memory, BEHAVIOR/SYSTEM under `identity`/`turboquant`

Any model that (a) loads via `AutoModelForCausalLM.from_pretrained(..., attn_implementation="eager")` and (b) returns a `past_key_values` where every layer exposes one `(key, value)` tensor pair already works for this entire subset — no `model_adapter.py` involvement. This is a strictly larger class than "the two wired families" — it's why `gemma3_270m` and `tinydeepseek_0.5b` (loaded via `AutoModelForCausalLM` without `trust_remote_code`, which resolves to transformers' *native* `deepseek_v3` architecture rather than the checkpoint's vendored custom code — see [MODEL_ARCHITECTURE_MATRIX.md](MODEL_ARCHITECTURE_MATRIX.md)) already pass FIDELITY/representation+memory today, mechanically, with zero adapter work.

### 8.2 Tier 1: generalize `load_attention_ops` from an if/elif chain to a registry + auto-derivation

**Current limitation:** every new family requires manually finding and importing that family's `apply_rotary_pos_emb`/`eager_attention_forward`/`ALL_ATTENTION_FUNCTIONS` symbols and hardcoding an `elif model_type == "...":` branch.

**Proposal:** most modern HF model families follow the exact same module-naming convention this code already exploits (`transformers.models.<model_type>.modeling_<model_type>`), and export the same three symbol names. A generic fallback branch could attempt:

```python
def _try_auto_import(model_type: str) -> AttentionOps | None:
    try:
        mod = importlib.import_module(f"transformers.models.{model_type}.modeling_{model_type}")
        return AttentionOps(
            model_type=model_type,
            apply_rotary_pos_emb=mod.apply_rotary_pos_emb,
            eager_attention_forward=mod.eager_attention_forward,
            all_attention_functions=mod.ALL_ATTENTION_FUNCTIONS,
            qk_norm_layout=_infer_qk_norm_layout(mod),   # see 8.3
            has_input_layernorm=True,                      # needs a real heuristic or explicit override table
            passes_sliding_window=False,
        )
    except (ImportError, AttributeError):
        return None
```

...used only when no explicit branch matches, with explicit branches remaining the source of truth for known-correct configurations (Qwen3/Qwen2/OLMo2 today, Granite/Gemma3 once added — §8.3/8.4). This turns "add a family" from "read the HF source, hand-write a branch" into "verify the auto-derived branch is correct, optionally override two fields" for the common case — real time savings, but **not a substitute for per-family verification**, since `has_input_layernorm`, `qk_norm_layout`, and sliding-window handling are genuine semantic differences auto-import can't infer from symbol presence alone.

### 8.3 Tier 1: add a third `qk_norm_layout` — fixes Falcon-H1

`project_qkv` currently branches on exactly two layouts. Falcon-H1-0.5B's attention module (confirmed live: `FalconH1Attention` has neither `q_norm` nor `k_norm`) needs a third:

```python
if ops.qk_norm_layout == "flat":
    ...  # OLMo2
elif ops.qk_norm_layout == "none":
    query_states = attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
else:  # "per_head" — Qwen3, Gemma3
    ...
```

Plus one new `load_attention_ops` branch importing from `transformers.models.falcon_h1.modeling_falcon_h1` (needs verification these exports exist with matching signatures — Falcon-H1's attention layer additionally carries its own `attention_in_multiplier`/`attn_out_multiplier`/`key_multiplier` gating scalars not present in Qwen3/OLMo2, so the calling convention may differ even once `qk_norm_layout="none"` is added). Smallest *adapter-only* fix in this document, but note Falcon-H1 also needs the Tier 3 hybrid-cache work below (§8.6) before it's fully correct — the adapter fix alone unblocks FIDELITY/attention and QJL/RocketKV online, but doesn't fix `iter_layer_kv` silently dropping its Mamba state.

### 8.4 Tier 2: per-layer RoPE tables — fixes Gemma3

**Current limitation:** every call site (`eval/fidelity/attention.py::_compute_layer_queries`, both `enable_*_online` patches) computes `(cos, sin)` **once** and reuses it for every layer — correct for Qwen3/OLMo2 (one global RoPE config) but wrong for Gemma3 (`Gemma3RotaryEmbedding.forward(x, position_ids, layer_type=None)` requires an explicit `layer_type` to select between `sliding_attention_inv_freq`/`full_attention_inv_freq`).

**Proposal:** extend `AttentionOps` with a `rope_table_fn: Callable[[layer_idx], tuple[str,str]] | None` (or simpler: a `layer_types: list[str] | None` field read once from `config.layer_types`), and change every call site from:

```python
position_embeddings = model.model.rotary_emb(hidden_states, position_ids)   # one shared table
...
cos, sin = position_embeddings
```

to computing (and caching) one `(cos, sin)` pair **per distinct layer type** up front, then selecting the right one per layer:

```python
rope_tables = {lt: model.model.rotary_emb(hidden_states, position_ids, layer_type=lt) for lt in set(ops.layer_types)}
...
cos, sin = rope_tables[ops.layer_types[layer_idx]]
```

This is a real (if mechanical) change to three files (`eval/fidelity/attention.py`, `framework/qjl_online.py`, `framework/rocketkv_online.py`), not a one-line fix — but it's a well-understood, already-shipped-in-transformers pattern, not new research.

### 8.5 Tier 2: MLA-native state — TinyDeepSeek's real blocker isn't `trust_remote_code`

The original version of this section (when the shortlist candidate was named MiniCPM4-0.5B) proposed a `trust_remote_code` opt-in flag as the blocker to solve. **For the current shortlist's MLA model, TinyDeepSeek-0.5B, that turned out not to be the actual gate**: `framework/model.py::ModelLayer.__init__` never passes `trust_remote_code`, and — confirmed live — `AutoModelForCausalLM.from_pretrained` on `models/tinydeepseek_0.5b` without it **succeeds**, resolving to transformers' native `DeepseekV3ForCausalLM` (the config's `model_type="deepseek_v3"` matches a real, already-supported transformers architecture) instead of the checkpoint's vendored `modeling_tinydeepseek.py`. The vendored custom code (which *does* fail to import against this repo's `transformers==5.8.1`) is simply never reached by the engine's actual load path. See [MODEL_ARCHITECTURE_MATRIX.md](MODEL_ARCHITECTURE_MATRIX.md) for the full trace, including the live-probed `DeepseekV3Attention` module (`kv_a_layernorm`, `kv_a_proj_with_mqa`, `kv_b_proj`, `kv_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim` — genuine MLA machinery) and the observed cache shape (asymmetric key/value last-dim: keys carry the nope+rope split, values don't).

The `trust_remote_code` policy proposal (opt-in config flag, loud warning, review-before-enable) is still generically valid engine hygiene for *some future* model that genuinely requires vendored code with no native-transformers fallback — just not the blocker for TinyDeepSeek specifically. TinyDeepSeek's actual remaining gap is a plain Tier 1 `load_attention_ops` branch for `model_type == "deepseek_v3"` (same category as Falcon-H1 above), **plus** a deeper question once that adapter exists: whether `iter_layer_kv`'s generic `(keys, values)` extraction — which today reads the native implementation's already-expanded per-head cache, not DeepSeek's actual compressed latent (`kv_lora_rank`-dim) representation — is scientifically the right thing to benchmark, or whether a real MLA-state type (per §8.6's typed-layer direction) is needed to compress the *native* latent representation instead of a post-hoc reconstruction.

### 8.6 Tier 3 (new engine capability, not an adapter): heterogeneous / hybrid caches

**This is the real ceiling, illustrated exactly by Falcon-H1-0.5B** (Qwen3.5-0.8B, the originally-named example, showed the same failure mode but has since been removed from `models/`). `iter_layer_kv`'s core assumption — every layer's cache entry has `.keys`/`.values` (or occupies one slot in parallel `.key_cache`/`.value_cache` lists) — is **misleadingly, not cleanly, false** for Falcon-H1: its cache layer class (`LinearAttentionAndFullAttentionLayer`) *does* expose `.keys`/`.values`, so `iter_layer_kv` doesn't crash (unlike Qwen3.5's `LinearAttentionLayer`, which has no such attributes at all) — but those attributes only cover the attention half of a genuinely dual-state layer. The other 24 (of 36) Mamba-mixer heads' worth of recurrent state per layer is invisible to the engine entirely, so today any compression run against Falcon-H1 would silently produce plausible-looking but incorrect memory/compression-ratio numbers rather than an honest error. This is **gate 1** (§8.1's "generic engine compatibility"), the most fundamental of the three gates — it fails before `model_adapter.py` is ever consulted, so no adapter registration can fix it, and Falcon-H1's case is *worse* than Qwen3.5's clean crash because it fails silently.

**What real support would require** (scoped, not a one-line change, but also not unbounded research):

1. **Layer classification.** `iter_layer_kv` (and everything built on it — `apply_compressor`, `decompress_to_legacy_cache`, `compress_token_slice`'s callers) would need to know, per layer, whether it's `"attention"` (has real K/V, compressible), `"recurrent"` (Mamba/linear-attention state, opaque), or — Falcon-H1's actual case — **both simultaneously on the same layer object**. `config.layer_types` is `["hybrid"] * 36` for Falcon-H1 (not a per-layer attention-vs-recurrent split like some other hybrid families use) — the type signal alone doesn't disambiguate; the fix needs to inspect the cache-layer class itself (`LinearAttentionAndFullAttentionLayer` vs. plain `DynamicLayer`) or the decoder-layer's submodules (presence of both `self_attn`-style projections and a `.mamba` submodule, both confirmed present on `FalconH1DecoderLayer`).
2. **Selective interception.** `KVCacheEngine` would compress/decompress only the attention-typed state, passing recurrent/Mamba state through the engine untouched (read on decompress, write back unmodified on compress) — architecturally a filter, not a rewrite of the core step loop.
3. **What's explicitly *not* included:** actually compressing recurrent/linear-attention state. That state isn't a growing K/V sequence; it's a fixed-size accumulator with completely different information-theoretic properties, and "how do you meaningfully compress a Mamba SSM state" is a genuinely open research question, not an engineering gap. Scoping the work to "pass hybrid layers through uncompressed, compress only the attention layers" keeps this a bounded engine change rather than a new research program.
4. **Near-term alternative, already noted in [SLM_COMPATIBILITY.md](SLM_COMPATIBILITY.md):** until (1)–(2) are built, an explicit guard at `ModelLayer`/`KVCacheEngine` construction time — reject any model whose decoder layer carries both attention and recurrent submodules (or whose cache layer class isn't a plain `DynamicLayer`/`DynamicSlidingWindowLayer`), with a clear error message — is strictly better than today's behavior for Falcon-H1 specifically, which is a **silent under-count**, not even a raised exception.

### 8.7 A conformance test as the real definition of "supported"

Whatever tier a new family lands in, the practical bar for calling it "supported" should be a repeatable check, not a manual smoke test — e.g. a `tests/test_model_adapter_conformance.py` pattern (parallel to the existing `tests/test_phase1_model.py`) that, given a `model_path`, asserts: model loads eager; `past_key_values` round-trips through `iter_layer_kv`; `load_attention_ops` resolves without raising; a manually-recomputed attention score via `model_adapter` matches the model's own real forward-pass attention output within floating-point tolerance (the strongest correctness check — confirms the adapter's projection/RoPE/norm reconstruction is bit-for-bit faithful to what the model actually does internally, not just "doesn't crash"). This would turn "is model X supported" from a documentation claim into something CI enforces.

### 8.8 Summary: effort ordering

| Tier | Fixes (current shortlist) | Scope | Effort |
|---|---|---|---|
| 0 | Nothing (already works) | `olmo2_1b`, `qwen3_0.6b` fully; `gemma3_270m`/`tinydeepseek_0.5b` FIDELITY/repr+mem and BEHAVIOR/SYSTEM under identity/turboquant | none |
| 1 | Falcon-H1 (adapter only, not the cache gap) | One `qk_norm_layout="none"` value + one `load_attention_ops` branch | small, contained |
| 1 | TinyDeepSeek (adapter only, not the latent-state question) | One `load_attention_ops` branch for `model_type="deepseek_v3"` | small, contained |
| 2a | Gemma3 | Per-layer RoPE table, touching 3 call sites | medium, mechanical, well-understood pattern |
| 3 | Falcon-H1 (full correctness — the silent Mamba-state undercount) | Layer-type-aware cache interception in the core engine (`iter_layer_kv` and everything built on it) | substantial, but bounded if scoped to "pass non-attention layers through uncompressed" rather than "compress everything" |
| 3 | TinyDeepSeek (benchmarking the *native* latent representation, not a reconstruction) | Same typed-state direction as Falcon-H1, applied to a compressed-latent state instead of recurrent state | substantial, open scientific question on top of the engineering work |

Full per-model live-probe evidence behind every row above: [MODEL_ARCHITECTURE_MATRIX.md](MODEL_ARCHITECTURE_MATRIX.md).

**Bottom line:** the framework's dense-transformer ceiling isn't a hard architectural wall for Granite or Gemma3 — both are reachable with contained, well-scoped changes to `framework/model_adapter.py` (and, for Gemma3, three call sites that assume one shared RoPE table). The real wall is hybrid/recurrent architectures (§8.6), which need a genuine new capability — selective, layer-type-aware interception — added to the core engine, not just a new adapter branch. That capability is buildable without solving the open research question of compressing recurrent state, by explicitly scoping it to attention-layer interception only.
