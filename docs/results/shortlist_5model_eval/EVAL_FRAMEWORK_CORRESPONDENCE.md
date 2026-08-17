# 5-Model Shortlist — Evaluation-Framework Correspondence

What actually happens when the real FIDELITY/BEHAVIOR/SYSTEM evaluation pipeline (`scripts/run_eval.py` → `eval/runner.py`) is pointed at each of the 5 architecture-matrix shortlist models (`olmo2_1b` MHA, `qwen3_0.6b` GQA, `gemma3_270m` MQA+local/global, `falcon_h1_0.5b` Hybrid Attn+Mamba2, `tinydeepseek_0.5b` MLA). This is the live, run-the-actual-CLI companion to the static architecture probe in [`../../architecture/MODEL_ARCHITECTURE_MATRIX.md`](../../architecture/MODEL_ARCHITECTURE_MATRIX.md) / [`models/ARCHITECTURE_REPORT.md`](../../../models/ARCHITECTURE_REPORT.md) — those establish *what each model's modules/cache look like*; this establishes *what the eval framework actually does when run against them, verbatim, with real numbers and real tracebacks*.

Run 2026-08-21: `python scripts/run_eval.py --compressor identity --context-length 128` against each model in turn (via a per-model `configs/model_<name>.yaml`, temporarily swapped into `configs/model.yaml`). `transformers==5.8.1`, this repo's `.venv`.

## Two real bugs found and fixed while getting these numbers

Both were pre-existing engine/environment defects, unrelated to the shortlist models themselves — surfaced only because this was the first time `scripts/run_eval.py` had actually been re-run locally in this environment recently:

1. **`configs/model.yaml`/`configs/model_qwen3.yaml` pointed at `models/qwen3_1.7b`**, which no longer exists after the `models/legacy/` reorg (see `models/ARCHITECTURE_REPORT.md`). Fixed: both now point at `models/legacy/qwen3_1.7b`.
2. **The `datasets` package (HF datasets, required by `data/loader.py` for WikiText-2) was not installed** in this venv despite being listed in `requirements.txt` — every local eval run failed at `EvaluationRunner.__init__` with `ImportError: cannot import name 'load_dataset' from 'datasets'`, for *every* model, not just the shortlist. Fixed: `pip install datasets` (resolved to `datasets==5.0.1`).

Both are now fixed in the working tree; anyone hitting either error should re-check `configs/model*.yaml` paths against the current `models/` layout and confirm `pip show datasets` succeeds.

## Results: models that ran end to end

`olmo2_1b` and `qwen3_0.6b` — both zero-adapter-work models — completed the full default run (FIDELITY always-on + BEHAVIOR/task_quality + SYSTEM/latency_throughput) successfully. Raw output: `results/shortlist_olmo2_1b_identity.{json,csv}`, `results/shortlist_qwen3_0.6b_identity.{json,csv}` (gitignored; below are the actual values from that run).

| Metric | olmo2_1b | qwen3_0.6b |
|---|---:|---:|
| key/value cosine similarity | 0.99999997 / 0.99999996 | 1.00000008 / 1.00000001 |
| attention cosine | 0.99982 | 0.99988 |
| compression ratio (identity) | 1.0× | 1.0× |
| perplexity (ctx=128) | 13.979 | 22.875 |
| tokens/sec | 16.90 | 17.31 |
| TTFT (ms) | 45.26 | 47.63 |
| end-to-end latency (ms) | 3788.0 | 3696.4 |

(Identity compressor is a correctness/plumbing check, not a compression result — ratio 1.0× and near-1.0 cosine values are expected and confirm the pipeline round-trips correctly for both models. The interesting numbers for these two models are already published in [`../qwen3_1.7b/`](../qwen3_1.7b/) and [`../olmo2_1b/`](../olmo2_1b/) at the legacy models' full context-length/compressor grid — this run is scoped to confirming the *shortlist* checkpoints specifically plumb through cleanly, at a fast ctx=128 identity smoke-test, not to replace those larger sweeps.)

**FIDELITY, BEHAVIOR, and SYSTEM all completed** for both models — this is real, current, evidence-based confirmation that `olmo2_1b`/`qwen3_0.6b` are fully supported end to end by the current engine, not just an inference from adapter-code inspection.

## Results: models that failed, and exactly where

All three remaining models fail inside `eval/fidelity/__init__.py::evaluate_fidelity` → `eval/fidelity/attention.py::evaluate_attention_fidelity` → `_compute_layer_queries` → `framework/model_adapter.py::load_attention_ops`. Because **FIDELITY always runs first and is not gated behind a CLI flag** (`eval/runner.py::run`), none of these three models ever reach BEHAVIOR or SYSTEM in a `run_eval.py` invocation today — the run aborts with an unhandled exception before those branches execute, even though (per the architecture probe) BEHAVIOR/SYSTEM under `identity`/`turboquant` would very likely work mechanically for two of the three if FIDELITY/attention were skipped or fixed.

### `gemma3_270m`

```
File "eval/fidelity/attention.py", line 151, in evaluate_attention_fidelity
    position_embeddings = model.model.rotary_emb(hidden_states[0], position_ids)
File ".../transformers/models/gemma3/modeling_gemma3.py", ... in forward
    inv_freq = getattr(self, f"{layer_type}_inv_freq")
AttributeError: 'Gemma3RotaryEmbedding' object has no attribute 'None_inv_freq'
```

Confirms exactly the mechanism documented in `models/ARCHITECTURE_REPORT.md`: the engine calls `rotary_emb` once globally (no `layer_type`), but Gemma3's rotary module requires one of `sliding_attention`/`full_attention` to select between its two independent buffer sets.

### `falcon_h1_0.5b`

```
File "eval/fidelity/attention.py", line 107, in _compute_layer_queries
    ops = load_attention_ops(model_layer.config)
File "framework/model_adapter.py", line 83, in load_attention_ops
    raise NotImplementedError(
NotImplementedError: Online attention adapters are not implemented for model_type='falcon_h1'. Supported: qwen3, olmo2.
```

Simple adapter-registry gate — model loads and forwards fine up to this point (weights load cleanly, ~2 min for 579 shards).

### `tinydeepseek_0.5b`

```
File "eval/fidelity/attention.py", line 107, in _compute_layer_queries
    ops = load_attention_ops(model_layer.config)
File "framework/model_adapter.py", line 83, in load_attention_ops
    raise NotImplementedError(
NotImplementedError: Online attention adapters are not implemented for model_type='deepseek_v3'. Supported: qwen3, olmo2.
```

**This is a significant, positive correction to the static architecture probe**, which (loading with `trust_remote_code=True` to inspect the vendored code directly) found an `ImportError` and concluded the model couldn't be assessed at all. Run through the engine's *actual* load path (`framework/model.py::ModelLayer`, which never passes `trust_remote_code`), the model loads cleanly in ~29s as transformers' **native** `DeepseekV3ForCausalLM` — the vendored `modeling_tinydeepseek.py` (whose `is_torch_fx_available` import fails against `transformers==5.8.1`) is never reached, because `model_type="deepseek_v3"` in `config.json` matches a real, already-supported transformers architecture. Confirmed live: `DeepseekV3Attention` exposes genuine MLA submodules (`kv_a_layernorm`, `kv_a_proj_with_mqa`, `kv_b_proj`, `kv_lora_rank=256`, `qk_nope_head_dim=32`, `qk_rope_head_dim=32`, `v_head_dim=32`), and a real forward pass succeeds — `iter_layer_kv` also succeeds (26 `DynamicLayer`s, keys shape `(1,4,5,64)`, values shape `(1,4,5,32)` — note the **asymmetric key/value last dimension**, 64 = `qk_nope_head_dim + qk_rope_head_dim`, 32 = `v_head_dim`, a structural difference from every other model in the shortlist where key/value share one `head_dim`). TinyDeepSeek's actual, sole blocker for a `run_eval.py` invocation today is the same one-line `load_attention_ops` adapter gate as Falcon-H1 and Gemma3, **not** an unrecoverable import failure.

(Caveat worth carrying forward: the cache being iterated here is the native implementation's already-expanded per-head K/V, not DeepSeek's actual compressed-latent representation — see the "what would count as really supporting MLA" discussion in `../../architecture/MODEL_ARCHITECTURE_MATRIX.md`.)

## Correspondence table — what would run today if FIDELITY/attention were skipped

Speculative (not run, since there's no `--skip-fidelity` flag today) but directly inferable from confirmed `iter_layer_kv` behavior in `models/ARCHITECTURE_REPORT.md`:

| Model | FIDELITY/representation+memory | FIDELITY/attention | BEHAVIOR+SYSTEM (identity/turboquant) | BEHAVIOR+SYSTEM (qjl/rocketkv) |
|---|:---:|:---:|:---:|:---:|
| olmo2_1b | ✅ (confirmed, ran) | ✅ (confirmed, ran) | ✅ (confirmed, ran) | ✅ (adapter exists) |
| qwen3_0.6b | ✅ (confirmed, ran) | ✅ (confirmed, ran) | ✅ (confirmed, ran) | ✅ (adapter exists) |
| gemma3_270m | ✅ (doesn't touch `rotary_emb`) | ❌ confirmed crash | ✅ likely (generic `iter_layer_kv` path) | ❌ (adapter gate) |
| tinydeepseek_0.5b | ✅ (doesn't touch `rotary_emb`) | ❌ confirmed `NotImplementedError` | ✅ likely (generic `iter_layer_kv` path, confirmed working) | ❌ (adapter gate) |
| falcon_h1_0.5b | ⚠️ likely runs, but **would silently undercount memory** — see below | ❌ confirmed `NotImplementedError` | ⚠️ likely runs, but **same silent undercount** | ❌ (adapter gate) |

Falcon-H1 is marked ⚠️ rather than ✅ even for the "would run" columns because `iter_layer_kv` succeeding on it is misleading, not correct: its cache layer (`LinearAttentionAndFullAttentionLayer`) exposes `.keys`/`.values` for the attention half only, so any compression/memory-accounting run today would silently exclude all Mamba recurrent state (24 of the model's attention+Mamba heads per layer) — producing a plausible-looking but wrong compression ratio, not an honest failure.

## Engine adoption and transformation plan

Full technical detail (code-level proposals, effort ordering, tier-by-tier scope) lives in [`ENGINE_INTERNALS.md §8`](../../architecture/ENGINE_INTERNALS.md#8-diversifying-to-other-architecture-families) — this is the ranked summary, cross-checked against the *live* results above rather than static analysis alone:

1. **Add `load_attention_ops` branches for `gemma3_text`, `falcon_h1`, `deepseek_v3`.** Confirmed live as the actual point of failure for all 3 blocked models. Falcon-H1 additionally needs a new `qk_norm_layout="none"` (confirmed no `q_norm`/`k_norm` exist on `FalconH1Attention`). Cheapest fix, unblocks the *adapter gate* for all three — but not sufficient on its own for Gemma3 or Falcon-H1 (see below).
2. **Per-layer RoPE selection in `eval/fidelity/attention.py` (and `framework/qjl_online.py`/`framework/rocketkv_online.py`).** Confirmed live as Gemma3's actual crash (`Gemma3RotaryEmbedding` genuinely holds two independent buffer sets). Required in addition to (1) for Gemma3 specifically.
3. **A real hybrid-cache abstraction in `framework/kv_cache.py`.** Confirmed live as necessary for Falcon-H1 to be *correctly* measured, not just to avoid a crash — today's `iter_layer_kv` silently drops its Mamba state rather than erroring, which is a worse failure mode than the other two models' clean exceptions. Required in addition to (1) for Falcon-H1 to be trustworthy, not merely non-crashing.
4. **A latent-KV (MLA) state type, same abstraction as (3).** Needed for TinyDeepSeek once (1) unblocks it, to benchmark DeepSeek's actual compressed-latent cache rather than the native implementation's already-materialized per-head reconstruction — an open scientific-scope question, not just an engineering task.
5. **Consider gating FIDELITY behind a `--skip-fidelity`-style flag**, newly motivated by this run: today, a single unimplemented adapter blocks BEHAVIOR/SYSTEM entirely even for models where those two branches would otherwise run cleanly (confirmed: `iter_layer_kv` succeeds standalone for all 3 blocked models). This isn't in `ENGINE_INTERNALS.md §8` — it's a smaller, orthogonal, immediately-actionable finding from actually running the CLI rather than reading the code.

None of items 1–4 are new relative to the static architecture-report analysis; what this document adds is **live confirmation** (real tracebacks, real successful runs, real numbers) rather than inference from module inspection alone — and one correction (TinyDeepSeek loads fine through the real engine path) plus one new finding (item 5) that only running the actual CLI surfaced.
