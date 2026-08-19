# 5-Model Shortlist — Evaluation-Framework Correspondence

What actually happens when the real FIDELITY/BEHAVIOR/SYSTEM evaluation pipeline (`scripts/run_eval.py` → `eval/runner.py`) is pointed at each of the 5 architecture-matrix shortlist models (`olmo2_1b` MHA, `qwen3_0.6b` GQA, `gemma3_270m` MQA+local/global, `falcon_h1_0.5b` Hybrid Attn+Mamba2, `tinydeepseek_0.5b` MLA). This is the live, run-the-actual-CLI companion to the static architecture probe in [`../../architecture/MODEL_ARCHITECTURE_MATRIX.md`](../../architecture/MODEL_ARCHITECTURE_MATRIX.md) / [`models/ARCHITECTURE_REPORT.md`](../../../models/ARCHITECTURE_REPORT.md) — those establish *what each model's modules/cache look like*; this establishes *what the eval framework actually does when run against them, verbatim, with real numbers and real tracebacks*.

Run 2026-08-21: `python scripts/run_eval.py --compressor identity --context-length 128` against each model in turn (via a per-model `configs/model_<name>.yaml`, temporarily swapped into `configs/model.yaml`). `transformers==5.8.1`, this repo's `.venv`.

**Update 2026-08-19 (94th commit):** TinyDeepSeek support landed — `deepseek_v3` adapter, `project_attention_states()` / MLA split-RoPE path, asymmetric K/V memory, and full eval-branch verification in `tests/test_tinydeepseek_reference.py`. **4 of 5 models pass Gate B**; TinyDeepSeek Gate C still fails (expanded KV, not native latent). See [Current status (2026-08-19)](#current-status-2026-08-19).

**Update 2026-08-18 (86th commit):** Gemma3 support landed — `gemma3_text` adapter, per-layer RoPE (`framework/rope.py`), and full eval-branch verification in `tests/test_gemma3_reference.py`. Live gate evaluation confirms **3 of 5 models pass all three compatibility gates** (`olmo2_1b`, `qwen3_0.6b`, `gemma3_270m`). Full audit: [`ENGINE_AND_EVALUATION_FRAMEWORKS_REDESIGN_PLAN.md § audit`](../../ENGINE_AND_EVALUATION_FRAMEWORKS_REDESIGN_PLAN.md#implementation-verification-audit-2026-08-18). The 2026-08-21 sections below remain as historical record for the initial probe.

## Two real bugs found and fixed while getting these numbers

Both were pre-existing engine/environment defects, unrelated to the shortlist models themselves — surfaced only because this was the first time `scripts/run_eval.py` had actually been re-run locally in this environment recently:

1. **`configs/model.yaml`/`configs/model_qwen3.yaml` pointed at `models/qwen3_1.7b`**, which no longer exists after the `models/legacy/` reorg (see `models/ARCHITECTURE_REPORT.md`). Fixed: both now point at `models/legacy/qwen3_1.7b`.
2. **The `datasets` package (HF datasets, required by `data/loader.py` for WikiText-2) was not installed** in this venv despite being listed in `requirements.txt` — every local eval run failed at `EvaluationRunner.__init__` with `ImportError: cannot import name 'load_dataset' from 'datasets'`, for *every* model, not just the shortlist. Fixed: `pip install datasets` (resolved to `datasets==5.0.1`).

Both are now fixed in the working tree; anyone hitting either error should re-check `configs/model*.yaml` paths against the current `models/` layout and confirm `pip show datasets` succeeds.

## Current status (2026-08-19)

Verified via live gate evaluation and `tests/test_*_reference.py`:

| Model | Gate A (loader/state) | Gate B (attention) | Gate C (state semantics) | Full eval (FIDELITY+BEHAVIOR+SYSTEM) |
|---|---|---|---|---|
| `olmo2_1b` | PASS | PASS | PASS | ✅ all compressors |
| `qwen3_0.6b` | PASS | PASS | PASS | ✅ all compressors |
| `gemma3_270m` | PASS | PASS | PASS | ✅ identity, TurboQuant, QJL, RocketKV (`test_gemma3_reference.py`) |
| `tinydeepseek_0.5b` | PASS | PASS | FAIL | ✅ identity, TurboQuant, QJL, RocketKV (`test_tinydeepseek_reference.py`; expanded KV disclosure) |
| `falcon_h1_0.5b` | PASS | PASS | PASS | ✅ identity, TurboQuant, QJL, RocketKV (`test_falcon_h1_reference.py`) |

### `tinydeepseek_0.5b` — now supported on expanded-cache path

Changes in 92–94th commits:
- `deepseek_v3` registered in `ATTENTION_ADAPTER_REGISTRY` (`qk_norm_layout="mla"`)
- `project_attention_states()` / `project_mla_qkv()` for split nope/RoPE + interleaved RoPE
- Asymmetric K/V memory: `resolve_key_head_dim()` / `resolve_value_head_dim()` (64/32)
- `get_model_eval_metadata` records `cache_representation="expanded_kv"` + MLA fields
- All eval branches pass in `tests/test_tinydeepseek_reference.py` (12 tests, ~29 min)
- Gate C **still fails** by design: visible cache is HF expanded K/V, not native `kv_lora_rank` latent

## Current status (2026-08-18) — superseded for TinyDeepSeek

<details>
<summary>Prior status before TinyDeepSeek adapter (94th commit)</summary>

| Model | Gate A | Gate B | Gate C | Full eval |
|---|---|---|---|---|
| `tinydeepseek_0.5b` | PASS | FAIL | FAIL | ❌ Gate B blocks FIDELITY |

</details>

### `gemma3_270m` — now fully supported

Changes in 86th commit:
- `gemma3_text` registered in `ATTENTION_ADAPTER_REGISTRY`
- `build_rope_context().get_rope(layer_idx)` selects sliding vs full RoPE per layer
- `get_model_eval_metadata` records per-layer `layer_attention` metadata
- All eval branches pass in `tests/test_gemma3_reference.py`

The historical RoPE crash documented below (2026-08-21 probe) is **resolved**.

## Results: models that ran end to end (2026-08-21 initial probe)

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

## Results: models that failed, and exactly where (2026-08-21 initial probe — historical)

At the time of the initial probe, all three remaining models failed inside `eval/fidelity/__init__.py::evaluate_fidelity` → `eval/fidelity/attention.py::evaluate_attention_fidelity` → `_compute_layer_queries` → `framework/model_adapter.py::load_attention_ops`. **Gemma3 is no longer in this category** (fixed 2026-08-18). By default **FIDELITY runs first** (`eval/runner.py::run`); use `run_fidelity=False` or `scripts/run_eval.py --skip-fidelity` to collect BEHAVIOR/SYSTEM on models where Gate A passes but Gate B does not.

### `gemma3_270m` — RESOLVED (was: RoPE crash)

<details>
<summary>Historical traceback (2026-08-21, fixed in 86th commit)</summary>

```
File "eval/fidelity/attention.py", line 151, in evaluate_attention_fidelity
    position_embeddings = model.model.rotary_emb(hidden_states[0], position_ids)
File ".../transformers/models/gemma3/modeling_gemma3.py", ... in forward
    inv_freq = getattr(self, f"{layer_type}_inv_freq")
AttributeError: 'Gemma3RotaryEmbedding' object has no attribute 'None_inv_freq'
```

</details>

Confirms the mechanism that motivated per-layer RoPE selection; now implemented in `framework/rope.py`.

### `falcon_h1_0.5b`

```
File "eval/fidelity/attention.py", line 107, in _compute_layer_queries
    ops = load_attention_ops(model_layer.config)
File "framework/model_adapter.py", line 83, in load_attention_ops
    raise NotImplementedError(
NotImplementedError: Online attention adapters are not implemented for model_type='falcon_h1'. Supported: gemma3_text, olmo2, qwen2, qwen3.
```

Simple adapter-registry gate — model loads and forwards fine up to this point (weights load cleanly, ~2 min for 579 shards).

### `tinydeepseek_0.5b` — RESOLVED (was: Gate B NotImplementedError)

<details>
<summary>Historical traceback (2026-08-21, fixed in 94th commit)</summary>

```
File "eval/fidelity/attention.py", line 107, in _compute_layer_queries
    ops = load_attention_ops(model_layer.config)
File "framework/model_adapter.py", line 83, in load_attention_ops
    raise NotImplementedError(
NotImplementedError: Online attention adapters are not implemented for model_type='deepseek_v3'. Supported: gemma3_text, olmo2, qwen2, qwen3.
```

</details>

**This is a significant, positive correction to the static architecture probe**, which (loading with `trust_remote_code=True` to inspect the vendored code directly) found an `ImportError` and concluded the model couldn't be assessed at all. Run through the engine's *actual* load path (`framework/model.py::ModelLayer`, which never passes `trust_remote_code`), the model loads cleanly in ~29s as transformers' **native** `DeepseekV3ForCausalLM`. Confirmed live: `DeepseekV3Attention` exposes genuine MLA submodules; `iter_layer_kv` succeeds (26 `DynamicLayer`s, keys `(1,4,T,64)`, values `(1,4,T,32)`). Full eval (identity, TurboQuant, QJL, RocketKV) verified in `tests/test_tinydeepseek_reference.py`. Gate C still fails (native latent not exposed); reports disclose `cache_representation="expanded_kv"`.

(Caveat worth carrying forward: the cache being iterated is HF's already-expanded per-head K/V, not DeepSeek's native compressed-latent representation.)

## Correspondence table — current status (2026-08-19)

| Model | FIDELITY/representation+memory | FIDELITY/attention | BEHAVIOR+SYSTEM (identity/turboquant) | BEHAVIOR+SYSTEM (qjl/rocketkv) |
|---|:---:|:---:|:---:|:---:|
| olmo2_1b | ✅ | ✅ | ✅ | ✅ |
| qwen3_0.6b | ✅ | ✅ | ✅ | ✅ |
| gemma3_270m | ✅ | ✅ | ✅ | ✅ |
| tinydeepseek_0.5b | ✅ | ✅ (expanded KV) | ✅ | ✅ |
| falcon_h1_0.5b | ✅ | ✅ (hybrid) | ✅ | ✅ |

Falcon-H1 memory accounting counts all visible state: `eval/fidelity/memory.py` uses `visible_state_bytes()` (attention K/V + Mamba recurrent/conv). Gate C passes; compression still targets attention K/V only.

## Correspondence table — initial probe (2026-08-21, superseded for Gemma3)

<details>
<summary>Historical table from first probe</summary>

| Model | FIDELITY/representation+memory | FIDELITY/attention | BEHAVIOR+SYSTEM (identity/turboquant) | BEHAVIOR+SYSTEM (qjl/rocketkv) |
|---|:---:|:---:|:---:|:---:|
| olmo2_1b | ✅ (confirmed, ran) | ✅ (confirmed, ran) | ✅ (confirmed, ran) | ✅ (adapter exists) |
| qwen3_0.6b | ✅ (confirmed, ran) | ✅ (confirmed, ran) | ✅ (confirmed, ran) | ✅ (adapter exists) |
| gemma3_270m | ✅ | ❌ (fixed 2026-08-18) | ✅ | ❌ (fixed 2026-08-18) |
| tinydeepseek_0.5b | ✅ | ❌ | ✅ likely | ❌ |
| falcon_h1_0.5b | ⚠️ | ❌ | ⚠️ | ❌ |

</details>

## Engine adoption and transformation plan

Full technical detail lives in [`ENGINE_INTERNALS.md §8`](../../architecture/ENGINE_INTERNALS.md#8-diversifying-to-other-architecture-families) and [`ENGINE_AND_EVALUATION_FRAMEWORKS_REDESIGN_PLAN.md`](../../ENGINE_AND_EVALUATION_FRAMEWORKS_REDESIGN_PLAN.md). Ranked summary, current as of 2026-08-18:

1. **Add `load_attention_ops` branches.** ✅ **Done for all five families** (`gemma3_text`, `deepseek_v3`, `falcon_h1`).
2. **Per-layer RoPE selection** — ✅ **done for Gemma3** (`build_rope_context().get_rope(layer_idx)` in FIDELITY/attention).
3. **Hybrid state interface** — ✅ **done (WP1):** `iter_layer_states()` + `visible_state_bytes()`. Falcon memory accounting fixed; Mamba compression remains passthrough by policy.
4. **MLA-native latent state** — still open: TinyDeepSeek Gate C fails until native `kv_lora_rank` interception lands (expanded cache benchmarked today with disclosure).
5. **`--skip-fidelity` on `scripts/run_eval.py`** — ✅ **implemented (WP1).**

What this document adds is **live confirmation** (tracebacks, successful runs, numbers) plus corrections (TinyDeepSeek loads via native `deepseek_v3` and full eval works on expanded cache, 94th commit; Falcon visible-state memory accounting; Gemma3 fully unblocked 2026-08-18).
