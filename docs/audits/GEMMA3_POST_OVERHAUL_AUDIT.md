# Gemma3-270M post-overhaul audit — fix checklist

Living document. Source: audit of `results/gemma3_full_eval_merged_20260904T234343Z.*`
(bundle `results/gemma3_full_eval_paper/`), performed 2026-10-01. Use it as the work
order for fixes and update **Status** / **Verified** as each item lands.

**Rule:** verification runs are *targeted* (single compressor, ctx 128, few trials,
one Modal job at most). **No full sweep until the user explicitly asks.**

Status legend: `TODO` · `IN PROGRESS` · `FIXED (local)` · `VERIFIED (Modal)` · `WONTFIX (documented)`

## Bundle verdict

The 18/18 grid is **not usable for the paper as it stands**. Do not cite any number
from it unless its row in §"What still holds" says so. After all fixes land, the whole
grid must be re-run on **one clean SHA** (only when the user says so).

## Fix checklist

| ID | Sev | Status | Problem | Verified |
|----|-----|--------|---------|----------|
| B1 | critical | FIXED (local) | Perplexity never sees compressed KV for default-path compressors | `tests/test_ppl_protocol.py` (fp32 Gemma3); Modal pending |
| B2 | critical | FIXED (local) | SnapKV / RocketKV perplexity leaks future tokens | `tests/test_eviction_causality.py`; Modal pending |
| B8 | critical | FIXED (local) | **New:** `qjl_decode` computed `S·b` instead of `Sᵀ·b` (QJL key reconstruction cos ≈ 0.004; TurboQuant value residual injected noise; crashes for `proj_dim ≠ head_dim`) | `tests/test_qjl.py::test_qjl_decode_reconstructs_*`; Modal pending |
| B7 | critical | FIXED (local) | **New:** `KVCacheEngine.generate` started from the engine's leftover `compressed_cache`, so reasoning / instruction-following trials (shared engine) prefilled each prompt on top of the previous trial's KV | `tests/test_engine_reuse.py`; Modal pending |
| B6 | high | FIXED (local) | **New:** decode after eviction used RoPE positions inferred from the shortened cache (affects retrieval / reasoning / instruction / throughput for SnapKV, RocketKV) | `tests/test_eviction_causality.py::test_decode_positions_*`; Modal pending |
| B3 | critical | FIXED (local) | QJL online payload double-append (since 2026-07-06) | local tests only (Modal blocked: workspace paused) |
| B4 | high | FIXED (local) | TurboQuant round trip not idempotent; snapshot path compounds error (+ sliding-window cache rebuild lost logical length) | `tests/test_sliding_window_cache.py`; Modal pending |
| B5 | high | FIXED (local) | Palu: attention fidelity via no-op `compress_kv`; false calibration metadata; ratio excludes shared factors | `tests/test_palu_fidelity.py`, `tests/test_palu.py`; Modal pending |
| S1 | high | FIXED (local) | SYSTEM baseline contaminated by attention patches (QJL, RocketKV) | `tests/test_system_baseline_unpatched.py`; Modal pending |
| S2 | med | FIXED (local) | TTFT includes cold-start; single sample, no warm-up | `tests/test_throughput_protocol.py`; Modal pending |
| S3 | med | FIXED (local) | `kernel_cost.attention_execution_time_ms` mislabeled; bytes-moved accounting wrong | `tests/test_system_modules.py`; Modal pending |
| S4 | med | FIXED (local) | KPI gating (reasoning / peak / gpu / baseline at ctx≥256) not reflected in job spec; inconsistent per row | `tests/test_kpi_plan.py`; Modal pending |
| P1 | high | FIXED (local) | Provenance: `git_sha` wrong for dirty-tree runs; 3 code states in one bundle | `tests/test_provenance.py`; Modal pending |
| P2 | low | FIXED (local) | `seed` missing from Phase-14 manifest | `tests/test_provenance.py` |
| F1 | med | FIXED (local) | Attention-fidelity window (128 cap) not recorded; differs across rows | `tests/test_fidelity_window.py` |
| R1 | med | FIXED (local) | Retrieval prompt 129/251/491 tokens vs labelled 128/256/512; n=5 trials | `tests/test_retrieval_prompt.py`; Modal pending |
| L1 | high | PARTIALLY ASSESSED → superseded | Legacy Qwen3-1.7B / OLMo2-1B results: check for B1 and B3 contamination | legacy TQ PPL inspected (below). 2026-11-24: re-measurement superseded by the clean reframe grid (R3 → R13); legacy numbers stay historical and invalid |

---

### B1 — Perplexity is vacuous for TurboQuant / Palu / identity
- **Where:** `eval/behavior/task_quality.py::evaluate_perplexity_result`. With `seq_len <= stride (512)` the loop does one multi-token `engine.step`; logits come from the uncompressed forward, compression happens after.
- **Evidence (fp32, CPU, Gemma3, ctx128):** harness PPL = 70.766925 for identity, TQ-b4, TQ-b2, Palu. With compressed KV in the decode loop: identity 28.52, TQ-b4 48.28 (+69%), TQ-b2 3.3e7, Palu 28.517 (prefill 64, score last 64).
- **Fix direction:** one definition for all methods: prefill P tokens → compress → teacher-forced decode of the remainder through the compressed cache. Prototype: session scratchpad `e1.py::incremental_ppl`. Keep the HF baseline computed on the same scored positions.
- **Acceptance:** identity engine PPL == HF baseline on the same tokens (fp32, ≤1e-4 rel); TQ-b2 PPL ≫ TQ-b4 ≫ identity; Palu ≈ identity; unit test asserting PPL changes when the compressor is lossy.
- **Done (2026-10-01):** `eval/behavior/task_quality.py` rewritten. Protocol for every compressor *and* the baseline: prefill `P = seq_len − 64` tokens (`ppl_prefill_length`, `DEFAULT_PPL_DECODE_TOKENS = 64`), compress, then teacher-force the last 64 tokens one at a time through the compressed cache; score NLL on logits `float()`-upcast. Baseline (`evaluate_perplexity_baseline*`) scores the same positions (`stride` kept for API compat, ignored). `eval/behavior/__init__.py` passes `max_length` to the baseline; `controlled_conditions.perplexity_scoring = prefill_then_compressed_decode_teacher_forcing`, `perplexity_decode_tokens` recorded. **Consequence: PPL is now measured on 64 tokens (was the whole window) and is NOT comparable with any previously published PPL.** Tests: `tests/test_ppl_protocol.py` (identity==baseline <1e-3; TQ b2 > b4 > identity).
- **Local numbers (fp32, CPU, Gemma3, ctx128 WikiText):** identity = baseline = 28.517 (n=64). Previously the harness reported 70.77 for every compressor.
- **Open:** QJL runs its estimator attention in prefill too (perturbs later layers' K/V); real QJL prefills exactly. Decide whether to keep (document) or make prefill exact.
- **Invalidates:** PPL for every compressor in the bundle (and the "FIDELITY ≠ BEHAVIOR" claim that leans on it).

### B2 — SnapKV / RocketKV PPL violates causality
- **Where:** `framework/snapkv_online.py` (evict inside prefill attention, then `align_attention_mask` slices to kept columns); same pattern in `framework/rocketkv_online.py`. The `evaluate_perplexity_result` docstring "fix" (SnapKV scored as no-op) introduced the scoring mode.
- **Evidence:** perturb token 120 → max |Δlogit| at positions <100: SnapKV 7.52, RocketKV 18.2 (identity/QJL 0). SnapKV cap 64 → PPL 299.8, cap 96 → 321.8 (worse with more budget). No-op budgets (cap/budget 10000) reproduce baseline exactly.
- **Fix direction:** prefill attention must be full and causal; evict *after* prefill (real SnapKV semantics); PPL scored via B1's prefill-then-compressed-decode protocol.
- **Acceptance:** causality probe (scratchpad `e5.py`) gives 0 for positions < perturbed index for every compressor; PPL monotone (non-increasing) in budget.
- **Done (2026-10-01):** `framework/snapkv_online.py` and `framework/rocketkv_online.py` now attend over the full causal K/V on multi-token (prefill) steps and only use eviction to decide what is *stored*; RocketKV keeps sparse attention (stage 1 maintain + stage 2 HSA) for `q_len == 1` decode. Verified locally: perturbation → Δlogit at earlier positions = 0.0 (was 7.5 / 18.2); no-op budgets == identity (28.517); PPL vs SnapKV cap (prefill 64): 60→28.76, 48→33.18, 32→41.60, 24→40.39, 16→41.23 (flat tail, noise-level non-monotonicity); RocketKV budget 60→43.4, 48→48.2, 32→53.0, 16→152 (its budget also caps the decode cache, unlike SnapKV; stage 2 only engages after stage 1 locks, by design). Tests: `tests/test_eviction_causality.py` (causality tests fail on the old code).

### B6 — Decode positions after eviction (new, found while fixing B2)
- **Where:** `KVCacheEngine.step/generate` passed `position_ids=None`; HF infers positions from the cache length, which shrinks under SnapKV/RocketKV eviction, so new tokens were rotated as if earlier in the sequence than their stored keys.
- **Evidence:** ctx 128, cap/budget 32, 6 decode steps: max |Δlogit| between implicit and logical positions 18.2 (SnapKV) / 16.3 (RocketKV), argmax changed; identity 0.
- **Fix:** `CompressedCache.logical_seq_len` (tokens processed) set by `KVCacheEngine.step`; when it differs from the stored length and the caller gave no `position_ids`, positions continue from the logical length. No behaviour change for non-evicting compressors.
- **Impact on existing data:** retrieval / reasoning / instruction-following / throughput numbers for SnapKV and RocketKV in the Gemma3 bundle (and legacy RocketKV sweeps, which use `generate`) were computed with wrong positions → also invalid. Likely a contributor to their 0.0 retrieval.

### B3 — QJL online payload double append
- **Where:** `framework/qjl_online.py::_ensure_key_payloads` (line ~35) appends the return value of `compressor.compress_key_token`, which already appends to `_online_key_payloads[layer]`. Prefill builds `[t0,t0,t1,t1,…]`, truncated to seq_len → keys scored against wrong sketches; decode keeps the misalignment. Present since commit `2d06095` (2026-07-06).
- **Evidence:** attention-output rel. error vs exact on identical tensors: 88.6% (as shipped) vs 4.6% (correct path), proj_dim 65536. PPL with scratch fix and proj_dim 65536: 66.96 (identity 70.77); as shipped 3125.
- **Fix direction:** drop the extra `append`; add regression test asserting `len(online_key_payloads(layer)) == seq_len` and per-token payload equals batch-encoded payload for that position.
- **After the B3 fix alone:** default QJL (m = head_dim = 256, 1-bit) on Gemma3 still gives PPL ≈ 6378 (old PPL protocol; 4370 under the new one, retrieval 0.0; a 16384-dim sketch gives 34.0 / 1.00 — near identity) because key norms reach ~130 (estimator noise ∝ ‖q‖‖k‖/√m). Report as a finding; consider documenting m needed to converge.
- **Acceptance:** converges toward identity as proj_dim grows; alignment test passes; MHA (OLMo2) numbers re-checked (L1).
- **Done (2026-10-01):** fix in `framework/qjl_online.py::_ensure_key_payloads`; regression tests `tests/test_qjl_online.py::test_ensure_key_payloads_one_payload_per_position` and `::test_qjl_online_attention_matches_exact_with_wide_sketch_mqa` (both fail on the old code, pass now; `tests/test_qjl*.py` 19 passed). **Pending:** targeted Modal check with preset `audit_verify` (identity, qjl_default, qjl_wide proj_dim 16384; ctx 128) — expect `qjl_wide` PPL near identity, `qjl_default` still high.

### B4 — TurboQuant non-idempotent; snapshot path compounds error
- **Where:** `framework/kv_engine.py::step` snapshot branch for models with `sliding_attention` layers recompresses the whole cache every step (violates "payload never recomputed"); `compressors/turboquant.py` / quantizers: V relative error 0.154 after 1 pass → 2.15 after 64 (b=4); b=2 → 1e7.
- **Fix direction:** either keep per-layer payload cursors so old tokens are not re-quantized on sliding-window models, or make compress∘decompress a projection (idempotent). Investigate why one-pass V error (0.154) exceeds K (0.096).
- **Acceptance:** repeated round-trip error stays ≈ one-pass error; Gemma3 TQ decode path does not re-quantize past tokens; unit test on idempotence.

- **Root cause refined (2026-10-01):** TQ "full" values = MSE quantization + 1-bit QJL residual (`use_qjl = FULL and mode == "value"`), i.e. the decoded value carries added random noise, so compress∘decompress can never be a projection — idempotence is unattainable by design; the correct fix is to **never re-quantize stored tokens**. Design note for the paper: the repo applies the QJL residual to *values* and plain MSE to *keys*, which is the opposite of the usual rationale (unbiased inner products matter for keys, MSE for values) — decide whether this is intended.
- **Done:** (1) `KVCacheEngine._compress_new_tokens` now uses a **per-layer append cursor** and drops the payloads a sliding-window layer evicted (`new_tokens` arg); the whole-cache snapshot branch in `_step_impl` is **removed**, restoring the "payload written once, never recomputed" invariant for Gemma3. (2) `decompress_to_legacy_cache(..., logical_seq_len=)` restores `DynamicSlidingWindowLayer.cumulative_length` after rebuilding the cache — without it `get_seq_length()`/mask offsets were wrong once the window filled (real window = 512, i.e. exactly the ctx-512 Gemma3 jobs: ctx+64 gen > 512). Found because the engine's identity PPL (7.97) disagreed with the full-forward baseline (2.109) on a small-window repro; after the fix both are 2.10919.
- **Local numbers (fp32, Gemma3, ctx128, new PPL protocol):** identity 28.517 · TQ-b4 full 33.09 · TQ-b3 108.8 · TQ-b2 1108 · TQ-b4 wht_quant 34.56 · Palu 28.517 (degenerate rank at ctx128, see B5). **Superseded by B8** (corrected QJL residual decode): TQ-b4 full 38.00 · b3 109.6 · b2 791.6; identity / wht_quant / Palu unchanged. The old snapshot path gave TQ-b4 48.28 / TQ-b2 3.3e7 (compounded error).
- **Tests:** `tests/test_sliding_window_cache.py` (identity == baseline after the window fills — fails on the old code; linear compress-work bound). Test uses `sliding_window=32` override because real window 512 is too slow on CPU.
- **Regression caught by the existing Gemma3 reference test (fixed):** restoring `cumulative_length = logical` unconditionally broke eviction compressors (RocketKV decode: mask 320 vs keys 286) because a sliding layer that is short only due to *eviction* must keep `cumulative_length == stored length`. Rule now: restore the logical length only when the layer's stored length has reached `sliding_window − 1` (`framework/kv_cache.py`). Test: `tests/test_eviction_causality.py::test_long_prompt_eviction_then_multistep_decode_builds_matching_masks`.
- **Open:** the `Counting` bound does not detect whole-cache recompression for compressors whose `compress()` bypasses `compress_kv`; revisit if another default-path compressor is added.

### B8 — `qjl_decode` computes S·b instead of Sᵀ·b (new, found while verifying B3 with a wide sketch)
- **Where:** `quantizers/qjl.py::qjl_decode` — docstring says `S^T b`, code did `einsum("ji,...i->...j", S, b)` = `S b`. For square `S` (m = head_dim, the default) it ran and returned an unrelated direction; for `proj_dim ≠ head_dim` it raised (`subscript i has size 16384 … 256`). Used by `QJLPipeline.decompress_tensor` (QJL keys) **and** `TurboQuantPipeline.decompress_tensor` (the 1-bit residual added to TurboQuant *values* in the `full` stage).
- **Evidence:** random Gaussian x (d=256): reconstruction cos(x, x̂) = **0.0041** with the old decode vs **0.627** (m=256) / **0.849** (m=1024) with Sᵀb. The earlier audit's claim that QJL key reconstruction "cos ≈ 0 is expected for 1-bit" was **wrong** — it was this bug.
- **Fix:** `einsum("ij,...i->...j", S, b)`; `tests/test_qjl.py::test_qjl_decode_scaling` (which pinned the wrong formula) corrected; new test asserts cos > 0.5 at m=d, > 0.8 at m=8d, and that a wide sketch no longer crashes.
- **Consequences measured locally (fp32, Gemma3, ctx128, PPL protocol):** TurboQuant `full` PPL b4 33.09 → **38.00**, b3 108.8 → **109.6**, b2 1108 → **791.6**; `wht_quant` (no residual) unchanged at 34.56. I.e. with a *correct* residual the 1-bit QJL term **helps at 2 bits but hurts at 4 bits** (it adds variance; the old decode's noise happened to be smaller). TQ one-pass value error 0.154 → 0.120 (keys unchanged 0.096).
- **Impact on existing data:** QJL `fidelity.representation.key_*` (cos 0.004, RMSE 5.2) in the Gemma3 bundle and legacy sweeps = decode bug, not a property of 1-bit QJL. **Every TurboQuant `full`-stage number (K attention aside) used the buggy residual: value RMSE/cos, PPL and retrieval of `tq_full_b*` in the Gemma3 bundle *and* the legacy Qwen3 / OLMo2 sweeps are invalid.** `tq_mse_b4` / `wht_quant` unaffected. TQ *key* metrics unaffected (keys never use the residual).
- **Design question for the paper (not changed):** the repo applies the QJL residual to **values** and plain MSE quantization to **keys** (`use_qjl = FULL and mode == "value"`). The usual rationale is the reverse (unbiased inner products matter for keys; values need low MSE). With the corrected decode, `full` is worse than `wht_quant` at b4 (38.0 vs 34.6), which supports revisiting it.

### B7 — Engine reuse leaks KV between prompts (new, found via the Gemma3 reference test)
- **Where:** `KVCacheEngine.step` uses `cache = compressed_cache or self.compressed_cache`; `generate()` passes `None` on its first step, so a *reused* engine (`evaluate_reasoning`, `evaluate_instruction_following` create one engine for all trials) continued from the previous trial's cache. Retrieval had already been worked around with a fresh engine per trial.
- **Evidence:** on the old code, identity `generate(b)` after `generate(a)` on the same engine ≠ `generate(b)` on a fresh engine (the second prompt was appended after the first). It surfaced as `RuntimeError: size of tensor a (287) must match tensor b (286)` for RocketKV in `evaluate_reasoning` once positions/logical lengths were tracked.
- **Fix:** `generate()` now resets `compressed_cache`, `_last_full_cache`, compressor state and the decode prefix before starting. Tests: `tests/test_engine_reuse.py` (identity + RocketKV; both fail on the old code).
- **Impact on existing data:** every reasoning / instruction-following number ever produced (Gemma3 bundle **and** legacy Qwen3 / OLMo2 sweeps) was measured with cross-trial context leakage → invalid. Reasoning is 0.0 everywhere on Gemma3-270M anyway; instruction-following compliance (1.0 / 0.667 / 0.0 / 0.167 in the bundle) cannot be trusted. Retrieval (fresh engine per trial) and perplexity / throughput (fresh engine per call) were not affected by this particular bug.

### B5 — Palu fidelity and cost records
- **Where:** `compressors/palu.py`: `compress_kv` is `tensor.clone()` so `eval/fidelity/attention.py` (else-branch) reports cos 1.0 / RMSE 0 / KL 0 while `representation` uses lossy `compress()` (K RMSE 0.33 @ ctx512). `offline_cost_metadata()` claims WikiText-2 / 2,097,152 tokens but `bind_model` is weight-only SVD. Reported ratio 2.0× excludes 16.5 MB shared factors (> whole KV 2.4–9.4 MB).
- **Fix direction:** give Palu an `attention_fidelity` hook that uses the real reconstruction; make calibration metadata truthful (`calibration_required=False` or actual data); report ratio with and without shared factors.
- **Acceptance:** Palu attention cos < 1 whenever representation error > 0; metadata matches code; both ratios in JSON.

- **Done (2026-10-01):** (1) `eval/fidelity/attention.py` has a Palu branch that uses the real whole-layer `compress()/decompress()` round trip (same path as the representation metric) and also yields `output_rmse` / KL. Local (fp32, ctx256): rate 0.5 → attn cos 0.99887, rmse 0.273, KL 0.044; rate 0.25 → cos 0.99376, rmse 0.638, KL 0.210 (previously exactly 1.0 / 0 / 0). (2) `PaluCompressor.offline_cost_metadata()` now `calibration_required=False`, no dataset/tokens (bind_model is a weight-only SVD); setup time/memory kept; taxonomy `calibration_free=True`. Tests that encoded the old claim were updated. (3) Ratio: **decision — keep excluding Palu's A/B factors from `compression_ratio`** (they replace `k_proj`/`v_proj` weights, per the existing rationale in `eval/fidelity/memory.py`), but every job now also reports `fidelity.memory.compression_ratio_including_shared` (Palu Gemma3 ctx256: 2.00 vs **0.44**; shared factors 16.5 MB). The paper must state which one it uses.
- **Open (paper decision):** if the factors are treated as additional (not replacing) weights, Palu *expands* memory at these context lengths.

### S1 — Contaminated SYSTEM baseline
- **Problem:** baseline latency QJL 9.5–15 s, RocketKV ~3 s vs ~1.5 s for the others → attention patches were still installed when the "uncompressed" baseline ran.
- **Done (2026-10-01):** `evaluate_throughput_baseline` calls `ensure_vanilla_attention(model)` first (the compressed pass re-installs patches through `make_kv_engine`). Test: patch via a QJL engine → run baseline → `forward` restored and `_qjl_online_enabled` cleared → compressed pass patched again (fails on old code). **Acceptance on Modal:** baseline e2e within noise across compressors at the same ctx.

### S2 — TTFT cold start / single sample
- **Problem:** identity TTFT 113/212/33 ms, TQ 999/1872/55 ms across rows; one timed pass, no warm-up.
- **Done:** `evaluate_throughput` / `evaluate_throughput_baseline` run an untimed warm-up (`DEFAULT_WARMUP_TOKENS=2`), then `DEFAULT_REPEATS=3` timed runs and report the median-elapsed run (stop early if a run exceeds `MAX_REPEAT_SECONDS=30`, e.g. QJL at long ctx). `ThroughputMetrics` gains `repeats` and `elapsed_seconds_all` (spread is visible in the JSON). **Cost:** SYSTEM throughput time ×~4 (warm-up + 3 runs) for fast compressors.
- **Open:** kernel-cost / bandwidth / peak-memory / GPU-util passes are still single-shot (they do not report TTFT).

### S3 — SYSTEM field semantics
- **Done:** `memory_bandwidth`: *read* = cache stored before the step, *write* = bytes appended by the step (whole prompt cache on prefill) — the old code charged 2× the post-step cache (writing one token was billed as rewriting the whole cache). New fields `kv_bytes_read`, `kv_bytes_written`; docstring states it counts compressed stored bytes only and is Python/launch-bound on small models (relative, not utilisation). `kernel_cost`: added `non_compression_time_ms`; `attention_execution_time_ms` kept as a documented legacy alias (total − compress/decompress; *not* an attention-kernel time) so `cost.online.attention_cost_ms` and downstream schemas do not break.
- **Open (paper):** describe `cost.online.attention_cost_ms` as "non-compression step time" in the text; Oaken layer description still says "attention".
- **Impact on existing data:** old `total_kv_bytes_moved` values are ~1.5–2× too large and not comparable with new runs.

### S4 — KPI gating
- **Problem:** `modal_app/worker.py` disabled reasoning / peak VRAM / GPU util at ctx ≥ 256 and `eval/system/__init__.py` dropped the baseline at ≥ 256 for *all* methods, but `job.run_reasoning` still said True and older rows (run before the gate existed) had the KPIs → inconsistent grid, silently.
- **Done (2026-10-01):** gating moved to config (`kpi_gating.skip_extra_passes_ctx_ge`, **null = off** in `configs/modal_gemma3.yaml`); the hard-coded `<256` baseline gate in `evaluate_system` is gone; `modal_app/job_spec.py::resolve_kpi_plan` decides the plan and the worker stores it as `payload["kpi_effective"]` (`gated`, effective flags, `skipped` list). Tests: `tests/test_kpi_plan.py`.
- **Open / risk:** the gate existed because ctx-512 identity/TQ/QJL hit A10G OOM. Several OOM causes were fixed since (B4 snapshot path, `ddp_cache_data` hydrate, decode-prefix retention), but this is **unproven on GPU** — the targeted Modal check at ctx 512 must confirm; if it OOMs, set the config value to 512 (the job JSON will then say what was skipped). `runner.py` still reloads the model before SYSTEM at ctx ≥ 256 on CUDA (memory workaround; warm-up in S2 absorbs its cold start).

### P1 — Provenance
- **Problem:** ctx-512 jobs recorded `git_sha=9bb0ee7` but contained code first committed in `c09be8a`: `KV_GIT_SHA` is HEAD at submission while Modal mounts the *working tree*. Bundle mixed `76cf309` / `7c7ce08` / `9bb0ee7`.
- **Done:** `eval/reproducibility/manifest.py::collect_git_state` → `{sha, dirty, dirty_file_count, diff_sha256}` (diff hash covers `git diff HEAD` **and untracked file contents**); `modal_app/settings.py` stamps `KV_GIT_DIRTY` / `KV_GIT_DIFF_SHA256` into the container env and prints a stderr warning on a dirty tree; every result JSON has `git_state`. Tests: `tests/test_provenance.py`.
- **Policy (manual):** before any real sweep commit first so `git_state.dirty == false`; compare `git_state.diff_sha256` across jobs of one bundle (all must match).

### P2 — seed
- **Problem:** `seed` "missing" for identity / SnapKV / RocketKV / Palu (no compressor seed) and never recorded for the eval generators.
- **Done:** `controlled_conditions.fixed.seed` (=0) and `.seeds` (retrieval / reasoning / instruction-following = 0; decoding greedy; data order deterministic); Phase-14 `manifest["seed"] = {"evaluation": 0, "compressor": <seed or None>}`; validation requires the evaluation seed.

### F1 — Attention-fidelity window
- **Done:** single source of truth `eval.fidelity.attention_fidelity_window` (`min(cfg, seq_len, ATTENTION_FIDELITY_MAX_TOKENS=128)`), used by both the evaluator and `controlled_conditions`; `AttentionMetrics.score_tokens` records it. Rows from before this change (windows 256/512 for eviction/Palu rows, 128 for the rest, JSON always claiming 512) remain non-comparable.

### R1 — Retrieval
- **Done:** `_build_prompt` binary-searches the filler length so prompts land within a few tokens of the requested context (test: `ctx-6 ≤ tokens ≤ ctx` for 128/256/512, question kept at the end; was 129/251/491) and the hard truncation (which could cut the question) is removed; default trials 5 → 20 (`DEFAULT_RETRIEVAL_TRIALS`). Standard error is still ~±0.1 at p≈0.5 — treat retrieval as coarse.
- **Not changed:** reasoning accuracy is 0.0 for every method on Gemma3-270M (floor effect). Decide whether to drop/replace it for sub-1B models.

### L1 — Legacy results (Qwen3-1.7B, OLMo2-1B)
- **Assessed (2026-10-01):**
  - **B1 does not affect the legacy TurboQuant numbers.** `results/phase5_modal_sweep_128_256_512_20260704T185851Z.csv` (Qwen3-1.7B): TQ perplexity depends strongly on bit width (ctx128: b2 19413.9, b3 96.5, b4 18.1, mse-b4 21.7 vs identity 14.2) → that harness did score under compression (older token-stepped PPL). The prefill-only scoring was introduced later with the multi-token-window change. **However the new protocol (prefill + 64-token compressed decode) is not comparable with the legacy token-stepped whole-window PPL** — do not mix them in one table.
  - **B3 (QJL) probably affects** every QJL sweep after 2026-07-06 (`phase5_modal_qjl_20260705T*` predate commit `2d06095`; `olmo2_phase5_qjl*` 2026-07-31 / 08-02 / 08-10 and `*_prodqjl_*` postdate it). **Not yet re-measured** — the OLMo2 (MHA) effect would need a local repro: the duplicated-append misalignment does not depend on GQA/MQA.
  - **B6 probably affects** legacy RocketKV `generate()`-based metrics (retrieval / throughput) because eviction shortens the cache. Not re-measured.
  - B4 (sliding-window) is N/A for Qwen3 / OLMo2 (no sliding layers).
- **Open:** Qwen3-1.7B (`models/legacy/qwen3_1.7b`) and OLMo2-1B (`models/olmo2_1b`) *are* present locally, so B3/B6/B7/B8 re-measurements are possible on CPU/MPS without Modal — not run yet. Additional legacy impact: **B7** (reasoning / instruction-following) and **B8** (every `tq_full_b*` row and QJL reconstruction metrics) apply to the legacy sweeps too.

## What still holds (after the fixes above)

- **Probably still valid (re-verify on the new clean run anyway):** TQ *key* representation / attention numbers (K rel. error 9.6% reproduced; KL ≈ ½σ² consistent); byte accounting for TQ / SnapKV / RocketKV (SnapKV/RocketKV "smoke" presets use an absolute 64-token budget → ratio scales with ctx); compatibility-gate metadata, identity FIDELITY, hardware profile; QJL *attention-score* fidelity (single-tensor estimator path; cos ≈ 0.72 at m=d is genuine estimator noise).
- **Invalid:** all perplexity numbers; QJL BEHAVIOR (B3); QJL key *reconstruction* metrics (B8); TQ `full` value metrics / PPL (B8); SnapKV/RocketKV/Palu BEHAVIOR and SYSTEM numbers (B2, B5, B6, S1); reasoning / instruction-following everywhere (B7); old SYSTEM baselines and bandwidth numbers (S1, S3).

## Precision note

Gemma3 is loaded in bf16 (`framework/model.py::_resolve_load_dtype`) although the paper contract says fp16; `controlled_conditions.paper_contract.deviates=true` records this. bf16 explains the ~0.2% engine-vs-HF PPL gap (fp32: 70.766929 vs 70.766925). Document in the paper, not a bug.

## Test-suite notes (2026-10-01)

- `tests/conftest.py` lowers the expensive production defaults for CPU smoke tests: `KV_RETRIEVAL_TRIALS=2`, `KV_THROUGHPUT_WARMUP_TOKENS=0`, `KV_THROUGHPUT_REPEATS=1`, `KV_PPL_DECODE_TOKENS=8`. Production defaults stay 20 / 2 / 3 / 64; `tests/test_ppl_protocol.py` restores the 64-token window. All four are env-overridable at runtime.
- **Cost note:** the new PPL protocol runs 64 compressed decode steps. For QJL every step decompresses the whole per-token cache (`decompress_to_legacy_cache` → `qjl_decode` per token per layer), so QJL PPL is ~minutes on CPU and ~2+ min on A10G at ctx 512. The QJL online path never reads the decompressed keys (it scores from payloads), so a placeholder/lazy key decode would cut this a lot — **not done** (optimization, out of audit scope; would need care for fidelity code that uses `decompress_kv`).
- **Final local status (2026-10-01, after B8, CPU):** 53 fast test files pass per file. Heavy real-model reference tests, run per case with the lowered defaults: `gemma3` identity / TurboQuant / RocketKV / QJL pass; `olmo2` (8 tests, 8 min) pass, TQ+QJL re-run after B8 pass; `falcon_h1` (20 min) pass, TQ+QJL re-run after B8 pass; `tinydeepseek_reference` pass, TQ+QJL re-run after B8 pass; `regression_validation` pass after the Phase-5 perplexity comparison was replaced by identity-vs-same-protocol-baseline. **One known failure, pre-existing:** `test_qwen3_reference::test_qwen3_vs_olmo2_measured_kv_ratio` (0.4375 vs 0.875) fails identically on a clean `HEAD` worktree — unrelated (probably checkpoint dtype).
- The reference files take 5–20 min each on CPU (QJL dominates); run them per case, not as one `pytest tests` (that exceeds a 30-min background limit).

## Modal verification log

Account: Modal profile **`faheemgurkani`** (`faheemgurkani@gmail.com`), activated 2026-10-01 via
`python -m modal profile activate faheemgurkani` (the `modal` executable is not on PATH; use `.venv/bin/python -m modal`).
Volumes `kv-engine-gemma3` / `kv-engine-results-gemma3` created on this account and Gemma3 checkpoint uploaded; `huggingface-secret` already present.
Targeted preset: `audit_verify` in `configs/modal_sweeps.yaml` (3 jobs at one ctx, no reasoning/kernel/bandwidth). Command:
`bash scripts/modal_gemma3_full_eval.sh` is a full-grid script — do **not** use it; instead
`KV_MODEL_CONFIG=configs/model_gemma3_270m.yaml KV_MODAL_CONFIG=configs/modal_gemma3.yaml .venv/bin/python scripts/run_taxonomy_smoke.py --modal --skip-dummy --sync --no-resume --skip-validate --preset audit_verify --context-lengths 128 --output audit_verify_b3`.
Record each targeted run here.

| Date | Fix IDs | Command | Result |
|------|---------|---------|--------|
| 2026-10-01 | B3 | `modal_gemma3_full_eval.sh setup` (ensure_model) | **BLOCKED** — `Workspace ac-51iKyqdATyYGN9iriAVw4P is paused`; volumes + upload succeeded, no compute possible until the workspace is resumed in the Modal dashboard |

## Change log

- 2026-10-01: document created from audit.
- 2026-10-01: working tree is **uncommitted** (nothing committed by the assistant); `git_state.dirty` will be true on any run until committed.
- 2026-10-01: B1–B8, S1–S4, P1, P2, F1, R1 fixed locally; L1 partially assessed (see sections); B3 fixed locally + regression tests; Modal profile switched; `audit_verify` preset added; Modal compute blocked by paused workspace.
- 2026-10-02: Modal profile re-confirmed: active = `faheemgurkani` (workspace `faheemgurkani`, `ac-51iKyqdATyYGN9iriAVw4P`, user `us-Fk5WNt4gN0R9PGZn6BYidR`). **No compute has ever run on any profile from this effort** (the workspace is paused; only volume creation/upload happened). Added `modal_app/settings.py::assert_expected_modal_profile` (called from `modal_runtime_env`, i.e. on every Modal launch); it raises unless the active profile is `faheemgurkani` (`$MODAL_PROFILE` or `~/.modal.toml`; override with `KV_MODAL_PROFILE`). Test: `tests/test_modal_profile_guard.py`. Still paused as of this entry.
- 2026-10-02: every `scripts/modal_*.sh` now runs `scripts/modal_assert_profile.py` first (aborts unless active profile is faheemgurkani); `scripts/modal_*.py` go through `modal_app`, whose launch path is guarded.
- 2026-11-24: all fixes committed and pushed as #201–#204 (HEAD `b452849`); the 2026-10-01 "uncommitted" entry is obsolete. Modal verification (R2) still pending. Project adopted the reframe ([`../RESEARCH_REFRAME_AND_PRIOR_ART.md`](../RESEARCH_REFRAME_AND_PRIOR_ART.md)); the "re-run the whole grid on one clean SHA" requirement is now work package R13 in [`../RESEARCH_REDESIGN_PLAN.md` → Part 0](../RESEARCH_REDESIGN_PLAN.md#part-0--reframe-adoption-and-master-status-2026-11-24). L1 re-measurement superseded by R13.
