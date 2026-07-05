# Current System State

Snapshot of the **KV-Cache Interception + Transformation Engine** as of Phase 5 Modal sweeps (July 2026). For architecture details see [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md); for Modal runtime see [MODAL_GPU_EVAL_DESIGN.md](MODAL_GPU_EVAL_DESIGN.md); for numbers see [PHASE5_EVAL_RESULTS.md](PHASE5_EVAL_RESULTS.md).

---

## 1. What this repo is

A **fixed evaluation stack** (Qwen3-1.7B, WikiText-2, Section A + B metrics) with **pluggable `KVCompressor` implementations**. TurboQuant, QJL, and RocketKV are plug-ins — not separate systems.

```text
Model (fixed) → KVCacheEngine (fixed) → KVCompressor (variable) → eval/ (fixed)
```

---

## 2. Compressor readiness

| Compressor | Code | Unit tests | Modal sweep | Notes |
|---|---|---|---|---|
| `identity` | ✅ | ✅ | ✅ baseline preset (3 jobs) | Shared baseline; run once, reuse for all methods |
| `turboquant` | ✅ | ✅ | ✅ 12 jobs | WHT + Lloyd-Max + optional QJL residual; scipy WHT on Modal |
| `qjl` | ✅ | ✅ | ✅ 3 jobs | 1-bit key signs; online uses key **reconstruct**; Section A uses **estimator** |
| `rocketkv` | ✅ | ✅ | ✅ 9 jobs | Token eviction; online HSA via `framework/rocketkv_online.py` |
| `kivi` | ❌ stub | — | — | `NotImplementedError`; deferred |

---

## 3. Phase 5 Modal sweeps — completed

All sweeps: **Qwen3-1.7B**, **WikiText-2**, ctx **128 / 256 / 512**, **Modal A10G**, `--no-resume` for publication consistency.

| Preset | Configs | Jobs | Completed |
|---|---|---:|---|
| `baseline` | `identity_baseline` | 3 | ✅ |
| `turboquant` | `tq_full_b2`, `tq_full_b3`, `tq_full_b4`, `tq_mse_b4` | 12 | ✅ |
| `qjl` | `qjl_default` | 3 | ✅ |
| `rocketkv` | `rocketkv_r25`, `rocketkv_r50`, `rocketkv_r75` | 9 | ✅ |

Sweep grids: `configs/modal_sweeps.yaml`  
Context lengths: `configs/model.yaml` → `[128, 256, 512]`

**Baseline separation:** identity results are **not** bundled with TurboQuant. Shared baseline lives in its own preset/bundle; method sweeps reference `../phase5_modal_baseline/` for comparison.

---

## 4. Configuration files

| File | Purpose |
|---|---|
| `configs/model.yaml` | Model path, context lengths |
| `configs/eval.yaml` | PPL stride, generated tokens, `attention_fidelity_tokens` (512) |
| `configs/modal.yaml` | GPU type (A10G), volumes, secrets, timeout |
| `configs/modal_sweeps.yaml` | Sweep presets: `baseline`, `turboquant`, `qjl`, `rocketkv` |

---

## 5. Modal infrastructure

| Artifact | Name | Role |
|---|---|---|
| Model volume | `kv-engine-qwen3` | Qwen3-1.7B weights (~3.2 GB) |
| Results volume | `kv-engine-results` | Per-job JSON (`{stem}.json` / `{stem}.error.json`) |
| Secret | `huggingface-secret` | `HF_TOKEN` for first-time model download |
| GPU | A10G (24 GB) | One GPU per job via `eval_worker.spawn_map()` |

**Entrypoints**

```bash
bash scripts/modal_setup_model.sh              # one-time model download
bash scripts/modal_run_sweep_baseline.sh       # identity (3 jobs)
bash scripts/modal_run_sweep.sh                # turboquant (12 jobs, default)
bash scripts/modal_run_sweep_qjl.sh            # qjl (3 jobs)
bash scripts/modal_run_sweep_rocketkv.sh       # rocketkv (9 jobs)
bash scripts/modal_smoke_eval.sh qjl           # single-job smoke @ ctx=128
bash scripts/modal_fetch_results.sh            # pull volume → results/modal_volume/
```

**Orchestration:** `modal_app/sweep.py::main` — flags: `--preset`, `--context-lengths`, `--labels`, `--sync`, `--no-resume`, `--output`

**Worker:** `modal_app/worker.py::eval_worker` — loads model from volume, runs `EvaluationRunner`, writes JSON to results volume.

---

## 6. Local vs Modal

| | Local (Mac MPS) | Modal (CUDA) |
|---|---|---|
| Use | pytest, smoke, dev | Full Phase 5 sweeps |
| Entry | `scripts/run_eval.py` | `modal_app/sweep.py` |
| Device | `get_eval_device()` → MPS/CPU | `KV_EVAL_DEVICE=cuda` |
| Model | `models/qwen3_1.7b/` | Volume `/models/qwen3_1.7b/` |
| Outputs | `results/` (gitignored) | Volume + local fetch |

Same eval code path: `eval/runner.py`, `eval/perplexity.py`, `framework/kv_engine.py`.

---

## 7. Key implementation details (online path)

### TurboQuant

- Incremental compress/decompress per token in `KVCacheEngine`.
- CUDA payloads stay on GPU when input is CUDA (`turboquant_pipeline._store_tensor`).
- WHT via scipy fallback on Modal (no `fast-hadamard-transform` wheel).

### QJL

- Keys: `sign(S @ k) + ||k||`; values passthrough FP16.
- Projection matrix regenerated from seed (not stored); counted in `shared_storage_bytes()`.
- Device fix: payloads store `original_device`; decompress restores to CUDA on Modal.
- **Section A:** `estimate_attention_scores()` for QK fidelity.
- **Section B:** `decompress_kv()` approximate reconstruction (known limitation).

### RocketKV

- **Section A:** full-layer `compress()` applies `TokenSelector` (SnapKV-style).
- **Section B:** raw per-token storage; sparsity in attention via `framework/rocketkv_online.py`:
  - Stage 1: permanent token filter
  - Stage 2: HSA dynamic top-k
  - Attention mask aligned to sparse key length (`align_attention_mask`)
- Baseline PPL/throughput run **before** `KVCacheEngine` construction (attention patch is model-global).

### Evaluation runner order

`eval/runner.py` runs **baseline metrics before compressed metrics** so RocketKV’s attention patch does not corrupt `perplexity_baseline`.

---

## 8. Local result bundles (gitignored)

Raw outputs under `results/` are **not committed**. After a sweep:

```text
results/
  phase5_modal_baseline/          # identity (3 jobs)
  phase5_modal_sweep_128_256_512/   # turboquant (12 jobs)
  phase5_modal_qjl/                 # qjl (3 jobs)
  phase5_modal_rocketkv/            # rocketkv (9 jobs)
  modal_volume/                     # full volume mirror
```

Each bundle contains: `jobs/*.json`, merged CSV/JSON, `logs/`, `manifest.json`.

**Version-controlled summary:** [PHASE5_EVAL_RESULTS.md](PHASE5_EVAL_RESULTS.md)

Restructure helper: `python scripts/restructure_modal_results.py`

---

## 9. Tests

| Suite | Covers |
|---|---|
| `tests/test_qjl.py`, `tests/test_qjl_online.py` | QJL roundtrip, estimator, shared storage |
| `tests/test_rocketkv.py`, `tests/test_rocketkv_online.py` | Token selection, HSA, mask alignment |
| `tests/test_online_inference.py` | Identity PPL parity, KVCacheEngine |
| `tests/test_incremental_kv_cache.py` | Incremental compress, TurboQuant online |
| `tests/test_turboquant_quality.py` | TurboQuant PPL regression |

Run locally: `pytest tests/ -q` (no full Modal sweep required).

---

## 10. Known limits & non-goals

- **KIVI:** not implemented.
- **QJL Section B:** online reconstruct path; estimator not wired into forward pass.
- **RocketKV Section A vs B:** offline fidelity does not reflect token eviction cost.
- **TurboQuant 2-bit @ ctx=128:** anomalously bad PPL; prefer ctx=512 for paper tables.
- **Modal WHT:** scipy fallback only; no CUDA `fast-hadamard-transform`.
- **FlashAttention:** disabled (`attn_implementation="eager"`) — required for KV interception.

---

## 11. Documentation index

| Document | Content |
|---|---|
| [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md) | Architecture, plug-in model, TurboQuant/QJL/RocketKV design |
| [MODAL_GPU_EVAL_DESIGN.md](MODAL_GPU_EVAL_DESIGN.md) | Modal GPU runtime, parallelism, storage, runbook |
| [PHASE5_EVAL_RESULTS.md](PHASE5_EVAL_RESULTS.md) | Full sweep numbers and findings |
| [CURRENT_STATE.md](CURRENT_STATE.md) | This file — readiness and setup snapshot |
| [README.md](../README.md) | Quick start, scripts, Modal commands |

Ignored locally (not in public docs): `docs/IMPLEMENTATION_PLAN.md`, `docs/EVALUATION_PLAN.md`.
