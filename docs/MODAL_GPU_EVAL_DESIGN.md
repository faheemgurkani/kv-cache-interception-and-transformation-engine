# Modal GPU Evaluation — Implemented System Design

NVIDIA CUDA evaluation for the **KV-Cache Interception + Transformation Engine**, deployed on [Modal](https://modal.com). The Mac M4 (MPS) local path is unchanged for development; **full Phase 5 sweeps run on Modal**.

This document describes what is **actually implemented and running today**, including parallelization strategy, storage, and known limits.

---

## 1. Executive Summary

| Aspect | Design |
|---|---|
| **Platform** | Modal serverless GPU containers |
| **GPU** | **A10G (24 GB)** per eval worker; fallbacks `L4`, `any` (`configs/modal.yaml`) |
| **Model** | Qwen3-1.7B (~3.2 GB), cached on Modal Volume |
| **Parallelism model** | **One sweep job = one GPU container** via `spawn_map()` (up to 30 concurrent jobs) |
| **Within-job eval** | Online PPL remains **sequential** (one token per step); Section A uses **one forward + windowed attention** |
| **Orchestration** | `modal_app/sweep.py::main` → `eval_worker.spawn_map(jobs)` |
| **Results** | Per-job JSON on `kv-engine-results` volume; merged locally to CSV/JSON |

```text
Local Mac (M4)                         Modal (NVIDIA A10G × N)
─────────────────                      ─────────────────────────
pytest, smoke, dev                     Full eval sweep (30 jobs)
scripts/run_eval.py @ ctx=128          modal_app/sweep.py::main
MPS / CPU                              CUDA via KV_EVAL_DEVICE=cuda
models/qwen3_1.7b/                     Volume /models/qwen3_1.7b/
results/ (local)                       Volume /results/*.json
```

**Rule:** Modal is an **orchestration + CUDA runtime layer**. The `KVCompressor` interface and metric definitions are shared with local eval.

---

## 2. Architecture

### 2.1 Component diagram

```text
┌──────────────────────────────────────────────────────────────────────────┐
│  LOCAL MACHINE                                                           │
│  modal run --detach modal_app/sweep.py::main                             │
│       │                                                                  │
│       ▼                                                                  │
│  build_sweep_jobs()  ──►  filter_existing_jobs()  ──►  spawn_map()     │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │  one remote call per job
         ┌─────────────────────┼─────────────────────┐
         ▼                     ▼                     ▼
   ┌───────────┐         ┌───────────┐         ┌───────────┐
   │ A10G #1   │         │ A10G #2   │   ...   │ A10G #30  │
   │ eval_worker│        │ eval_worker│        │ eval_worker│
   └─────┬─────┘         └─────┬─────┘         └─────┬─────┘
         │                     │                     │
         └─────────────────────┼─────────────────────┘
                               ▼
              ┌────────────────────────────────────┐
              │  SHARED EVAL CODE (same as local)  │
              │  eval/runner.py                    │
              │  eval/perplexity.py  (cache carry) │
              │  framework/kv_engine.py            │
              │  compressors/*  quantizers/*       │
              └────────────────────────────────────┘
                               │
         ┌─────────────────────┴─────────────────────┐
         ▼                                           ▼
  Volume: kv-engine-qwen3                    Volume: kv-engine-results
  /models/qwen3_1.7b/                        /results/{label}_ctx*.json
```

### 2.2 File layout

```text
modal_app/
  image.py          CUDA container image, volumes, env
  settings.py       Loads configs/modal.yaml; resolves project root in container
  worker.py         ensure_model (CPU), eval_worker (A10G), list_completed_jobs
  sweep.py          Local entrypoints: main (spawn/sync), merge_local
  job_spec.py       EvalJobSpec + 5-config × N-context grid
  merge.py          Flatten worker JSON → CSV/JSON reports
configs/modal.yaml  GPU type, timeout, volume/secret names
scripts/
  modal_setup_model.sh
  modal_run_sweep.sh
  modal_fetch_results.sh
requirements-modal.txt   Python deps (torch installed separately with cu124)
```

### 2.3 Container image (`modal_app/image.py`)

Built from `debian_slim` + PyTorch **cu124**:

| Step | Purpose |
|---|---|
| `pip_install("torch", index_url=...cu124)` | CUDA PyTorch |
| `pip_install_from_requirements(requirements-modal.txt)` | transformers, datasets, scipy, … |
| `.env({...})` | `KV_EVAL_DEVICE=cuda`, `PYTHONPATH`, `KV_PROJECT_ROOT`, alloc conf |
| `.add_local_dir(repo → /root/kv-cache-engine)` | **Must be last** — mounts code at container start |

**Not in image:** `fast-hadamard-transform` (no prebuilt wheel for torch 2.6; source build failed on Modal). WHT uses **scipy fallback** on CUDA (`quantizers/hadamard.py`).

### 2.4 Workers (`modal_app/worker.py`)

| Function | GPU | Role |
|---|---|---|
| `ensure_model` | None (CPU) | Idempotent Qwen3 download into model volume |
| `eval_worker` | A10G (+ fallbacks) | One `(compressor, context_length)` eval; writes JSON |
| `list_completed_jobs` | None | List result stems on volume (for resume) |

Each `eval_worker` run:

1. `model_volume.reload()` / `results_volume.reload()`
2. Load model from `/models/qwen3_1.7b/`
3. `EvaluationRunner.run(context_length, …)` — same code as local
4. Write `{result_stem}.json` or `.error.json` to `/results/`
5. `results_volume.commit()`

Timeout: **4 hours** per job (`configs/modal.yaml`).

---

## 3. Parallelizability — What Is and Is Not Parallel

Modal parallelism is **job-level** (multiple checkout lanes). Within each job, online PPL stays **sequential by design**.

### 3.1 Analogy

```text
Current eval (within one job)  = one cashier, one item at a time
Modal spawn_map (across jobs)  = 30 checkout lanes open at once
A10G GPU                       = a faster cashier in each lane
```

NVIDIA helps most when you add **lanes** (Modal) and **keep data on GPU**; batched online PPL would change what the metric measures.

### 3.2 Parallelism matrix

| Strategy | Implemented? | How | Speedup |
|---|---|---|---|
| **Different configs in parallel** | ✅ **Yes** | `eval_worker.spawn_map()` — identity, tq_b2, tq_b3, tq_b4, tq_mse | **Main win** (~linear with GPU count) |
| **Different context lengths in parallel** | ✅ **Yes** | Same grid: 128 … 32768 each get their own container | **Main win** |
| **Keep compress/decompress on GPU** | ✅ **Yes** | `turboquant_pipeline._store_tensor()` stays on CUDA when input is CUDA | High per job |
| **Sliding-window cache carry (PPL)** | ✅ **Yes** | `eval/perplexity.py` — no O(n²) prefix replay across strides | 2–8× at 4K+ |
| **Identity PPL fix (attention mask)** | ✅ **Yes** | `eval/perplexity.py` + `framework/kv_engine.py` | Correctness |
| **Section A: single forward pass** | ✅ **Yes** | `eval/fidelity.py` — one model forward for tensor + attention + memory | Cuts VRAM vs 3× forward |
| **Section A: windowed QK^T fidelity** | ✅ **Yes** | Last 512 tokens (`attention_fidelity_tokens` in `configs/eval.yaml`) | Avoids O(n²) VRAM on A10G |
| **Resume / skip completed jobs** | ✅ **Yes** | `list_completed_jobs()` + `filter_existing_jobs()` | Saves re-work |
| **Batched multi-token PPL forwards** | ❌ No | Still 1 token → decompress → forward → compress per step | Would approximate metric |
| **Batched Section A across layers** | ❌ No | Per-layer loop; tensors freed + `empty_cache()` each layer | Minor gain; not done |
| **CUDA `fast-hadamard-transform`** | ❌ No | Scipy WHT on GPU tensors instead | Medium; blocked by wheel/build |
| **Multi-GPU layer split** | ❌ No | One model copy per job | Hard; not needed for 1.7B |
| **Multiple sequences batched** | ❌ No | `batch_size: 1` in eval config | N/A for current metrics |

### 3.3 Job grid (30 checkout lanes)

Each job = one `(compressor, bitwidth, stage, context_length)` tuple.

```text
5 configs × 6 context lengths = 30 jobs

Configs:
  identity_baseline
  tq_full_b2, tq_full_b3, tq_full_b4
  tq_mse_b4

Context lengths:
  128, 512, 4096, 8192, 16384, 32768
```

Orchestrator (`modal_app/sweep.py::main`):

```python
jobs = build_sweep_jobs(context_lengths=[...], labels=...)
if resume:
    completed = set(list_completed_jobs.remote())
    jobs = filter_existing_jobs(jobs, completed)

job_dicts = [job.to_dict() for job in jobs]

# Detached (default for full sweep):
eval_worker.spawn_map(job_dicts)

# Blocking + local merge:
results = list(eval_worker.map(job_dicts))
write_merged_reports(ok, Path("results"), output)
```

Modal runs each job in a **separate container with its own GPU** — no distributed PyTorch required.

### 3.4 Expected wall-clock

| Setup | Full 30-job sweep |
|---|---|
| Mac M4 serial | Days |
| Mac + cache-carry fix only | ~1–2 days serial |
| **Modal 30× A10G parallel** | **~3–8 hours** (longest single job dominates) |

Longest jobs: ctx=32768 online PPL (many sequential steps). Use `--detach` so the local client can disconnect without killing workers.

---

## 4. Storage and Secrets

Batch evaluation only — **no live serving endpoint**.

| Artifact | Persist on Modal? | Name / path |
|---|---|---|
| Qwen3-1.7B weights | ✅ Yes | Volume `kv-engine-qwen3` → `/models/qwen3_1.7b/` |
| Per-job eval JSON | ✅ Yes | Volume `kv-engine-results` → `/results/{stem}.json` |
| HF token | ✅ Secret | `huggingface-secret` (`HF_TOKEN`) |
| WikiText-2 cache | ❌ No | Rebuilt per job (~10 MB) |
| TurboQuant centroids | ❌ No | Deterministic from seed + bitwidth |
| Compressed KV (runtime) | ❌ No | Ephemeral inside worker |
| Repo / Python deps | Image + mount | `add_local_dir` → `/root/kv-cache-engine` |

Result filename pattern: `{label}_ctx{length}_b{bitwidth}_{stage}.json`

Failed jobs write `{stem}.error.json` with traceback; resume **retries** failed jobs (only successful `.json` stems are skipped).

---

## 5. Shared CUDA Code Path (Local + Modal)

These changes apply on **both** platforms where noted.

### 5.1 Device selection (`framework/device.py`)

| Environment | Device |
|---|---|
| Local (default) | MPS on Apple Silicon, else CPU |
| Modal | `KV_EVAL_DEVICE=cuda` → CUDA |

`ModelLayer` uses `get_eval_device()`.

### 5.2 Online PPL fixes (`eval/perplexity.py`, `framework/kv_engine.py`)

| Fix | Status |
|---|---|
| Carry compressed cache across stride windows | ✅ |
| Explicit / auto attention mask in online loop | ✅ |
| Cache trim at `max_length` | ✅ |

**Verified on Modal:** identity @ ctx=512 → PPL **~14.12** vs baseline **~14.11** (not the old ~46 bug).

### 5.3 GPU-native TurboQuant (`quantizers/turboquant_pipeline.py`)

On CUDA, compression payloads stay on GPU (`_store_tensor`). MPS still uses CPU payloads for stability.

### 5.4 Section A memory strategy (`eval/fidelity.py`, `eval/attention_score_error.py`)

Long context OOM on A10G was caused by:

- Three separate full forwards (tensor, attention, memory)
- Full `seq_len × seq_len` attention score matrices at 8K+

**Fixes applied:**

1. **Single forward** in `evaluate_fidelity()` — shared `past_key_values` + `hidden_states`
2. **Windowed QK^T fidelity** — last `attention_fidelity_tokens` (default **512**) from `configs/eval.yaml`
3. Per-layer tensor cleanup + `torch.cuda.empty_cache()`
4. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in Modal image env

Online PPL at 32K still needs sequential steps and significant VRAM; Section A no longer builds full 8K×8K matrices.

---

## 6. Configuration

### 6.1 `configs/modal.yaml`

```yaml
gpu: A10G
gpu_fallbacks: [A10G, L4, any]
timeout_hours: 4
volumes:
  model: kv-engine-qwen3
  results: kv-engine-results
secrets:
  huggingface: huggingface-secret
```

### 6.2 `configs/eval.yaml` (Modal-relevant)

```yaml
perplexity_stride: 512
generated_tokens: 64
attention_fidelity_tokens: 512   # Section A QK^T window for long ctx
```

### 6.3 Container path resolution (`modal_app/settings.py`)

Workers mount code at `/root/kv-cache-engine` but import `modal_app` from `/root/modal_app`. Settings resolves config via:

1. `KV_PROJECT_ROOT` / `PYTHONPATH`
2. Fallback `/root/kv-cache-engine`

---

## 7. Operational Runbook

```bash
cd /path/to/kv-cache-compression-benchmark
source .venv/bin/activate
pip install modal

# One-time: cache model on Modal Volume (~3.2 GB)
bash scripts/modal_setup_model.sh
# equivalent: modal run modal_app/worker.py::ensure_model

# Full 30-job sweep (detached — recommended)
bash scripts/modal_run_sweep.sh
# equivalent: modal run --detach modal_app/sweep.py::main

# Subset example
modal run --detach modal_app/sweep.py::main \
  --context-lengths 128,512 --labels identity_baseline

# Sync smoke (blocks until done; merges locally)
modal run modal_app/sweep.py::main --sync \
  --context-lengths 128 --labels identity_baseline --no-resume

# After jobs finish — fetch + merge
bash scripts/modal_fetch_results.sh
modal run modal_app/sweep.py::merge_local --input-dir results/modal_volume

# Resume partial sweep (skips successful .json on volume)
modal run --detach modal_app/sweep.py::main
```

Monitor runs in the [Modal dashboard](https://modal.com/apps).

---

## 8. Implementation Status

| Phase | Work | Status |
|---|---|---|
| **A** | Sliding-window cache carry | ✅ Done |
| **B** | Identity PPL @ ctx≥512 (attention mask) | ✅ Verified on Modal @ 512 |
| **C** | GPU TurboQuant payloads on CUDA | ✅ Done (scipy WHT, not FHT CUDA) |
| **D** | `framework/device.py` CUDA path | ✅ Done |
| **E** | Modal image, volumes, worker, secrets | ✅ Done |
| **F** | `spawn_map` orchestrator + merge | ✅ Done |
| **G** | Full 30-job grid complete | ⚠️ In progress — long-ctx jobs may need re-run after OOM fixes |

### Provisioning issues encountered (resolved)

| Issue | Resolution |
|---|---|
| `fast-hadamard-transform` build / 404 wheel | Dropped from image; scipy WHT fallback |
| `configs/modal.yaml` not found in worker | `KV_PROJECT_ROOT` + `settings.project_root()` |
| `add_local_dir` after build steps | Moved `add_local_dir` to last image step |
| `spawn_map` returned `None` | Do not iterate return value; fire-and-forget |
| Section A OOM @ 8K+ on A10G | Single forward + 512-token attention window |
| Local client timeout on long sync jobs | Use `--detach` for production sweeps |

---

## 9. What Modal Does Not Solve

| Limitation | Reason |
|---|---|
| **Autoregressive PPL order** | Scored tokens must be processed sequentially for correct online metric |
| **Eager attention** | Required for KV intercept; acceptable on CUDA |
| **Instant 32K PPL** | Still many sequential steps per job |
| **Batched PPL without metric change** | Would skip per-step compress semantics |

Future optional work (not implemented):

- Prefix warm-up batch forwards (non-scored tokens only)
- Batched Section A across layer groups
- `fast-hadamard-transform` when torch 2.6 wheels exist, or pinned torch version
- L40S workers if full-sequence Section A is required without windowing

---

## 10. References

- [Modal GPU guide](https://modal.com/docs/guide/gpu)
- [Modal scale / map / spawn_map](https://modal.com/docs/guide/scale)
- [Modal volumes](https://modal.com/docs/guide/volumes)
- [Modal CUDA](https://modal.com/docs/guide/cuda)
- Local dual-runtime overview: [SYSTEM_DESIGN.md §7.5](SYSTEM_DESIGN.md#75-dual-runtime-local-mps--modal-nvidia-cuda)
