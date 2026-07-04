# Modal GPU Evaluation Design

Redesign for running the TurboQuant evaluation sweep on **Modal NVIDIA GPUs**, while keeping the existing **Mac M4 (MPS) local path** unchanged.

Based on Modal docs (Context7: `/websites/modal`) and profiling of the current eval loop.

---

## 1. Problem Summary (and fixes applied)

| Bottleneck | Where | Status |
|---|---|---|
| Missing attention mask in online loop | `eval/perplexity.py`, `framework/kv_engine.py` | **Fixed** — explicit mask + auto mask in `step()` |
| Cache reset every stride window | `eval/perplexity.py` | **Fixed** — single incremental cache across windows |
| TurboQuant payloads forced to CPU | `quantizers/turboquant_pipeline.py` | **Fixed on CUDA** — GPU-resident payloads when input is CUDA |
| No CUDA device path | `framework/device.py` | **Fixed** — `KV_EVAL_DEVICE=cuda` for Modal |
| Serial sweep | local scripts | Modal `.map()` added in `modal_app/` |
| Eager attention | `framework/model.py` | Unchanged (required for KV intercept) |

Modal fixes **hardware speed** and **job-level parallelism**. Code changes fix **correctness** and **device placement**.

---

## 1b. Modal storage: what to persist vs live inference

This is **batch offline evaluation**, not a serving API. Each Modal container runs one job and exits.

| Artifact | Store on Modal Volume? | Why |
|---|---|---|
| **Qwen3-1.7B weights** (~3.2 GB) | **Yes** — `kv-engine-qwen3` | Download once; avoid 3 GB pull per job |
| **Eval JSON/CSV results** | **Yes** — `kv-engine-results` | Collect outputs after parallel sweep |
| **HF_TOKEN** | **Secret** — `huggingface-secret` | First-time model download only |
| **WikiText-2 cache** | **No** (optional HF cache in container) | Rebuilt cheaply; ~10 MB |
| **TurboQuant centroids** | **No** | Deterministic from `seed` + `bitwidth` |
| **Compressed KV caches** | **No** | Ephemeral per job; not reusable across configs |
| **Live inference endpoint** | **No** | Not needed — eval workers are stateless batch jobs |
| **Full repo / venv** | **Baked in image** | Via `add_local_dir` in `modal_app/image.py` |

**Verdict:** Modal needs **model weights + result storage + secrets**. Everything else is **live inference inside a short-lived worker** — load model, run eval, write JSON, exit.

```bash
# Secret already created as huggingface-secret (HF_TOKEN)
bash scripts/modal_setup_model.sh
bash scripts/modal_run_sweep.sh          # detached spawn_map (30 jobs)
bash scripts/modal_fetch_results.sh      # pull JSON from kv-engine-results
modal run modal_app/sweep.py::merge_local --input-dir results/modal_volume
```

---

## 2. Recommended Modal GPU

For **Qwen3-1.7B** (~3.2 GB fp16) with KV-cache eval up to **32K context**:

| GPU | VRAM | Best for | Notes |
|---|---:|---|---|
| **L4** | 24 GB | Cost-effective parallel sweep | Enough for 1.7B + 32K KV on one worker |
| **A10G** | 24 GB | **Recommended default** | Good inference $/hr; widely available on Modal |
| **L40S** | 48 GB | Chunked/batched forwards | Headroom for `chunk_size=32–64` experiments |
| **A100** | 40–80 GB | Overkill for single job | Use only if batching many sequences |
| **H100** | 80 GB | Not cost-effective here | Model is too small to justify |

### Verdict

**Primary choice: `gpu="A10G"`** per eval worker.

- Fits 1.7B + 32K uncompressed KV (~3.6 GB) with comfortable margin on 24 GB.
- Modal supports `gpu="A10G"`, fallbacks via `gpu=["A10G", "L4", "any"]`.
- For the full sweep grid, run **many A10G workers in parallel** via `.map()` / `.spawn_map()` — cheaper than one H100 serially.

**Secondary choice: `gpu="L40S"`** if implementing **multi-token chunk forwards** (Section 4.2) with larger activation memory.

---

## 3. Architecture: Two Runtimes, One Codebase

```text
┌─────────────────────────────────────────────────────────────────┐
│  LOCAL (unchanged)          │  MODAL (new)                      │
│  Mac M4 / MPS / CPU         │  NVIDIA CUDA workers              │
├─────────────────────────────┼───────────────────────────────────┤
│  scripts/run_eval.py        │  modal_app/sweep.py               │
│  scripts/run_turboquant_    │    @app.local_entrypoint()        │
│    sweep.py                 │    → spawn_map(eval_worker, jobs) │
│  framework/device.py        │  modal_app/worker.py              │
│    prefer_mps=True          │    @app.function(gpu="A10G")      │
│                             │    → run_single_eval(job)         │
├─────────────────────────────┴───────────────────────────────────┤
│  SHARED (paper-independent)                                       │
│  eval/runner.py  framework/kv_engine.py  compressors/*            │
│  eval/perplexity.py (optimized)  quantizers/* (GPU path)        │
└─────────────────────────────────────────────────────────────────┘
```

**Rule:** Modal is an **orchestration + CUDA runtime layer**. No changes to `KVCompressor` interface or metric definitions.

---

## 4. Code Changes (Required)

### 4.1 Device abstraction (`framework/device.py`)

Add CUDA detection without breaking MPS default:

```python
def get_device(prefer_mps: bool = True, prefer_cuda: bool = False) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def get_eval_device() -> torch.device:
    """Modal sets MODAL_GPU=CUDA; local default stays MPS."""
    import os
    if os.environ.get("KV_EVAL_DEVICE") == "cuda" or os.environ.get("MODAL_GPU"):
        return get_device(prefer_cuda=True, prefer_mps=False)
    return get_device(prefer_mps=True)
```

`ModelLayer` reads `get_eval_device()` instead of hard-coded MPS.

---

### 4.2 Fix sliding-window PPL (highest impact, all platforms)

**Current bug:** each stride window restarts `cache=None` and re-steps through the **entire** window token-by-token. At ctx=4096, stride=512 → 8 windows × thousands of redundant steps.

**Fix:** carry compressed cache across windows; only evaluate loss on new `trg_len` tokens.

```text
Before (per window):
  cache = None
  for t in 0..window_len:          # re-processes prefix every window
      step(token_t)

After:
  cache persists across begin_loc
  for t in prev_end..end_loc:      # only NEW tokens
      step(token_t)
  compute loss on t in [eval_start, end_loc)
```

**Expected speedup:** ~2–8× at 4K+ context (removes O(n²) prefix replay).

Files: `eval/perplexity.py`

---

### 4.3 Chunk forward within incremental loop (CUDA path)

Still **online** (compress after each new token), but amortize decompress + forward:

| Mode | Behavior | Valid for online PPL? |
|---|---|---|
| **Current** | 1 token forward per step | Yes (baseline) |
| **Chunk warm-up** | Batch-forward prefix once, then 1-token steps for eval region | Yes, if prefix tokens are not scored |
| **Multi-token eval** | Forward `chunk_size` new tokens per step, compress each | Approximate; use only for smoke tests |

**Recommended:** prefix warm-up (above) + optional `chunk_size=1` for scored region.

Do **not** batch all 4096 tokens in one forward — that bypasses per-step compression semantics.

---

### 4.4 GPU-native TurboQuant (`quantizers/turboquant_pipeline.py`)

Current path forces CPU for cross-device stability:

```python
vector_norm=vector_norm.detach().cpu()   # ← kills GPU pipeline
```

**Change:** device-aware payloads:

```python
def _store_tensor(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    if device.type == "cuda":
        return t.detach()          # stay on GPU
    return t.detach().cpu()        # MPS/CPU path unchanged
```

`decompress_tensor`: use `payload.vector_norm.to(device=target_device)` instead of always CPU.

**CUDA-only:** enable `fast-hadamard-transform` in Modal image (builds with nvcc).

Files: `quantizers/turboquant_pipeline.py`, `quantizers/hadamard.py`, `requirements-cuda.txt`

---

### 4.5 Batched Section A fidelity (offline metrics)

Attention RMSE and tensor RMSE can batch across layers on GPU:

```python
# eval/attention_score_error.py — optional batched path
for layer_batch in chunks(layers, batch_size=4):
    scores = vectorized_attention_fidelity(layer_batch)
```

Section A is cheap vs PPL; lower priority than 4.2–4.4.

---

### 4.6 Identity engine PPL regression (fix before trusting long-ctx PPL)

At ctx=512, identity online PPL was 46 vs baseline 14 — caused by missing attention mask in the online loop.

**Fixed:** explicit mask in `eval/perplexity.py` + auto mask in `framework/kv_engine.py`. Confirm on Modal with the `identity_baseline` job at ctx=512 before trusting long-ctx numbers.

---

## 5. Modal Setup

### 5.1 Project layout (implemented)

```text
modal_app/
  __init__.py
  image.py          # CUDA image + volumes + add_local_dir
  settings.py       # loads configs/modal.yaml
  worker.py         # ensure_model + eval_worker (A10G)
  sweep.py          # local_entrypoint: spawn_map / sync merge
  job_spec.py       # EvalJobSpec + 5-config grid
  merge.py          # flatten worker JSON → CSV/JSON
configs/modal.yaml  # GPU, timeout, volume names, secret name
scripts/
  modal_setup_model.sh
  modal_run_sweep.sh
  modal_fetch_results.sh
requirements-modal.txt  # CUDA deps (torch via cu124 index in image)
```

### 5.2 Container image

```python
# modal_app/image.py
import modal

cuda_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("fast-hadamard-transform")  # CUDA builds on Modal
    .env({
        "KV_EVAL_DEVICE": "cuda",
        "HF_HUB_CACHE": "/models",
        "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
    })
)
```

Use Modal **Volume** for model weights (avoid re-download per job):

```python
model_vol = modal.Volume.from_name("kv-engine-qwen3", create_if_missing=True)

@app.function(
    gpu="A10G",
    image=cuda_image,
    volumes={"/models": model_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=4 * 60 * 60,  # 4 hours per long-ctx job
)
def eval_worker(job: dict) -> dict:
    ...
    model_vol.commit()  # after first download
```

### 5.3 Secrets & env

| Item | Local (M4) | Modal |
|---|---|---|
| `HF_TOKEN` | `.env` | `modal.Secret.from_name("huggingface-secret")` |
| Device | MPS auto | `KV_EVAL_DEVICE=cuda` |
| Model path | `models/qwen3_1.7b/` | Volume `/models/qwen3_1.7b/` |

Setup once:

```bash
modal secret create huggingface-secret HF_TOKEN=hf_...   # already exists
bash scripts/modal_setup_model.sh
modal run --detach modal_app/sweep.py
# or: bash scripts/modal_run_sweep.sh
```

---

## 6. Parallel Sweep on Modal

### 6.1 Job grid (one GPU per job)

Each job = one `(compressor, bitwidth, stage, context_length)` tuple.

Full TurboQuant grid from `EVALUATION_PLAN.md`:

```text
5 configs × 6 context lengths = 30 jobs
(identity, tq_b2, tq_b3, tq_b4, tq_mse) × (128, 512, 4096, 8192, 16384, 32768)
```

### 6.2 Orchestrator

```python
# modal_app/sweep.py
import modal
from modal_app.worker import app, eval_worker
from modal_app.job_spec import build_sweep_jobs

@app.local_entrypoint()
def main(detach: bool = False):
    jobs = build_sweep_jobs(
        context_lengths=[128, 512, 4096, 8192, 16384, 32768],
    )
    # Parallel: up to 30 concurrent A10G containers
    if detach:
        for result in eval_worker.spawn_map(jobs):
            print(result.object_id)
    else:
        results = list(eval_worker.map(jobs))
        merge_and_write_results(results)
```

Modal `.map()` / `.spawn_map()` runs jobs on **separate containers** — natural multi-GPU parallelism without writing distributed PyTorch.

### 6.3 Expected wall-clock (rough)

| Setup | ctx=4096 full grid (30 jobs) |
|---|---|
| Mac M4 serial | ~3–7 days |
| Modal 30× A10G parallel | ~3–8 hours (longest single job dominates) |
| Mac + algorithm fix (4.2) alone | ~1–2 days serial |

---

## 7. What Stays on Mac M4

| Use case | Runtime |
|---|---|
| Unit tests (`pytest`) | Local MPS/CPU |
| Quick smoke (ctx=128) | Local |
| TurboQuant stage debugging | Local |
| Code development | Local |
| Full production sweep | **Modal** |

No requirement to remove MPS path. `get_eval_device()` selects backend automatically.

---

## 8. Implementation Order

| Phase | Work | Platform | Status |
|---|---|---|---|
| **A** | Fix sliding-window cache carry (`perplexity.py`) | Local + Modal | ✅ |
| **B** | Fix identity PPL regression at ctx≥512 (attention mask) | Local + Modal | ✅ (verify on Modal) |
| **C** | GPU-native TurboQuant payloads (CUDA branch) | Modal (+ local CPU fallback) | ✅ |
| **D** | `framework/device.py` + `ModelLayer` CUDA path | Both | ✅ |
| **E** | Modal image, volume, worker, secrets | Modal only | ✅ |
| **F** | `spawn_map` sweep orchestrator + merge | Modal only | ✅ |
| **G** | Re-run full grid; merge into `results/` | Modal | Run on Modal |

---

## 9. Modal GPU utilization checklist

To **fully leverage** NVIDIA on Modal:

- [x] All compress/decompress tensors stay on **CUDA** (no `.cpu()` in hot path on CUDA)
- [x] `fast-hadamard-transform` installed in Modal image
- [x] Model loaded **once per worker** via `ensure_model()` + volume cache
- [x] Sliding-window cache **carried** across strides (Section 4.2)
- [x] One sweep job per GPU via `.spawn_map()` — 30 configs parallel, not serial in one container
- [x] Volume for model weights + `volume.commit()` after download
- [x] `timeout` = 4 h per job (`configs/modal.yaml`)
- [x] `gpu=["A10G", "L4", "any"]` fallback in `configs/modal.yaml`
- [x] Resume: skip jobs whose JSON already exists on results volume
- [x] Merge results to CSV/JSON via `merge.py` + `modal_fetch_results.sh`

---

## 10. What Modal does **not** solve

- **Autoregressive order** — scored tokens still processed sequentially (by design)
- **Eager attention overhead** — still required for KV intercept (acceptable on CUDA)
- **Correctness** — faster hardware does not fix identity PPL bug or bad metrics

---

## 11. References

- Modal GPU types: `gpu="A10G"`, `gpu="A100:2"`, `gpu=["H100", "A100", "any"]` — [Modal GPU guide](https://modal.com/docs/guide/gpu)
- Parallel jobs: `Function.map()`, `Function.spawn_map()` — [Modal scale guide](https://modal.com/docs/guide/scale)
- Volumes + secrets: [Modal volumes guide](https://modal.com/docs/guide/volumes)
- CUDA PyTorch on Modal: [Modal CUDA guide](https://modal.com/docs/guide/cuda)

---

## 12. Runbook (copy-paste)

```bash
cd /path/to/kv-cache-compression-benchmark
source .venv/bin/activate
pip install modal

# One-time model cache on Modal Volume
bash scripts/modal_setup_model.sh

# Full 30-job sweep (detached)
bash scripts/modal_run_sweep.sh

# Monitor in Modal dashboard; when done:
bash scripts/modal_fetch_results.sh
modal run modal_app/sweep.py::merge_local --input-dir results/modal_volume

# Resume after partial run (skips JSON already on volume)
modal run --detach modal_app/sweep.py --resume

# Sync subset and merge locally (blocks until done)
modal run modal_app/sweep.py --sync --context-lengths 128,512 --labels identity_baseline
```
