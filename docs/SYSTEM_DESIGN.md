# System Design — KV-Cache Interception + Transformation Engine

This document records the engineering decisions behind the **KV-Cache Interception + Transformation Engine**. The goal is a **NeurIPS-style systems replication**: a fixed model and evaluation stack, with pluggable compression methods.

---

## 0. Mental Model (Read This First)

**This project is NOT:** a TurboQuant script, a single-paper reimplementation, or a model modification toolkit.

**This project IS:** a **KV-cache interception + transformation engine** inside an LLM forward pass.

```text
Tokenizer → Model Forward → KV Cache → (intercept here) → Attention → Next tokens
                                              │
                                    KVCompressor (plug-in)
```

- **`KVCacheEngine`** (`framework/kv_engine.py`) owns the intercept point: decompress stored KV → model step → compress new KV.
- **`KVCompressor`** (`compressors/base.py`) is the only swap-in surface per paper.
- **TurboQuant** is the first fully implemented `KVCompressor`; KIVI, QJL, and RocketKV are additional plug-ins, not separate systems.

---

## 1. Global Architecture (Non-Negotiable)

The system is organized into four fixed layers plus one pluggable compression layer. **Only the compression layer changes between papers.**

```text
                ┌──────────────────────────┐
                │   TOKENIZED INPUT TEXT   │
                └────────────┬─────────────┘
                             │
                     HuggingFace Model          ← Model Layer (fixed)
                             │
                 (Transformer Layers)
                             │
                 ┌───────────▼───────────┐
                 │  KV Cache Interceptor │  ← framework/kv_engine.py
                 └───────────┬───────────┘
                             │
                 KVCompressor (plug-in)         ← Compression Layer (variable)
                             │
        ┌──────────────┬──────────────┬──────────────┐
        │              │              │              │
   TurboQuant      KIVI          QJL           RocketKV
        │              │              │              │
        └──────────────┴──────────────┴──────────────┘
                             │
                     Attention Forward
                             │
                        Next Tokens
                             │
                 ┌───────────▼───────────┐
                 │  Evaluation + Report  │  ← eval/ + reporting/ (fixed)
                 └───────────────────────┘
```

| Layer | Directory | Changes per paper? |
|---|---|---|
| Model | `framework/model.py` | No |
| KV interception | `framework/kv_engine.py`, `framework/kv_cache.py` | No |
| Compression | `compressors/`, `quantizers/` | **Yes** |
| Evaluation | `eval/` | No |
| Reporting | `reporting/` | No |

### Design rationale

- **Zero duplication:** KIVI, QJL, and RocketKV plug into the same `KVCompressor` interface and `KVCacheEngine` without touching eval code.
- **True KV-level control:** We intercept `past_key_values` between forward steps rather than modifying model weights or attention kernels.
- **Apple Silicon compatible:** PyTorch + Transformers + MPS; no CoreML/Ollama/GGUF for the research path.

---

## 2. Model Layer (Fixed)

### 1.1 Model choice

**Primary model:** `Qwen/Qwen3-1.7B`

| Criterion | Why it matters |
|---|---|
| `use_cache=True` | Required for KV-cache research |
| Clean `past_key_values` API | Direct access to K/V tensors per layer |
| GQA (8 KV heads, 128 head dim) | Realistic compression target |
| 32k context | Long-context experiments on M4 Air |
| HuggingFace native | No GGUF/Ollama — full tensor access |

**Alternate (documented for future):** `Phi-3-mini` — same criteria apply.

### 1.2 Load configuration

Implemented in `framework/model.py`:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float16,
    attn_implementation="eager",
)
model.config.use_cache = True
```

Config defaults live in `configs/model.yaml` under `turboquant.attn_implementation` and `torch_dtype`.

### 1.3 Why `attn_implementation="eager"` matters

This is a **hard requirement** for KV-cache interception research.

| Backend | Problem for this project |
|---|---|
| **FlashAttention / SDPA (fused)** | KV tensors are fused inside optimized kernels; K/V cannot be reliably read or replaced between steps |
| **Eager attention** | Standard PyTorch matmuls; `past_key_values` is materialized as explicit tensors per layer |

**Consequences without eager mode:**

1. `outputs.past_key_values` may be incomplete, opaque, or backend-specific.
2. Compression hooks that read `(K, V)` after each forward pass break silently or produce wrong shapes.
3. Round-trip compress → decompress → re-inject into the cache fails unpredictably.

**Decision:** Always load with `attn_implementation="eager"`. Accept the throughput penalty — this is a research instrument, not a production deployment path.

---

## 3. KV Cache Interception System

The core engine is `KVCacheEngine` in `framework/kv_engine.py`.

### 2.1 Where KV appears in HuggingFace

Every causal forward pass with caching:

```python
outputs = model(input_ids, past_key_values=past_kv, use_cache=True)
pkv = outputs.past_key_values  # DynamicCache in Transformers 5.x
```

Per layer (Qwen3-1.7B example):

```text
K, V ∈ R[B, num_kv_heads, seq_len, head_dim]
     = R[1, 8, T, 128]
```

Transformers 5.x exposes this via `DynamicCache.layers[i].keys` and `.values`. Legacy tuple-of-tuples format is handled in `framework/kv_cache.py`.

### 2.2 Hook system

```python
class KVCacheEngine:
    def __init__(self, model, compressor: KVCompressor):
        self.model = model
        self.compressor = compressor
        self.compressed_cache = None
```

The compressor is a **plug-in**. TurboQuant, KIVI, QJL, and RocketKV all implement the same interface.

### 2.3 Forward interception loop (incremental)

Online steps **append** one compressed payload per new token. Old tokens are never re-compressed.

```python
def step(self, input_ids, compressed_cache):
    prev_seq = compressed_cache.seq_length if compressed_cache else 0
    past_kv = decompress_to_legacy_cache(compressed_cache, compressor, config, device)
    outputs = model(input_ids, past_key_values=past_kv, use_cache=True)

    # Only compress newly appended positions [prev_seq : total_seq)
    for layer_idx, (k, v) in enumerate(iter_layer_kv(outputs.past_key_values)):
        append compress_token_slice(k, v, token_idx, ...) for token_idx in range(prev_seq, k.shape[2])

    return outputs.logits, CompressedCache(layers=incremental_payload_lists)
```

**Critical:** Re-compressing the full cache every step caused error accumulation → KV norm explosion → NaN perplexity. Incremental storage fixes this.

### 2.4 Key insight

> **The model is not modified. The engine modifies KV flow between the forward pass and the next token step.**

The model always sees decompressed float16 K/V. Compression affects **storage and bandwidth**, not the model architecture. This keeps the design paper-agnostic: swap `TurboQuantCompressor` for `KIVICompressor` and the engine unchanged.

### 2.5 DynamicCache rebuild

Transformers 5.x requires `DynamicCache(ddp_cache_data=layer_pairs, config=model.config)` when re-injecting decompressed tensors — not a raw tuple. See `decompress_to_legacy_cache()` in `framework/kv_cache.py`.

---

## 4. Compression Layer Interface

All methods implement `KVCompressor` in `compressors/base.py`:

```python
class KVCompressor(ABC):
    def compress_kv(self, tensor, layer=0, mode="key") -> object: ...
    def decompress_kv(self, payload, mode="key") -> Tensor: ...

    def compress(self, key, value, layer=0) -> CompressedKV: ...
    def decompress(self, compressed) -> tuple[Tensor, Tensor]: ...
```

- **`compress_kv` / `decompress_kv`:** Single-tensor API matching the interception loop (mode = `"key"` or `"value"`).
- **`compress` / `decompress`:** Layer-level wrapper used by eval and memory profiling.

---

## 5. TurboQuant Compression Layer

TurboQuant is **isolated** from the model and eval layers:

```text
quantizers/hadamard.py         → WHT + padding
quantizers/lloyd_max.py        → Lloyd-Max centroids + quantize/dequantize
quantizers/qjl.py              → random projection residual codec
quantizers/turboquant_pipeline.py  → full per-tensor pipeline + stages
compressors/turboquant.py      → TurboQuantCompressor plug-in
```

### 4.1 Mathematical objects

Each KV tensor: **K, V ∈ R[B, H, T, D]**

Operations apply along the **last dimension D** (head dimension). Qwen3 uses D=128 (already a power of 2).

### 4.2 Pipeline (8 steps)

| Step | Operation | Implementation | Notes |
|---|---|---|---|
| 0 | Pad D → 2^k | `pad_to_power_of_two()` | Required for WHT when D is not power of 2 |
| 1 | Walsh-Hadamard | `hadamard_transform()` | y = Hx, H orthogonal; scipy fallback on MPS |
| 2 | Normalize | `x / √D` on unit-norm vectors after WHT | Preserves variance in rotated domain |
| 2b | Gamma scale | `γ = amax(y) / max(centroid)` | Per-vector dynamic range for Lloyd-Max |
| 2c | Vector norm | store `‖x‖`, normalize before WHT | PolarQuant-style magnitude preservation |
| 3 | Lloyd-Max quantize | KMeans centroids + nearest index | Offline Gaussian fit, k = 2^bitwidth |
| 4 | Dequantize | `centroids[indices]` | MSE reconstruction x̂ |
| 5 | Residual | r = x − x̂ | TurboQuant's two-part encoding |
| 6 | QJL encode | b = sign(Sr) | **Values only** in FULL stage; QJL on keys hurts attention |
| 7 | QJL decode | r̂ = √(π/2)/d · Sᵀb · ‖r‖ | Unbiased sign estimator |
| 8 | Reconstruct | x_final = x̂ + r̂ → inverse WHT → unpad | Return to original space |

Full forward:

```text
pad → WHT → ÷√D → Lloyd-Max → dequant → residual → QJL → store
decompress: QJL decode → add MSE → ×√D → inverse WHT → unpad
```

### 4.3 Storage format (`TurboQuantTensorPayload`)

```python
{
    "indices": int8 container,   # accounted as bitwidth bits each (not 8 bits)
    "qjl_bits": int8 container,  # accounted as 1 bit per sign (not 8 bits)
    "norm_r": float32,           # residual norms (32 bits each)
    "metadata": 32 bytes/tensor, # dims, stage, bitwidth, shape
}
```

Shared once per compressor (not per layer): Lloyd-Max **centroid table** (`shared_storage_bytes()`).

Size accounting: `framework/storage_accounting.py` → `storage_bits()` / `storage_bytes()` on payload.

Reported metrics (`eval/memory.py`): raw KV bytes, compressed bytes, compression ratio, **effective bits/KV element**.

### 4.4 Stage ablation (do not skip)

Implement and validate **incrementally**:

| Stage | Flag | What it tests |
|---|---|---|
| WHT only | `wht_only` | Lossless transform roundtrip |
| WHT + quant | `wht_quant` | Lloyd-Max error |
| WHT + quant + residual | `wht_quant_residual` | Residual magnitude |
| Full (+ QJL) | `full` | Complete TurboQuant |

```bash
python scripts/validate_turboquant.py --phase stages
python scripts/run_eval.py --compressor turboquant --stage wht_only --context-length 512
python scripts/run_eval.py --compressor turboquant --stage full --context-length 512
```

### 4.5 Apple Silicon note

`fast-hadamard-transform` requires CUDA. On MPS we use a **scipy Hadamard fallback** (`quantizers/hadamard.py`). Decompression runs on CPU for cross-device stability, then tensors move to MPS for the model forward pass.

---

## 5.1 QJL Compression Layer

QJL is **independent** of TurboQuant — it applies random projection directly to key vectors without WHT or Lloyd-Max.

```text
quantizers/qjl.py              → projection_matrix, qjl_encode, qjl_decode (shared primitives)
quantizers/qjl_pipeline.py     → QJLPipeline, QJLTensorPayload
compressors/qjl.py             → QJLCompressor plug-in
```

### Pipeline

| Step | Operation | Notes |
|---|---|---|
| 1 | `k̃ = sign(S @ k)` | Random Gaussian S regenerated from seed |
| 2 | Store | sign bits + `‖k‖` only |
| 3 | Values | Uncompressed fp16 passthrough |
| 4 | Attention | Asymmetric estimator: `q·k ≈ √(π/2) · (‖k‖/m) · ⟨Sq, k̃⟩` |

Full forward:

```text
compress:  k → sign(S @ k) + ||k||     (keys)
           v → passthrough              (values)
attention: estimate_attention_scores()  (preferred)
decompress: approximate k̂ via sign estimator  (engine compatibility)
```

### Storage format (`QJLTensorPayload`)

```python
{
    "sign_bits": int8 container,   # 1 bit per sign (not 8)
    "vector_norm": float32,        # ||k|| (32 bits)
    "metadata": 24 bytes/tensor,
}
```

Projection matrix S is **never stored** — regenerated from `(seed, head_dim)`.

### Eval integration

`eval/attention_score_error.py` calls `compressor.estimate_attention_scores()` when available, giving paper-faithful inner-product measurement for Section A metrics.

**Full reference:** [docs/QJL_AND_ROCKETKV.md](QJL_AND_ROCKETKV.md)

---

## 5.2 RocketKV Compression Layer

RocketKV **drops tokens** rather than quantizing vectors. It is fully independent of TurboQuant and QJL.

```text
quantizers/rocketkv.py         → TokenSelector, HybridSparseAttention, RocketKVLayerPayload
compressors/rocketkv.py        → RocketKVCompressor plug-in
```

### Stage 1 — Permanent filtering

```text
[prefix tokens | observation window]
     ↓
TokenSelector: score prefix by dot-product with mean window key
     ↓
Keep top keep_ratio of prefix + all window tokens
     ↓
{selected_indices, kept_K, kept_V}
```

### Stage 2 — Hybrid Sparse Attention (HSA)

```text
query → approximate scores (head reduction) → top-k → union permanent indices
```

Exposed via `RocketKVCompressor.select_dynamic_tokens()`. Stage 2 is implemented but not yet wired into `KVCacheEngine.generate()` — online eval currently uses Stage 1 filtering via decompress.

### Storage format (`RocketKVLayerPayload`)

```python
{
    "selected_indices": int64,
    "keys": float16,           # retained K tensor
    "values": float16,         # retained V tensor
    "original_seq_len": int,
}
```

No quantization. Memory savings = `T' / T` reduction in sequence length.

**Full reference:** [docs/QJL_AND_ROCKETKV.md](QJL_AND_ROCKETKV.md)

---

## 6. Evaluation Pipeline (Fixed Across All Papers)

**Do not change eval code when adding a new compression method.** Results are split into two sections:

### Section A — Compression Fidelity (offline)

Validates implementation quality without the autoregressive loop.

| Metric | Module | Method |
|---|---|---|
| **Tensor RMSE** | `eval/fidelity.py` | Compress/decompress K/V snapshots; mean RMSE per layer |
| **Attention RMSE** | `eval/attention_score_error.py` | Compare `QK^T / √d` before vs after K compression; layer-wise MSE/RMSE/cosine/max |
| **Memory** | `eval/memory.py` | Raw KV bytes vs bit-accurate payload + shared metadata; ratio + effective bits/KV |

Orchestrator: `evaluate_fidelity()` in `eval/fidelity.py`

### Section B — Inference Impact (online)

Validates usefulness with compressed KV **inside** the generation loop.

| Metric | Module | Method |
|---|---|---|
| **Perplexity** | `eval/perplexity.py` | Sliding-window NLL; each token via `KVCacheEngine.step()` (compress → store → decompress) |
| **Throughput** | `eval/throughput.py` | `KVCacheEngine.generate()` tokens/sec and ms/token |

Optional baselines (uncompressed HF path): `evaluate_perplexity_baseline()`, `evaluate_throughput_baseline()` — pass `--include-baselines` to `run_eval.py`.

Orchestrator: `eval/runner.py` → `EvaluationRunner`

Reporting: `reporting/reporter.py` → JSON (`section_a_fidelity`, `section_b_inference`) + CSV in `results/`

### 6.1 Long-context evaluation

WikiText-2 documents are short. We **concatenate samples** until `target_length` (4K–32K) via `data/loader.py::build_long_context_ids()`. Standard practice in KV-cache papers.

### 6.2 Datasets

| Phase | Dataset |
|---|---|
| Phase 1 | WikiText-2 (`Salesforce/wikitext`, `wikitext-2-raw-v1`) |
| Phase 2 | WikiText-2 + small C4 subset |

---

## 7. Execution Order

Follow this order to avoid getting stuck:

| Phase | Goal | Command / check |
|---|---|---|
| **1 — Model** | Load, generate, confirm KV exists | `scripts/verify_kv_cache.py` |
| **2 — Intercept** | Extract K/V, print shapes | `scripts/validate_turboquant.py --phase intercept` |
| **3 — Baseline** | Identity compressor, perplexity | `run_eval.py --compressor identity` |
| **4 — TurboQuant** | WHT → quant → QJL step-by-step | `validate_turboquant.py --phase stages` |
| **5 — Full eval** | Memory + speed + perplexity | `run_eval.py --compressor turboquant --stage full` |
| **6 — QJL** | Key sign-projection compressor | `pytest tests/test_qjl.py`; `run_eval.py --compressor qjl` |
| **7 — RocketKV** | Token selection compressor | `pytest tests/test_rocketkv.py`; `run_eval.py --compressor rocketkv` |
| **8 — Sweep** | Compare all methods | Local: `run_eval.py`; **full grid: Modal** (`bash scripts/modal_run_sweep.sh`) |

---

## 7.5 Dual Runtime: Local (MPS) + Modal (NVIDIA CUDA)

The same eval code runs on two backends. Only orchestration and device selection differ.

| | **Local (Mac M4)** | **Modal (NVIDIA)** |
|---|---|---|
| Device | MPS / CPU via `get_eval_device()` | CUDA via `KV_EVAL_DEVICE=cuda` |
| Use case | Dev, pytest, ctx=128 smoke | Full Phase 5 sweep (30 jobs) |
| Entry | `scripts/run_eval.py` | `modal_app/sweep.py::main` |
| Model path | `models/qwen3_1.7b/` | Volume `/models/qwen3_1.7b/` |
| Parallelism | Serial (one job at a time) | **Job-level:** up to 30 A10G workers via `spawn_map()` |
| Within-job PPL | Sequential token loop | Same — sequential by design |
| Docs | This file | [MODAL_GPU_EVAL_DESIGN.md](MODAL_GPU_EVAL_DESIGN.md) |

**Parallelism model:** Modal adds **checkout lanes** (one GPU per config × context job). It does **not** batch online PPL tokens across a sequence — that would change the metric. See Modal doc §3 for the full parallelism matrix.

---

## 8. Explicit Non-Goals

| Technology | Why excluded (for now) |
|---|---|
| **CoreML** | Deployment-focused; no K/V tensor access |
| **Apple Intelligence / CoreAI** | Inference APIs only; no cache introspection |
| **Ollama / GGUF** | No `past_key_values`; cannot intercept KV |
| **MLX** | Optional future ablation (PyTorch-MPS vs MLX throughput) |
| **FlashAttention** | Breaks KV interception (see §1.3) |

---

## 9. File Map

```text
framework/
  model.py          ModelLayer — eager attn, fp16, MPS
  kv_engine.py      KVCacheEngine — intercept loop
  kv_cache.py       iter/decompress/rebuild DynamicCache
  device.py         MPS / CUDA / CPU selection (KV_EVAL_DEVICE)

modal_app/          Modal NVIDIA sweep (see docs/MODAL_GPU_EVAL_DESIGN.md)
  worker.py         eval_worker @ A10G — one job per GPU
  sweep.py          spawn_map orchestrator

compressors/
  base.py           KVCompressor ABC
  turboquant.py     TurboQuant plug-in
  qjl.py            QJL plug-in
  rocketkv.py       RocketKV plug-in
  identity.py       No-compression baseline
  registry.py       Factory: get_compressor(name, bitwidth, stage)

quantizers/
  hadamard.py       WHT + pad/unpad
  lloyd_max.py      Lloyd-Max centroids
  qjl.py            Sign-projection primitives (shared)
  qjl_pipeline.py   Standalone QJL pipeline
  rocketkv.py       TokenSelector + HSA
  turboquant_pipeline.py  Stage enum + per-tensor pipeline

docs/
  QJL_AND_ROCKETKV.md   QJL + RocketKV implementation reference

eval/               Paper-independent metrics
reporting/          JSON/CSV export
```

---

## 10. System Guarantees

This design provides:

- A clean modular research engine with one plug-in point (`KVCompressor`)
- Zero eval duplication across TurboQuant, KIVI, QJL, RocketKV
- True KV-level control of LLM inference without model surgery
- Publication-grade metric structure (quality / memory / throughput)
- Reproducible on Apple Silicon with documented eager-attention tradeoff
- **NVIDIA CUDA path** on Modal for full parallel eval sweeps (A10G, job-level `spawn_map`)

---

## 11. Design Verification Checklist

Verification against common KV-compression mistakes (checked against current codebase).

### Mistake 1: Compressing full KV globally

**Rule:** compress per layer, per head, per token vector — not one flattened cache blob.

| Check | Status | Evidence |
|---|---|---|
| Per layer | ✅ | `apply_compressor()` loops `iter_layer_kv()`; one `compress()` call per layer |
| Per head + token | ✅ | TurboQuant ops apply on last dim `D` (head_dim); each `(batch, head, token)` vector transformed independently |
| Not global | ✅ | No code flattens all layers/heads/tokens into a single matrix |

### Mistake 2: Ignoring attention compatibility

**Rule:** validate `q @ k.T` impact, not just tensor reconstruction error.

| Check | Status | Evidence |
|---|---|---|
| Reconstruction error tested | ✅ | `eval/fidelity.py`, `test_turboquant.py` |
| Attention score test (`q @ k.T`) | ✅ | `eval/attention_score_error.py`; `test_attention_score_error.py` |
| Perplexity with compressed KV | ✅ | `eval/perplexity.py` via `KVCacheEngine.step()`; `test_online_inference.py` |
| Throughput with compressed KV | ✅ | `eval/throughput.py` via `KVCacheEngine.generate()`; `test_online_inference.py` |

**Verified (identity baseline):** attention RMSE < 1e-3; online PPL within 5% of uncompressed baseline.

### Mistake 3: Storing full S matrix

**Rule:** never persist `d × d` projection matrix; use fixed seed + implicit regeneration.

| Check | Status | Evidence |
|---|---|---|
| S not in payload | ✅ | `TurboQuantTensorPayload` stores only `indices`, `qjl_bits`, `norm_r` |
| Regenerated from seed | ✅ | `projection_matrix(dim, seed=seed+dim)` in `quantizers/qjl.py` |
| Runtime cache only | ✅ | `_projections` dict in `TurboQuantPipeline` — in-memory, not serialized |

### Mistake 4: Incorrect Hadamard scaling

**Rule:** WHT must be orthonormal (`H^T H = I`) plus explicit `÷√D` normalization.

| Check | Status | Evidence |
|---|---|---|
| Orthonormal WHT (CPU/MPS) | ✅ | Scipy path: `H / √n`; roundtrip error ~1e-6 |
| Normalize + inverse pairing | ✅ | Compress: `÷√D` after WHT; decompress: `×√D` before inverse WHT |
| CUDA FHT path | ⚠️ | Modal uses scipy WHT fallback; `fast-hadamard-transform` not in Modal image |

### Reusability across papers

The `KVCompressor` interface is the single swap point. Other methods plug in without touching model, engine, or eval.

| Paper | How it maps | Status |
|---|---|---|
| **KIVI** | Replace `compress_kv()` with asymmetric scalar INT quant only (skip WHT/QJL) | 🔜 stub in `compressors/kivi.py` |
| **QJL** | Standalone key sign-projection; `estimate_attention_scores()` for inner products | ✅ `compressors/qjl.py`, `quantizers/qjl_pipeline.py` |
| **RocketKV** | Token selection + eviction; no vector quantization | ✅ `compressors/rocketkv.py`, `quantizers/rocketkv.py` |
| **TurboQuant** | Full pipeline in `quantizers/turboquant_pipeline.py` | ✅ implemented |

```text
KVCacheEngine (fixed)
       │
       ├── TurboQuantCompressor   → WHT + Lloyd-Max + QJL
       ├── KIVICompressor         → scalar quant only
       ├── QJLCompressor          → projection sign only
       └── RocketKVCompressor     → token selection + eviction
```

Same engine, same eval runner, same reporting — only `compressors/` changes.

---

## 12. Methodology Gap Closure (Verified)

All three evaluation gaps identified in the architecture review are closed and tested.

| Gap | Fix | Test |
|---|---|---|
| No `QK^T` metric | `eval/attention_score_error.py` | `test_attention_score_error.py` |
| PPL on uncompressed KV | `eval/perplexity.py` → `KVCacheEngine.step()` | `test_online_inference.py` |
| Throughput on uncompressed path | `eval/throughput.py` → `KVCacheEngine.generate()` | `test_online_inference.py` |

**Verification run (`pytest tests/ -v`):** 18/18 passed.

**Identity sanity checks:**
- Attention RMSE < 1e-3 (compression path preserves inner products)
- Online PPL within 5% of uncompressed baseline
- Throughput reports `online_compressed_kv=True`

**TurboQuant:** attention RMSE is now measured per layer (expected > 0); use Section A metrics before comparing papers.
