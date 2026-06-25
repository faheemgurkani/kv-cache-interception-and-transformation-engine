# System Design

This document records the engineering decisions behind the KV-cache compression benchmark. The goal is a **NeurIPS-style systems replication**: a fixed model and evaluation stack, with pluggable compression methods.

---

## 0. Global Architecture (Non-Negotiable)

The system is organized into four layers. **Only the compression layer changes between papers.**

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

## 1. Model Layer (Fixed)

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
| **FlashAttention / SDPA (fused)** | KV tensors are fused inside optimized kernels; you cannot reliably read or replace K/V between steps |
| **Eager attention** | Standard PyTorch matmuls; `past_key_values` is materialized as explicit tensors per layer |

**Consequences if you skip eager mode:**

1. `outputs.past_key_values` may be incomplete, opaque, or backend-specific.
2. Compression hooks that read `(K, V)` after each forward pass break silently or produce wrong shapes.
3. Round-trip compress → decompress → re-inject into the cache fails unpredictably.

**Decision:** Always load with `attn_implementation="eager"`. Accept the throughput penalty — this is a research instrument, not a production deployment path.

---

## 2. KV Cache Interception System

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

### 2.3 Forward interception loop

```python
def step(self, input_ids, compressed_cache):
    # 1. Decompress stored cache → float K/V for the model
    past_kv = decompress_to_legacy_cache(compressed_cache, compressor, config, device)

    # 2. Standard forward
    outputs = model(input_ids, past_key_values=past_kv, use_cache=True)

    # 3. Compress updated cache after forward
    for layer_idx, (k, v) in enumerate(iter_layer_kv(outputs.past_key_values)):
        compressed_layers.append(compressor.compress(k, v, layer=layer_idx))

    return outputs.logits, CompressedCache(layers=compressed_layers)
```

### 2.4 Key insight

> **You are NOT modifying the model. You are modifying KV flow between the forward pass and the next token step.**

The model always sees decompressed float16 K/V. Compression affects **storage and bandwidth**, not the model architecture. This keeps the design paper-agnostic: swap `TurboQuantCompressor` for `KIVICompressor` and the engine unchanged.

### 2.5 DynamicCache rebuild

Transformers 5.x requires `DynamicCache(ddp_cache_data=layer_pairs, config=model.config)` when re-injecting decompressed tensors — not a raw tuple. See `decompress_to_legacy_cache()` in `framework/kv_cache.py`.

---

## 3. Compression Layer Interface

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

## 4. TurboQuant Compression Layer

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
| 2 | Normalize | `x / √D` | **Critical** — preserves variance; valid Lloyd-Max centroids |
| 3 | Lloyd-Max quantize | KMeans centroids + nearest index | Offline Gaussian fit, k = 2^bitwidth |
| 4 | Dequantize | `centroids[indices]` | MSE reconstruction x̂ |
| 5 | Residual | r = x − x̂ | TurboQuant's two-part encoding |
| 6 | QJL encode | b = sign(Sr) | Fixed-seed Gaussian S ∈ R^{d×d}, reused |
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
    "indices": int8,      # Lloyd-Max cluster indices
    "qjl_bits": int8,     # sign(Sr) ∈ {-1, +1}
    "norm_r": float32,    # ||r|| per vector (for QJL decode)
    "original_dim": int,  # before padding
    "padded_dim": int,
}
```

Centroids are shared (cached per bitwidth), not stored per layer.

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

## 5. Evaluation Pipeline (Fixed Across All Papers)

**Do not change eval code when adding a new compression method.**

| Metric | Module | Method |
|---|---|---|
| **Quality** | `eval/perplexity.py` | Sliding-window perplexity on WikiText-2; `ppl = exp(mean NLL)` |
| **Memory** | `eval/memory.py` | Uncompressed KV bytes vs compressed payload bytes; compression ratio |
| **Speed** | `eval/throughput.py` | `tokens/sec = generated_tokens / elapsed` |

Orchestrator: `eval/runner.py` → `EvaluationRunner`

Reporting: `reporting/reporter.py` → JSON + CSV in `results/`

### 5.1 Long-context evaluation

WikiText-2 documents are short. We **concatenate samples** until `target_length` (4K–32K) via `data/loader.py::build_long_context_ids()`. Standard practice in KV-cache papers.

### 5.2 Datasets

| Phase | Dataset |
|---|---|
| Phase 1 | WikiText-2 (`Salesforce/wikitext`, `wikitext-2-raw-v1`) |
| Phase 2 | WikiText-2 + small C4 subset |

---

## 6. Execution Order

Follow this order to avoid getting stuck:

| Phase | Goal | Command / check |
|---|---|---|
| **1 — Model** | Load, generate, confirm KV exists | `scripts/verify_kv_cache.py` |
| **2 — Intercept** | Extract K/V, print shapes | `scripts/validate_turboquant.py --phase intercept` |
| **3 — Baseline** | Identity compressor, perplexity | `run_eval.py --compressor identity` |
| **4 — TurboQuant** | WHT → quant → QJL step-by-step | `validate_turboquant.py --phase stages` |
| **5 — Full eval** | Memory + speed + perplexity | `run_eval.py --compressor turboquant --stage full` |
| **6 — Other papers** | Plug KIVI/QJL/RocketKV into same engine | Implement `compressors/{method}.py` |

---

## 7. Explicit Non-Goals

| Technology | Why excluded (for now) |
|---|---|
| **CoreML** | Deployment-focused; no K/V tensor access |
| **Apple Intelligence / CoreAI** | Inference APIs only; no cache introspection |
| **Ollama / GGUF** | No `past_key_values`; cannot intercept KV |
| **MLX** | Optional future ablation (PyTorch-MPS vs MLX throughput) |
| **FlashAttention** | Breaks KV interception (see §1.3) |

---

## 8. File Map

```text
framework/
  model.py          ModelLayer — eager attn, fp16, MPS
  kv_engine.py      KVCacheEngine — intercept loop
  kv_cache.py       iter/decompress/rebuild DynamicCache
  device.py         MPS/CPU selection

compressors/
  base.py           KVCompressor ABC
  turboquant.py     TurboQuant plug-in
  identity.py       No-compression baseline
  registry.py       Factory: get_compressor(name, bitwidth, stage)

quantizers/
  hadamard.py       WHT + pad/unpad
  lloyd_max.py      Lloyd-Max centroids
  qjl.py            QJL encode/decode
  turboquant_pipeline.py  Stage enum + per-tensor pipeline

eval/               Paper-independent metrics
reporting/          JSON/CSV export
```

---

## 9. System Guarantees

If you follow this design:

- Clean modular research engine with one plug-in point (`KVCompressor`)
- Zero eval duplication across TurboQuant, KIVI, QJL, RocketKV
- True KV-level control of LLM inference without model surgery
- Publication-grade metric structure (quality / memory / throughput)
- Reproducible on Apple Silicon with documented eager-attention tradeoff
