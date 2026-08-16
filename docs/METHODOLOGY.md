# Methodology

Complete methodology for the **KV Cache Interception and Transformation Engine**: system architecture, compression plug-ins, online inference paths, and evaluation protocol. Verified against the implementation in `framework/`, `compressors/`, `quantizers/`, and `eval/`.

Equations: [MATHEMATICS_AND_ALGORITHMS.md](MATHEMATICS_AND_ALGORITHMS.md) · Results: [RESULTS_COMPLETE.md](RESULTS_COMPLETE.md) · Reproduce: [REPRODUCIBILITY.md](REPRODUCIBILITY.md)

---

## 1. Experimental setup

| Parameter | Value | Source |
|---|---|---|
| Model | Qwen3-1.7B (`Qwen/Qwen3-1.7B`) | `configs/model.yaml` |
| Precision | FP16 weights/activations | `framework/model.py` |
| Attention | `attn_implementation="eager"` | Required for KV interception |
| GQA | 8 KV heads, 28 layers, head dim 128 | Model config |
| Dataset | WikiText-2 test (`wikitext-2-raw-v1`) | `configs/eval.yaml` |
| Context lengths | 128, 256, 512 | `configs/model.yaml` |
| PPL stride | 512 | `configs/eval.yaml` |
| FIDELITY attention window | 512 tokens (trailing) | `attention_fidelity_tokens` |
| Throughput (SYSTEM) | 64 generated tokens | `generated_tokens` |
| Batch size | 1 | `configs/eval.yaml` |
| GPU (reference sweeps) | Modal NVIDIA A10G | `configs/modal.yaml` |

Context construction: WikiText-2 test split is tokenized and concatenated to exactly `context_length` tokens via `data/loader.py` (`build_long_context_ids`).

---

## 2. System architecture

### 2.1 Fixed pipeline

```text
Tokenizer → Model forward → past_key_values → KVCacheEngine → KVCompressor → Attention → logits
```

Only `KVCompressor` (and its quantizer backend) changes between methods. Model weights, eval orchestrator, and metric definitions are shared.

### 2.2 KVCacheEngine (`framework/kv_engine.py`)

**Purpose:** Intercept KV tensors between autoregressive steps; store compressed payloads; decompress before each forward.

**Incremental compression:** Each new token position is compressed **once** when produced. Prior positions are never re-compressed (re-compressing the full cache caused NaN PPL in early development).

**Step loop (`engine.step`):**

1. Decompress `CompressedCache` → HuggingFace `DynamicCache` (if cache exists).
2. Forward one token with `past_key_values`, explicit `attention_mask`, and `position_ids`.
3. Extract new K/V slices from `past_key_values`; append compressed payloads per layer.
4. Return logits and updated `CompressedCache`.

**Method-specific online hooks:**

| Compressor | Hook | File |
|---|---|---|
| RocketKV | Patches Qwen3 eager attention for sparse HSA | `framework/rocketkv_online.py` |
| QJL | Patches attention to use asymmetric QJL estimator | `framework/qjl_online.py` |
| TurboQuant / identity | Standard decompress → full attention | — |

### 2.3 Compressor interface (`compressors/base.py`)

Every plug-in implements:

- `compress_kv(tensor, layer, mode)` / `decompress_kv(payload, mode)` — per-tensor path
- `compress(key, value, layer)` / `decompress(compressed)` — paired K/V path
- Optional: `reconstruction_error`, `attention_fidelity`, `estimate_attention_scores`, `reset_state`, `shared_storage_bytes`

### 2.4 Storage accounting (`framework/storage_accounting.py`)

Compressed size includes:

- Payload bits (indices, sign bits, norms, FP16 tensors)
- Per-payload metadata (TurboQuant 32 B, QJL 24 B, RocketKV 32 B)
- Shared metadata (TurboQuant Lloyd-Max centroids, QJL projection seeds)

Memory ratio: `uncompressed_bytes / compressed_bytes` where uncompressed is FP16 K+V for all layers and tokens from one forward pass.

---

## 3. TurboQuant methodology

**Reference implementation:** `quantizers/turboquant_pipeline.py` → `compressors/turboquant.py`

### 3.1 Pipeline stages

| Stage | Name | Stored |
|---|---|---|
| `wht_only` | WHT ablation | Rotated unit-norm coefficients `y` |
| `wht_quant` | WHT + Lloyd-Max | Quantized indices + γ scale + vector norm |
| `wht_quant_residual` | + residual norm | Above + ‖residual‖ (no QJL bits) |
| `full` | Production | Lloyd-Max + QJL on residual |

### 3.2 Per-vector procedure

1. **Pad** last dimension to next power of two (`quantizers/hadamard.py`).
2. **Unit normalize:** \( \hat{x} = x / \|x\|_2 \).
3. **WHT:** \( y = H \hat{x} \) (orthonormal Hadamard).
4. **Feature normalize:** \( y \leftarrow y / \sqrt{d} \).
5. **γ scaling:** per-vector amax scale so coefficients fit Lloyd-Max codebook.
6. **Quantize:** nearest centroid index (2/3/4-bit → 4/8/16 centroids from KMeans on Gaussian samples, seed 42).
7. **Residual (full stage):** \( r = y - \hat{y}_{\text{MSE}} \); QJL sign-encode \( r \) with Gaussian projection \( S \).
8. **Decompress:** inverse chain — dequantize → add QJL decode of residual → inverse WHT → multiply by stored vector norm → unpad.

### 3.3 Modal vs local WHT

- **CUDA Modal:** scipy Sylvester Hadamard (`scipy.linalg.hadamard`) — no `fast-hadamard-transform` in image.
- **Local CUDA (optional):** `fast-hadamard-transform` if installed.

### 3.4 Sweep configs (Phase 5)

| Label | bitwidth | stage |
|---|---|---|
| `tq_full_b2` | 2 | full |
| `tq_full_b3` | 3 | full |
| `tq_full_b4` | 4 | full |
| `tq_mse_b4` | 4 | wht_quant (no QJL residual) |

---

## 4. QJL methodology

**Reference implementation:** `quantizers/qjl_pipeline.py` → `compressors/qjl.py` · Online: `framework/qjl_online.py`

### 4.1 Key compression

For each key vector \( k \in \mathbb{R}^d \):

1. Store \( \|k\|_2 \) (FP32).
2. Draw fixed Gaussian \( S \in \mathbb{R}^{m \times d} \) (seed 42 + head_dim; default \( m = d \)).
3. Encode \( b = \mathrm{sign}(S k) \in \{-1,+1\}^m \) (strict sign, not round/threshold).
4. Values: FP16 passthrough (uncompressed).

### 4.2 Decode (offline / baseline path)

Reconstruct \( \hat{k} = \frac{\sqrt{\pi/2}}{m} S^\top b \cdot \|k\| \).

### 4.3 Asymmetric attention estimator (Section A + online Section B)

Do **not** decompress keys for attention. Use the literature ProdQJL estimator (Zandieh et al., Def. 3.1 / Eq. 4):

\[
q \cdot k \approx \frac{\sqrt{\pi/2}}{m} \|k\|_2 \cdot \langle S q,\, \mathrm{sign}(S k) \rangle
\]

where \( S q \in \mathbb{R}^m \) is the **float** JL projection of the query (never sign-quantized). Only the key sketch is binarized. Signing both sides estimates angle, not an unbiased inner product, and is therefore incorrect for attention.

**GQA:** Scores computed **per query head**; each query head maps to its KV head group (`qi // group`). Query heads within a GQA group are **not** averaged (earlier bug fix).

**Batched keys:** Online path concatenates per-token key payloads along sequence dim for efficient matmul.

### 4.4 Sweep config

| Label | bitwidth | seed |
|---|---|---|
| `qjl_default` | 1 (sign bits) | 42 |

---

## 5. RocketKV methodology

**Reference implementation:** `quantizers/rocketkv.py` → `compressors/rocketkv.py` · Online: `framework/rocketkv_online.py`

Token eviction — not vector quantization. Kept tokens remain FP16.

### 5.1 Stage 1 — permanent token filter (`TokenSelector`)

SnapKV-inspired prefix scoring:

1. Split sequence into **prefix** (all but last `window_size` tokens) and **window** (trailing `window_size`).
2. Score each prefix position: mean over heads of \( \langle k_i, \bar{k}_{\text{window}} \rangle \) where \( \bar{k}_{\text{window}} \) is mean window key per head.
3. Keep top `prefix_budget = token_budget - window_size` prefix tokens + entire window.
4. **Lock after budget:** Once global sequence length ≥ `token_budget`, selected prefix indices become **permanent** and are unioned with the trailing window on subsequent steps (`maintain_with_permanent`).

### 5.2 Stage 2 — Hybrid Sparse Attention (`HybridSparseAttention`)

At decode time, select up to `hsa_budget` tokens for attention:

1. Approximate scores: mean over GQA query groups of \( Q K^\top \) at current query position.
2. Take top-k dynamic indices; **union** with permanent (stage-1) global indices.
3. If union exceeds budget, prioritize permanent tokens then fill from dynamic scores.
4. Physical cache eviction: only selected K/V written to sparse cache (`_write_sparse_cache`).

### 5.3 Global index tracking

Local cache indices differ from global sequence positions after eviction. The compressor maintains `current_global` and `permanent_prefix_global` per layer; attention masks are aligned/truncated to stored cache length.

### 5.4 FIDELITY

`RocketKVCompressor.reconstruction_error` and `attention_fidelity` measure **post-selection** kept tokens (not identity on full cache). This avoids misleading RMSE = 0 when tokens are evicted offline.

### 5.5 Sweep configs (Phase 5)

| Label | token_budget | hsa_budget | window_size |
|---|---:|---:|---:|
| `rocketkv_r256` | 256 | 256 | 32 |
| `rocketkv_r512` | 512 | 512 | 32 |
| `rocketkv_r1024` | 1024 | 1024 | 32 |

When `token_budget ≥ seq_len`, no eviction occurs (compression ratio ≈ 1.0).

---

## 6. Evaluation methodology

Orchestrator: `eval/runner.py` · Modal worker: `modal_app/worker.py` (same code path).

### 6.1 Section A — offline fidelity (`eval/fidelity.py`)

Single forward pass with `use_cache=True`, `output_hidden_states=True`:

| Metric | Module | Definition |
|---|---|---|
| Key / value RMSE | `eval/fidelity.py` | Mean over layers of RMSE(compress→decompress) |
| Attention MSE/RMSE/cosine/max | `eval/attention_score_error.py` | Compare \(QK^\top/\sqrt{d}\) before vs after compression |
| Memory | `eval/memory.py` | Uncompressed vs compressed bytes, ratio, eff. bits/KV |

**Attention fidelity window:** Last `attention_fidelity_tokens` (512) query/key positions to limit \(O(n^2)\) memory.

**Method-specific attention path:**

- QJL / RocketKV: `compressor.attention_fidelity()` (estimator or post-selection)
- Others: compress → decompress → standard matmul

**State isolation:** `compressor.reset_state()` after Section A before Section B.

### 6.2 Section B — online inference

**Order constraint:** Baseline PPL runs **before** `KVCacheEngine` construction (RocketKV/QJL patch attention).

| Metric | Module | Protocol |
|---|---|---|
| Perplexity (baseline) | `eval/perplexity.py` | Sliding-window NLL, stride 512, standard HF forward |
| Perplexity (compressed) | `eval/perplexity.py` | Token-by-token `engine.step`, single persistent cache, explicit mask + position_ids |
| Throughput | `eval/throughput.py` | `engine.generate`, 64 new tokens, wall-clock tok/s |

**PPL formula:** \( \mathrm{PPL} = \exp\left(\frac{1}{N}\sum_i \mathrm{NLL}_i\right) \) over scored tokens in sliding windows.

**Compressed PPL scoring:** For each stride window, score tokens from `prev_end_loc + 1` through `end_loc - 1` using logits from incremental decode.

### 6.3 Shared baseline

Identity baseline (`identity_baseline`) runs once under Modal preset `baseline`. Method jobs report `perplexity_baseline` per job (should match shared baseline within noise).

---

## 7. Phase 5 sweep design

| Preset | Jobs | Configs × contexts |
|---|---|---|
| `baseline` | 3 | identity × {128,256,512} |
| `turboquant` | 12 | 4 configs × {128,256,512} |
| `qjl` | 3 | qjl_default × {128,256,512} |
| `rocketkv` | 9 | r256/r512/r1024 × {128,256,512} |

Parallelism: one Modal GPU per job (`eval_worker.spawn_map`). Within-job PPL is sequential by design.

Result bundles: `results/phase5_modal_*/` with `jobs/*.json`, merged CSV/JSON, `manifest.json`.

---

## 8. Known methodological limits

See [CURRENT_STATE.md](CURRENT_STATE.md):

- Single model, single dataset, ctx ≤ 512
- Section A does not always predict Section B PPL
- TurboQuant online throughput dominated by per-step compress/decompress
- QJL/RocketKV catastrophic PPL on Qwen3-1.7B under this pipeline reflects measured behavior, not implementation shortcuts (post-audit)
