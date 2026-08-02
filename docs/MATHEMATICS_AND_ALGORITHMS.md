# Mathematics and Algorithms

Formal definitions for the KV interception engine, compression methods, and evaluation metrics. Symbols match the implementation in `quantizers/` and `eval/`.

Methodology context: [METHODOLOGY.md](METHODOLOGY.md)

---

## 1. Notation

| Symbol | Meaning |
|---|---|
| \(B\) | Batch size (always 1 in eval) |
| \(L\) | Number of decoder layers (28) |
| \(H_q, H_k\) | Query / KV head counts (GQA: \(H_q > H_k\)) |
| \(d\) | Head dimension (128) |
| \(T, T_q, T_k\) | Sequence / query / key length |
| \(K, V\) | Key / value tensors, shape \([B, H_k, T, d]\) |
| \(Q\) | Query tensor, shape \([B, H_q, T_q, d]\) |
| \(H\) | Orthonormal Walsh–Hadamard matrix, \(H^\top H = I\) |
| \(S\) | Gaussian random projection, \(S \in \mathbb{R}^{m \times d}\) |

---

## 2. Attention (reference)

**Scaled dot-product scores:**

\[
A = \frac{Q K^\top}{\sqrt{d}} \in \mathbb{R}^{B \times H_q \times T_q \times T_k}
\]

**GQA head expansion** (when \(H_q \neq H_k\)): repeat each KV head \(H_q / H_k\) times before matmul.

**Softmax attention weights:** \( \mathrm{softmax}(A) \) along \(T_k\).

Implementation: `eval/attention_score_error.py::attention_scores`, `expand_kv_heads`.

---

## 3. TurboQuant

### 3.1 Padding and WHT

Pad vector \(x \in \mathbb{R}^{d_0}\) to \(d = 2^{\lceil \log_2 d_0 \rceil}\):

\[
x_{\text{pad}} = [x \;\|\; 0_{d-d_0}]
\]

Unit normalization and transform:

\[
\hat{x} = \frac{x_{\text{pad}}}{\|x_{\text{pad}}\|_2}, \qquad y = H \hat{x}
\]

Feature normalization (`normalize_features`):

\[
y \leftarrow \frac{y}{\sqrt{d}}
\]

Implementation: `hadamard.py`, `lloyd_max.py::normalize_features`.

### 3.2 Lloyd–Max quantization

**Centroids:** \( \mathcal{C} = \{c_1, \ldots, c_K\} \), \(K = 2^b\) for bitwidth \(b\), fitted via KMeans on \(10^6\) standard normal samples (seed 42).

**Per-vector scale** (`compute_gamma`):

\[
\gamma = \frac{\max_j |y_j|}{\max_k |c_k|}
\]

**Quantize / dequantize:**

\[
i = \arg\min_k |y_j / \gamma - c_k|, \qquad \hat{y}_{\text{MSE}} = c_i \cdot \gamma
\]

### 3.3 Residual and full pipeline

\[
r = y - \hat{y}_{\text{MSE}}
\]

**Full stage** adds QJL on residual (Section 4):

\[
y_{\text{rec}} = \hat{y}_{\text{MSE}} + \widehat{r}_{\text{QJL}}
\]

**Inverse to original space:**

\[
x_{\text{pad}} = H^\top (y_{\text{rec}} \cdot \sqrt{d}) \cdot \|x_{\text{pad}}\|_2
\]

Then unpad to \(d_0\). Implementation: `TurboQuantPipeline._from_rotated`.

### 3.4 Algorithm (compress)

```text
COMPRESS_TURBOQUANT(x, bitwidth b, stage):
  x_pad, d0 ← PAD_TO_POW2(x)
  y, ‖x‖, γ ← ROTATE(x_pad)           # unit norm + WHT + normalize + γ
  if stage = WHT_ONLY: store y; return
  idx ← QUANTIZE(y/γ, centroids[b])
  y_mse ← DEQUANTIZE(idx, centroids) * γ
  if stage = WHT_QUANT: store idx, γ, ‖x‖; return
  r ← y - y_mse
  if stage = WHT_QUANT_RESIDUAL: store idx, γ, ‖x‖, ‖r‖; return
  b_bits ← SIGN(S @ r)
  store idx, γ, ‖x‖, b_bits, ‖r‖
```

---

## 4. QJL (Johnson–Lindenstrauss sketch)

### 4.1 Projection

\[
S \in \mathbb{R}^{m \times d}, \quad S_{ij} \sim \mathcal{N}(0,1), \quad \text{seed} = 42 + d
\]

### 4.2 Encode

\[
z = S k, \qquad b = \mathrm{sign}(z) \in \{-1,+1\}^m
\]

where \(\mathrm{sign}(0) = +1\) via `torch.where(z >= 0, 1, -1)`.

Store \(\|k\|_2\) separately.

### 4.3 Decode (symmetric reconstruction)

\[
\hat{k} = \frac{\sqrt{\pi/2}}{m} \, S^\top b \cdot \|k\|
\]

Implementation: `qjl.py::qjl_decode`.

### 4.4 Asymmetric inner-product estimator

For query \(q\) and compressed key \((b_k, \|k\|)\):

\[
S_q = \mathrm{sign}(Sq)
\]

\[
q \cdot k \approx \frac{\sqrt{\pi/2}}{m} \|k\| \cdot (S_q^\top b_k)
\]

**Per query head** (GQA index \( \text{kv} = \lfloor \text{qi} / \text{group} \rfloor \)):

\[
\hat{A}_{b,h_q,t,t'} = \frac{1}{\sqrt{d}} \cdot \frac{\sqrt{\pi/2}}{m} \|k_{t'}\| \cdot \langle S_q^{(h_q,t)}, b_k^{(\text{kv},t')} \rangle
\]

Implementation: `QJLPipeline._estimate_from_signs`, `estimate_attention_scores`.

### 4.5 Algorithm (online attention)

```text
QJL_ATTENTION(Q, compressed_keys, S):
  for each query head h_q:
    kv ← MAP_GQA(h_q)
    Sq ← sign(S @ q[h_q])
    for each key position t':
      score[h_q,t,t'] ← (sqrt(π/2)/m) * ||k[t']|| * dot(Sq[t], sign_bits[kv,t'])
  return score / sqrt(d)
```

---

## 5. RocketKV

### 5.1 Stage 1 — prefix scoring

Window \(W\) = last \(w\) tokens; prefix \(P\) = tokens \([0, T-w)\).

\[
\bar{k} = \frac{1}{w}\sum_{t \in W} k_t
\]

\[
\text{score}_i = \frac{1}{H_k}\sum_{h} \langle k_{i,h}, \bar{k}_h \rangle, \quad i \in P
\]

Keep top \(\min(|P|, B - w)\) prefix indices plus all window indices, where \(B =\) `token_budget`.

### 5.2 Stage 1 — lock after budget

When global length \(T \geq B\), permanent set \(\mathcal{P}\) = globals of selected prefix at lock step. Later steps:

\[
\mathcal{I} = \mathcal{P} \cup W \quad \text{(truncated to } B \text{ tokens)}
\]

### 5.3 Stage 2 — HSA selection

Approximate score at current query position \(t_q\):

\[
s_i = \frac{1}{H_k}\sum_{h} \left\langle q_{h,t_q}, k_{h,i} \right\rangle
\]

(with GQA: average query groups per KV head)

Top-k dynamic set \(D = \mathrm{topk}(s, k)\). Final attention set:

\[
\mathcal{A} = \mathrm{union}(\mathcal{P}, D), \quad |\mathcal{A}| \leq B_{\text{HSA}}
\]

If \(|\mathcal{A}| > B_{\text{HSA}}\), retain all permanent tokens and fill remaining slots from \(D \setminus \mathcal{P}\) by score.

### 5.4 Algorithm (online step)

```text
ROCKETKV_STEP(layer, new_k, new_v, state):
  append new_k, new_v to logical cache
  if not state.locked and len(cache) >= token_budget:
      (_, cache, perm_globals) ← SELECT_WITH_BUDGET(cache, token_budget)
      state.locked ← true; state.permanent ← perm_globals
  elif state.locked:
      cache ← MAINTAIN_WITH_PERMANENT(cache, state.permanent, token_budget)
  on attention(query):
      (k_sparse, v_sparse, idx) ← HSA_SELECT(query, cache, hsa_budget, state.permanent)
      return standard_attention(query, k_sparse, v_sparse)
```

---

## 6. KV cache engine

**Compressed cache** per layer: list of per-token payloads \(\{p_k^{(t)}, p_v^{(t)}\}_{t=1}^{T}\).

**Decompress** (conceptually):

\[
K[:,:,t,:] = \mathrm{decompress}(p_k^{(t)}), \quad V[:,:,t,:] = \mathrm{decompress}(p_v^{(t)})
\]

**Incremental append** at step \(t\):

\[
p_k^{(t)} = \mathrm{compress\_kv}(K[:,:,t,:]), \quad \text{payloads}_{1:t-1} \text{ unchanged}
\]

---

## 7. Evaluation metrics

### 7.1 Tensor RMSE

\[
\mathrm{RMSE}_K = \sqrt{\frac{1}{|\Omega|}\sum_{(l,i) \in \Omega} \|K_l^{(i)} - \hat{K}_l^{(i)}\|_F^2}
\]

Mean over layers (and heads/tokens in implementation). Same for values.

### 7.2 Attention score error

For float scores \(A\) and compressed scores \(\hat{A}\) (same shape, trailing 512-token window):

\[
\mathrm{MSE} = \frac{1}{|\mathcal{I}|}\sum_{i \in \mathcal{I}} (A_i - \hat{A}_i)^2, \quad \mathrm{RMSE} = \sqrt{\mathrm{MSE}}
\]

\[
\cos(A, \hat{A}) = \frac{A \cdot \hat{A}}{\|A\|_2 \|\hat{A}\|_2}, \quad \max\_\mathrm{err} = \max_i |A_i - \hat{A}_i|
\]

Report layer-wise then arithmetic mean across layers.

### 7.3 Memory

\[
\text{compression\_ratio} = \frac{\text{uncompressed\_bytes}}{\text{compressed\_bytes}}
\]

\[
\text{effective\_bits\_per\_KV} = \frac{8 \cdot \text{compressed\_bytes}}{\text{num\_KV\_elements}}
\]

where `num_KV_elements` = \(2 \times L \times T \times H_k \times d\) (keys + values).

Analytical FP16 cache size:

\[
\text{bytes} = B \cdot 2 \cdot L \cdot T \cdot H_k \cdot d \cdot 2 \quad \text{(FP16 = 2 bytes)}
\]

### 7.4 Perplexity (sliding window)

For windows starting at `begin_loc` with stride \(S\):

\[
\mathcal{L} = \frac{1}{N}\sum_{i} \mathrm{NLL}(x_i \mid x_{<i}), \quad \mathrm{PPL} = e^{\mathcal{L}}
\]

Labels mask: tokens before `prev_end_loc` within each window set to `-100` (baseline path).

**Compressed path:** NLL computed only on tokens scored incrementally via `engine.step` logits (see `eval/perplexity.py`).

### 7.5 Throughput

\[
\text{tokens/sec} = \frac{N_{\text{gen}}}{t_{\text{elapsed}}}, \quad \text{latency\_ms/token} = \frac{1000 \cdot t_{\text{elapsed}}}{N_{\text{gen}}}
\]

with \(N_{\text{gen}} = 64\) from config.

---

## 8. Storage bit accounting

| Payload field | Bits |
|---|---|
| Lloyd-Max index | `bitwidth` per coefficient |
| QJL sign | 1 per dimension |
| FP32 scalar (norm, γ) | 32 each |
| FP16 passthrough | 16 per element |
| Metadata | fixed bytes × 8 per payload |

Shared centroids (TurboQuant): counted in `shared_metadata_bytes` once per job.

Implementation: `framework/storage_accounting.py`, payload `storage_bits()` methods.

---

## 9. File map (equations → code)

| Topic | Primary file |
|---|---|
| WHT | `quantizers/hadamard.py` |
| Lloyd-Max | `quantizers/lloyd_max.py` |
| TurboQuant pipeline | `quantizers/turboquant_pipeline.py` |
| QJL encode/decode/estimate | `quantizers/qjl.py`, `quantizers/qjl_pipeline.py` |
| RocketKV selection | `quantizers/rocketkv.py` |
| Online QJL attention | `framework/qjl_online.py` |
| Online RocketKV attention | `framework/rocketkv_online.py` |
| KV engine | `framework/kv_engine.py` |
| Section A metrics | `eval/fidelity.py`, `eval/attention_score_error.py`, `eval/memory.py` |
| Section B metrics | `eval/perplexity.py`, `eval/throughput.py` |
