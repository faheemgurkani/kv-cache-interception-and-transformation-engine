Yes. Based on everything you have provided, the right way to approach this is **not to redesign KVBench around the five models**. The existing engine should remain the default path, and we should add **model-specific compatibility layers and narrowly scoped evaluation extensions** around it.

The five-model target matrix is:

| Model                 | Architecture                 | Current state                                                                 | Required work                                              |
| --------------------- | ---------------------------- | ----------------------------------------------------------------------------- | ---------------------------------------------------------- |
| **OLMo2-1B**          | MHA, 16Q/16KV                | ✅ Fully supported (all three gates pass)                                      | Compatibility validation only                              |
| **Qwen3-0.6B**        | GQA, 16Q/8KV                 | ✅ Fully supported (all three gates pass)                                      | Compatibility validation only                              |
| **Gemma3-270M**       | MQA + sliding/full attention | ✅ Fully supported (`gemma3_text` adapter + per-layer RoPE; 86th commit)       | Compatibility validation only                              |
| **TinyDeepSeek-0.5B** | MLA                          | ⚠️ Loads, cache readable; Gates B + C fail                                     | `deepseek_v3` adapter; native-latent refinement later      |
| **Falcon-H1-0.5B**    | Attention + Mamba2 hybrid    | ⚠️ Loads, hybrid state visible + counted (Gate C pass); Gate B fails         | `falcon_h1` adapter (`qk_norm_layout="none"`)              |

The key architectural principle should be:

> **Existing model paths remain unchanged. New architectures are added through adapters, capability metadata, and specialized state handling only where required.**

This is particularly important because the current evaluation framework has **three distinct compatibility gates**: generic model/cache compatibility, attention-adapter compatibility, and model-specific attention quirks.

---

## Implementation verification audit (2026-08-18)

**Executive verdict:** The **§1–2 infrastructure narrative is implemented, active, and tested**. The codebase is prepared for the revision described in the opening text. Companion docs updated to match: [`models/ARCHITECTURE_REPORT.md`](../../models/ARCHITECTURE_REPORT.md), [`docs/results/shortlist_5model_eval/EVAL_FRAMEWORK_CORRESPONDENCE.md`](results/shortlist_5model_eval/EVAL_FRAMEWORK_CORRESPONDENCE.md), [`docs/architecture/MODEL_ARCHITECTURE_MATRIX.md`](architecture/MODEL_ARCHITECTURE_MATRIX.md).

**Evidence base:** live gate evaluation against all five checkpoints; `tests/test_{model_capabilities,compatibility_gates,model_adapter_registry,rope,memory_accounting}.py` (22 infra tests); `tests/test_{olmo2,qwen3,gemma3}_reference.py`; commits **79–86** on `main` (`1397f46` = 86th commit).

### Pillar 1 — system supports the revision (§1–2)

**Architectural principle (lines 13–15) — confirmed.** Legacy KV path not replaced:

| Plan requirement | Code evidence |
| ---------------- | ------------- |
| OLMo2/Qwen3 path unchanged | No model-specific branches in `framework/kv_engine.py` |
| `iter_layer_kv()` remains default | Universal cache reader in `framework/kv_cache.py`; used by compressors and FIDELITY |
| Extensions via adapters + capabilities | `ATTENTION_ADAPTER_REGISTRY`, `ModelCapabilities`, `iter_layer_states()` |

Extended stack live:

```text
Model → ATTENTION_ADAPTER_REGISTRY → iter_layer_kv / iter_layer_states → Evaluation
```

Exported from `framework/__init__.py`: `ModelCapabilities`, `CompatibilityGate`, `evaluate_compatibility_gates`, `iter_layer_states`, `visible_state_bytes`, `build_rope_context`.

**Three compatibility gates — confirmed** in `framework/compatibility.py`:

| Plan gate | Code enum | What it checks |
| --------- | --------- | -------------- |
| Generic model/cache compatibility | `LOADER_STATE` | Load, forward, `iter_layer_states()` discovers layers |
| Attention-adapter compatibility | `ATTENTION` | `adapter_registered` + `load_attention_ops()` succeeds |
| Model-specific state semantics | `STATE_SEMANTICS` | All visible state accounted for; native-latent flagged |

Eval runner integrates capabilities on every run (`eval/runner.py` → `model_capabilities`, `model_metadata`).

**§2 capability contract — confirmed.** `ModelCapabilities` in `framework/model_capabilities.py` matches the plan's conceptual fields plus gate flags. All five shortlist families registered; tested in `tests/test_model_capabilities.py`.

**Config / setup — ready.** Per-model YAML: `configs/model_{olmo2_1b,qwen3_0.6b,gemma3_270m,tinydeepseek_0.5b,falcon_h1_0.5b}.yaml`. All five checkpoints under `models/`.

### Pillar 2 — grounded in reality (doc ↔ code)

**Live gate results** (real checkpoints, 2026-08-18):

| Model | Gate A (loader/state) | Gate B (attention) | Gate C (state semantics) |
| ----- | --------------------- | ------------------ | ------------------------ |
| OLMo2-1B | PASS | PASS | PASS |
| Qwen3-0.6B | PASS | PASS | PASS |
| Gemma3-270M | PASS | PASS | PASS |
| TinyDeepSeek-0.5B | PASS | FAIL | FAIL |
| Falcon-H1-0.5B | PASS | FAIL | PASS |

Falcon empirical proof: `visible_bytes=14,708,736` vs `attn_bytes=36,864` — Mamba state visible and counted; `hybrid=True`.

**§2 capability table (lines 109–118) vs code — aligned:**

| Capability row | Code match |
| -------------- | ---------- |
| MHA / GQA / MQA / MLA families | `attention_family` values correct |
| Q/K norm layouts | Matches `qk_norm_layout` (flat / per-head / mla / none) |
| Global vs per-layer RoPE | `rope_mode`: global / per_layer_type / split_nope_rope |
| Per-layer attention type (Gemma3) | `per_layer_attention_type=True` + `get_layer_attention_metadata()` |
| Dual-state / recurrent (Falcon) | `StateType.HYBRID`, `has_recurrent_state=True` |
| Latent KV (TinyDeepSeek) | `native_latent_cache=True`, disclosure string, Gate C fails by design |

**Memory math — grounded.** `eval/fidelity/memory.py` implements the plan formula, including asymmetric K/V (`value_head_dim`) for MLA. Reference tests: Qwen3 GQA ratio 8/16 (`test_qwen3_gqa_memory_head_ratio_is_half_of_mha`); Gemma3 MQA ratio 1/16 vs OLMo2 (`test_gemma3_mqa_memory_head_ratio_is_quarter_of_mha`).

### Pillar 3 — anomalies fixed and flagged

**Fixed in code (active):**

1. Explicit three-gate framework — no implicit failures deep in eval.
2. `ModelCapabilities` registry — avoids scattered `if qwen3/elif olmo2/...` (plan lines 124–132).
3. Typed state interface — `framework/state_interface.py` (`AttentionKVState`, `RecurrentState`, `LayerState`).
4. Falcon memory undercount — fixed via `visible_state_bytes()`; Gate C passes.
5. Gemma3 adapter + per-layer RoPE — `gemma3_text` in registry; `build_rope_context().get_rope(layer_idx)` in FIDELITY/attention.
6. TinyDeepSeek native-latent disclosure — flagged in capabilities + Gate C message.
7. `--skip-fidelity` — documented in `eval/runner.py` for models blocked at Gate B.

**Flagged, intentionally open (matches plan intent):**

- **TinyDeepSeek:** Gate B (no `deepseek_v3` adapter) + Gate C (expanded cache ≠ native latent).
- **Falcon-H1:** Gate B only (no `falcon_h1` adapter; `qk_norm_layout="none"` scaffold in `project_qkv`, builder not registered).

### Verified work in place (nothing lost)

| Work item | Status | Evidence |
| --------- | ------ | -------- |
| WP1 infrastructure | Done | Commits 79–82; 22 unit tests pass |
| OLMo2 reference + conformance | Done | `tests/test_olmo2_reference.py`; Gates A/B/C PASS |
| Qwen3 parameterized conformance | Done | `tests/test_qwen3_reference.py` (85th commit); Gates PASS |
| Gemma3 adapter + RoPE + eval metadata | Done | 86th commit; `tests/test_gemma3_reference.py`; Gates PASS; all compressors |
| All work committed | Done | Working tree clean; `1397f46` |

### Bottom line (three criteria)

1. **System ready for §1–2 narrative?** **Yes.** Registry, capabilities, gates, state interface, RoPE abstraction, eval metadata, configs in place.
2. **Grounded in reality?** **Yes** (code and docs synced 2026-08-18). Live gates match opening table above.
3. **Anomalies fixed/flagged?** **Yes in implementation.** Remaining work is §14+ (TinyDeepSeek/Falcon adapters), not missing §1–137 infrastructure.

---

# 1. First: establish the compatibility architecture

Before going model by model, I would make one small conceptual change to the engine.

Currently, the architecture is effectively:

```text
Model
  ↓
Model Adapter
  ↓
Attention Operations
  ↓
KV Cache
  ↓
Compressor
  ↓
Evaluation
```

Do **not** replace this.

Instead, extend it to:

```text
                         Model
                           │
                           ▼
                 Model Adapter Registry
                           │
             ┌─────────────┴─────────────┐
             │                           │
       Existing path               New adapter path
             │                           │
       Qwen3 / OLMo2          Gemma / DeepSeek / Falcon
             │                           │
             └─────────────┬─────────────┘
                           ▼
                    State Interface
                           │
              ┌────────────┼────────────┐
              │            │            │
          KV State      MLA State    Hybrid State
              │            │            │
              ▼            ▼            ▼
         Existing       Specialized   Specialized
         compressors    handling       handling
              │            │            │
              └────────────┼────────────┘
                           ▼
                    Evaluation Framework
```

The important part is that **the existing KV path remains the default**.

So something like:

```python
iter_layer_kv()
```

should continue working exactly as it does for OLMo2 and Qwen3.

You add capabilities around it rather than replacing it.

---

# 2. Compatibility contract

I recommend defining a lightweight internal contract for every supported model.

Something conceptually like:

```python
ModelCapabilities(
    attention_family="mha",
    kv_layout="standard",
    qk_norm_layout="per_head",
    rope_mode="global",
    has_recurrent_state=False,
    native_latent_cache=False,
)
```

The exact implementation is up to the existing code style, but the **capability information** is important.

For the five models:

| Capability               | OLMo2 | Qwen3    | Gemma3                | TinyDeepSeek          | Falcon-H1             |
| ------------------------ | ----- | -------- | --------------------- | --------------------- | --------------------- |
| Standard K/V cache       | ✓     | ✓        | ✓                     | ✓*                    | ✓*                    |
| MHA/GQA/MQA/MLA          | MHA   | GQA      | MQA                   | MLA                   | GQA (attention half)  |
| Q/K norm                 | flat  | per-head | per-head              | MLA-specific          | none                  |
| Global RoPE              | ✓     | ✓        | ✗                     | split (nope + rope)   | ✓ (attention half)    |
| Per-layer attention type | ✗     | ✗        | ✓ (sliding/full)      | ✗                     | ✗                     |
| Dual-state layer         | ✗     | ✗        | ✗                     | ✗                     | ✓ (attn + Mamba)      |
| Recurrent state          | ✗     | ✗        | ✗                     | ✗                     | ✓                     |
| Latent KV                | ✗     | ✗        | ✗                     | ✓ (native; cache expanded*) | ✗             |

`*` **TinyDeepSeek:** HF eager materializes expanded per-head K/V (`D_k ≠ D_v`); the visible cache is not the native `kv_lora_rank` latent. Gate C **fails** until MLA-native interception lands (expanded-cache accounting is still correct).

`*` **Falcon-H1:** Attention K/V is visible via `.keys`/`.values`; Mamba `recurrent_states` + `conv_states` live on the same cache layer. Memory accounting counts all visible components; compression policy remains attention-K/V only (Mamba passthrough).

This capability layer prevents future additions from turning the engine into:

```python
if qwen3:
elif olmo2:
elif gemma:
elif falcon:
...
```

everywhere.

---

# 3. Model 1: OLMo2-1B

## Current status

**No functional engine change is required.**

You have already demonstrated:

* 1,484,916,736 parameters
* 16 layers
* hidden size 2048
* 16 Q heads
* 16 KV heads
* genuine MHA
* standard cache
* `olmo2` adapter already registered
* `iter_layer_kv()` works
* global RoPE is correct
* FIDELITY works
* BEHAVIOR works
* SYSTEM works

The actual run produced:

* PPL: **13.98 @ context 128**
* attention cosine: **0.9998**
* throughput: **16.9 tok/s**
* identity compression ratio: **1.0×**

So OLMo2 is not a model that needs to be "made compatible."

It is your **compatibility baseline**.

---

## Engine: changes

### Required

**None.**

Do not touch the OLMo2 execution path.

That is important for backward compatibility.

The existing:

```text
model_type == "olmo2"
        ↓
existing attention adapter
        ↓
existing RoPE
        ↓
existing KV iteration
        ↓
existing compressor
```

should remain intact.

### Recommended addition

Add a **conformance test**, not a code modification.

For example:

```text
test_olmo2_adapter_conformance()
```

should verify:

1. Model loads.
2. Eager attention loads.
3. Forward pass succeeds.
4. Cache has 16 layers.
5. Every layer exposes K and V.
6. K/V heads = 16.
7. K/V shape is expected.
8. RoPE computation succeeds.
9. `iter_layer_kv()` returns exactly 16 pairs.
10. Identity reconstruction is exact.

This becomes the regression test for the legacy path.

---

# 4. OLMo2 evaluation framework changes

Again, **no functional changes**.

But OLMo2 should become the **reference implementation for evaluation correctness**.

The framework should record:

```text
Model: OLMo2-1B
Attention: MHA
Q heads: 16
KV heads: 16
RoPE: global
State: standard KV
Adapter: olmo2
```

Then all evaluation branches should run:

```text
FIDELITY
 ├── representation
 ├── memory
 └── attention

BEHAVIOR
 └── identity / TurboQuant / ...

SYSTEM
 └── QJL / RocketKV / ...
```

The existing framework already treats FIDELITY/attention and QJL/RocketKV as adapter-dependent while generic representation/memory and identity/TurboQuant use the generic cache path. 

### Acceptance criterion

The existing numbers should remain unchanged within normal run-to-run noise.

That gives you a very important regression guarantee:

> **Adding support for the new architectures must not alter the established OLMo2 results.**

---

# 5. Model 2: Qwen3-0.6B

## Current status

Again, **fully supported**.

You measured:

* 596,049,920 parameters
* 28 layers
* 16 Q heads
* 8 KV heads
* genuine 2:1 GQA
* bfloat16
* Qwen3 architecture
* standard cache
* global RoPE
* per-head Q/K normalization

And the existing `qwen3` adapter already handles the architecture.

The important point is that **Qwen3-1.7B and Qwen3-0.6B share the same engine path**.

Therefore Qwen3-0.6B should not receive a new model-specific implementation.

---

## Engine: changes

### Required

None.

Keep:

```python
if model_type == "qwen3":
```

as the common Qwen3 path.

Do not create:

```python
if model_name == "qwen3_0.6b":
```

That would be the wrong abstraction.

Architecture family, not checkpoint identity, should determine the adapter.

---

## What should be added instead?

A parameterized Qwen3 conformance test:

```text
Qwen3-1.7B
Qwen3-0.6B
       │
       ▼
same adapter
       │
       ▼
same attention implementation
```

Verify:

```text
Q heads = 16
KV heads = 8
KV/Q ratio = 0.5
```

and that:

```text
iter_layer_kv()
```

returns 28 layers.

---

# 6. Qwen3 evaluation framework

The evaluation framework should likewise require **no new algorithm-specific logic**.

But there should be a model metadata check.

For example:

```text
attention_family = GQA
num_q_heads = 16
num_kv_heads = 8
```

This is important for memory accounting.

For standard KV caching, ignoring datatype and batch size for simplicity:

[
M_{KV}
======

L \times T \times H_{KV} \times D \times 2 \times b
]

where:

* (L) = number of layers
* (T) = cached tokens
* (H_{KV}) = KV heads
* (D) = head dimension
* (b) = bytes per element
* factor 2 = K + V

Thus Qwen3's 8 KV heads versus OLMo2's 16 means, all else equal:

[
\frac{M_{Qwen}}{M_{OLMo2}}
==========================

# \frac{8}{16}

0.5
]

for the per-layer KV-head component.

That difference is **part of the scientific variable KVBench wants to study**, so the evaluation framework should report the actual architecture rather than normalize it away.

This aligns with the existing paper's observation that compression rankings can depend on attention layout, particularly GQA versus MHA. 

---

# 7. Model 3: Gemma3-270M

This is the first model requiring actual additions.

## Architecture

Gemma3 has:

```text
4 Q heads
1 KV head
```

so:

[
H_{KV}/H_Q = 1/4
]

This is genuine MQA.

But the more important feature is:

```text
Layer
 ├── sliding-window attention
 └── full attention
```

with the model alternating between local and global attention.

And the live model confirmed two separate RoPE buffers:

```text
sliding_attention_inv_freq
full_attention_inv_freq
```

Therefore the existing assumption:

```text
one model
    ↓
one RoPE table
    ↓
all layers
```

is invalid.

---

# 8. Gemma3 engine changes

There are **two required changes**.

## Change 1: adapter registration

Add:

```text
gemma3_text
```

to the model adapter registry.

Conceptually:

```python
if model_type == "qwen3":
    ...
elif model_type == "olmo2":
    ...
elif model_type == "gemma3_text":
    ...
```

But I would prefer eventually converting this registry into a mapping:

```python
ATTENTION_ADAPTERS = {
    "qwen3": Qwen3Adapter,
    "olmo2": Olmo2Adapter,
    "gemma3_text": Gemma3Adapter,
}
```

This is an **additive refactor**, provided the old dispatch behavior is preserved.

---

# 9. Gemma3 engine change 2: per-layer-type RoPE

This is the important part.

The current assumption is essentially:

```python
cos, sin = rotary_emb(hidden_states, position_ids)
```

Gemma3 needs:

```python
cos, sin = rotary_emb(
    hidden_states,
    position_ids,
    layer_type=...
)
```

or the equivalent API required by the actual Transformers implementation.

### Do not recompute RoPE for every layer

That would work conceptually but is unnecessarily expensive.

Instead:

```text
Initialize model
       │
       ▼
Compute two RoPE contexts
       │
       ├── sliding RoPE
       │
       └── full RoPE
       │
       ▼
During layer iteration
       │
       ├── sliding layer → sliding table
       │
       └── full layer → full table
```

So:

[
R_l =
\begin{cases}
R_{\text{sliding}}, & l \in \mathcal{S}\
R_{\text{full}}, & l \in \mathcal{F}
\end{cases}
]

where (\mathcal{S}) and (\mathcal{F}) are the layer-type sets.

---

# 10. Where Gemma3 RoPE changes are needed

This is especially important because fixing only the attention evaluator will **not** be sufficient.

You identified three relevant execution locations:

### A. `eval/fidelity/attention.py`

Currently assumes one global RoPE context.

Change:

```text
global rope
```

to:

```text
rope_context[layer_type]
```

---

### B. QJL online path

The QJL online attention computation eventually calls the attention adapter.

It must receive the correct layer-specific RoPE.

---

### C. RocketKV online path

Same requirement.

The RocketKV attention computation must not accidentally use the global/full or global/sliding table for every layer.

---

## Recommended abstraction

Instead of exposing:

```python
get_rope()
```

use:

```python
get_rope(layer_idx)
```

Internally:

```python
layer_type = model.config.layer_types[layer_idx]
return rope_context[layer_type]
```

This is backwards compatible because:

```python
get_rope(layer_idx)
```

for OLMo2/Qwen3 can simply return the same global context for every layer.

So:

```text
Old models:
layer 0 → same global RoPE
layer 1 → same global RoPE
...

Gemma3:
layer 0 → sliding
layer 1 → sliding
...
layer 5 → full
...
```

No old model changes behavior.

---

# 11. Gemma3 cache handling

The existing cache reader already succeeds.

That is significant.

You confirmed:

```text
sliding layer → DynamicSlidingWindowLayer
full layer    → DynamicLayer
```

and both expose K/V.

Therefore **do not rewrite `iter_layer_kv()` just for Gemma3.**

Instead, preserve:

```python
iter_layer_kv()
```

for standard KV extraction.

If needed, add metadata:

```python
iter_layer_states()
```

or:

```python
get_layer_attention_metadata(layer_idx)
```

that exposes:

```text
attention_type = sliding/full
window_size = ...
```

But do not force the existing KV iterator to become architecture-aware unless necessary.

---

# 12. Gemma3 evaluation framework

Gemma3 needs evaluation metadata beyond simply:

```text
KV cache exists
```

For every layer, record:

```text
layer_idx
attention_type
kv_heads
head_dim
window_size
rope_type
```

Then FIDELITY/attention must reconstruct attention using the same positional semantics as the original model.

This is essential because otherwise you could obtain a numerical "attention preservation" score against an attention calculation that does **not correspond to the model's actual attention mechanism**.

That would make the metric invalid.

---

# 13. Gemma3 evaluation acceptance test

Before allowing benchmark results:

```text
Load
 ↓
Forward
 ↓
Cache extraction
 ↓
Layer classification
 ↓
RoPE selection
 ↓
Original attention
 ↓
Reconstructed attention
 ↓
Identity compression
 ↓
Compare logits
```

Identity compression should establish:

[
\Delta_{\text{logit}}
\approx 0
]

and:

[
\text{cosine}(A_{\text{original}},A_{\text{identity}})
\approx 1
]

within the numerical tolerance appropriate for bfloat16.

Only after this should TurboQuant/QJL/RocketKV be enabled.

---

# 14. Model 4: TinyDeepSeek-0.5B

This is actually much easier than it initially appeared.

## Current status

The important correction you established is:

> The custom repository implementation failing does not mean the model is unusable by KVBench.

The engine's actual loading path succeeds.

You confirmed:

* model loads
* forward pass works
* MLA internals exist
* `kv_lora_rank=256`
* `kv_a_layernorm`
* `kv_b_proj`
* RoPE/non-RoPE Q/K components
* cache extraction works
* keys and values have structurally different final dimensions

The actual blocker is simply:

```text
model_type='deepseek_v3'
```

not being registered with the attention adapter.

---

# 15. TinyDeepSeek engine changes

## Change 1: adapter registration

Add:

```text
deepseek_v3
```

to the adapter registry.

Conceptually:

```python
ATTENTION_ADAPTERS = {
    ...
    "deepseek_v3": DeepSeekMLAAdapter,
}
```

The existing built-in implementation should remain the loader.

Do not activate the broken custom-code path merely to support the model.

---

# 16. MLA attention adapter

There is one important distinction here.

Do **not** pretend MLA is ordinary MHA/GQA.

The adapter should understand the model's attention projections sufficiently to compute the attention used by FIDELITY/attention.

The conceptual structure is:

```text
Hidden state
     │
     ├── latent KV projection
     │
     ├── decoupled RoPE component
     │
     └── query projections
```

The attention adapter must reproduce the model's native attention computation.

However, according to your current testing, the built-in implementation exposes an expanded cache:

```text
K → dimension 64
V → dimension 32
```

rather than the raw compact latent vector.

That means the **first implementation can support functional experimentation**, but it should not yet claim:

> "KVBench compresses MLA's native latent KV state."

It would actually be:

> "KVBench compresses the K/V representation exposed by the current implementation."

That distinction needs to be documented.

---

# 17. TinyDeepSeek evaluation framework

The framework needs a new architecture label:

```text
attention_family = MLA
```

and metadata such as:

```text
kv_lora_rank = 256
qk_nope_head_dim = ...
qk_rope_head_dim = ...
v_head_dim = ...
```

The memory accounting must use the **actual tensors being stored by the cache implementation**.

If the implementation exposes:

[
K \in \mathbb{R}^{T \times 64}
]

and:

[
V \in \mathbb{R}^{T \times 32}
]

then the reported cache memory must be based on those actual dimensions:

[
M =
T \cdot b \cdot
(H_KD_K + H_VD_V)
]

rather than blindly assuming:

[
M = 2TH_{KV}D b
]

because that conventional formula assumes identical K/V dimensions.

---

# 18. TinyDeepSeek: important methodological safeguard

Add a field to the evaluation report:

```text
cache_representation = expanded_kv
```

rather than:

```text
cache_representation = native_mla_latent
```

until the true latent-state implementation exists.

This prevents a scientifically misleading claim.

Then later:

```text
expanded KV
      ↓
native latent state
```

can become an explicit ablation.

That is actually potentially interesting research:

> Does compression of the expanded MLA representation provide the same benefit as compression of MLA's native latent state?

But that is **not required to establish basic TinyDeepSeek support**.

---

# 19. Model 5: Falcon-H1-0.5B

This is the most important model from an engine-design perspective.

## Architecture

Each layer contains:

```text
Falcon-H1 layer
       │
       ├── Attention
       │     ├── 8 Q heads
       │     └── 2 KV heads
       │
       └── Mamba2
             └── recurrent state
```

The two mechanisms operate **in parallel in the same layer**.

Therefore:

```text
one layer ≠ one KV pair
```

That is the fundamental difference.

---

# 20. Falcon-H1 engine change 1: adapter registration

Add:

```text
falcon_h1
```

to the adapter registry.

The attention adapter must support:

```text
Q heads = 8
KV heads = 2
Q/K normalization = none
```

The last item matters because the current implementation does not have a named path for "no Q/K norm."

So add an explicit enum/configuration:

```python
qk_norm_layout = "none"
```

rather than overloading another normalization mode.

This should be backwards compatible:

```text
existing:
flat
per_head

new:
none
```

---

# 21. Falcon-H1 engine change 2: dual-state cache representation

This is the substantial change.

The existing conceptual model is:

```python
layer -> (key, value)
```

Falcon requires:

```python
layer -> {
    attention_state,
    recurrent_state
}
```

But **do not break the existing `iter_layer_kv()` API.**

Instead, introduce an additional interface.

For example:

```python
iter_layer_states()
```

conceptually yielding:

```python
LayerState(
    layer_idx=0,
    attention=AttentionState(...),
    recurrent=MambaState(...),
)
```

Then retain:

```python
iter_layer_kv()
```

as a compatibility view:

```python
for state in iter_layer_states():
    if state.attention is not None:
        yield state.attention.key, state.attention.value
```

This is the key backward-compatibility decision.

Existing models continue to behave exactly as before.

---

# 22. Falcon-H1 compression policy

For the first supported version:

```text
Attention state → compress
Mamba state     → preserve exactly
```

This must be explicit.

You are **not** claiming to compress Falcon-H1's entire inference state.

You are testing:

> KV compression applied to the attention component of a hybrid architecture while preserving the recurrent state.

Mathematically, if total inference state is:

[
S_t = (K_t,V_t,R_t)
]

where (R_t) is the Mamba state, then your transformation is:

[
\mathcal{C}(S_t)
================

(\mathcal{C}_{KV}(K_t,V_t), R_t)
]

and not:

[
\mathcal{C}(S_t)
================

(\mathcal{C}*{KV}(K_t,V_t),\mathcal{C}*{R}(R_t))
]

The latter is explicitly outside the current scope.

---

# 23. Falcon-H1 memory accounting

This is where the existing implementation **must not simply be reused**.

If the framework reports:

[
M_{\text{reported}} = M_{KV}
]

while the actual state is:

[
M_{\text{actual}} =
M_{KV}+M_{Mamba}
]

then the framework systematically underestimates memory.

Your own live testing showed exactly this risk.

Therefore the framework should distinguish:

### Attention KV memory

[
M_{KV}
]

### Recurrent-state memory

[
M_R
]

### Total inference-state memory

[
M_{\text{state}}
================

M_{KV}+M_R
]

Then compression ratio should be reported carefully.

If only attention KV is compressed:

[
M_{\text{compressed}}
=====================

M_{KV,c}+M_R
]

and:

[
CR_{\text{total}}
=================

\frac{M_{KV}+M_R}
{M_{KV,c}+M_R}
]

This is fundamentally different from:

[
CR_{KV}
=======

\frac{M_{KV}}
{M_{KV,c}}
]

**Both can be useful, but they must not be conflated.**

I would report both:

```text
KV compression ratio
Total inference-state reduction
```

for Falcon-H1.

---

# 24. Falcon-H1 evaluation framework

The evaluation framework needs to become state-aware.

Instead of assuming:

```text
model → KV cache → compressor
```

the evaluator should conceptually execute:

```text
model
 ↓
inference state
 ↓
 ┌───────────────────────┐
 │                       │
Attention state      Mamba state
 │                       │
compress              preserve
 │                       │
 └───────────┬───────────┘
             ↓
       model continuation
```

---

# 25. Falcon-H1 FIDELITY

FIDELITY should report at least:

### Attention representation

[
RMSE(K,K')
]

[
RMSE(V,V')
]

### Attention preservation

[
\cos(A,A')
]

### Mamba preservation

Since the Mamba state is deliberately untouched:

[
R'_t = R_t
]

should be asserted rather than treated as a compression metric.

### Total memory

Report:

```text
Original attention state
Original recurrent state
Compressed attention state
Preserved recurrent state
Total original state
Total compressed state
```

This makes the benchmark honest.

---

# 26. Falcon-H1 BEHAVIOR

The online decode loop must preserve the recurrent state exactly.

At each token:

```text
Previous state
      │
      ├── compressed K/V
      │
      └── exact Mamba state
             │
             ▼
          next token
```

The evaluation should verify that introducing the KV compressor does not accidentally:

* reset Mamba state
* duplicate Mamba state
* move it to the wrong device
* change its dtype
* detach it incorrectly
* omit it during generation

This is especially important because these bugs could produce seemingly valid generations while actually changing the model's recurrent computation.

---

# 27. Falcon-H1 SYSTEM evaluation

Throughput should include the actual hybrid forward pass.

Do **not** report a throughput measurement based on an artificially simplified attention-only state.

Likewise, memory should include the preserved Mamba state.

The system evaluation should therefore distinguish:

```text
attention compression overhead
+
Mamba state overhead
+
overall generation throughput
```

This prevents the benchmark from making Falcon-H1 look artificially memory-efficient merely because it ignored half of its state.

---

# 28. What should happen to the compressor API?

This is one of the most important backwards-compatibility decisions.

Do **not** immediately rename:

```python
compress_kv()
```

to:

```python
compress_state()
```

throughout the repository.

That risks breaking the existing TurboQuant/QJL/RocketKV implementations.

Instead:

```text
Existing API
compress_kv(K, V)
        │
        ▼
Existing compressors
```

remains untouched.

Then add:

```text
State-aware dispatch
        │
        ├── AttentionState
        │       ↓
        │   compress_kv()
        │
        └── RecurrentState
                ↓
             passthrough
```

Thus:

```python
compress_state(state):
    if state.type == ATTENTION_KV:
        return compressor.compress_kv(...)
    elif state.type == RECURRENT:
        return state
```

Conceptually.

This gives Falcon-H1 support **without requiring every existing compressor to understand Mamba**.

---

# 29. Evaluation framework architecture I recommend

The evaluation framework should have a common execution protocol:

```text
                    Model
                      │
                      ▼
              Compatibility Probe
                      │
          ┌───────────┼────────────┐
          │           │            │
       Loader      Attention      State
                     adapter      adapter
          │           │            │
          └───────────┼────────────┘
                      ▼
                 FIDELITY
                      │
          ┌───────────┼─────────────┐
          │           │             │
      Representation Attention    Memory
          │           │             │
          └───────────┼─────────────┘
                      ▼
                 BEHAVIOR
                      │
                      ▼
                 SYSTEM
```

The important thing is that **Fidelity, Behavior, and System should not each independently rediscover how a model works.**

They should consume the same compatibility abstraction.

---

# 30. Add a model compatibility manifest

I strongly recommend adding something like:

```yaml
model: gemma3_270m

architecture:
  family: mqa
  q_heads: 4
  kv_heads: 1

attention:
  adapter: gemma3_text
  qk_norm: ...
  rope: per_layer_type

cache:
  type: standard_kv
  sliding_layers: true

evaluation:
  fidelity_attention: supported
  behavior_identity: supported
  behavior_turboquant: supported
  behavior_qjl: supported
  behavior_rocketkv: supported
  system: supported
```

For Falcon:

```yaml
model: falcon_h1_0.5b

architecture:
  family: hybrid

attention:
  adapter: falcon_h1
  q_heads: 8
  kv_heads: 2
  qk_norm: none

state:
  attention: compressible
  recurrent: passthrough

evaluation:
  total_state_accounting: required
```

This makes compatibility **declarative rather than scattered through the evaluator.**

---

# 31. The three compatibility gates should become explicit

Your current framework already effectively has three gates. 

I would formalize them:

## Gate A — Loader/state compatibility

```text
Can model load?
Can it forward?
Can state be discovered?
```

Required for:

* representation
* memory
* identity
* TurboQuant

---

## Gate B — Attention compatibility

```text
Can evaluator reproduce model attention correctly?
```

Required for:

* FIDELITY/attention
* QJL
* RocketKV

This is why Gemma3 currently fails even though its cache can be read.

---

## Gate C — State semantics compatibility

```text
Does the framework account for every inference state component?
```

This becomes essential for Falcon-H1.

A model should not be marked "supported" merely because:

```python
iter_layer_kv()
```

doesn't crash.

---

# 32. Final model-by-model implementation plan

Here is the practical implementation order I would use.

| Phase | Model        | Engine                                                 | Evaluation                              | Difficulty |
| ----- | ------------ | ------------------------------------------------------ | --------------------------------------- | ---------- |
| **0** | OLMo2        | No change                                              | Add regression/conformance test         | Very low   |
| **0** | Qwen3-0.6B   | No change                                              | Add regression/conformance test         | Very low   |
| **1** | TinyDeepSeek | Register `deepseek_v3` adapter                         | Add MLA metadata + tests                | Low        |
| **2** | Gemma3       | Register adapter + layer-aware RoPE                    | Layer-aware attention evaluation        | Medium     |
| **3** | Falcon-H1    | Register adapter + no-QK-norm + dual-state abstraction | Hybrid state/memory/behavior evaluation | High       |

This ordering is slightly different from simply implementing models in shortlist order because **TinyDeepSeek gives the largest architectural gain per engineering effort**.

---

# 33. The backward-compatibility strategy

This is the most important part of the whole plan.

### Rule 1 — Existing adapters remain untouched

```text
OLMo2Adapter → unchanged
Qwen3Adapter → unchanged
```

### Rule 2 — Existing compressor API remains unchanged

```python
compress_kv(K, V)
```

continues to work.

### Rule 3 — Existing KV iterator remains valid

```python
iter_layer_kv()
```

continues to work for conventional KV models.

### Rule 4 — New state API is additive

Add:

```python
iter_layer_states()
```

rather than replacing:

```python
iter_layer_kv()
```

### Rule 5 — Global RoPE remains the default

```python
get_rope(layer_idx)
```

can internally return the same global RoPE for OLMo2/Qwen3.

Gemma3 overrides the behavior.

### Rule 6 — Compressors remain unaware of model architectures

They receive the representation they already understand.

### Rule 7 — Unsupported state types fail explicitly

Never silently ignore them.

For example:

```text
Unsupported inference state:
Mamba recurrent state

Attention KV compression will not be evaluated
until state policy is explicitly declared.
```

That is much safer than Falcon's current silent undercount.

---

# 34. The final supported-state model

The engine should conceptually evolve from:

```text
Layer
 └── K,V
```

to:

```text
Layer
 └── InferenceState
       │
       ├── AttentionKV
       │
       ├── LatentKV
       │
       └── RecurrentState
```

But this does **not** mean you rewrite everything.

The compatibility hierarchy should be:

```text
InferenceState
      │
      ├── ConventionalKVState
      │      └── existing KVBench path
      │
      ├── MLAState
      │      └── MLA adapter
      │
      └── HybridState
             ├── AttentionKVState
             └── RecurrentState
```

The existing OLMo2/Qwen3 implementation effectively becomes the first subclass.

---

# 35. Final acceptance matrix

Before declaring the five-model benchmark ready, I would require this exact progression:

| Model        | Load | Forward | Cache/state | Attention | Identity | TurboQuant | QJL | RocketKV | Correct memory |
| ------------ | ---: | ------: | ----------: | --------: | -------: | ---------: | --: | -------: | -------------: |
| OLMo2        |    ✅ |       ✅ |           ✅ |         ✅ |        ✅ |          ✅ |   ✅ |        ✅ |              ✅ |
| Qwen3-0.6B   |    ✅ |       ✅ |           ✅ |         ✅ |        ✅ |          ✅ |   ✅ |        ✅ |              ✅ |
| Gemma3       |    ✅ |       ✅ |           ✅ |         ✅ |        ✅ |          ✅ |   ✅ |        ✅ |              ✅ |
| TinyDeepSeek |    ✅ |       ✅ |          ✅* |         ❌ |        ❌ |          ❌ |   ❌ |        ❌ |             ✅* |
| Falcon-H1    |    ✅ |       ✅ |           ✅ |         ❌ |        ❌ |          ❌ |   ❌ |        ❌ |              ✅ |

`*` **TinyDeepSeek — cache/state and memory:** Gate A passes (`iter_layer_kv` succeeds). Gate C fails because the visible cache is HF's expanded per-head K/V, not the native `kv_lora_rank` latent. "Correct memory" means **correct accounting of the exposed/expanded cache** until MLA-native interception lands.

`*` **TinyDeepSeek — eval branches blocked at Gate B:** FIDELITY/attention, identity, TurboQuant, QJL, and RocketKV all require `load_attention_ops` for `deepseek_v3`, which is not yet registered.

For Falcon-H1, "correct memory" means:

[
M_{\text{total}}
================

M_{\text{attention}}
+
M_{\text{Mamba}}
]

not merely the attention KV component.

---

# 36. What I would actually implement

If the goal is to get from the **current repository to a scientifically defensible five-model KVBench**, I would make the work packages:

### WP1 — Compatibility infrastructure

Add:

```text
ModelCapabilities
StateType
StateAdapter
layer-aware attention metadata
compatibility/conformance tests
```

**No behavior change for existing models.**

---

### WP2 — TinyDeepSeek

Add:

```text
deepseek_v3 adapter
MLA attention handling
MLA metadata
expanded-KV disclosure
```

Then run the complete:

```text
FIDELITY → BEHAVIOR → SYSTEM
```

pipeline.

---

### WP3 — Gemma3

Add:

```text
gemma3_text adapter
per-layer-type RoPE
layer attention metadata
```

Then validate identity attention before running compression experiments.

---

### WP4 — Falcon-H1

Add:

```text
falcon_h1 adapter
qk_norm = none
InferenceState abstraction
dual-state layer representation
attention-state compression
Mamba-state passthrough
total-state memory accounting
```

Then validate that the recurrent state is unchanged through compressed decoding.

---

### WP5 — Regression validation

Run the existing configurations again:

```text
OLMo2
Qwen3-1.7B
Qwen3-0.6B
```

and confirm that the new architecture support did **not** change their:

* perplexity
* attention cosine
* memory
* throughput
* compression behavior

The original KVBench design deliberately fixes the model, decode loop, and metrics while changing only the compressor, so preserving that controlled execution protocol is important. 

---

# Bottom line

The correct engineering philosophy is:

> **Do not redesign KVBench to accommodate the five models. Extend KVBench's existing abstractions just enough to correctly represent the inference state of each architecture.**

That gives you:

```text
                    KVBench

                       │
          ┌────────────┼─────────────┐
          │            │             │
     Conventional    Latent        Hybrid
        KV            KV            State
          │            │             │
       ┌──┼──┐         │             │
       │  │  │         │             │
      MHA GQA MQA      MLA       Attention+Mamba
       │  │  │         │             │
     OLMo Qwen Gemma  DeepSeek     Falcon
```

with the **existing OLMo2/Qwen3 path preserved**, **TinyDeepSeek added through a small adapter**, **Gemma3 through adapter + layer-aware RoPE**, and **Falcon-H1 through a bounded dual-state extension**.

Most importantly, the evaluation framework should never equate **"the code ran"** with **"the measurement is valid."** Falcon-H1 is the clearest example: its current cache reader can run while silently ignoring the Mamba state. Your new compatibility layer should therefore establish **semantic correctness and memory-accounting correctness**, not merely eliminate exceptions.

That is what makes the five-model matrix defensible as a benchmark rather than just a collection of models.
