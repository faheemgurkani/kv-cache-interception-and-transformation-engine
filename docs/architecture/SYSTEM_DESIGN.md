# System Design — KV Cache Interception and Transformation Engine

**Purpose:** compression task analysis and benchmarking. Fixed model + eval stack; **only `compressors/` changes per method.**

This is a **generic evaluation framework**, not a TurboQuant-only implementation. The engine provides unified KV interception, plug-in compressors, and three evaluation branches — **FIDELITY** (did the transformation preserve the KV representation and attention behavior?), **BEHAVIOR** (does the model still behave correctly after KV transformation?), and **SYSTEM** (does the compression actually make inference better?). Published KV-compression papers propose new algorithms; this repo standardizes how those algorithms are measured under online inference.

## Architecture

```text
Tokenizer → Model Forward → KV Cache → KVCacheEngine → KVCompressor → Attention → Next tokens
                                                                    │
                                              TurboQuant | QJL | RocketKV | identity
```

| Layer | Directory | Swappable? |
|---|---|---|
| Model | `framework/model.py` | No |
| KV interception | `framework/kv_engine.py`, `framework/kv_cache.py` | No |
| Compression | `compressors/`, `quantizers/` | **Yes** |
| Evaluation | `eval/`, `reporting/` | No |

## Model

**Qwen3-1.7B** — GQA (8 KV heads, head dim 128), `use_cache=True`, HuggingFace native.

```python
AutoModelForCausalLM.from_pretrained(..., dtype=torch.float16, attn_implementation="eager")
```

**Eager attention is required.** FlashAttention / fused SDPA hide or fuse `past_key_values`; compression hooks need explicit per-layer K/V tensors.

## KVCacheEngine

`framework/kv_engine.py` intercepts between steps:

1. Decompress stored payloads → `DynamicCache`
2. Model forward with `past_key_values`
3. Compress **only new token positions** (incremental — re-compressing full cache caused NaN PPL)

Model always sees decompressed FP16 K/V. Swap compressors without touching the engine or eval code.

## KVCompressor interface

`compressors/base.py`:

```python
class KVCompressor(ABC):
    def compress_kv(self, tensor, layer=0, mode="key") -> object: ...
    def decompress_kv(self, payload, mode="key") -> Tensor: ...
    def compress(self, key, value, layer=0) -> CompressedKV: ...
    def decompress(self, compressed) -> tuple[Tensor, Tensor]: ...
```

## Compressors

### TurboQuant

`quantizers/turboquant_pipeline.py` → `compressors/turboquant.py`

Pipeline: pad → WHT → Lloyd-Max → residual → optional QJL on values → store.

Stages: `wht_only`, `wht_quant`, `wht_quant_residual`, `full`. Shared Lloyd-Max centroids via `shared_storage_bytes()`.

### QJL

`quantizers/qjl_pipeline.py` → `compressors/qjl.py`

Keys: `sign(S @ k) + ||k||` (Gaussian S, fixed seed; strict ±1 signs). Values: FP16 passthrough.

FIDELITY: `attention_fidelity()` / `estimate_attention_scores()`. BEHAVIOR: online **estimator** via `framework/qjl_online.py`.

### RocketKV

`quantizers/rocketkv.py` → `compressors/rocketkv.py`

Stage 1: permanent token filter (SnapKV-style). Stage 2: HSA dynamic top-k.

Online: `framework/rocketkv_online.py` patches Qwen3 eager attention. Baseline metrics run **before** engine construction.

### KIVI

Stub in `compressors/kivi.py` — not implemented.

## Evaluation

Three branches instead of an offline/online split — `eval/runner.py` (`EvaluationRunner.run()`) always runs FIDELITY; BEHAVIOR and SYSTEM sub-metrics are opt-in flags.

**FIDELITY** (`eval/fidelity/`, single offline forward pass): tensor RMSE/relative-error/cosine similarity (`representation.py`), QK^T score MSE/RMSE/cosine/max-error + attention-output RMSE + attention-distribution KL divergence (`attention.py`), compression ratio + actual bytes + metadata overhead (`memory.py`), hybrid recurrent exact-preservation (`recurrent.py`).

**BEHAVIOR** (`eval/behavior/`, through `KVCacheEngine`): sliding-window perplexity (`task_quality.py`, on by default), needle-in-haystack retrieval (`retrieval.py`, on by default), instruction-following format compliance (`instruction_following.py`, on by default), synthetic arithmetic reasoning (`reasoning.py`, opt-in).

**SYSTEM** (`eval/system/`, through `KVCacheEngine`): TTFT + inter-token latency (mean/p50/p99) + tokens/sec + end-to-end latency (`latency_throughput.py`, on by default), peak CUDA memory (`vram.py`, opt-in), analytical KV-bandwidth (`memory_bandwidth.py`, opt-in), compress/decompress vs. rest-of-forward-pass time (`kernel_cost.py`, opt-in), best-effort NVML sampling (`gpu_utilization.py`, opt-in).

Orchestrator: `eval/runner.py`. WikiText-2 samples concatenated to target length via `data/loader.py`.

Results: [Qwen3-1.7B](../results/qwen3_1.7b/PHASE5_EVAL_RESULTS.md) · [OLMo2-1B](../results/olmo2_1b/PHASE5_EVAL_RESULTS.md) · [5-model shortlist](../results/shortlist_5model_eval/) · Reproducibility: [REPRODUCIBILITY.md](../reproducibility/REPRODUCIBILITY.md). Known limits: [CURRENT_STATE.md](../methodology/CURRENT_STATE.md).

Methodology: [METHODOLOGY.md](../methodology/METHODOLOGY.md) · Equations: [MATHEMATICS_AND_ALGORITHMS.md](../methodology/MATHEMATICS_AND_ALGORITHMS.md).

## Runtimes

| | Local | Modal |
|---|---|---|
| Device | MPS / CPU | CUDA (`KV_EVAL_DEVICE=cuda`) |
| Use | pytest, smoke | Full sweeps |
| Entry | `scripts/run_eval.py` | `modal_app/sweep.py` |

Modal details: [REPRODUCIBILITY.md §11](../reproducibility/REPRODUCIBILITY.md)

## Non-goals

CoreML, Ollama/GGUF (no KV access), MLX, FlashAttention, multi-GPU layer split.

## File map

```text
framework/     model, kv_engine, kv_cache, device, rocketkv_online
compressors/   identity, turboquant, qjl, rocketkv, kivi (stub)
quantizers/    hadamard, lloyd_max, turboquant_pipeline, qjl_pipeline, rocketkv
eval/          fidelity/, behavior/, system/, runner
modal_app/     worker, sweep, job_spec, merge
configs/       model.yaml, eval.yaml, modal.yaml, modal_sweeps.yaml
```
