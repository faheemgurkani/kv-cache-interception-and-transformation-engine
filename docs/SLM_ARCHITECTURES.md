# SLM Architectures vs. Engine Support

Brief summary of what architecture "family" each of the 6 local models actually is, and what class of architecture the KV-cache interception engine can support. Full metric-by-metric detail: [SLM_COMPATIBILITY.md](SLM_COMPATIBILITY.md).

## The 6 models, by architecture type

| Model | Architecture type | Detail |
|---|---|---|
| **Qwen3-1.7B** | Plain dense transformer, GQA | 16Q/8KV heads, per-head Q/K-norm, single shared RoPE |
| **OLMo 2 1B** | Plain dense transformer, MHA | 16Q/16KV heads (no GQA), flat Q/K-norm, post-norm, single shared RoPE |
| **Granite 4.0 350M** | Plain dense transformer, GQA (this checkpoint only) | 16Q/4KV heads, **no** Q/K-norm, single shared RoPE — packaged as a "MoE-Hybrid" class but this checkpoint has 0 active MoE/Mamba layers |
| **Gemma3-270M** | Plain dense transformer, GQA — but **dual RoPE** | 4Q/1KV heads, per-head Q/K-norm, alternates local (`sliding_attention`) / global (`full_attention`) layers, each with its own RoPE table |
| **MiniCPM4-0.5B** | Plain dense transformer, GQA (unconfirmed — custom code) | 16Q/2KV heads, `longrope` RoPE scaling, ships its own Python modeling code (`trust_remote_code`) instead of using a transformers-native class |
| **Qwen3.5-0.8B** | **Hybrid** linear-attention + full-attention | 24 layers: 18 recurrent linear-attention layers (no K/V cache at all) + 6 standard attention layers |

Five of six are, at core, ordinary dense decoder-only transformers (differing only in GQA ratio, norm placement, and RoPE details). One (Qwen3.5) is architecturally a different class entirely — a hybrid model that mixes recurrent linear-attention state with conventional K/V attention.

## What the engine can support

**By design, today:** dense, decoder-only transformers where **every layer** does standard scaled-dot-product attention over an explicit, growing K/V cache — i.e. `attn_implementation="eager"` must expose real per-layer `(key, value)` tensors for `past_key_values`, one pair per layer, no exceptions. Two further requirements narrow it inside that class:

- **Wired today** (`framework/model_adapter.py`, hardcoded to `{"qwen3", "qwen2", "olmo2"}`): a **single, model-wide RoPE table** reused across all layers, and one of two known Q/K-norm layouts (`"per_head"` or `"flat"`). This gate is required for FIDELITY's attention metrics and for QJL/RocketKV's online attention patch — not for FIDELITY's representation/memory metrics or for BEHAVIOR/SYSTEM under `identity`/`turboquant`, which only need the broader "dense transformer with a standard K/V cache" property above.
- **Not wired, but addable without new engine design** (Granite, Gemma3): still dense transformers, just needing a new adapter branch (Granite: register `model_type` + a "no Q/K-norm" layout) or a moderate extension (Gemma3: per-layer RoPE selection instead of one shared table).
- **Not addable as an adapter — needs a decision, not code** (MiniCPM4): `trust_remote_code` is a trust boundary the maintainers must choose to cross, not a technical gap.

**Not supported, and not close** (Qwen3.5): any model where a layer's "cache" isn't a K/V tensor pair at all — recurrent/linear-attention state, Mamba/SSM state, or MoE routing with genuinely active experts. The engine's core assumption — intercept K/V between decode steps, run it through a compressor, decompress before the next forward — has no equivalent operation for a fixed-size recurrent state; extending to that class is a new architecture-support project, not a bug fix.

## One-line summary

The engine supports **dense, decoder-only, attention-only transformers with a uniform per-layer K/V cache**. Within that class, everything works if the model additionally has one shared RoPE table and a known Q/K-norm layout (Qwen3, OLMo2 today); dense transformers that break only those two assumptions (Granite, Gemma3) are reachable with contained adapter work; anything with recurrent/hybrid layers that don't produce a K/V cache at all (Qwen3.5's linear-attention layers) is outside the architecture class the engine was built for.
