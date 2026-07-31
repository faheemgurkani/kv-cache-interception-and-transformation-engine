# OLMo 2 1B — architecture probe (KVBench)

**Model:** `allenai/OLMo-2-0425-1B` → `models/olmo2_1b`  
**Status:** Wired for Phase-5 Modal sweeps (Identity / TurboQuant / QJL / RocketKV × ctx 128/256/512).

## Architecture vs Qwen3 reference

| Property | Qwen3-1.7B (ref) | OLMo 2 1B |
|---|---|---|
| HF class | `Qwen3ForCausalLM` | `Olmo2ForCausalLM` |
| Layers | 28 | 16 |
| Attention | GQA 16Q / 8KV | **MHA 16Q / 16KV** |
| `head_dim` | 128 (on config) | 128 (module; **not** on config JSON) |
| QK-norm | Yes (`q_norm`/`k_norm` **per-head** after view) | Yes (`q_norm`/`k_norm` **flat** on full projection before view) |
| Norm placement | Pre-norm (`input_layernorm`) | **Post-norm** (no `input_layernorm`) |
| RoPE θ | — | 500000 |
| Max position | — | 4096 |
| Sliding window | Optional on Qwen3 attn | None |
| Eager attn API | `position_embeddings` + `past_key_values.update` | Same signature |

## Algorithm coupling

| Algorithm | Offline (A) | Online (B) | Notes |
|---|---|---|---|
| Identity | ✓ | ✓ | Stock eager path |
| TurboQuant | ✓ | ✓ | `d=128` pow2; decompress → stock attn |
| QJL | ✓ | ✓ | Adapter: flat QK-norm + `modeling_olmo2` RoPE |
| RocketKV | ✓ | ✓ | Same adapter; MHA (`n_rep=1`); no sliding-window kwargs |
| KIVI | Stub | N/A | Not implemented |

## Code changes for this model

- `framework/model_adapter.py` — family registry (`qwen3` / `olmo2`)
- `framework/qjl_online.py`, `framework/rocketkv_online.py` — use adapter (also fixes per-layer closure bind)
- `eval/attention_score_error.py` — post-norm + derived `head_dim`
- `configs/model.yaml` → OLMo2; Qwen3 backup in `configs/model_qwen3.yaml`
- `configs/modal.yaml` → volumes `kv-engine-olmo2` / `kv-engine-results-olmo2`
- `modal_app/worker.py` — dynamic `_model_dir()` from `local_path` basename

## Modal bring-up

```bash
bash scripts/modal_setup_model.sh
bash scripts/modal_run_sweep_baseline.sh
PRESET=turboquant OUTPUT=olmo2_phase5_turboquant bash scripts/modal_run_sweep.sh
bash scripts/modal_run_sweep_qjl.sh          # set OUTPUT if desired
bash scripts/modal_run_sweep_rocketkv.sh
bash scripts/modal_fetch_results.sh
```
