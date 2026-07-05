# Known Limits

Online-path caveats and non-goals. Setup and commands: [README](../README.md). Architecture: [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md).

| Topic | Limit |
|---|---|
| **KIVI** | Stub only (`NotImplementedError`) |
| **QJL Section B** | Uses key reconstruct in forward pass; Section A uses attention estimator — metrics diverge |
| **RocketKV Section A vs B** | Offline fidelity ignores token eviction; Section B captures it |
| **TurboQuant 2-bit @ ctx=128** | Anomalously bad PPL; use ctx≥256 for comparisons |
| **TurboQuant online speed** | ~0.08 tok/s @ ctx=512 (per-step compress/decompress) |
| **Modal WHT** | Scipy fallback; no CUDA `fast-hadamard-transform` |
| **Attention** | `attn_implementation="eager"` required — FlashAttention breaks KV intercept |
| **Baseline eval order** | Baseline PPL runs before RocketKV attention patch (`eval/runner.py`) |

Raw job JSON lives under `results/` (gitignored). Version-controlled numbers: [PHASE5_EVAL_RESULTS.md](PHASE5_EVAL_RESULTS.md).
