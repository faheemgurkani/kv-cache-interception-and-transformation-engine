# Shared identity baseline (Modal Phase 5)

No-compression reference runs for comparative analysis across all KV-cache methods.

| Context | Label | Compressor |
| --- | --- | --- |
| 128 | `identity_baseline` | `identity` |
| 256 | `identity_baseline` | `identity` |
| 512 | `identity_baseline` | `identity` |

**Source:** extracted from the original Modal app `ap-ek9dIxujlrECcfFaOa3ok3` (same eval pipeline as TurboQuant / RocketKV sweeps).

**Reuse:** cite these rows as the baseline when comparing TurboQuant, QJL, RocketKV, or KIVI — do not re-run identity per method.

**Files:**
- `jobs/` — per-job Modal JSON payloads
- `phase5_modal_baseline_*.csv` / `.json` — merged tables for the paper
