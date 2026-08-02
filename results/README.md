# Results

Published **summary** evaluation bundles for KVBench Phase-5 sweeps live here.

| Path | Contents |
|---|---|
| `phase5_modal_baseline/` | Qwen3 identity baseline (merged CSV/JSON + `manifest.json`) |
| `phase5_modal_sweep_128_256_512/` | Qwen3 TurboQuant grid |
| `phase5_modal_qjl/` | Qwen3 QJL |
| `phase5_modal_rocketkv/` | Qwen3 RocketKV |
| `olmo2_phase5_*` | OLMo 2 1B counterpart bundles + `olmo2_phase5_summary.json` / inventory CSV |

Human-readable tables: [`docs/PHASE5_EVAL_RESULTS.md`](../docs/PHASE5_EVAL_RESULTS.md), [`docs/OLMO2_RESULTS_COMPLETE.md`](../docs/OLMO2_RESULTS_COMPLETE.md), [`docs/RESULTS_COMPLETE.md`](../docs/RESULTS_COMPLETE.md).

## Not tracked in git

- `modal_volume/` / `modal_volume_olmo2/` — raw Modal downloads (large)  
- `**/jobs/` — per-job JSON copies inside bundles (regenerate via fetch + restructure scripts)  
- Model weights under `models/`  

Re-fetch from Modal:

```bash
bash scripts/modal_fetch_results.sh
python scripts/restructure_modal_results.py          # Qwen3 volume
python scripts/restructure_olmo2_modal_results.py    # OLMo2 volume
```
