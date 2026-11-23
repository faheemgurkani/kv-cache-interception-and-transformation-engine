# Results

> ⚠️ **Historical (2026-11-24).** All Phase-5 bundles listed here (Qwen3-1.7B, OLMo2-1B) and the Gemma3 `gemma3_full_eval_*` bundles were invalidated by the Oct-2026 audit ([`docs/audits/GEMMA3_POST_OVERHAUL_AUDIT.md`](../docs/audits/GEMMA3_POST_OVERHAUL_AUDIT.md)). Do not cite them. New results will come from the clean-commit grid (R13 in [`docs/RESEARCH_REDESIGN_PLAN.md` → Part 0](../docs/RESEARCH_REDESIGN_PLAN.md#part-0--reframe-adoption-and-master-status-2026-11-24)).

Published **summary** evaluation bundles for the KV Cache Interception and Transformation Engine (KVBench Phase-5 sweeps) live here.

| Path | Contents |
|---|---|
| `phase5_modal_baseline/` | Qwen3 identity baseline (merged CSV/JSON + `manifest.json`) |
| `phase5_modal_sweep_128_256_512/` | Qwen3 TurboQuant grid |
| `phase5_modal_qjl/` | Qwen3 QJL (ProdQJL Aug 2026; invalid — audit B3/B8) |
| `phase5_modal_qjl_prodqjl/` | Qwen3 QJL ProdQJL re-run bundle (`jobs/`, `logs/`, manifest) |
| `phase5_modal_rocketkv/` | Qwen3 RocketKV |
| `olmo2_phase5_qjl/` | OLMo 2 QJL (ProdQJL Aug 2026; invalid — audit B3/B8) |
| `olmo2_phase5_qjl_prodqjl/` | OLMo 2 QJL ProdQJL re-run bundle |
| `olmo2_phase5_*` | Other OLMo 2 1B bundles + `olmo2_phase5_summary.json` / inventory CSV |

Human-readable tables: [`docs/results/qwen3_1.7b/PHASE5_EVAL_RESULTS.md`](../docs/results/qwen3_1.7b/PHASE5_EVAL_RESULTS.md), [`docs/results/olmo2_1b/RESULTS_COMPLETE.md`](../docs/results/olmo2_1b/RESULTS_COMPLETE.md), [`docs/results/qwen3_1.7b/RESULTS_COMPLETE.md`](../docs/results/qwen3_1.7b/RESULTS_COMPLETE.md).

## Not tracked in git

- `modal_volume/` / `modal_volume_olmo2/` — raw Modal downloads (large)  
- `modal_volume_qjl_prodqjl_qwen3/` / `modal_volume_qjl_prodqjl_olmo2/` — ProdQJL re-run raw fetches  
- `qjl_prodqjl_rerun.log` — local orchestration log for the 6-job re-sweep  
- `**/jobs/` — per-job JSON copies inside bundles (regenerate via fetch + restructure scripts)  
- Model weights under `models/`  

Re-fetch from Modal:

```bash
bash scripts/modal_fetch_results.sh
python scripts/restructure_modal_results.py          # Qwen3 volume
python scripts/restructure_olmo2_modal_results.py    # OLMo2 volume
```
