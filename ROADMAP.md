# Roadmap — KV Cache Interception and Transformation Engine

Research manuscript (**KVBench**) in preparation. Since 2026-11-24 the study follows the adopted research framing: **do KV-cache compression results transfer across attention architectures, and what does a compression ratio buy once the full inference state and offline cost are counted?** ([docs/RESEARCH_REFRAME_AND_PRIOR_ART.md](docs/RESEARCH_REFRAME_AND_PRIOR_ART.md)).

This page is a public summary. Detailed work packages and their status (R0–R16, E1–E8) live in [docs/RESEARCH_REDESIGN_PLAN.md → Part 0](docs/RESEARCH_REDESIGN_PLAN.md#part-0--reframe-adoption-and-master-status-2026-11-24).

## Now: re-validation

- [x] Fix the evaluation bugs found in the Oct-2026 audit (committed #201–#204; [audit](docs/audits/GEMMA3_POST_OVERHAUL_AUDIT.md))
- [ ] Verify the fixes on GPU (Modal `audit_verify` preset)
- [ ] Freeze hypotheses H1–H6 before any grid run

## Next: the cross-architecture study

- [ ] Longer contexts (2K–4K) across all shortlist models
- [ ] Gated-DeltaNet hybrid model (e.g. Qwen3.5 sub-2B) alongside Falcon-H1 (Mamba-2 hybrid)
- [ ] Native MLA latent interception (TinyDeepSeek Gate C)
- [ ] Multiple compression budgets × seeds per compressor
- [ ] Rank statistics (Spearman / Kendall τ, bootstrap CIs, held-out prediction)
- [ ] Claimed-vs-realized savings table (nominal KV ratio vs total inference state)
- [ ] Calibration-cost break-even analysis
- [ ] RULER / needle-in-a-haystack-style long-context retrieval task
- [ ] Pilot (3 architectures × 4 compressors × 3 seeds at 2K), then the full clean-commit grid

## Engine and compressors

- [x] SnapKV plug-in
- [x] Compression taxonomy (Phase 4, `compressors/taxonomy.py`)
- [x] Palu plug-in (G-LRD latent cache)
- [x] Per-architecture adapters: OLMo2 (MHA), Qwen3 (GQA), Gemma3 (MQA + sliding/global), DeepSeek-V3/TinyDeepSeek (MLA, expanded KV), Falcon-H1 (Mamba-2 hybrid)
- [ ] KIVI plug-in (replace current stub)
- [ ] AdaKV plug-in
- [ ] CI smoke tests (identity @ ctx=128 on CPU/MPS where feasible)

## Systems / infrastructure

- [ ] Multi-provider support (local CUDA, Modal, and optional additional cloud runners)
- [ ] Clearer dynamic model/volume parameterization for multi-model sweeps
- [ ] Optional fused-attention paths only where KV tensors remain observable
- [ ] Packaged install (`pip install` / pyproject extras for modal)

## Documentation & community

- [ ] Paper rewrite to the adopted framing (after the clean grid)
- [ ] Public summary tables regenerated from the clean-commit bundles
- [ ] Contributor tutorials for new compressors and SLM adapters
- [ ] Published arXiv manuscript + updated `CITATION.cff`
- [ ] Example notebooks for inspecting FIDELITY/BEHAVIOR/SYSTEM artifacts

## Explicitly out of scope (for now)

- Claiming novelty for TurboQuant / QJL / RocketKV / SnapKV / Palu algorithms themselves
- Exhaustive 7B–70B factorial sweeps (the study is deliberately small-model, where all five attention classes exist as open checkpoints)
- Serving-engine integration (vLLM / SGLang)
- **Phase 5:** heterogeneous/adaptive per-layer/head/token plugin API
- A broad workload matrix (summarization, code, agents); only a long-context retrieval task is planned
