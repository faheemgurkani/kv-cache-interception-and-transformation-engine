# Roadmap — KV Cache Interception and Transformation Engine

Research manuscript (**KVBench**) in preparation. This roadmap lists planned extensions to the engine and evaluation grid; ordering may change as the study evolves.

## Near term

- [ ] Additional SLMs (e.g. Granite dense ~350M, MiniCPM4-0.5B, Gemma 3 270M where licenses allow)
- [ ] Per-architecture attention adapters for non–QK-norm models
- [ ] KIVI plug-in (replace current stub)
- [x] SnapKV plug-in
- [x] Compression taxonomy (Phase 4, `compressors/taxonomy.py`)
- [x] Palu plug-in (G-LRD latent cache)
- [ ] AdaKV plug-in
- [ ] Public summary leaderboard tables regenerated from `results/` bundles
- [ ] CI smoke tests (identity @ ctx=128 on CPU/MPS where feasible)

## Evaluation expansion

- [ ] LongBench
- [ ] RULER
- [ ] Needle-in-a-haystack / retrieval-style probes
- [ ] Longer contexts (2K–8K) within SLM VRAM budgets
- [ ] Additional generation-quality online metrics beyond WikiText-2 PPL

## Systems / infrastructure

- [ ] Multi-provider support (local CUDA, Modal, and optional additional cloud runners)
- [ ] Clearer dynamic model/volume parameterization for multi-model sweeps
- [ ] Optional fused-attention paths only where KV tensors remain observable
- [ ] Packaged install (`pip install` / pyproject extras for modal)

## Documentation & community

- [ ] Contributor tutorials for new compressors and SLM adapters
- [ ] Published arXiv manuscript + updated `CITATION.cff`
- [ ] Phase 6 paper rewrite: controlled interception narrative + FIDELITY/BEHAVIOR/SYSTEM (code/docs done; `docs/research_paper_writeup/` deferred)
- [ ] Example notebooks for inspecting FIDELITY/BEHAVIOR/SYSTEM artifacts

## Explicitly out of scope (for now)

- Claiming novelty for TurboQuant / QJL / RocketKV algorithms themselves  
- Exhaustive 7B–70B factorial sweeps as the primary paper claim  
- Hybrid / linear-attention models that are not standard KV-cache causal LMs (e.g. Qwen3.5 hybrid) without a redesign of the interception contract  
- **Phase 5:** heterogeneous/adaptive per-layer/head/token plugin API  
- **Phase 8:** unified multi-budget compression–quality curve sweeps (existing per-method grids are sufficient)
