# Literature alignment — Phase 29

Maps **recent KV-cache papers** to KVBench **engine capabilities** and **paper rewrite targets**. These papers **shape methodology and positioning**; most are **not implemented** as compressors.

**Staging BibTeX** (merge into `reference.bib` at paper rewrite): [`staging_entries.bib`](staging_entries.bib)  
**Audit CLI:** `python scripts/audit_bibliography.py`  
**Roadmap spec:** `docs/RESEARCH_REDESIGN_PLAN.md` Phases 29–33

## Priority literature map

| # | Paper | Suggested key | Engine touchpoint | Paper section | Implement? |
| - | ----- | ------------- | ----------------- | ------------- | ---------- |
| 1 | Oaken — ISCA 2025 | `oaken2025` | `eval/cost/oaken_taxonomy.py` (Phase 26) | Related Work §4; COST subsection | No — cite taxonomy |
| 2 | SCOPE — ACL 2025 | `scope2025` | SYSTEM TTFT/ITL vs BEHAVIOR PPL split | Related Work §4; future prefill/decode note | No |
| 3 | RocketKV — ICML 2025 | `rocketkv` | `compressors/rocketkv.py`, case study | Methods + Results | ✅ case study |
| 4 | TurboAttention — MLSys 2025 | `turboattention2025` | SYSTEM `kernel_cost.py` (attention path) | Related Work §3–§4 | No |
| 5 | R-KV — NeurIPS 2025 | `rkv2025` | BEHAVIOR `reasoning.py` (opt-in) | Related Work §4; Phase 11 future | No |
| 6 | Pitfalls — ACL 2026 | `chen2026pitfalls` | BEHAVIOR default stack | Related Work §4; F1/F4 | No — cited |
| 7 | OjaKV — ACL 2026 | `ojakv2026` | `cost.benchmark_dimensions.stateful` (Phase 27) | Related Work §3; calibration table | No |
| 8 | HybridKV — ACL 2026 | `hybridkv2026` | Taxonomy D+E; Phase 5 deferred | Related Work §3 | No |
| 9 | Benchmarking KV-Cache Optimizations — 2026 | `kvbench2026serving` | Phase 30 contrast only | Intro + Related Work §4 | No — **must differentiate** |
| 10 | KVCache Cache in the Wild — USENIX ATC 2025 | `kvcachewild2025` | Phase 28 workload scope | Related Work §4; Discussion F6 | No |
| 11 | CacheBlend — EuroSys 2025 | `cacheblend2025` | Phase 28 RAG/serving future | Related Work §4 | No |

## Narrative roles (do not conflate)

| Role | Papers | KVBench response |
| ---- | ------ | ---------------- |
| **Behavioral evaluation gap** | Pitfalls, SCOPE | Three-branch BEHAVIOR; F1/F4 findings |
| **Serving / system benchmarks** | `kvbench2026serving`, Oaken, TurboAttention | Phase 30: controlled interception vs serving stack |
| **Workload realism** | Cache in the Wild, CacheBlend, R-KV, Short-RL | Phase 28 scope sentence; F6 unanswered |
| **Cost / calibration** | Oaken, TurboQuant Lloyd-Max | Phase 26 Oaken layers + Phase 27 table |
| **Adaptive / stateful compression** | OjaKV, HybridKV, RocketKV | Phase 27 `stateful` column; RocketKV case study |

## Closest competing benchmark (Phase 30)

**Their focus:** evaluate existing KV optimizations across **workloads** and **system-level metrics** in long-context **serving**.

**KVBench focus:** controlled **KV interception/transformation layer** — same incremental autoregressive decode path, matched FIDELITY / BEHAVIOR / SYSTEM analysis, plug-in factorial on SLMs.

Canonical contrast sentence: see Phase 30 in `RESEARCH_REDESIGN_PLAN.md`.

## Conceptual model (Phase 32)

Canonical end-to-end diagram: [`SYSTEM_DESIGN.md` §Phase 32 conceptual model](../architecture/SYSTEM_DESIGN.md#phase-32-conceptual-model-end-to-end-story).

## One-sentence identity (Phase 33)

> KVBench is not primarily a benchmark asking which KV-cache compression method wins; it is a controlled inference-time experimentation framework for understanding how different KV transformations trade representation fidelity, model behavior, memory efficiency, and actual generation performance under matched conditions.
