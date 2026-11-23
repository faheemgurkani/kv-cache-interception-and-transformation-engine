# Documentation

Public documentation for the **KV Cache Interception and Transformation Engine** (Apache-2.0). The reproducible benchmarking protocol and case-study write-up are branded **KVBench** (research manuscript in preparation — see [conference_101719.tex](research_paper_writeup/conference_101719.tex)).

See also the root [README.md](../README.md), [ROADMAP.md](../ROADMAP.md), and [CONTRIBUTING.md](../CONTRIBUTING.md).

## Start here: framing and status

| Doc | Purpose |
|---|---|
| [RESEARCH_REFRAME_AND_PRIOR_ART.md](RESEARCH_REFRAME_AND_PRIOR_ART.md) | **Adopted research framing (2026-11-24)**: transfer across attention architectures + full-state accounting; contributions C1–C3, hypotheses H1–H6, prior-art survey, venue verdict |
| [RESEARCH_REDESIGN_PLAN.md → Part 0](RESEARCH_REDESIGN_PLAN.md#part-0--reframe-adoption-and-master-status-2026-11-24) | **Master status**: work packages R0–R16 (done / not done) and which Phase 1–33 items still apply |
| [ENGINE_AND_EVALUATION_FRAMEWORKS_REDESIGN_PLAN.md → Part 0](ENGINE_AND_EVALUATION_FRAMEWORKS_REDESIGN_PLAN.md#part-0--post-audit-status-and-reframe-engine-work-2026-11-24) | Engine work items E1–E8 and per-model support vs valid results |
| [audits/GEMMA3_POST_OVERHAUL_AUDIT.md](audits/GEMMA3_POST_OVERHAUL_AUDIT.md) | Oct-2026 evaluation audit: bugs, fixes, which results are invalid |
| [literature/FRAMING_VALIDATION_2026-11-24.md](literature/FRAMING_VALIDATION_2026-11-24.md) | Evidence behind the framing decision (2025–2026 literature, 46 PDFs in `literature/papers/`) |

**Results status:** no valid study results exist yet. The legacy Qwen3-1.7B / OLMo2-1B pages under `results/` are historical and marked invalid.

## architecture/ — what the engine is and what it supports

| Doc | Purpose |
|---|---|
| [SYSTEM_DESIGN.md](architecture/SYSTEM_DESIGN.md) | Interception engine architecture (high-level, shortest entry point) |
| [ENGINE_INTERNALS.md](architecture/ENGINE_INTERNALS.md) | Complete implementation walkthrough — every file, the full execution flow, and the tier-based engineering plan (§8) for supporting model architectures beyond dense transformers |
| [SLM_COMPATIBILITY.md](architecture/SLM_COMPATIBILITY.md) | Historical: the original 6-candidate probe (Qwen3-1.7B, OLMo2-1B, Granite, Gemma3, MiniCPM4, Qwen3.5) that preceded the current shortlist |
| [MODEL_ARCHITECTURE_MATRIX.md](architecture/MODEL_ARCHITECTURE_MATRIX.md) | **Current** model set: legacy models + 5-model architecture-matrix shortlist (MHA/GQA/MQA/MLA/Hybrid) + planned GDN hybrid; engine-support status per model |

Deep per-model probe (measured params/dtypes, live module/cache internals): [`models/ARCHITECTURE_REPORT.md`](../models/ARCHITECTURE_REPORT.md).

## methodology/ — how evaluation works

| Doc | Purpose |
|---|---|
| [METHODOLOGY.md](methodology/METHODOLOGY.md) | Experimental setup, per-compressor methodology, FIDELITY/BEHAVIOR/SYSTEM protocol — the authoritative version of this content |
| [MATHEMATICS_AND_ALGORITHMS.md](methodology/MATHEMATICS_AND_ALGORITHMS.md) | Equations and pseudocode for every compressor and metric |
| [CURRENT_STATE.md](methodology/CURRENT_STATE.md) | Known limits and research scope |

## reproducibility/ — running it yourself

| Doc | Purpose |
|---|---|
| [REPRODUCIBILITY.md](reproducibility/REPRODUCIBILITY.md) | Local + Modal reproduction steps, seeds, config source-of-truth, verification checklist, and (§11) the Modal GPU infra reference |

## results/ — measured numbers, by model (all historical; see banner)

> ⚠️ Every page below is either invalidated by the Oct-2026 audit (legacy Phase-5) or an engine-compatibility check (shortlist). No valid study results exist yet; they will come from the clean grid (R13).

| Doc | Purpose |
|---|---|
| [results/qwen3_1.7b/PHASE5_EVAL_RESULTS.md](results/qwen3_1.7b/PHASE5_EVAL_RESULTS.md) | Qwen3-1.7B Phase-5 summary tables (historical, invalid) |
| [results/qwen3_1.7b/RESULTS_COMPLETE.md](results/qwen3_1.7b/RESULTS_COMPLETE.md) | Qwen3-1.7B full metrics, per-layer stats, run logs (historical, invalid) |
| [results/olmo2_1b/PHASE5_EVAL_RESULTS.md](results/olmo2_1b/PHASE5_EVAL_RESULTS.md) | OLMo2-1B (`allenai/OLMo-2-0425-1B`) Phase-5 summary tables incl. the August ProdQJL re-sweep (historical, invalid) |
| [results/olmo2_1b/RESULTS_COMPLETE.md](results/olmo2_1b/RESULTS_COMPLETE.md) | OLMo2-1B full metrics + run logs (historical, invalid) |
| [results/shortlist_5model_eval/](results/shortlist_5model_eval/) | Engine-compatibility runs against the 5-model shortlist (identity smoke tests, gates, reference tests) — not study results |

Merged CSV/JSON summary bundles: [`../results/`](../results/README.md) (repo-root `results/`, distinct from `docs/results/` above — the root directory holds gitignored raw job JSON/CSV; `docs/results/` holds the version-controlled written tables).

## Directory layout

```
docs/
  README.md                        — this file
  architecture/                    — what the engine is, what models it supports
  methodology/                     — how evaluation works (protocol, math, limits)
  reproducibility/                 — how to run it yourself (local + Modal)
  results/
    qwen3_1.7b/                    — legacy model, primary published results
    olmo2_1b/                      — legacy model, second published results
    shortlist_5model_eval/         — current 5-model shortlist, live eval-framework run
  audits/                          — post-overhaul audit checklists (Gemma3 run: bugs found, fixes, verification log);
                                     DOCS_PLANS_RESULTS_ALIGNMENT_AUDIT.md (plans/results vs RESEARCH_REFRAME_AND_PRIOR_ART.md, 2026-11-24)
  literature/                      — literature alignment, staging bib, FRAMING_VALIDATION_2026-11-24.md,
                                     papers/ (downloaded PDFs + extracted text; not for commit)
  research_paper_writeup/          — arXiv manuscript source (unchanged)
  RESEARCH_REFRAME_AND_PRIOR_ART.md — research framing, prior-art survey, publication verdict
                                     (formerly MISC_Research.md; updated 2026-11-24)
```
