# Documentation

Public documentation for the **KV Cache Interception and Transformation Engine** (Apache-2.0). The reproducible benchmarking protocol and case-study write-up are branded **KVBench** (research manuscript in preparation — see [conference_101719.tex](research_paper_writeup/conference_101719.tex)).

See also the root [README.md](../README.md), [ROADMAP.md](../ROADMAP.md), and [CONTRIBUTING.md](../CONTRIBUTING.md).

## architecture/ — what the engine is and what it supports

| Doc | Purpose |
|---|---|
| [SYSTEM_DESIGN.md](architecture/SYSTEM_DESIGN.md) | Interception engine architecture (high-level, shortest entry point) |
| [ENGINE_INTERNALS.md](architecture/ENGINE_INTERNALS.md) | Complete implementation walkthrough — every file, the full execution flow, and the tier-based engineering plan (§8) for supporting model architectures beyond dense transformers |
| [SLM_COMPATIBILITY.md](architecture/SLM_COMPATIBILITY.md) | Historical: the original 6-candidate probe (Qwen3-1.7B, OLMo2-1B, Granite, Gemma3, MiniCPM4, Qwen3.5) that preceded the current shortlist |
| [MODEL_ARCHITECTURE_MATRIX.md](architecture/MODEL_ARCHITECTURE_MATRIX.md) | **Current** model set: the 2 legacy models + 5-model architecture-matrix shortlist (MHA/GQA/MQA/MLA/Hybrid), engine-support status per model, ranked adoption/transformation plan |

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

## results/ — published numbers, by model

| Doc | Purpose |
|---|---|
| [results/qwen3_1.7b/PHASE5_EVAL_RESULTS.md](results/qwen3_1.7b/PHASE5_EVAL_RESULTS.md) | Qwen3-1.7B Phase-5 summary tables + findings narrative |
| [results/qwen3_1.7b/RESULTS_COMPLETE.md](results/qwen3_1.7b/RESULTS_COMPLETE.md) | Qwen3-1.7B full metrics, per-layer stats, run logs |
| [results/olmo2_1b/PHASE5_EVAL_RESULTS.md](results/olmo2_1b/PHASE5_EVAL_RESULTS.md) | OLMo2-1B Phase-5 summary tables (current QJL numbers — August ProdQJL re-sweep) |
| [results/olmo2_1b/RESULTS_COMPLETE.md](results/olmo2_1b/RESULTS_COMPLETE.md) | OLMo2-1B full metrics + run logs (QJL rows reconciled against the re-sweep; original July numbers kept struck through for record) |
| [results/shortlist_5model_eval/](results/shortlist_5model_eval/) | **New**: live evaluation-framework run against the 5-model architecture-matrix shortlist — real numbers for the 2 models that work, exact tracebacks for the 3 that don't |

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
  research_paper_writeup/          — arXiv manuscript source (unchanged)
```
