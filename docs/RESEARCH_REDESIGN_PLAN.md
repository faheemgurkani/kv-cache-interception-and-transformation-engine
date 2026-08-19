# KVBench: Complete Research Improvement Roadmap

## Completeness record — Phases 1–10 (verified 2026-08-19)

Tracks **engine** (code + tests), **documentation** (in-repo docs), and **paper** (`docs/research_paper_writeup/conference_101719.tex`). Phases **5**, **8**, **11**, **12**, and **13** are flagged **not planned / future extension** — design reference only.

**Executive verdict:** Phases **1–4**, **6**, **7**, **9**, **10**, and **14** are **complete in the engine and documentation**. The paper still reflects the **pre-redesign** framing for branches 1–7 and lacks reproducibility citations for Pareto (Phase 9) and extended SYSTEM hardware columns (Phase 10). **Paper changes are documented only** in [Paper alignment guide](#paper-alignment-guide--codebase--conference_101719tex) and per-phase **Paper change log** subsections below — apply when revised experimental results are ready. Phases **5**, **8**, **11**, **12**, and **13** require **no paper or engine work** for the current case-study scope.

| Phase | Engine | Docs | Paper | Primary evidence |
| ----- | ------ | ---- | ----- | ---------------- |
| **1** Three-branch framework | ✅ Done | ✅ Done | 📝 Pending | `eval/runner.py`, `eval/{fidelity,behavior,system}/` |
| **2** FIDELITY branch | ✅ Done | ✅ Done | 📝 Pending | `eval/fidelity/{representation,attention,memory,recurrent}.py` |
| **2** BEHAVIOR branch | ✅ Done | ✅ Done | 📝 Pending | `eval/behavior/{task_quality,retrieval,instruction_following,reasoning}.py` |
| **2** SYSTEM branch | ✅ Done | ✅ Done | 📝 Pending | `eval/system/{latency_throughput,vram,kernel_cost,...}.py` |
| **3** Cost accounting | ✅ Done | ✅ Done | 📝 Pending | `eval/cost/accounting.py`, `EvaluationResult.cost` |
| **4** Taxonomy + SnapKV/Palu | ✅ Done | ✅ Done | 📝 Pending | `compressors/taxonomy.py`, `snapkv`, `palu` |
| **5** Adaptive plugin API | ⏸ Not planned | ⏸ Flagged | — | Do not implement or follow up |
| **6** Interception as contribution | ✅ Done | ✅ Done | 📝 Pending | `eval/controlled_conditions.py` (Phase 6 principle) |
| **7** Controlled experimental axes | ✅ Done | ✅ Done | 📝 Pending | `controlled_conditions` Phase 7 export (`phase: "7"`) |
| **8** Unified budget curves | ⏸ Not planned | ⏸ Flagged | — | Existing per-method sweeps sufficient |
| **9** Pareto analysis | ✅ Done | ✅ Done | 📝 Pending | `eval/pareto/`, `scripts/analyze_pareto.py` |
| **10** Hardware-aware eval (single GPU) | ✅ Done | ✅ Done | 📝 Pending | `eval/hardware/`, Modal A10G path |
| **14** Reproducibility harness | ✅ Done | ✅ Done | 📝 Pending | `controlled_conditions` + `cost` + YAML configs + `REPRODUCIBILITY.md` |
| **11** Realistic workload dimension | ⏸ Future extension | ⏸ Flagged | — | Current WikiText + default BEHAVIOR scope sufficient |
| **12** Workload scaling (2K–32K, batch) | ⏸ Future extension | ⏸ Flagged | — | ctx 128–512 / batch 1 / 64 tok gen sufficient for paper |
| **13** Serving-engine validation (vLLM/SGLang) | ⏸ Not planned | ⏸ Flagged | — | Controlled KVBench path sufficient; no vLLM/SGLang integration |

**Cross-cutting tests:** `tests/test_eval_runner.py`, `tests/test_controlled_conditions.py`, `tests/test_cost_accounting.py`, `tests/test_taxonomy.py`, `tests/test_pareto_analysis.py`, `tests/test_hardware_profile.py`, `tests/test_modal_merge_hardware.py`, `tests/test_behavior_modules.py`, `tests/test_system_modules.py`, `tests/test_*_reference.py`.

**Paper rewrite hub:** [`conference_101719.tex`](research_paper_writeup/conference_101719.tex) — full section-by-section spec in [Paper alignment guide](#paper-alignment-guide--codebase--conference_101719tex) below.

**Intentional engine gaps (documented, not paper blockers):**

- BEHAVIOR retrieval/instruction/reasoning use **synthetic in-repo generators**, not LongBench/RULER (`CURRENT_STATE.md`) — **Phase 11 deferred**; do not expand workloads for the current paper.
- Context lengths capped at **128 / 256 / 512**, batch **1**, **64** generated tokens — **Phase 12 deferred**; sufficient for controlled SLM case study.
- No vLLM/SGLang serving validation — **Phase 13 not planned**; controlled `KVCacheEngine` path is the systems scope.
- SYSTEM peak VRAM / GPU util collected on Modal CUDA reference path (Phase 10); optional locally via `--hardware-metrics`.
- Reasoning is opt-in (`--reasoning`); skip flags: `--skip-retrieval`, `--skip-instruction-following`.
- Hybrid FIDELITY/`recurrent` extension: `eval/fidelity/recurrent.py` (Falcon-H1).

Authoritative metric definitions: [`docs/methodology/METHODOLOGY.md`](methodology/METHODOLOGY.md) §1.1, §6.

---

## Paper alignment guide — codebase ↔ `conference_101719.tex`

**Purpose:** Document exactly what the paper still says vs. what the engine/docs now implement, so a later rewrite stays aligned. **Do not edit the `.tex` file until revised experimental results are ready** — this section is the specification for that pass.

**When to apply:** After the next evaluation sweep completes (new job JSON/CSV bundles under `results/`). Order of work: (1) run experiments → (2) update result tables/figures from bundles → (3) apply framing/terminology changes below → (4) compile PDF.

**Paper file:** [`docs/research_paper_writeup/conference_101719.tex`](research_paper_writeup/conference_101719.tex)  
**Code truth sources:** `eval/runner.py`, `eval/controlled_conditions.py`, `eval/{fidelity,behavior,system,cost}/`, `compressors/taxonomy.py`, `docs/methodology/METHODOLOGY.md`

Phases **5**, **8**, **11**, **12**, and **13:** no paper changes (flagged not planned / future extension).

### Phases 9–12 — feasibility vs paper (2026-08-19)

| Phase | Paper today | Code today | Phase goal | Already doing it? |
| ----- | ----------- | ---------- | ---------- | ----------------- |
| **9** Pareto | ✅ Figure + discussion (`Fig.~\ref{fig:pareto}`, L475–481, L617) | ✅ Auto frontier export (`eval/pareto/`) | Reproducible pipeline analysis | **~90%** — paper has figure; engine now automates regeneration |
| **10** CUDA hardware | ✅ A10G Modal sweeps (L222–229) | ✅ Single-GPU path + peak VRAM/GPU util | Deep HW metrics on CUDA | **~80%** — A10G done; VRAM/GPU util not yet in paper tables |
| **11** Workloads | WikiText + PPL/tok/s only | + synthetic BEHAVIOR tasks (default in engine) | Diverse external workloads | **⏸ Deferred** — current scope OK; no LongBench/RULER |
| **12** Scaling | ctx 128–512, batch 1, 64 tok | Same caps in `configs/` | 2K–32K, batch/gen grids | **⏸ Deferred** — explicit future work in `METHODOLOGY.md` / `ROADMAP.md` |

**Bottom line:** Phase **9 is largely done in the paper** (figure exists); Phase **10** needs optional table/caption updates at re-sweep. Phases **11–12** are **future extensions only** — the current WikiText case-study grid and ctx 128–512 setup are sufficient for paper framing; the engine is ahead on BEHAVIOR workload *types*, but the paper correctly reports only PPL/tok/s for now.

### Global terminology map

| Concept | Paper today | Codebase today | Paper should say |
| ------- | ----------- | -------------- | ---------------- |
| Evaluation split | Section A (offline) + Section B (online) | **FIDELITY** / **BEHAVIOR** / **SYSTEM** | Three independent branches; retire “Section A/B” except in a one-line legacy note if needed |
| Primary contribution | “Benchmarking framework” / “dual metrics” | Controlled **interception-and-transformation engine** | Engine + protocol as contribution; methods are case studies |
| Quality under compression | Section B = PPL + throughput only | BEHAVIOR: PPL + retrieval + instruction following (+ reasoning opt-in) | BEHAVIOR subsection; PPL in results; task probes in methodology (+ optional appendix numbers) |
| Runtime efficiency | Throughput under Section B | SYSTEM: TTFT, ITL, tok/s, latency (+ VRAM/bandwidth opt-in) | Separate **SYSTEM** subsection; tok/s tables move under SYSTEM |
| Cost | Not mentioned | `EvaluationResult.cost` (compression / offline / online) | New **Cost accounting** subsection |
| Method taxonomy | Three families in prose | A–E taxonomy + SnapKV/Palu plug-ins | Taxonomy table in Methodology; empirical results still TQ/QJL/RocketKV unless re-swept |
| Controlled comparison | “Identical conditions” in prose | `controlled_conditions` JSON per job (Phase 7) | Explicit **Controlled conditions** table + reproducibility sentence |
| Sweep name | “Phase-5 grid”, “27 jobs” | Same historical bundles; engine now richer | Keep job counts if bundles unchanged; rename to “evaluation grid” or keep Phase-5 label with footnote |

### Section-by-section change log

#### Title (L31)

| | |
| --- | --- |
| **Current** | `KVBench: Bridging Offline Fidelity and Online Inference Evaluation…` |
| **Codebase** | Three-branch eval; interception engine naming in `README.md` |
| **Change** | Consider: *KVBench: A Controlled KV Interception Engine for Fidelity, Behavior, and System Evaluation of Cache Compression in SLMs* (or keep “Bridging…” with subtitle mentioning three branches) |
| **Needs new results?** | No — framing only |
| **Phase** | 1, 6 |

#### Abstract (L44–46)

| | |
| --- | --- |
| **Current** | “benchmarking framework”; “dual metrics: Section A … Section B”; offline does not predict online |
| **Codebase** | FIDELITY/BEHAVIOR/SYSTEM; controlled interception; cost + taxonomy exist in code |
| **Change** | Replace “dual Section A/B” with three-branch names. Lead with “controlled interception engine.” Keep empirical claims (TQ/QJL/RocketKV, Qwen3/OLMo2) until re-sweep replaces numbers. Add one clause: “only the compressor plug-in varies under matched conditions.” |
| **Needs new results?** | Numbers: yes when re-sweeping; framing: no |
| **Phase** | 1, 6, 7 |

#### Keywords (L48–50)

| | |
| --- | --- |
| **Current** | `offline fidelity, online inference` |
| **Change** | Add: `KV interception, fidelity evaluation, system metrics` (optional: `cost accounting`) |
| **Needs new results?** | No |
| **Phase** | 1 |

#### Introduction (L52–60)

| | |
| --- | --- |
| **Current** | L58 “What: Section A … Section B”; L60 contributions (1) dual Section A/B protocol |
| **Codebase** | L58 already says “interception-and-transformation engine” — good. “What” axis is outdated. |
| **Change** | **L58 *What* bullet:** FIDELITY (representation/attention/memory) + BEHAVIOR (PPL, retrieval, instruction following) + SYSTEM (latency/throughput). **L60 contribution (1):** “three-branch evaluation protocol” not “dual Section A/B”. **Contribution (3):** “FIDELITY does not predict BEHAVIOR” (not offline→online). |
| **Needs new results?** | Framing no; empirical paragraph yes if models/methods change |
| **Phase** | 1, 2, 6 |

#### Related Work (L62–81)

| | |
| --- | --- |
| **Current** | “offline fidelity and online quality always reported together”; cites Palu/SnapKV in eviction/sketching subsections |
| **Codebase** | Palu/SnapKV implemented as plug-ins; not in paper results |
| **Change** | L66, L78, L81: replace “offline/online” with “fidelity/behavior/system”. **Do not** claim SnapKV/Palu empirical results unless sweeps are run. Optional sentence: “The engine also hosts SnapKV and Palu plug-ins (taxonomy categories A and C) for future sweeps.” |
| **Needs new results?** | SnapKV/Palu claims: yes if included in results |
| **Phase** | 1, 4 |

#### §Methodology opening (L83–86, `\label{sec:methodology}`)

| | |
| --- | --- |
| **Current** | “benchmarking harness”; “dual offline/online evaluation contract” |
| **Change** | “controlled interception-and-transformation environment”; “three-branch FIDELITY / BEHAVIOR / SYSTEM contract” |
| **Needs new results?** | No |
| **Phase** | 1, 6 |

#### §Design Principles (L88–96, `\label{subsec:design}`)

| | |
| --- | --- |
| **Current** | Bullet “Dual evaluation. Section A … Section B …” |
| **Codebase** | Three branches; plug-in isolation matches |
| **Change** | Replace dual bullet with: **Three-branch evaluation.** FIDELITY, BEHAVIOR, and SYSTEM are always reported together (with BEHAVIOR/SYSTEM sub-metrics configurable). Add bullet or sentence: **Controlled comparison.** Same model, input, decode loop, hardware; only compressor varies (`controlled_conditions` in export JSON). |
| **Needs new results?** | No |
| **Phase** | 1, 6, 7 |

#### Fig. pipeline caption (L98–102, `\label{fig:pipeline}`)

| | |
| --- | --- |
| **Current** | “Section A offline fidelity and Section B online quality” |
| **Change** | Caption lists FIDELITY / BEHAVIOR / SYSTEM; mention cost block if figure updated. Regenerate figure asset if diagram still shows two boxes. |
| **Needs new results?** | Figure asset: optional; caption text: no |
| **Phase** | 1, 3, 6 |

#### §Plug-in Interface (L153–155, `\label{subsec:plugins}`)

| | |
| --- | --- |
| **Current** | Lists hooks; three methods only |
| **Codebase** | `offline_cost_metadata`, `theoretical_compression_ratio`, taxonomy on `EvaluationResult` |
| **Change** | Add cost hooks and taxonomy metadata to hook list. Mention `controlled_conditions` export. |
| **Needs new results?** | No |
| **Phase** | 3, 4, 7 |

#### **NEW** §Compression taxonomy (insert after `\label{subsec:plugins}`, before Case-Study Methods)

| | |
| --- | --- |
| **Current** | Not present |
| **Codebase** | `compressors/taxonomy.py` categories A–E |
| **Change** | Add table mapping category → mechanism → case-study plug-in (TQ=B, QJL=B+E, RocketKV=D+E, SnapKV=A, Palu=C+E). State empirical evaluation covers TQ/QJL/RocketKV only. |
| **Needs new results?** | No for taxonomy table; yes to add SnapKV/Palu result rows |
| **Phase** | 4 |

#### §Case-Study Methods (L165–167, `\label{subsec:methods}`)

| | |
| --- | --- |
| **Current** | “three published families” |
| **Change** | Keep three for **results** unless re-sweep adds methods. Cross-reference taxonomy table. |
| **Needs new results?** | Yes to expand beyond three |
| **Phase** | 4 |

#### §Experiments opening + setup (L216–229)

| | |
| --- | --- |
| **Current** | “dual Section A/B”; lists model, WikiText, batch 1, A10G; “Each run records Section A and Section B metrics” |
| **Codebase** | Phase 7 exports full axis checklist in JSON |
| **Change** | Add **Table: Controlled experimental conditions** (model, tokenizer, dataset/split, ctx lengths, batch, gen length 64, PPL stride 512, greedy decode, A10G, metrics enabled). Replace “Section A and Section B” with three branches + cost. Add: “Per-job JSON includes `controlled_conditions` (fixed vs. variable axes).” |
| **Needs new results?** | Table structure: no; row values yes if setup changes |
| **Phase** | 7 |

#### §Evaluation Protocol (L255–285, `\label{subsec:eval_protocol}`)

| | |
| --- | --- |
| **Current** | Two subsubsections: **Section A: Offline Fidelity** (L267–274), **Section B: Online Inference** (L276–284) |
| **Codebase** | See `METHODOLOGY.md` §6.1–6.3, §6.5 |
| **Change** | **Rename & split into four subsubsections:** |

**→ FIDELITY** (replace L267–274):

| Paper has | Codebase also has | Action |
| --------- | ----------------- | ------ |
| Tensor RMSE | Relative recon error, cosine | Add to item list |
| Attention MSE/RMSE/cosine/max | Attention-output RMSE, KL divergence | Add to item list |
| Memory ratio, eff. bits | Metadata overhead, shared bytes | Add metadata bullet |
| — | `recurrent` (hybrid) | Footnote: Falcon-H1 only |

**→ BEHAVIOR** (replace L276–282 PPL-only scope):

| Paper has | Codebase also has | Action |
| --------- | ----------------- | ------ |
| Sliding-window PPL | Same | Keep Algorithm 2 (`\label{alg:ppl}`) |
| — | Needle-in-haystack retrieval | **Optional (Phase 11 deferred):** one-sentence protocol in methodology only if space; **no result table required** for current submission |
| — | Instruction-following compliance | Same — optional methodology mention; WikiText PPL remains primary BEHAVIOR metric in results |
| — | Reasoning (opt-in) | Future work only |

**→ SYSTEM** (split from old Section B throughput, L284):

| Paper has | Codebase also has | Action |
| --------- | ----------------- | ------ |
| tok/s, ms/token | TTFT, ITL p50/p99, end-to-end latency | Add definitions |
| — | Peak VRAM, kernel cost, bandwidth | Appendix or “CUDA extended metrics” |

**→ COST** (new, after SYSTEM):

| Paper has | Codebase has | Action |
| --------- | ------------ | ------ |
| — | compression / offline / online tree | Add subsection mirroring Phase 3 diagram in this doc; cite TurboQuant calibration vs QJL/RocketKV calibration-free |

| **Needs new results?** | PPL/tok/s numbers yes; protocol structure no |
| **Phase** | 1, 2, 3 |

#### §Online Evaluation Procedure (L286–315, `\label{subsec:online_proc}`)

| | |
| --- | --- |
| **Current** | Algorithm 2 for PPL; implementation notes for TQ/QJL/RocketKV |
| **Change** | Keep for BEHAVIOR/PPL. Add pointer: SYSTEM throughput uses same `KVCacheEngine.generate` greedy loop (64 tokens). L274 “reset before Section B” → “reset compressor state before BEHAVIOR/SYSTEM passes.” |
| **Needs new results?** | No |
| **Phase** | 2 |

#### Results §Qwen3 / §OLMo2 (L319+, `\label{sec:qwen3}`, `\label{sec:olmo2}`)

| | |
| --- | --- |
| **Current** | Table captions “Section A fidelity”, “Section B online metrics/PPL” |
| **Change** | Rename captions: **FIDELITY**, **BEHAVIOR (perplexity)**, **SYSTEM (throughput/latency)**. Split combined tables if BEHAVIOR and SYSTEM were merged. **Replace numeric cells only from new sweep bundles** — do not hand-edit. |
| **Needs new results?** | **Yes** for all numeric cells |
| **Phase** | 1, 2 |

#### Figures: offline-vs-online, Pareto (L475+, L608–615)

| | |
| --- | --- |
| **Current** | Axis labels “Section A” vs “Section B”; `plot_offline_vs_online.pdf`. **Pareto:** `plot_pareto.pdf` at T=512 — memory ratio vs log PPL ratio, marker area ∝ tok/s, empirical front (L475–481); cited in Discussion L617. |
| **Codebase** | `scripts/analyze_pareto.py`, `eval/pareto/analysis.py`, `ResultReporter.save_pareto()` — regenerates 2D/3D front from job bundles + writes `pareto_ctx512.json`. |
| **Change** | **Offline-vs-online figure:** regenerate with FIDELITY vs BEHAVIOR labels. **Pareto figure:** *concept unchanged* — re-export via CLI from post-rewrite bundles (do not hand-draw). Update caption footnote: “Pareto front computed by `scripts/analyze_pareto.py` from job JSON.” Optionally cite optimal set from `pareto_ctx512.json` in text. |
| **When** | During paper rewrite pass **after** re-sweep bundles land (same timing as result tables). Pareto can be regenerated **without** new GPU jobs if existing Phase-5 JSON is reused. |
| **Why** | Paper already demonstrates the trade-off analysis (Phase 9 goal met visually); engine gap was reproducibility — now closed. Rewrite pass should align figure provenance with automated export, not change the scientific claim. |
| **Needs new results?** | **Only if** sweep grid/methods change; otherwise replot from existing JSON |
| **Phase** | 1, 2, 9 |

#### §Experiments setup — hardware block (L216–229, extends Phase 7 table)

| | |
| --- | --- |
| **Current** | “NVIDIA A10G GPUs (Modal)”; throughput/tok/s in SYSTEM tables; **no** peak VRAM or GPU utilization columns. |
| **Codebase** | Modal worker collects `hardware` block + peak VRAM + GPU util (`eval/hardware/`, Phase 10). Merge CSV columns: `peak_vram_*`, `gpu_util_*`, `reference_gpu`. |
| **Change** | In setup paragraph (L222–229): add one sentence — single-GPU Modal A10G reference path; peak device memory and NVML GPU utilization collected per job (not multi-GPU tier matrix). In **SYSTEM tables** or appendix: add optional columns `peak_vram_allocated_mb`, `gpu_util_mean_pct` for T=512 representative configs. **Do not** claim A100/H100/4090 comparisons. |
| **When** | At **re-sweep** when new Modal jobs include hardware metrics (already enabled in worker). Can defer columns to appendix if page-limited. |
| **Why** | Satisfies inference-engineering credibility (“actual GPU behavior”) without expanding scope to multi-GPU tiers. Existing A10G sweeps already satisfy “one CUDA experiment.” |
| **Needs new results?** | **Yes** for numeric VRAM/GPU util cells; **no** for setup prose |
| **Phase** | 7, 10 |

#### **NEW** §Reproducibility / artifact availability (insert after `\label{subsec:exp_config}`, Phase 14)

| | |
| --- | --- |
| **Current** | L229 mentions “hyperparameters, and timestamps” in passing; no structured config export |
| **Codebase** | Per-job `controlled_conditions`, `cost`, `hardware`; YAML configs; `REPRODUCIBILITY.md` |
| **Change** | Subsection or appendix: (1) **Table: Standardized run configuration** (Phase 14 YAML fields); (2) sentence that JSON replay is supported; (3) repo URL, config paths, `git rev-parse HEAD` pin, Modal reproduce commands. |
| **When** | Paper rewrite — **no new GPU jobs required** for this prose |
| **Why** | Literature comparison problem (different models/tasks/budgets) — reproducibility is explicit contribution |
| **Needs new results?** | No |
| **Phase** | 14 |

#### §Discussion (L595–623, `\label{sec:discussion}`)

| | |
| --- | --- |
| **Current** | L598 “benchmarking and evaluation study”; L608 “Offline fidelity does not predict online quality”; L617 TurboQuant throughput as mechanism story |
| **Change** | L598: “controlled experimentation framework” — contribution is matched interception, not ranking winners. L608: **FIDELITY does not predict BEHAVIOR**; SYSTEM explains TurboQuant speed penalty separately. L623 implications: report all three branches + cost + controlled conditions. |
| **Needs new results?** | Empirical paragraphs yes; framing no |
| **Phase** | 1, 2, 3, 6 |

#### §Conclusion (L625–629, `\label{sec:conclusion}`)

| | |
| --- | --- |
| **Current** | “benchmarking framework”; “dual Section A/Section B metrics” |
| **Change** | List: (i) interception engine, (ii) plug-in API, (iii) FIDELITY/BEHAVIOR/SYSTEM/Cost, (iv) controlled conditions export, (v) case-study findings. **Future work (Phases 11–12):** external benchmarks (LongBench/RULER), long-context scaling (2K–32K), batch/concurrency grids — explicitly **out of scope** for current paper. |
| **Needs new results?** | Findings bullet yes; structure no |
| **Phase** | 1–10 (future work cites 11–12 only) |

### What to pull from codebase when writing

| Paper element | Source in repo |
| ------------- | -------------- |
| Controlled conditions table | Any `result.to_dict()["controlled_conditions"]["fixed"]` from a reference job |
| Phase 14 reproducibility manifest | Map fields via [Phase 14 field mapping](#engine-encapsulation--field-mapping) |
| FIDELITY metric definitions | `docs/methodology/METHODOLOGY.md` §6.1 |
| BEHAVIOR protocols | §6.2 + `eval/behavior/*.py` docstrings |
| SYSTEM metrics | §6.3 |
| Cost tree | §6.5 + `eval/cost/accounting.py` |
| Taxonomy table | `compressors/taxonomy.py` `METHOD_TAXONOMY` |
| Pareto optimal set | `python scripts/analyze_pareto.py … --context-length 512` → `pareto_ctx512.json` |
| Hardware + VRAM/GPU util | `result.to_dict()["hardware"]`, `system.peak_memory`, `system.gpu_utilization`; Modal merge CSV |
| Result numbers | `results/phase5_modal_*`, `results/olmo2_phase5_*` (or post-rewrite bundles) |

### Minimal vs full paper update (choose at rewrite time)

| Tier | When | Scope |
| ---- | ---- | ----- |
| **Minimal** | Re-sweep done; tight page limit | Terminology pass (Section A/B → three branches) + controlled conditions table + caption renames; keep 3 methods; **regenerate Pareto from CLI** (Phase 9) |
| **Full** | Re-sweep + appendix space | Above + Cost subsection + taxonomy table + **SYSTEM VRAM/GPU util columns** (Phase 10) + optional BEHAVIOR protocol prose (Phase 11 — no new numbers) |

---

## Phase 1: Redesign the Core Evaluation Framework

### 1. Move away from the simple "Offline vs Online" split ✅ **Done**

**Previous (pre-redesign) conceptual structure:**

```text
KVBench
 ├── Offline
 │    ├── Reconstruction error
 │    ├── Attention error
 │    └── Memory
 │
 └── Online
      ├── PPL
      ├── Throughput
      └── Decode
```

That split was **too coarse**. The engine now uses **three primary evaluation dimensions**:

```text
                         KVBench
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
       FIDELITY           BEHAVIOR          SYSTEM
          │                 │                 │
    Representation      Task Quality       Latency / TTFT
    Attention           PPL (default)      Throughput / ITL
    Memory              Retrieval (default)  Peak VRAM*
    Recurrent†          Instruction (default) Memory BW*
                        Reasoning*           Kernel Cost*
                                             GPU Util*

    * SYSTEM / reasoning opt-in (CLI flags); use --skip-retrieval / --skip-instruction-following to disable defaults
    † hybrid models only (Falcon-H1); eval/fidelity/recurrent.py
```

This turns KVBench from a **compression test harness** into a **multi-dimensional inference benchmark**.

**Code:** `eval/runner.py` · **CLI:** `scripts/run_eval.py` · **Docs:** `METHODOLOGY.md` §6, `SYSTEM_DESIGN.md` §Evaluation.

### Completeness record

| Track | Status | Detail |
| ----- | ------ | ------ |
| **Engine** | ✅ Done | Three-branch orchestrator; legacy Section A/B accessors retained on `EvaluationResult` for back-compat only. |
| **Documentation** | ✅ Done | `README.md`, `SYSTEM_DESIGN.md`, `METHODOLOGY.md` §6, `CLAUDE.md`. |
| **Paper** | 📝 Documented | See [Paper alignment guide — Title, Abstract, Introduction](#paper-alignment-guide--codebase--conference_101719tex). Apply after re-sweep. |

---

# Phase 2: Create Three Explicit Evaluation Branches

## 2. Fidelity Evaluation ✅ **Done**

Answer:

> **Did the transformation preserve the KV representation and attention behavior?**

Measure (all implemented in `eval/fidelity/`):

| Planned metric | Module | Status |
| -------------- | ------ | ------ |
| KV reconstruction RMSE | `representation.py` | ✅ |
| Relative reconstruction error | `representation.py` | ✅ |
| Cosine similarity | `representation.py` | ✅ |
| Attention-output RMSE | `attention.py` | ✅ |
| Attention distribution divergence (KL) | `attention.py` | ✅ |
| Compression ratio | `memory.py` | ✅ |
| Actual memory reduction | `memory.py` | ✅ |
| Metadata/storage overhead | `memory.py` (`shared_metadata_bytes`) | ✅ |
| Hybrid recurrent preservation (extension) | `recurrent.py` | ✅ |

Runs on a **single offline forward pass** by default (`run_fidelity=True`). Explicitly named **Fidelity Evaluation** in code and docs (replaces legacy "offline" branch).

**Tests:** FIDELITY sub-metrics asserted in `tests/test_*_reference.py`, `tests/test_regression_validation.py`, `tests/test_eval_runner.py`.

### Completeness record

| Track | Status | Detail |
| ----- | ------ | ------ |
| **Engine** | ✅ Done | All planned metrics + hybrid `recurrent.py` extension. |
| **Documentation** | ✅ Done | `METHODOLOGY.md` §6.1; `MATHEMATICS_AND_ALGORITHMS.md`. |
| **Paper** | 📝 Documented | [§Evaluation Protocol → FIDELITY](#evaluation-protocol-l255285-labelsubseceval_protocol); [Results caption renames](#results-qwen3--olmo2-l319-labelsecqwen3-labelsecolmo2). |

---

## 3. Behavioral Evaluation ✅ **Done** (PPL default; task metrics opt-in)

Answer:

> **Does the model still behave correctly after KV transformation?**

| Planned capability | Module | Default | Status |
| ------------------ | ------ | ------- | ------ |
| Perplexity | `behavior/task_quality.py` | **on** | ✅ |
| Long-context retrieval | `behavior/retrieval.py` | **on** | ✅ |
| Instruction following | `behavior/instruction_following.py` | **on** | ✅ |
| Reasoning (Option C — bonus) | `behavior/reasoning.py` | opt-in | ✅ |

Plan recommendation was **PPL + long-context retrieval + instruction following** — all three run **by default** in `EvaluationRunner.run()` and `scripts/run_eval.py`. Use `--skip-retrieval` / `--skip-instruction-following` for faster smoke runs. Reasoning remains opt-in (`--reasoning`).

All BEHAVIOR metrics run through **`KVCacheEngine`** (compressed KV drives real decode), not a single forward pass.

**Tests:** Full recommendation stack exercised in `tests/test_{olmo2,qwen3,gemma3,tinydeepseek,falcon_h1}_reference.py` (`test_*_all_eval_branches` with all BEHAVIOR flags). Default-path PPL only in `tests/test_eval_runner.py` and WP5 regression.

**Caveat:** Synthetic in-repo task generators — legible failure modes, not external benchmark scale (`CURRENT_STATE.md`).

### Completeness record

| Track | Status | Detail |
| ----- | ------ | ------ |
| **Engine** | ✅ Done | PPL + retrieval + instruction following default; reasoning opt-in. All via `KVCacheEngine`. |
| **Documentation** | ✅ Done | `METHODOLOGY.md` §6.2; module docstrings in `eval/behavior/`. |
| **Paper** | 📝 Documented | [§Evaluation Protocol → BEHAVIOR](#evaluation-protocol-l255285-labelsubseceval_protocol); synthetic task protocols optional until re-sweep. |

---

## 4. System Evaluation ✅ **Done** (latency default; rest opt-in)

Answer:

> **Does the compression actually make inference better?**

| Planned metric | Module | Default | Status |
| -------------- | ------ | ------- | ------ |
| TTFT | `system/latency_throughput.py` | **on** | ✅ |
| Inter-token latency (ITL mean/p50/p99) | `system/latency_throughput.py` | **on** | ✅ |
| Decode latency | `system/latency_throughput.py` | **on** | ✅ |
| Tokens/sec | `system/latency_throughput.py` | **on** | ✅ |
| End-to-end latency | `system/latency_throughput.py` | **on** | ✅ |
| Peak VRAM | `system/vram.py` | opt-in (`--peak-memory`) | ✅ (CUDA only) |
| Actual KV memory | `eval/runner.py` → `SystemMetrics.actual_kv_memory_bytes` | on when SYSTEM runs | ✅ |
| Compression/decompression time | `system/kernel_cost.py` | opt-in | ✅ |
| Attention execution time (proxy) | `system/kernel_cost.py` | opt-in | ✅ |
| GPU utilization | `system/gpu_utilization.py` | opt-in | ✅ (CUDA + pynvml) |
| Memory bandwidth | `system/memory_bandwidth.py` | opt-in | ✅ |

The **4× compression vs 3× with lower overhead** tradeoff is exactly why SYSTEM is separate from FIDELITY — documented in `eval/system/__init__.py`, `METHODOLOGY.md` §6.3, and empirical Phase-5 results.

**Tests:** Throughput in `tests/test_online_inference.py`; full SYSTEM stack in reference tests; default TTFT/ITL/tok-s in smoke/regression.

**Caveats:** RocketKV bypasses timed `compress_kv` wrapper in `kernel_cost` (reads as attention time). VRAM/GPU util unavailable on MPS/CPU.

### Completeness record

| Track | Status | Detail |
| ----- | ------ | ------ |
| **Engine** | ✅ Done | TTFT/ITL/tok-s/end-to-end default; VRAM, bandwidth, kernel cost, GPU util opt-in. |
| **Documentation** | ✅ Done | `METHODOLOGY.md` §6.3; `eval/system/__init__.py` rationale. |
| **Paper** | 📝 Documented | [§Evaluation Protocol → SYSTEM](#evaluation-protocol-l255285-labelsubseceval_protocol); [§Discussion L617](#discussion-l595623-labelsecdiscussion). |

---

# Phase 3: Add Explicit Cost Accounting ✅ **Done**

For **every compression plugin**, report:

```text
METHOD
│
├── Compression
│   ├── theoretical compression ratio
│   ├── actual memory reduction
│   └── metadata overhead
│
├── Offline cost
│   ├── calibration required?
│   ├── calibration dataset
│   ├── calibration tokens
│   ├── calibration time
│   └── calibration memory
│
└── Online cost
    ├── compression time
    ├── decompression time
    ├── attention cost
    └── end-to-end decode cost
```

**Code:** `eval/cost/accounting.py` · **Hooks:** `compressors/base.py` (`offline_cost_metadata`, `theoretical_compression_ratio`) · **Runner:** `EvaluationResult.cost` · **CLI:** on by default; `--skip-cost` to disable · **Online detail:** `--kernel-cost` for compress/decompress/attention breakdown.

**Tests:** `tests/test_cost_accounting.py`; cost block asserted in `tests/test_eval_runner.py`.

Recent work such as Oaken explicitly separates offline preparation from online inference cost, while calibration-free methods show that calibration requirements themselves are an important methodological variable.

### Completeness record

| Track | Status | Detail |
| ----- | ------ | ------ |
| **Engine** | ✅ Done | `eval/cost/accounting.py`; hooks on `compressors/base.py`; `--skip-cost`; `--kernel-cost` for online breakdown. |
| **Documentation** | ✅ Done | `METHODOLOGY.md` §6.5; cost fields in CSV/JSON export. |
| **Paper** | 📝 Documented | [§Evaluation Protocol → COST (new)](#evaluation-protocol-l255285-labelsubseceval_protocol). |

---

# Phase 4: Add a Compression Taxonomy ✅ **Done**

Don't treat every method as simply:

> "KV compression."

The engine classifies methods by mechanism via `compressors/taxonomy.py` (`CompressionCategory` A–E, `METHOD_TAXONOMY`, exposed as `EvaluationResult.taxonomy`).

For example:

### A. Eviction

* H2O
* Scissorhands
* **SnapKV** ✅ (`compressors/snapkv.py`)
* etc.

### B. Quantization

* QJL
* TurboQuant
* AsymKV
* XQuant

### C. Projection / dimensionality reduction

* **Palu** ✅ (`compressors/palu.py`)
* MiniCache

### D. Hybrid compression

* RocketKV
* HqeKV
* HybridKV

### E. Compression + modified attention

Some methods don't merely compress the cache. They also change how attention operates.

This distinction is particularly important for methods such as RocketKV and Palu (RoPE path).

**Code:** `compressors/taxonomy.py` · **New plug-ins:** `snapkv`, `palu` · **Tests:** `tests/test_taxonomy.py`, `tests/test_snapkv.py`, `tests/test_palu.py`

### Completeness record

| Track | Status | Detail |
| ----- | ------ | ------ |
| **Engine** | ✅ Done | Categories A–E; `EvaluationResult.taxonomy`; SnapKV + Palu plug-ins registered. KIVI remains stub. |
| **Documentation** | ✅ Done | `METHODOLOGY.md` §5a/5b; `SYSTEM_DESIGN.md`; `CURRENT_STATE.md`. |
| **Paper** | 📝 Documented | [NEW §Compression taxonomy](#new-compression-taxonomy-insert-after-labelsubsecplugins-before-case-study-methods); [§Case-Study Methods](#case-study-methods-l165167-labelsubsecmethods). No SnapKV/Palu results unless re-swept. |

---

# Phase 5: Upgrade the Plugin Architecture ⏸ **Not planned**

> **Status (2026-08-19):** Out of scope. This section is **design reference only** — do **not** implement, schedule, or follow up on Phase 5. The current one-plug-in / one-global-config model is sufficient for the paper and near-term roadmap. Proceed from Phase 4 directly to Phase 6+.

Your current engine should not assume:

> One compressor = one global transformation.

Modern KV methods are becoming increasingly adaptive.

The plugin API should support:

### Layer-specific decisions

```text
Layer 1 → 4-bit
Layer 2 → 4-bit
Layer 3 → 2-bit
...
```

### Head-specific decisions

```text
Head 1 → retain
Head 2 → quantize
Head 3 → evict
```

### Token-specific decisions

```text
Token A → retain
Token B → compress
Token C → evict
```

### Stateful/online decisions

```text
t1 → policy
t2 → update policy
t3 → update policy
...
```

This is important because recent methods increasingly use heterogeneous and adaptive policies.

**Partial overlap today (no Phase 5 work required):** token-level eviction (RocketKV, SnapKV), per-head voting (SnapKV), stateful online paths (RocketKV, QJL), and layer-wise storage in the engine are handled **inside individual plug-ins**, not via a general policy API.

### Completeness record

| Track | Status | Detail |
| ----- | ------ | ------ |
| **Engine** | ⏸ Not planned | One plug-in / one global config remains the contract. |
| **Documentation** | ⏸ Flagged | This section retained as **design reference only**. |
| **Paper** | — | No paper update required. |

---

# Phase 6: Make the Interception Engine the Central Methodological Contribution ✅ **Code/docs done · 📝 Paper deferred**

> **Status (2026-08-19):** **Implementation complete** — controlled interception contract in `eval/controlled_conditions.py`, emitted on every `EvaluationResult` as `controlled_conditions`; `framework/kv_engine.py` and evaluation docs frame FIDELITY / BEHAVIOR / SYSTEM under matched conditions. **Paper update deferred** — `docs/research_paper_writeup/conference_101719.tex` still uses “benchmarking framework” + Section A/B naming; rewrite abstract, contributions, and figure captions later.

This is important for your paper.

Don't describe KVBench simply as:

> "a framework for comparing KV compression methods."

Instead, emphasize:

```text
                    SAME MODEL
                        │
                    SAME INPUT
                        │
                 SAME DECODE LOOP
                        │
                ┌───────┴───────┐
                │               │
          KV INTERCEPTION   KV INTERCEPTION
                │               │
          Method A          Method B
                │               │
                └───────┬───────┘
                        ↓
                  SAME INFERENCE
                        ↓
             FIDELITY / BEHAVIOR / SYSTEM
                        ↓
                 FAIR COMPARISON
```

The key methodological value is:

> **Different KV transformations are executed through the same inference path under matched conditions.**

That controlled environment is much more important than simply saying "we benchmark several methods."

**Code:** `eval/controlled_conditions.py` · **Runner field:** `EvaluationResult.controlled_conditions` · **Export:** `to_dict()["controlled_conditions"]` · **Tests:** `tests/test_controlled_conditions.py`

### Completeness record

| Track | Status | Detail |
| ----- | ------ | ------ |
| **Engine** | ✅ Done | Principle + diagram semantics enforced in runner/engine docstrings. |
| **Documentation** | ✅ Done | `METHODOLOGY.md` §1.1; `SYSTEM_DESIGN.md` Phase 6 block; `README.md` lead paragraph. |
| **Paper** | 📝 Documented | [§Design Principles](#design-principles-l8896-labelsubsecdesign); [Fig. pipeline](#fig-pipeline-caption-l98102-labelfigpipeline); [§Discussion](#discussion-l595623-labelsecdiscussion); [§Conclusion](#conclusion-l625629-labelsecconclusion). |

---

# Phase 7: Introduce Controlled Experimental Conditions ✅ **Done**

> **Status (2026-08-19):** **Implementation complete** — every controlled axis is exported in `eval/controlled_conditions.py` (`REQUIRED_FIXED_AXES` checklist, hardware/tokenizer/input/decoding/metrics profiles, method-specific `compression_budget`). Validated by `tests/test_controlled_conditions.py`. Paper controlled-conditions table deferred with Phase 6 narrative update.

Make the benchmark explicitly control:

* model
* tokenizer
* prompt
* dataset
* context length
* generation length
* compression budget
* hardware
* batch size
* decoding configuration
* evaluation metrics

Then the comparison becomes:

> **Only the KV transformation changes.**

This makes causal comparison much stronger.

Your methodology can explicitly say:

> Same model + same input + same decode loop + same hardware + different KV transformation.

**Code:** extends Phase 6 — `detect_hardware()`, `build_tokenizer_metadata()`, `build_input_construction()`, `build_decoding_configuration()`, `build_evaluation_metrics_profile()`, `extract_compression_budget()`, `validate_controlled_contract()` · **Export:** `controlled_conditions.phase == "7"` in every job JSON.

### Completeness record

| Track | Status | Detail |
| ----- | ------ | ------ |
| **Engine** | ✅ Done | `REQUIRED_FIXED_AXES` validated per run; hardware, tokenizer, input construction, decoding, metrics, compression budget exported. Env: `KV_EVAL_DEVICE`, `KV_HARDWARE_PROFILE`. |
| **Documentation** | ✅ Done | `METHODOLOGY.md` §1.1 (full axis table); `CURRENT_STATE.md`; `tests/test_controlled_conditions.py` (12 tests). |
| **Paper** | 📝 Documented | [§Experiments setup + controlled conditions table](#experiments-opening--setup-l216229). |

---

# Phase 8: Add Multiple Compression Budgets ⏸ **Not planned**

> **Status (2026-08-19):** Out of scope. This section is **design reference only** — do **not** implement, schedule, or follow up on Phase 8. Current method-specific budget sweeps (TurboQuant bitwidth/stages, RocketKV token budgets) in existing results are sufficient for the paper; proceed from Phase 7 directly to Phase 9+.

Don't only test one compression setting.

For example:

```text
1×
2×
4×
8×
16×
```

or equivalent bit/retention settings.

Then show:

> How does quality degrade as compression increases?

This allows you to compare **compression-quality curves**, rather than isolated numbers.

### Completeness record

| Track | Status | Detail |
| ----- | ------ | ------ |
| **Engine** | ⏸ Not planned | Per-method budget grids (TurboQuant bitwidth/stages, RocketKV r256/r512/r1024) already exist. |
| **Documentation** | ⏸ Flagged | This section retained as **design reference only**. |
| **Paper** | — | Existing Phase-5 sweep tables already show multiple budgets per method; no unified cross-method curve framework needed. |

---

# Phase 9: Add Pareto Analysis ✅ **Done**

> **Status (2026-08-19):** **Implementation complete** — `eval/pareto/analysis.py` computes 2D (paper-style) and 3D (quality/memory/speed) Pareto frontiers from `EvaluationResult`, `to_dict()`, or legacy Phase-5 bundle JSON. CLI: `scripts/analyze_pareto.py`. Reporting: `ResultReporter.save_pareto()` / `reporting/pareto_report.py`. Tests: `tests/test_pareto_analysis.py`. Paper figure can be regenerated from bundles; `.tex` update deferred.

### Feasibility snapshot (paper vs code)

| | Paper | Code (now) | Gap |
| --- | ----- | ---------- | --- |
| **Proposed** | Plot quality vs memory vs speed; mark Pareto-optimal methods | First-class reproducible analysis in pipeline | — |
| **Already in place** | ✅ `Fig.~\ref{fig:pareto}` (`plot_pareto.pdf`, L475–481): memory ratio vs log PPL ratio, marker size ∝ tok/s, empirical front; Discussion L617 | ✅ `eval/pareto/`, CLI, reporter | Paper **ahead on visualization**; code was manual → **now automated** |
| **Remaining gap** | Caption does not cite reproducible export path | — | One caption/footnote sentence at rewrite |
| **Feasibility** | **High** — mostly provenance + regeneration; scientific content already in PDF |

This is another strong improvement.

Instead of only producing tables:

```text
Method A = X
Method B = Y
Method C = Z
```

plot:

```text
Quality
  ↑
  │       ● A
  │
  │   ● B
  │
  │ ● C
  └────────────────→
       Memory / Speed
```

Identify **Pareto-optimal methods** across:

* quality
* memory
* throughput

This makes your results much more analytical.

**Code:** `eval/pareto/analysis.py` · `eval/pareto/plot.py` · **CLI:** `python scripts/analyze_pareto.py results/phase5_modal_*/*.json --context-length 512` · **Reporter:** `ResultReporter.save_pareto()` · **Tests:** `tests/test_pareto_analysis.py`

### Paper change log — when, why, what

| When | Why | What to change in `conference_101719.tex` |
| ---- | --- | ----------------------------------------- |
| **At paper rewrite** (can precede re-sweep if reusing Phase-5 JSON) | Engine now exports frontiers reproducibly; paper figure was built manually | Regenerate `plot_pareto.pdf` via `scripts/analyze_pareto.py` — **keep axes and semantics** (memory ratio, log₁₀ PPL ratio, marker area ∝ tok/s) |
| **Same pass** | Reviewers expect reproducible analysis artifacts | Add caption footnote or `\S`Methods sentence: optimal set from `pareto_ctx512.json`; cite CLI command in reproducibility appendix |
| **Discussion L617** | Align narrative with automated frontier | Optional: replace hand-enumerated “no single winner” examples with IDs from `pareto_ctx512.json` optimal set — **only if** numbers change after re-sweep |
| **Do not** | Phase 9 scientific claim already satisfied | Do **not** add new experiments solely for Pareto; do **not** change figure semantics to 3D unless appendix space allows |

**No codebase changes required** for Phase 9 beyond what is already merged.

### Completeness record

| Track | Status | Detail |
| ----- | ------ | ------ |
| **Engine** | ✅ Done | 2D front (max compression ratio, min log₁₀ PPL ratio) + 3D front (+ tok/s). Loads new `to_dict()` and legacy bundle JSON. |
| **Documentation** | ✅ Done | This section + `METHODOLOGY.md` §6.7. |
| **Paper** | 📝 ~90% done | Figure + discussion exist. **Pending:** regenerate from CLI + reproducibility citation. See [Figures: Pareto](#figures-offline-vs-online-pareto-l475-l608615). |

---

# Phase 10: Add Hardware-Aware Evaluation ✅ **Done**

> **Status (2026-08-19):** **Implementation complete** — single-GPU Modal A10G reference path with `eval/hardware/profile.py`, automatic peak VRAM + GPU util on Modal, hardware block in every job JSON, merge/reporter CSV columns. Multi-GPU tier matrix **not planned**. Paper table columns deferred to re-sweep.

Your current Apple MPS development environment is fine for building the engine.

But for an inference-engineering paper, add at least:

> **one NVIDIA CUDA experiment**

Preferably:

* A100
* H100
* RTX 4090/5090

You don't need a huge model.

A **1B–3B model is perfectly reasonable** if the goal is controlled inference-engineering evaluation. 

The important thing is to measure actual:

* latency
* memory
* throughput
* GPU execution behavior

### Scope decision (2026-08-19)

**In scope:** one **single-GPU** NVIDIA CUDA reference path — Modal `@app.function(gpu=…)` with `a10g` primary and `[a10g, l4, any]` fallbacks per [Modal GPU docs](https://modal.com/docs/guide/gpu). Each `eval_worker` job runs on **one** container GPU; peak VRAM and GPU utilization are collected automatically on Modal.

**Out of scope (not planned):** multi-GPU tier matrix (A100 / H100 / RTX 4090 side-by-side sweeps). To compare tiers later, rerun the same sweep after editing `configs/modal.yaml` `gpu` / `gpu_fallbacks` manually — no automated matrix runner.

### Implementation

| Component | Path | Role |
| --------- | ---- | ---- |
| Hardware profile | `eval/hardware/profile.py` | `HardwareProfile`, `collect_hardware_profile()`, `nvidia-smi` + torch CUDA props |
| Runner export | `eval/runner.py` | `EvaluationResult.hardware`; auto-enables peak VRAM + GPU util when `KV_COLLECT_HARDWARE_METRICS=1` or `KV_EXECUTION_PLATFORM=modal` |
| Modal image env | `modal_app/image.py` | `KV_EVAL_DEVICE=cuda`, `KV_EXECUTION_PLATFORM=modal`, `KV_HARDWARE_PROFILE`, `KV_COLLECT_HARDWARE_METRICS=1` |
| Modal worker | `modal_app/worker.py` | CUDA run with `--peak-memory` / `--gpu-utilization` equivalent flags; stamps `reference_gpu`, `modal_gpu` |
| Config | `configs/modal.yaml` | `hardware:` block documents single-GPU policy; `gpu: a10g` |
| Merge CSV | `modal_app/merge.py` | Flattened hardware + peak VRAM + GPU util columns |
| Reporter CSV | `reporting/reporter.py` | Hardware + SYSTEM VRAM/GPU util columns |
| CLI | `scripts/run_eval.py` | `--hardware-metrics` (local CUDA smoke) |
| Tests | `tests/test_hardware_profile.py` | Profile collection + env gating |

**Environment variables:**

| Variable | Purpose |
| -------- | ------- |
| `KV_EVAL_DEVICE=cuda` | Force CUDA device (Modal image) |
| `KV_EXECUTION_PLATFORM=modal` | Marks Modal reference sweeps; enables hardware metrics by default |
| `KV_HARDWARE_PROFILE=NVIDIA A10G` | Configured/reference GPU label in JSON |
| `KV_COLLECT_HARDWARE_METRICS=1` | Opt-in peak VRAM + GPU util locally |
| `MODAL_GPU_REQUEST` | Primary Modal `gpu=` request from config |

### Feasibility snapshot (paper vs code)

| | Paper | Code (now) | Gap |
| --- | ----- | ---------- | --- |
| **Proposed** | ≥1 NVIDIA CUDA run; ideally A100/H100/4090 | Single-GPU Modal `a10g` + SYSTEM + hardware export | Multi-GPU matrix **out of scope** |
| **Already in place** | ✅ A10G Modal sweeps (L222–229); tok/s, latency | ✅ `KV_EVAL_DEVICE=cuda`, worker, peak VRAM/GPU util auto-on Modal | Peak VRAM/GPU util **not in paper tables** |
| **Feasibility** | **High** for A10G (done); tier matrix **not planned** | — | Optional appendix columns at re-sweep |

### Paper change log — when, why, what

| When | Why | What to change in `conference_101719.tex` |
| ---- | --- | ----------------------------------------- |
| **At re-sweep** (Modal jobs with Phase 10 worker) | Code now collects peak VRAM + GPU util; paper tables omit them | Add `peak_vram_allocated_mb` and/or `gpu_util_mean_pct` to SYSTEM table or appendix for representative T=512 configs |
| **Setup §Experiments L222–229** | Document controlled hardware axis | One sentence: single NVIDIA A10G GPU per job (Modal); no multi-GPU tier comparison. Cross-ref Phase 7 controlled-conditions table. |
| **Do not** | Scope decision 2026-08-19 | Do **not** add A100/H100/4090 matrix claims or new GPU-tier experiments for this paper |
| **Optional (no new GPU jobs)** | Provenance only | Mention `hardware` block in reproducibility paragraph (`result.to_dict()["hardware"]`) |

**No further codebase changes required** for Phase 10 unless Modal GPU type is changed manually in `configs/modal.yaml`.

### Completeness record

| Track | Status | Detail |
| ----- | ------ | ------ |
| **Engine** | ✅ Done | Single-GPU Modal A10G path; hardware block + peak VRAM + GPU util collected/reported/merged. Multi-GPU matrix explicitly **not planned**. |
| **Documentation** | ✅ Done | This section + `METHODOLOGY.md` §6.8 + `configs/modal.yaml` comments. |
| **Paper** | 📝 ~80% done | A10G CUDA sweeps satisfy “one CUDA experiment.” **Pending:** VRAM/GPU util columns + setup sentence. See [§Experiments setup — hardware](#experiments-setup--hardware-block-l216229-extends-phase-7-table). |

---

# Phase 11: Add a Realistic Workload Dimension ⏸ **Future extension — not planned**

> **Status (2026-08-19):** **Out of scope for current paper and engine roadmap.** This section is **design reference only** — do **not** implement LongBench/RULER/C4 integration, expand sweep grids with new workload types, or add BEHAVIOR task results to the paper for this submission. The **current setup is sufficient:** WikiText-2 PPL + throughput in the paper; synthetic retrieval/instruction/reasoning in the engine (default/off) for legible failure modes without external benchmark dependency.

### Feasibility snapshot (why deferred)

| | Paper | Code | Decision |
| --- | ----- | ---- | -------- |
| **Proposed** | Beyond WikiText — short/long ctx, long output, retrieval, reasoning, IF | BEHAVIOR has PPL + **synthetic** retrieval, IF, reasoning (`eval/behavior/`) | **Engine ahead; paper intentionally narrower** |
| **Gap** | No retrieval/reasoning/IF numbers in paper | No LongBench/RULER | Acceptable — case study focuses on controlled PPL/tok/s under WikiText |
| **Feasibility if pursued later** | Medium — wire existing BEHAVIOR into sweeps + text | LongBench/RULER = more work | **Not scheduled** |

Don't rely entirely on WikiText-2.

The benchmark should eventually include different workload types:

### Short context

Tests normal inference.

### Long context

Tests the actual KV-cache bottleneck.

### Long-output generation

Tests decode-heavy workloads.

### Retrieval-heavy context

Tests whether important information survives compression.

### Reasoning

Tests whether compression behaves differently during long generation.

### Instruction following

Tests behavioral degradation.

Recent research strongly suggests workload characteristics matter.

### What stays as-is (no action)

- **Paper:** WikiText-2 PPL + SYSTEM throughput/latency only — correct scope for current case study.
- **Engine:** Default synthetic BEHAVIOR tasks remain available (`--skip-retrieval`, `--skip-instruction-following`, `--reasoning`) but **not required** in Modal sweeps or paper tables.
- **Paper rewrite (Phases 1–7):** BEHAVIOR protocol paragraphs in [§Evaluation Protocol](#evaluation-protocol-l255285-labelsubseceval_protocol) may **describe** synthetic tasks in methodology without reporting numbers — optional, not blocking.

### Paper change log

| When | Why | What |
| ---- | --- | ---- |
| **Current submission** | Phase 11 deferred | **No paper changes** for workload expansion |
| **Future extension only** | If external benchmarks added later | New §Workload types + result tables; cite LongBench/RULER — requires new experiments |

### Completeness record

| Track | Status | Detail |
| ----- | ------ | ------ |
| **Engine** | ⏸ Sufficient | Synthetic BEHAVIOR modules exist; no LongBench/RULER wiring planned. |
| **Documentation** | ⏸ Flagged | This section + `METHODOLOGY.md` §6.2 + `CURRENT_STATE.md` limits. |
| **Paper** | — | WikiText-only results **intentional**; mention diverse workloads only in Conclusion **future work** (see [§Conclusion](#conclusion-l625629-labelsecconclusion)). |

---

# Phase 12: Add Workload Scaling ⏸ **Future extension — not planned**

> **Status (2026-08-19):** **Out of scope.** Do **not** implement 2K–32K context sweeps, batch/concurrency grids, or generation-length matrices for this paper. Context **128 / 256 / 512**, batch **1**, **64** generated tokens (`configs/model.yaml`, `configs/eval.yaml`) are the controlled case-study grid — explicitly noted as future work in `METHODOLOGY.md` §9 and `ROADMAP.md`.

Ideally evaluate across:

```text
Context length:
2K → 4K → 8K → 16K → 32K

Batch/concurrency:
1 → 2 → 4 → 8 ...

Generation:
short → medium → long
```

You don't need every combination.

Even a small matrix would demonstrate:

> **The best KV method depends on the workload.**

### Feasibility snapshot (why deferred)

| | Paper | Code | Decision |
| --- | ----- | ---- | -------- |
| **Proposed** | ctx 2K→32K, batch 1→8, short→long generation | Runner accepts `context_length` / `generated_tokens`; configs cap at 512 ctx | **Scaling study = future work** |
| **Already in place** | ctx 128–512, batch 1, 64 tok gen | Same limits | Sufficient for SLM controlled eval |
| **Feasibility if pursued later** | Medium–hard — VRAM/runtime/cost grow fast | Engine can run longer ctx in principle | **Not scheduled** |

### What stays as-is (no action)

- **No codebase changes** — `configs/model.yaml` context list and `generated_tokens: 64` remain the paper-aligned grid.
- **No paper changes** — do not claim long-context or batch scaling results.

### Paper change log

| When | Why | What |
| ---- | --- | ---- |
| **Current submission** | Phase 12 deferred | **No paper changes** |
| **Conclusion future work only** | Acknowledge limitation | One sentence: ctx ≤512, batch 1; long-context scaling tracked in `ROADMAP.md` |
| **Future extension** | If 4K+ sweeps run | New context-length rows + VRAM/latency scaling figures — requires new Modal budget |

### Completeness record

| Track | Status | Detail |
| ----- | ------ | ------ |
| **Engine** | ⏸ Sufficient | Parameterized runner; no 2K+ sweep automation planned. |
| **Documentation** | ⏸ Flagged | `METHODOLOGY.md` §8–9; `ROADMAP.md` long-context item. |
| **Paper** | — | Current 128–512 grid **intentional**; no scaling claims beyond reported ctx. |

---

# Phase 13: Add a Serving-Engine Validation Path ⏸ **Not planned**

> **Status (2026-08-20):** **Out of scope.** Do **not** implement vLLM, SGLang, or other serving-engine integrations for this paper. KVBench's controlled interception environment (`KVCacheEngine`, plug-in compressors, three-branch eval) is the systems contribution — validating inside a production serving stack is **future work only**, not required for current claims or paper framing.

You don't need to turn KVBench into vLLM.

Instead:

```text
KVBench
   │
   ├── Controlled research environment   ← current scope (sufficient)
   │
   └── Optional serving integration      ← NOT planned (Phase 13)
           │
           ├── vLLM
           └── SGLang
```

The idea is:

> First establish controlled results inside KVBench, then validate selected findings inside a real serving engine.

That validation path would strengthen a **future** systems claim; it is **not** a blocker for the current submission.

The recent literature increasingly connects compression to actual serving systems and memory-management architectures.

### What stays as-is (no action)

- **Engine:** No vLLM/SGLang adapters, no serving endpoints — `modal_app/worker.py` one-shot eval jobs remain the CUDA reference path.
- **Paper:** Do not claim serving-engine validation; controlled KVBench results are the empirical basis.

### Paper change log

| When | Why | What |
| ---- | --- | ---- |
| **Current submission** | Phase 13 deferred | **No paper changes** |
| **Conclusion future work only** | Acknowledge optional extension | One sentence: selected findings could be validated in vLLM/SGLang in future work |

### Completeness record

| Track | Status | Detail |
| ----- | ------ | ------ |
| **Engine** | ⏸ Not planned | No serving integration; controlled eval path complete (Phases 1–10). |
| **Documentation** | ⏸ Flagged | This section retained as **design reference only**. |
| **Paper** | — | No serving-engine claims; mention only in future work if at all. |

---

# Phase 14: Add Reproducibility as a First-Class Feature ✅ **Done**

> **Status (2026-08-20):** **Implementation complete** — every evaluation job exports a standardized reproducibility harness via `controlled_conditions` (Phase 7), `cost` (Phase 3 calibration block), `hardware` (Phase 10), version-controlled YAML configs, and `docs/reproducibility/REPRODUCIBILITY.md`. **Paper update deferred** — `.tex` still describes hyperparameters in prose only; controlled-conditions table + artifact paragraph pending.

Your benchmark should record a standardized configuration:

```yaml
model:
context_length:
generation_length:
hardware:
batch_size:
compression_method:
compression_ratio:
calibration:
dataset:
seed:
precision:
```

Every result should be reproducible from this configuration.

### Engine encapsulation — field mapping

The Phase 14 YAML checklist is **not a separate file format**; it is assembled from existing exports and configs:

| Phase 14 field | Engine source | JSON / config path |
| -------------- | ------------- | ------------------ |
| `model` | `get_model_eval_metadata()` | `controlled_conditions.fixed.model` · top-level `model` in `to_dict()` |
| `context_length` | runner arg | `controlled_conditions.fixed.context_length` · `context_length` |
| `generation_length` | `configs/eval.yaml` `generated_tokens` | `controlled_conditions.fixed.generation_length` |
| `hardware` | Phase 10 profile | `controlled_conditions.fixed.hardware` · `hardware` |
| `batch_size` | `configs/eval.yaml` | `controlled_conditions.fixed.batch_size` |
| `compression_method` | compressor plug-in | `controlled_conditions.variable.compressor` · `compression_budget.compression_method` |
| `compression_ratio` | measured + theoretical | **Outcome:** `fidelity.memory.compression_ratio` · **Theoretical:** `cost.compression.theoretical_compression_ratio` |
| `calibration` | compressor offline hooks | `cost.offline` (`calibration_required`, `calibration_dataset`, `calibration_time_ms`, …) |
| `dataset` | WikiText-2 config | `controlled_conditions.fixed.dataset` |
| `seed` | compressor pipeline (QJL/TurboQuant) | `controlled_conditions.variable.compression_budget.seed` · Modal `job.compressor_kwargs` |
| `precision` | `ModelLayer.torch_dtype` | `controlled_conditions.fixed.precision` (e.g. `float16`) |

**Version-controlled sweep grid (source of truth before run):**

| File | Role |
| ---- | ---- |
| `configs/model.yaml` | Model path, context lengths |
| `configs/eval.yaml` | Dataset split, PPL stride, generated tokens, batch size |
| `configs/modal_sweeps.yaml` | Compressor presets, bitwidths, budgets, seeds |
| `configs/modal.yaml` | GPU, hardware collection policy |

**Artifacts per job:** `result.to_dict()` JSON (local or Modal volume) includes `controlled_conditions`, `cost`, `hardware`, `fidelity`/`behavior`/`system`, timestamps; Modal payloads add `job`, `started_at`, `finished_at`. Merge: `modal_app/merge.py` CSV.

**Manual step (not auto-exported):** record `git rev-parse HEAD` when publishing numbers (`REPRODUCIBILITY.md` §2).

**Code:** `eval/controlled_conditions.py` (`build_controlled_conditions`, `extract_compression_budget`, `format_model_precision`) · `eval/runner.py` · `docs/reproducibility/REPRODUCIBILITY.md` · **Tests:** `tests/test_controlled_conditions.py`

This is particularly important because one of the central problems in the literature is that different papers use different:

* models
* tasks
* budgets
* serving stacks

making direct comparison difficult.

### Paper change log — when, why, what (`docs/research_paper_writeup/conference_101719.tex`)

| When | Section (label) | Why | What to change |
| ---- | --------------- | --- | -------------- |
| **Paper rewrite** (no new experiments required for prose) | **§Experimental Configuration** (`\label{subsec:exp_config}`, L220–229) | Paper lists setup in bullets but not as a reproducibility contract | Add **Table: Standardized run configuration** (model, dataset/split, ctx lengths, gen length 64, batch 1, FP16, greedy decode, A10G, PPL stride 512). Sentence: *“Every job JSON includes `controlled_conditions` (fixed vs. variable axes) enabling exact replay.”* |
| **Same pass** | **§Design Principles** (`\label{subsec:design}`, L88–96) | Reproducibility is a design principle, not an afterthought | New bullet: **Reproducible configuration export** — fixed model/input/decode/hardware vs. variable compressor + budget; per-job JSON + YAML configs in repo. |
| **Same pass** | **§Evaluation Protocol** (`\label{subsec:eval_protocol}`, L255+) | Readers need to know what is held constant | Short paragraph: only `compression_method` / budget varies; cite `variable.compression_budget` fields (bitwidth, stage, RocketKV budgets, QJL seed 42). |
| **Same pass** | **§Plug-in Interface** (`\label{subsec:plugins}`, L153–155) | Calibration differs by method | Note TurboQuant offline Lloyd-Max vs QJL/RocketKV calibration-free; point to `cost.offline` block. |
| **At re-sweep or artifact submission** | **NEW §Reproducibility** (insert after `\label{subsec:exp_config}` or appendix) | Artifact reviewers expect explicit availability statement | Bullets: repo URL, config files, Modal sweep scripts, `results/` bundle layout, `git` SHA pin, CLI reproduce commands from `REPRODUCIBILITY.md`. |
| **Abstract / Conclusion** (L44–46, L625–629) | Framing | Claim reproducibility as contribution | One clause: *“standardized per-job configuration export”* alongside controlled interception. |
| **Do not** | Results tables (L319+) | Numbers unchanged | Do **not** block rewrite on new sweeps — configuration table is structural |

### Completeness record

| Track | Status | Detail |
| ----- | ------ | ------ |
| **Engine** | ✅ Done | Phase 14 YAML fields mapped to `controlled_conditions` + `cost` + `hardware` + configs; `precision` and `seed` in export. |
| **Documentation** | ✅ Done | This section + `REPRODUCIBILITY.md` + `METHODOLOGY.md` §1.1 cross-ref. |
| **Paper** | 📝 Pending | Controlled-conditions table + reproducibility subsection + artifact paragraph. See table above. |

---

# Phase 15: Redefine the Main Research Question

This is probably the **single most important conceptual change**.

### Current implicit question:

> **Which KV-cache compression method performs best?**

Change it to:

> **How should KV-cache transformations be evaluated under controlled and realistic inference conditions?**

Then:

> **KVBench is the instrument for answering that question.**

This turns your work from:

**"another KV compression comparison"**

into:

**"an inference-aware methodology for evaluating KV transformations."**



---

# Phase 16: Reframe the Core Problem Statement

The new problem should be:

```text
Existing KV-cache research
        ↓
Many different algorithms
        ↓
Different implementations
Different models
Different workloads
Different metrics
Different hardware
        ↓
Results are difficult to compare
        ↓
Compression ratio ≠ memory savings
Memory savings ≠ speedup
Tensor fidelity ≠ behavior
Offline quality ≠ online quality
        ↓
Need controlled evaluation
```

This is the heart of the revised paper.

---

# Phase 17: Reframe the Novelty

Do **not** claim:

> "KV-cache compression has never been benchmarked."

That is now difficult to defend because 2026 work explicitly benchmarks KV optimizations across quality and system performance. 

Instead claim something closer to:

> **Existing KV-cache studies evaluate individual compression mechanisms under heterogeneous implementations and experimental conditions. KVBench provides a controlled interception-and-transformation environment in which different KV transformations can be executed through a common incremental autoregressive decode loop, enabling representation-level, behavioral, and system-level comparisons under matched conditions.**



This is a much safer novelty claim.

---

# Phase 18: Clarify Exactly What KVBench Is

I would position it as:

> **A unified KV-cache inference benchmarking and transformation framework**

or:

> **An extensible inference-time KV-cache compression evaluation engine**

or:

> **A modular KV-cache inference optimization and benchmarking framework**

I would **not** call it a full "inference engine."

vLLM/SGLang are serving engines.

KVBench is an **inference-time KV transformation and evaluation layer**. 

---

# Phase 19: Reframe the Domain Positioning

Your research belongs under:

```text
LLM Systems
    ↓
LLM Inference
    ↓
KV-Cache Optimization
    ↓
Compression / Transformation
    ↓
Inference Evaluation Infrastructure
    ↓
KVBench
```

So the paper is legitimately an:

> **LLM inference-systems / inference-engineering paper**

with a specific focus on **KV-cache optimization and evaluation**.

---

# Phase 20: Completely Restructure Related Work

I recommend four sections.

## 1. KV-Cache Eviction

Discuss:

* H2O
* Scissorhands
* StreamingLLM
* SnapKV
* PyramidKV
* Ada-KV

---

## 2. KV Representation Compression

Discuss:

* MiniCache
* QJL
* Palu
* Outlier Tokens
* KVSink
* AsymKV
* XQuant
* TurboQuant

---

## 3. Architecture- and Serving-Aware KV Optimization

Discuss:

* MHA → GQA
* PagedEviction
* HqeKV
* RocketKV
* HybridKV
* potentially CompressKV

---

## 4. KV-Cache Evaluation and Benchmarking

This is the **new critical section**.

Discuss:

* Oaken
* SCOPE
* The Pitfalls of KV Cache Compression
* Benchmarking KV-Cache Optimizations...
* relevant serving/workload studies
* CacheBlend
* KVCache Cache in the Wild

Then end the section with:

> **What is still missing?**

And introduce KVBench.

This creates the logical chain:

```text
Many compression techniques
          ↓
Fragmented evaluation
          ↓
Behavioral failures discovered
          ↓
Serving/runtime effects discovered
          ↓
Need controlled evaluation
          ↓
KVBench
```



---

# Phase 21: Rewrite the Introduction Around This Story

The Introduction should no longer primarily say:

> KV cache is large → compression is useful → we implemented several methods.

Instead:

### Paragraph 1

KV cache is a major LLM inference bottleneck.

### Paragraph 2

Many compression/transformation approaches now exist.

### Paragraph 3

However, **compression ratio is not equivalent to inference benefit**.

### Paragraph 4

Recent work demonstrates:

* behavioral degradation
* workload dependence
* hardware/runtime effects
* serving-specific effects

### Paragraph 5

Therefore, evaluating KV transformations requires a **controlled, multidimensional inference environment**.

### Paragraph 6

Introduce KVBench.

### Paragraph 7

State contributions.

This would make the paper feel much more like a **research methodology/system paper**.

---

# Phase 22: Rewrite the Contributions

I would aim for contributions roughly like:

### Contribution 1

**A unified KV interception and transformation framework** that allows heterogeneous KV optimization methods to operate inside the same autoregressive inference loop.

### Contribution 2

**A multidimensional evaluation methodology** separating representation fidelity, behavioral quality, and system performance.

### Contribution 3

**A controlled comparison protocol** that normalizes model, workload, decode configuration, hardware and compression budget.

### Contribution 4

**A cross-method empirical study** revealing where compression ratio, fidelity, behavioral quality and runtime efficiency diverge.

### Contribution 5

**An extensible plugin architecture** supporting quantization, eviction, projection and adaptive/stateful KV transformations.

Only claim the ones your experiments actually support.

---

# Phase 23: Change the Results Narrative

Don't write:

> TurboQuant achieved the lowest reconstruction error.

Then:

> QJL achieved X throughput.

Instead ask:

### Finding 1

Does better KV reconstruction imply better model quality?

### Finding 2

Does higher compression imply lower memory?

### Finding 3

Does lower memory imply higher throughput?

### Finding 4

Does offline quality predict online generation quality?

### Finding 5

Does the best method change with context length?

### Finding 6

Does the best method change with workload?

### Finding 7

What is the actual quality-memory-speed Pareto frontier?

This transforms your results from a **leaderboard** into **research findings**.

---

# Phase 24: Add Cross-Dimensional Analysis

This is particularly valuable.

Analyze correlations such as:

```text
Compression Ratio ↔ Memory Reduction
Compression Ratio ↔ PPL
Reconstruction Error ↔ PPL
Reconstruction Error ↔ Task Accuracy
Memory Reduction ↔ Throughput
Online Overhead ↔ Throughput
```

Then ask:

> Which metrics actually predict real inference performance?

This could become one of the most interesting empirical contributions of the paper.

---

# Phase 25: Add a "Compression Trade-off" Figure

A central figure could show:

```text
                 QUALITY
                    ↑
                    │
                    │
             ● A    │
                    │       ● B
                    │
        ● C         │
                    │
                    └────────────────→
                         MEMORY / SPEED
```

Or a 3D/paired plot showing:

> **Quality ↔ Memory ↔ Speed**

This visually communicates the central thesis much better than another large table.

---

# Phase 26: Add an Explicit "Offline Does Not Mean Cheap" Discussion

This is an important refinement from Oaken.

Your current offline/online terminology can accidentally imply:

> offline = free

Instead distinguish:

### Offline evaluation

What happens **before/around inference**.

### Offline preprocessing cost

What the method itself needs to calculate.

### Online transformation cost

What happens during generation.

### Online attention cost

What the transformed cache does to actual attention execution.

### End-to-end serving cost

What the user actually experiences.

This is a much more rigorous interpretation of "offline vs online." 

---

# Phase 27: Add Calibration as a Benchmark Dimension

Every method should report:

| Property             | Example    |
| -------------------- | ---------- |
| Calibration required | Yes/No     |
| Calibration data     | WikiText-2 |
| Calibration tokens   | X          |
| Calibration time     | X sec      |
| Calibration memory   | X GB       |
| Stateful             | Yes/No     |
| Online overhead      | X ms/token |

This makes comparisons much fairer. 

---

# Phase 28: Add a Workload-Aware Discussion

Recent systems research suggests real KV workloads are not uniform.

Therefore discuss:

> KV reuse/compression behavior can depend heavily on workload characteristics.

This gives you justification for eventually including:

* long-context prompts
* RAG
* multi-turn conversations
* long generation
* reasoning

rather than treating WikiText-2 as representative of all inference.



---

# Phase 29: Use Recent Literature to Strengthen, Not Replace, Your Existing Work

The important papers I would now explicitly incorporate into the story are:

### Highest priority

1. **Oaken — ISCA 2025**

   * offline/online hybrid
   * hardware/runtime cost

2. **SCOPE — ACL 2025**

   * long-context generation
   * prefill/decode evaluation

3. **RocketKV — ICML 2025**

   * compression + actual attention execution

4. **TurboAttention — MLSys 2025**

   * compression must account for attention execution

5. **R-KV — NeurIPS 2025**

   * reasoning workloads

6. **The Pitfalls of KV Cache Compression — ACL 2026**

   * behavioral evaluation

7. **OjaKV — ACL 2026**

   * stateful/online adaptation

8. **HybridKV — ACL 2026**

   * heterogeneous adaptive compression

9. **Benchmarking KV-Cache Optimizations... — 2026**

   * closest competing benchmark

10. **KVCache Cache in the Wild — USENIX ATC 2025**

* workload realism

11. **CacheBlend — EuroSys 2025**

* serving/RAG workload

These aren't all things you need to implement. Some are there to **shape your methodology and positioning**.

---

# Phase 30: Explicitly Address the Closest Competing Benchmark

This is critical.

The 2026:

> **Benchmarking KV-Cache Optimizations Across Task Quality and System Performance for Long-Context Serving**

is close enough to KVBench that you must explicitly discuss it.

Don't hide it.

Instead say, in substance:

> That work evaluates existing KV optimizations across workloads and system-level metrics. KVBench differs by providing a controlled KV interception/transformation layer in which different transformations execute through a common incremental autoregressive decode path, allowing matched representation-level, behavioral, and runtime analysis.

That distinction should appear in **Related Work and Introduction**, not just in a footnote.



---

# Phase 31: Clean the Bibliography

### Definitely keep

* H2O
* Scissorhands
* StreamingLLM
* SnapKV
* PyramidKV
* MiniCache
* QJL
* Palu
* Outlier Tokens
* KVSink
* AsymKV
* XQuant
* Qwen3
* OLMo 2
* TurboQuant
* HqeKV
* The Pitfalls
* KVBench Serving
* WikiText

### Verify/fix

* Ada-KV
* KVSink venue
* RocketKV metadata
* CompressKV
* PatternKV
* MHA→GQA

### Remove unless specifically needed

* anonymous Cost-Optimal GQA
* QJL-CS anonymous preprint
* Expected Attention anonymous/unverified
* Short-RL

And fix the **SnapKV venue from ACL to NeurIPS 2024**.



---

# Phase 32: The Final Conceptual Model of KVBench

After all these changes, I would want the paper to communicate this:

```text
                         KVBench
                            │
                 KV INTERCEPTION LAYER
                            │
                    Original KV Cache
                            │
             ┌──────────────┼──────────────┐
             │              │              │
          Eviction      Quantization    Projection
             │              │              │
             └──────────────┼──────────────┘
                            │
                     Transformed KV
                            │
                    Same Decode Loop
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
     FIDELITY            BEHAVIOR             SYSTEM
        │                   │                   │
    Reconstruction       PPL                 TTFT
    Attention            Retrieval           ITL
    Similarity           Instruction         Throughput
    Memory               Reasoning           VRAM
                        Robustness           Runtime
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ↓
                 QUALITY / MEMORY / SPEED
                            ↓
                  Pareto + Workload Analysis
                            ↓
                    Research Findings
```

That is the **new KVBench story** I would aim for.

---

# Phase 33: The new paper narrative in one sentence

If I had to reduce the entire revision to one sentence:

> **KVBench is not primarily a benchmark asking which KV-cache compression method wins; it is a controlled inference-time experimentation framework for understanding how different KV transformations trade representation fidelity, model behavior, memory efficiency, and actual generation performance under matched conditions.**

That is a substantially stronger research identity.

---

## What I would consider the priority order

If you cannot implement everything, do it in this order:

### 🔴 Must do

1. **Redesign evaluation into Fidelity / Behavior / System.**
2. **Add explicit online/offline cost accounting.**
3. **Add at least one realistic task-level evaluation.**
4. **Add actual end-to-end latency/VRAM measurements.**
5. **Reframe novelty around controlled KV interception and evaluation.**
6. **Rewrite Introduction around the evaluation problem.**
7. **Rewrite Related Work into the four categories.**
8. **Explicitly differentiate KVBench from the 2026 benchmarking paper.**

### 🟠 Strongly recommended

9. Add CUDA/NVIDIA evaluation.
10. Add multiple compression budgets.
11. Add Pareto analysis.
12. Add calibration accounting.
13. Support layer/head/token/stateful plugins.
14. Add long-context workload variation.
15. Add reproducibility configuration.

### 🟡 Good extensions

16. vLLM/SGLang validation.
17. RAG workload.
18. reasoning workload.
19. safety/robustness evaluation.
20. real serving traces.

The important thing is **not to implement all 20 blindly**. The first eight are the changes that most fundamentally improve the scientific story. The remaining items can be layered on depending on time and compute.
