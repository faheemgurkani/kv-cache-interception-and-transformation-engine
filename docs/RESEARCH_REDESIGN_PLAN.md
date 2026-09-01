# KVBench: Complete Research Improvement Roadmap

## Completeness record — Phases 1–33 (verified 2026-08-20)

Tracks **engine** (code + tests), **documentation** (in-repo docs), and **paper** (`docs/research_paper_writeup/conference_101719.tex`). Phases **5**, **8**, **11**, **12**, and **13** are flagged **not planned / future extension** — design reference only.

**Executive verdict:** Phases **1–4**, **6**, **7**, **9**, **10**, **14**, **24**, **25**, **26**, **27**, **31** (audit tooling), and **32** (conceptual model docs) are **complete in the engine and documentation** (re-audited 2026-08-20). **Phases 28–30**, **33** are **paper-only** framing. **Phases 15–23**, **29–30**, **33** have per-phase **Paper change log** subsections. **Phase 31** bib cleanup is **paper-pending** but **`scripts/audit_bibliography.py`** + tests enforce invariants. The paper still uses Section A/B naming, lacks Oaken/calibration/workload prose (Phases 26–28), serving-benchmark contrast (Phase 30), and updated pipeline figure (Phase 32). **Paper changes documented only** in this file — apply at rewrite. Phases **5**, **8**, **11**, **12**, **13** require **no work** for current scope.

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
| **14** Reproducibility harness | ✅ Done | ✅ Done | 📝 Pending | `eval/reproducibility/manifest.py`, `tests/test_reproducibility_harness.py` |
| **15** Main research question | — | 📝 Spec only | 📝 Pending | Paper framing only — see Phase 15 paper change log |
| **16** Core problem statement | — | 📝 Spec only | 📝 Pending | Problem cascade + metric decoupling — Phase 16 |
| **17** Novelty reframe | — | 📝 Spec only | 📝 Pending | Safe novelty claim — Phase 17 |
| **18** What KVBench is | — | 📝 Spec only | 📝 Pending | Terminology boundary (not vLLM/serving) — Phase 18 |
| **19** Domain positioning | — | 📝 Spec only | 📝 Pending | LLM inference-systems / KV eval infrastructure — Phase 19 |
| **20** Related Work restructure | — | 📝 Spec only | 📝 Pending | Four-section Related Work + eval gap — Phase 20 |
| **21** Introduction narrative | — | 📝 Spec only | 📝 Pending | Seven-paragraph Intro story — Phase 21 |
| **22** Contributions rewrite | — | 📝 Spec only | 📝 Pending | Five-contribution taxonomy + Intro packaging — Phase 22 |
| **23** Results narrative | — | 📝 Spec only | 📝 Pending | Seven research findings vs leaderboard — Phase 23 |
| **24** Cross-dimensional analysis | ✅ Done | ✅ Done | 📝 Pending | `eval/cross_dim/`, `scripts/analyze_cross_dim.py` |
| **25** Trade-off figures | ✅ Done | ✅ Done | 📝 Pending | `plot_tradeoff_*.pdf` + Phase 9 Pareto — Phase 25 |
| **26** Oaken cost taxonomy | ✅ Done | ✅ Done | 📝 Pending | `eval/cost/oaken_taxonomy.py`, `cost.oaken_layers` export |
| **27** Calibration dimensions | ✅ Done | ✅ Done | 📝 Pending | `cost.benchmark_dimensions`, `export_method_benchmark_table.py` |
| **28** Workload-aware discussion | ⏸ Paper only | ✅ Scoped | 📝 Pending | WikiText scope + Phase 11 deferred — Phase 28 |
| **29** Recent literature map | — | ✅ Done | 📝 Pending | `docs/literature/LITERATURE_ALIGNMENT.md`, `staging_entries.bib` |
| **30** Serving benchmark contrast | — | ✅ Done | 📝 Pending | Explicit `kvbench2026serving` differentiation — Phase 30 |
| **31** Bibliography cleanup | ✅ Audit CLI | ✅ Done | 📝 Pending | `scripts/audit_bibliography.py`, `tests/test_bibliography_audit.py` |
| **32** Conceptual model | ✅ Pipeline | ✅ Done | 📝 Pending | `SYSTEM_DESIGN.md` §Phase 32; regen pipeline figure |
| **33** One-sentence identity | — | ✅ Done | 📝 Pending | Abstract + Conclusion canonical sentence — Phase 33 |
| **11** Realistic workload dimension | ⏸ Future extension | ⏸ Flagged | — | Current WikiText + default BEHAVIOR scope sufficient |
| **12** Workload scaling (2K–32K, batch) | ⏸ Future extension | ⏸ Flagged | — | ctx 128–512 / batch 1 / 64 tok gen sufficient for paper |
| **13** Serving-engine validation (vLLM/SGLang) | ⏸ Not planned | ⏸ Flagged | — | Controlled KVBench path sufficient; no vLLM/SGLang integration |

**Cross-cutting tests:** `tests/test_eval_runner.py`, `tests/test_controlled_conditions.py`, `tests/test_reproducibility_harness.py`, `tests/test_cost_accounting.py`, `tests/test_oaken_benchmark_dimensions.py`, `tests/test_taxonomy.py`, `tests/test_pareto_analysis.py`, `tests/test_cross_dim_analysis.py`, `tests/test_hardware_profile.py`, `tests/test_modal_merge_hardware.py`, `tests/test_behavior_modules.py`, `tests/test_system_modules.py`, `tests/test_bibliography_audit.py`, `tests/test_*_reference.py`.

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

**When to apply:** After the next evaluation sweep completes (new job JSON/CSV bundles under `results/`). Order of work: (1) run experiments → (2) update result tables/figures from bundles **including Oaken-style cost fields** (see [Post-sweep reminder — Oaken cost](#post-sweep-reminder--oaken-cost-phases-3-26-27)) → (3) apply framing/terminology changes below → (4) compile PDF.

### Post-sweep reminder — Oaken cost (Phases 3, 26, 27)

**Do not skip when compiling results or editing the paper.** Oaken (ISCA 2025) motivates separating **offline preprocessing cost** (calibration, codebooks) from **online per-token cost** (compress/decompress during decode). KVBench already collects this in the engine — the paper does not report it yet.

| What | Status | Action at paper/results time |
| ---- | ------ | ------------------------------ |
| **Offline eval (FIDELITY)** | ✅ Every job | Do **not** conflate with “free” offline work — FIDELITY is quality measurement, not deployment cost |
| **Offline preprocessing** | ✅ `cost.offline` | Report TurboQuant calibration (`calibration_required`, `calibration_time_ms`, …) vs QJL/RocketKV calibration-free |
| **Online transformation** | ⚠️ Partial in standard sweeps | `cost.online` populated; **compress/decompress/attention split** needs `--kernel-cost` (opt-in; not default on Modal) |
| **End-to-end decode** | ✅ Every job | Use `system.throughput.ttft_ms`, `itl_ms_*`, `end_to_end_latency_ms` + `cost.online.end_to_end_decode_cost_ms` |
| **Oaken five-layer taxonomy** | ✅ `cost.oaken_layers` | Cite layers in new **§COST** subsection (rewrite step 11); Discussion **F3** |
| **Phase 27 calibration table** | ✅ Export CLI | Run `python scripts/export_method_benchmark_table.py` → `method_benchmark_dimensions.csv` for appendix/table |
| **Oaken hardware / serving stack** | ❌ Out of scope | Cite in Related Work §4 only — methodology alignment, not replication |

**From existing sweep JSON (no re-run required):** compare methods on `cost` + `oaken_layers`, calibration flags in `benchmark_dimensions`, and TTFT/ITL/end-to-end latency.

**Optional follow-up (subset re-run):** add `--kernel-cost` locally or to Modal worker for full online compress / decompress / attention breakdown before final tables.

**Paper targets:** cite `oaken2025`; add **§COST** after SYSTEM; Related Work evaluation gap; practitioner checklist — report offline preprocessing separately from FIDELITY (Phase 26 rewrite step 11).

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
| Primary contribution | “Benchmarking framework” / “dual metrics” | Controlled **interception-and-transformation engine** | Engine + protocol as contribution; methods are case studies (**Phases 15, 17, 18**) |
| Problem statement | Fragmentation mentioned in Intro | Metric decoupling cascade (**Phase 16**) | Explicit problem paragraph + optional figure |
| Novelty | Implicit “shared yardstick” | Controlled env + three branches under matched conditions (**Phase 17**) | Do not claim “first benchmark” |
| What KVBench is | “Benchmarking framework” / “harness” | Evaluation layer at KV boundary; not serving engine (**Phase 18**) | Define once; contrast vLLM/SGLang in one sentence |
| Domain / venue | Implicit SLM compression comparison | **LLM inference-systems** paper; KV eval infrastructure (**Phase 19**) | Keywords + Intro ladder; scope Conclusion to SLM inference engineering |
| Introduction narrative | Bottleneck + method dump + Section A/B + horse-race contributions (**Phase 21**) | Seven-paragraph methodology story; FIDELITY/BEHAVIOR/SYSTEM; instrument framing | Rewrite L52–60 per Phase 21 before Related Work |
| Contributions block | (1) dual Section A/B engine; (2) 27-job study co-primary; (3) offline≠online (**Phase 22**) | Protocol-first: engine + protocol + controlled export; demonstrations second; safe scope | Rewrite L60 per Phase 22 taxonomy; mirror Abstract + Conclusion |
| Results narrative | Method-by-method leaderboard (“TurboQuant achieved…”) (**Phase 23**) | Seven research findings (F1–F7); tables as evidence; F6 future work only | Experiments L215–218 + Discussion L595–623 restructure; step 9 |
| Related Work structure | Algorithm-family subsections + standalone Positioning (**Phase 20**) | Four sections: Eviction / Representation / Arch-Serving / **Evaluation** + *What is still missing?* | Restructure L62–81; add Oaken/SCOPE/CacheBlend/Cache-in-the-Wild bibs |
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
| **Change** | Consider: *KVBench: A Controlled KV Interception Engine for Fidelity, Behavior, and System Evaluation of Cache Compression in SLMs* (or keep “Bridging…” with subtitle mentioning three branches). **Phase 15:** de-emphasize “comparing methods” in title if space allows — foreground *evaluation* / *controlled* / *instrument*. |
| **Needs new results?** | No — framing only |
| **Phase** | 1, 6 |

#### Abstract (L44–46)

| | |
| --- | --- |
| **Current** | “benchmarking framework”; “dual metrics: Section A … Section B”; offline does not predict online |
| **Codebase** | FIDELITY/BEHAVIOR/SYSTEM; controlled interception; cost + taxonomy exist in code |
| **Change** | Replace “dual Section A/B” with three-branch names. Lead with “controlled interception engine.” **Phase 15:** frame KVBench as evaluation **instrument**; case studies demonstrate branch divergence — close with “methodology for evaluating KV transformations,” not “yardstick for comparing methods.” Keep empirical claims until re-sweep replaces numbers. |
| **Needs new results?** | Numbers: yes when re-sweeping; framing: no |
| **Phase** | 1, 6, 7, 15 |

#### Keywords (L48–50)

| | |
| --- | --- |
| **Current** | `offline fidelity, online inference` |
| **Change** | Add: `LLM inference systems`, `KV-cache optimization`, `inference engineering` (Phase 19). Also: `KV interception`, `fidelity evaluation`, `system metrics` (optional: `cost accounting`) |
| **Needs new results?** | No |
| **Phase** | 1, 19 |

#### Introduction (L52–60)

| | |
| --- | --- |
| **Current** | L58 “What: Section A … Section B”; L60 contributions (1) dual Section A/B protocol |
| **Codebase** | L58 already says “interception-and-transformation engine” — good. “What” axis is outdated. |
| **Change** | **Phase 21** seven-paragraph outline + **Phase 22** contribution bullets (see [Phase 22](#phase-22-rewrite-the-contributions--paper-only)). Merge L54–58 into ¶1–4; KVBench ¶6; contributions ¶7 = Phase 22 **minimal (3)** or **full (4–5)** packaging. **Phase 16** gap in ¶3–5. **Phase 18** boundary one sentence in ¶6. |
| **Needs new results?** | Framing no; empirical paragraph yes if models/methods change |
| **Phase** | 1, 2, 6, 15, 16, 18, 19, 21, 22 |

#### Related Work (L62–81)

| | |
| --- | --- |
| **Current** | “offline fidelity and online quality always reported together”; cites Palu/SnapKV in eviction/sketching subsections |
| **Codebase** | Palu/SnapKV implemented as plug-ins; not in paper results |
| **Change** | **Phase 20:** restructure into four subsections (see below). L66, L78, L81: FIDELITY/BEHAVIOR/SYSTEM terminology. **Phase 17:** canonical novelty in §4 closing. **Phase 19:** systems framing opening. Retire standalone `\subsection{Positioning of KVBench}` — fold into §4 *What is still missing?* |
| **Needs new results?** | SnapKV/Palu empirical claims: yes if sweeps run; Related Work restructure: no |
| **Phase** | 1, 4, 17, 19, 20 |

#### §Methodology opening (L83–86, `\label{sec:methodology}`)

| | |
| --- | --- |
| **Current** | “benchmarking harness”; “dual offline/online evaluation contract” |
| **Change** | “Controlled interception-and-transformation **evaluation environment**” (Phase 18); three-branch contract (Phase 1). Not a new compressor (Phase 17). |
| **Needs new results?** | No |
| **Phase** | 1, 6, 17, 18 |

#### §Design Principles (L88–96, `\label{subsec:design}`)

| | |
| --- | --- |
| **Current** | Bullet “Dual evaluation. Section A … Section B …” |
| **Codebase** | Three branches; plug-in isolation matches |
| **Change** | Add **Problem** bullet (Phase 16). **Three-branch evaluation** + **Controlled comparison** (Phases 1, 6, 7). |
| **Needs new results?** | No |
| **Phase** | 1, 6, 7, 16 |

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
| **Change** | Add **Table: Controlled experimental conditions** (model, tokenizer, dataset/split, ctx lengths, batch, gen length 64, PPL stride 512, greedy decode, A10G, metrics enabled). Replace “Section A and Section B” with three branches + cost. Add: “Per-job JSON includes `controlled_conditions` (fixed vs. variable axes).” **Phase 23:** opening sentence frames section as answering **Findings 1–7**, not reporting a leaderboard. |
| **Needs new results?** | Table structure: no; row values yes if setup changes |
| **Phase** | 7, 23 |

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
| — | compression / offline / online tree | Add subsection mirroring Phase 3 diagram + **Phase 26 Oaken five-layer taxonomy** + **Phase 27 benchmark dimensions table** (from `method_benchmark_dimensions.csv`) |

| **Needs new results?** | PPL/tok/s numbers yes; protocol structure no |
| **Phase** | 1, 2, 3, 26, 27 |

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
| **Current** | Method-by-method paragraphs (“TurboQuant results”, “QJL results”) read as **leaderboard**; Table captions “Section A fidelity”, “Section B online metrics/PPL” |
| **Change** | **Phase 23:** keep tables as **evidence appendix**; shorten per-method prose to 1–2 sentences each pointing to **finding numbers** (F1–F7). Rename captions: **FIDELITY**, **BEHAVIOR (perplexity)**, **SYSTEM (throughput/latency)**. Optionally add cross-method summary table up front (`tab:cross`) framed as “answers to evaluation questions,” not rankings. **Replace numeric cells only from new sweep bundles** — do not hand-edit. |
| **Needs new results?** | **Yes** for all numeric cells; narrative restructure **no** |
| **Phase** | 1, 2, 9, 23 |

#### Figures: offline-vs-online, Pareto (L475+, L608–615)

| | |
| --- | --- |
| **Current** | Axis labels “Section A” vs “Section B”; `plot_offline_vs_online.pdf`. **Pareto:** `plot_pareto.pdf` at T=512 — memory ratio vs log PPL ratio, marker area ∝ tok/s, empirical front (L475–481); cited in Discussion L617. |
| **Codebase** | `scripts/analyze_pareto.py`, `eval/pareto/analysis.py`, `ResultReporter.save_pareto()` — regenerates 2D/3D front from job bundles + writes `pareto_ctx512.json`. |
| **Change** | **Offline-vs-online figure:** regenerate with FIDELITY vs BEHAVIOR labels. **Pareto figure (Phase 9 / F7):** re-export via `scripts/analyze_pareto.py`. **Trade-off figure (Phase 25):** add `plot_tradeoff_ctx512.pdf` (Quality↔Memory + Quality↔Speed panels) via `scripts/analyze_cross_dim.py`. **Correlation appendix (Phase 24):** optional table from `correlations_ctx512.json`. Update captions with CLI provenance. |
| **When** | During paper rewrite pass **after** re-sweep bundles land (same timing as result tables). Can replot from existing Phase-5 JSON without new GPU jobs. |
| **Why** | Paper already demonstrates trade-off analysis; engine now automates Pareto + cross-dim export. |
| **Needs new results?** | **Only if** sweep grid/methods change; otherwise replot from existing JSON |
| **Phase** | 1, 2, 9, 23, 24, 25 |

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
| **Change** | **Phase 23:** restructure body as **seven research findings** (F1–F7) with `\textbf{Finding N: …?}` headers — see [Phase 23](#phase-23-change-the-results-narrative--paper-only). Migrate existing L608–621 content into finding blocks; retire method-centric paragraph titles. L598: contribution is **how to evaluate**, not ranking winners. **Phase 15:** opening sentence states research question. L617 Pareto = **F7** (“no single winner on all axes”). L623 implications: report FIDELITY + BEHAVIOR + SYSTEM + cost + controlled conditions. |
| **Needs new results?** | Empirical paragraphs yes; framing no |
| **Phase** | 1, 2, 3, 6, 15, 23 |

#### §Conclusion (L625–629, `\label{sec:conclusion}`)

| | |
| --- | --- |
| **Current** | “benchmarking framework”; “dual Section A/Section B metrics” |
| **Change** | Restate Phase 15 research question in first sentence. Enumerate **Phase 22** C1–C3 in prose (engine, three-branch protocol, controlled export). C4 as case-study **demonstrations** — not “findings = best method”. **Future work (Phases 11–12):** external benchmarks, long-context scaling — out of scope. **Last sentence:** “methodology for evaluating KV transformations,” not “yardstick for comparing methods.” |
| **Needs new results?** | Findings bullet yes; structure no |
| **Phase** | 1–10, 15, 22 (future work cites 11–12 only) |

### What to pull from codebase when writing

| Paper element | Source in repo |
| ------------- | -------------- |
| Controlled conditions table | Any `result.to_dict()["controlled_conditions"]["fixed"]` from a reference job |
| Phase 14 reproducibility manifest | Map fields via [Phase 14 field mapping](#engine-encapsulation--field-mapping) |
| FIDELITY metric definitions | `docs/methodology/METHODOLOGY.md` §6.1 |
| BEHAVIOR protocols | §6.2 + `eval/behavior/*.py` docstrings |
| SYSTEM metrics | §6.3 |
| Cost tree | §6.5 + `eval/cost/accounting.py` |
| Oaken five-layer snapshot | `result.to_dict()["cost"]["oaken_layers"]` from any job JSON (Phase 26) |
| Offline vs online cost fields | `cost.offline` / `cost.online` in merged sweep CSV/JSON (Phase 3); kernel split if `--kernel-cost` was used |
| Taxonomy table | `compressors/taxonomy.py` `METHOD_TAXONOMY` |
| Pareto optimal set | `python scripts/analyze_pareto.py … --context-length 512` → `pareto_ctx512.json` |
| Cross-dim correlations | `python scripts/analyze_cross_dim.py … --context-length 512` → `correlations_ctx512.json` |
| Trade-off figure | `results/cross_dim/plot_tradeoff_ctx512.pdf` (Phase 25); Pareto: `plot_pareto_ctx512.pdf` (Phase 9/F7) |
| Method benchmark table | `python scripts/export_method_benchmark_table.py` → `method_benchmark_dimensions.csv` (Phase 27) |
| Hardware + VRAM/GPU util | `result.to_dict()["hardware"]`, `system.peak_memory`, `system.gpu_utilization`; Modal merge CSV |
| Result numbers | `results/phase5_modal_*`, `results/olmo2_phase5_*` (or post-rewrite bundles) |

### Minimal vs full paper update (choose at rewrite time)

| Tier | When | Scope |
| ---- | ---- | ----- |
| **Minimal** | Re-sweep done; tight page limit | Terminology pass + controlled conditions table + **Phases 15–22 prose**; **Phase 23** findings structure in Experiments + Discussion; regenerate Pareto (Phase 9) |
| **Full** | Re-sweep + appendix space | Above + Cost subsection + taxonomy table + **SYSTEM VRAM/GPU util columns** (Phase 10) + optional BEHAVIOR protocol prose (Phase 11 — no new numbers) + optional **problem-cascade figure** (Phase 16) + reproducibility subsection (Phase 14) + **correlation table** from `correlations_ctx512.json` (Phase 24) + **trade-off figure** (Phase 25) + **Oaken + calibration table** (Phases 26–27, rewrite step 11) + **workload scope paragraph** (Phase 28, step 12) |

### Phases 15–22 — recommended `.tex` rewrite order

Apply in **one framing pass** before editing result numbers (can precede re-sweep):

| Step | Phase | Primary `.tex` targets |
| ---- | ----- | -------------------- |
| 1 | **15** | Abstract, Intro (question + contributions), Discussion opening, Conclusion |
| 2 | **16** | Intro gap (L56–57), optional `\S Problem`, Design Principles, Discussion bridge |
| 3 | **17** | Abstract, Intro, Related Work L66/L78/L81, Case-Study disclaimer L167, Conclusion |
| 4 | **18** | Abstract, Intro L58, Methodology opening L83–86, `\label{subsec:engine}` L106, pipeline caption |
| 5 | **19** | Intro opening L54, Keywords, Related Work framing, Experiments opening, Conclusion scope |
| 6 | **21** | **Rewrite `\label{sec:introduction}`** (L52–60) as seven-paragraph story **before** finalizing Related Work cross-refs |
| 7 | **22** | **Finalize `\textbf{Contributions.}` block** (L60) using five-contribution taxonomy; mirror in Abstract + Conclusion L627–629 |
| 8 | **20** | **Restructure `\label{sec:related}`** (L62–81) into four subsections; merge/remove old `\subsection{Positioning}` into §4 closing |

**No new GPU jobs** for steps 1–8. Step **21 before 22** (Intro narrative before contribution bullets); step **22 before 20** (contributions locked before Related Work positioning). Step **20 last** in framing pass so §4 *What is still missing?* closes into KVBench.

### Phase 23 — results narrative pass (after framing or with re-sweep)

| Step | Phase | Primary `.tex` targets |
| ---- | ----- | -------------------- |
| 9 | **23** | `\label{sec:experiments}` opening L215–218; `\label{sec:qwen3}` / `\label{sec:olmo2}` paragraph leads L319+; **NEW** `\subsection{Research Findings}` or restructure `\label{sec:discussion}` L595–623 as seven finding blocks; Pareto + offline-vs-online captions |

Apply step **9** when result tables/figures are updated (re-sweep or replot from existing JSON). **Finding 6 (workload)** is **future work only** — do not claim multi-workload answers (Phase 11 deferred).

| Step | Phase | Primary `.tex` targets |
| ---- | ----- | -------------------- |
| 10 | **24–25** | Regenerate `plot_pareto.pdf`, `plot_tradeoff.pdf`, optional appendix correlation table from CLI; Discussion **F1/F3/F7** cites `correlations_ctx512.json` weak predictors |

### Phases 26–28 — cost, calibration, workload scope (rewrite pass 2)

| Step | Phase | Primary `.tex` targets |
| ---- | ----- | -------------------- |
| 11 | **26–27** | **NEW §COST** after SYSTEM (~L285): Oaken five-layer taxonomy + Phase 27 calibration/stateful/overhead table from `method_benchmark_dimensions.csv`; Design Principles L95; Discussion **F3** Layers 3–5 |
| 12 | **28** | **§Experiments setup** scope sentence (WikiText only); **§Discussion Finding 6** disclaimer; **NEW `\paragraph{Workload scope and limits.}`** before Implications |

Apply steps **11–12** in the same pass as step **10** (no new GPU jobs — tables/figures from existing JSON + `export_method_benchmark_table.py`).

### Phases 29–33 — literature, bib, identity (rewrite pass 1–2)

| Step | Phase | Primary `.tex` targets |
| ---- | ----- | -------------------- |
| 13 | **29–30** | Merge `staging_entries.bib` → `reference.bib`; Related Work §4 + Intro ¶4 cites; **`\paragraph{Relation to serving benchmarks.}`** (Phase 30) |
| 14 | **31** | Run `scripts/audit_bibliography.py`; drop/replace anonymous cites; delete stale SnapKV ACL comment (L641) |
| 15 | **32–33** | Regenerate `\label{fig:pipeline}`; Abstract + Conclusion open with Phase 33 canonical sentence |

Step **13** with Phase 20 step **8**; step **14** before final PDF; step **15** after Phase 22 contributions locked.

Cross-link [Paper alignment guide](#paper-alignment-guide--codebase--conference_101719tex) for line-level detail.

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

### Paper change log — section by section (`conference_101719.tex`)

| When | Section | Why | What to change |
| ---- | ------- | --- | -------------- |
| **Rewrite pass 1, step 0** (before Phase 15) | **Title** (L31) | “Bridging offline/online” encodes obsolete two-branch split | Prefer three-branch subtitle or *Controlled KV Interception Engine for Fidelity, Behavior, and System Evaluation* |
| **Same pass** | **Abstract** (L45) | “dual metrics: Section A … Section B” | Replace with **FIDELITY / BEHAVIOR / SYSTEM**; one clause per branch |
| **Same pass** | **Keywords** (L49) | `offline fidelity, online inference` only | Add `KV interception`, `fidelity evaluation`, `system metrics` |
| **Same pass** | **Intro L58 “What” axis** | Still Section A/B pairing | Rewrite “What” as three independent branches (not offline/online dichotomy) |
| **Same pass** | **§Methodology opening** (L83–86) | “dual offline/online evaluation contract” | “Three-branch evaluation contract” under controlled interception |
| **Same pass** | **§Design Principles** (L95) | Bullet “Dual evaluation. Section A … Section B …” | **Three-branch evaluation** bullet listing FIDELITY / BEHAVIOR / SYSTEM |
| **Same pass** | **Fig. pipeline caption** (L101) | “Section A offline … Section B online” | Caption lists three branches; optional cost block if figure asset updated |
| **Same pass** | **§Evaluation Protocol** (L267–284) | Two subsubsections Section A/B | Split/rename per [§Evaluation Protocol](#evaluation-protocol-l255285-labelsubseceval_protocol) — four subsubsections |
| **Do not** | Results tables | Branch rename is structural | Keep numeric results until re-sweep; terminology pass only |

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

### Paper change log — Phase 2 branches (`conference_101719.tex`)

| When | Section | Branch | What to change |
| ---- | ------- | ------ | -------------- |
| **Rewrite pass 1** | **§Evaluation Protocol** (L267–274) | **FIDELITY** | Rename from “Section A: Offline Fidelity”; add relative recon error, cosine, KL, metadata overhead bullets from `METHODOLOGY.md` §6.1 |
| **Same pass** | **§Evaluation Protocol** (L276–282) | **BEHAVIOR** | Rename from “Section B” PPL-only scope; keep Alg. 2 PPL; optional one-line synthetic retrieval/IF protocol (**no result table** — Phase 11 deferred) |
| **Same pass** | **§Evaluation Protocol** (L284) | **SYSTEM** | Split tok/s from old Section B; add TTFT, ITL p50/p99, end-to-end latency; move throughput tables under SYSTEM in Results |
| **Same pass** | **§Experiments** (L216) | All three | Replace “Each run records Section A and Section B metrics” with three-branch + cost sentence |
| **Same pass** | **§Discussion L608–621** | FIDELITY vs BEHAVIOR | Cite branch names when discussing QJL/RocketKV decoupling (Phase 23 step 9 expands into F1–F7) |
| **Do not** | BEHAVIOR result tables | Engine has synthetic tasks; paper grid is WikiText PPL | Phase 28 scope sentence only — no new BEHAVIOR numbers |

### Phase 2 — consolidated completeness record

| Track | Status | Detail |
| ----- | ------ | ------ |
| **Engine** | ✅ Done | FIDELITY (`eval/fidelity/`), BEHAVIOR (`eval/behavior/`), SYSTEM (`eval/system/`) — all modules wired in `EvaluationRunner.run()`. |
| **Documentation** | ✅ Done | `METHODOLOGY.md` §6.1–6.3; per-branch completeness records above. |
| **Paper** | 📝 Pending | §Evaluation Protocol four-way split; Results subsection headers FIDELITY/BEHAVIOR/SYSTEM. See table above + [global protocol spec](#evaluation-protocol-l255285-labelsubseceval_protocol). |

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

**Code:** `eval/cost/accounting.py` · `eval/cost/oaken_taxonomy.py` (Phase 26) · `eval/cost/benchmark_dimensions.py` (Phase 27) · **Hooks:** `compressors/base.py` · **Runner:** `EvaluationResult.cost` (includes `oaken_layers` + `benchmark_dimensions`) · **Export:** `scripts/export_method_benchmark_table.py` · **CLI:** on by default; `--skip-cost` to disable · **Online detail:** `--kernel-cost`

**Tests:** `tests/test_cost_accounting.py`; cost block asserted in `tests/test_eval_runner.py`.

Recent work such as Oaken explicitly separates offline preparation from online inference cost, while calibration-free methods show that calibration requirements themselves are an important methodological variable.

### Completeness record

| Track | Status | Detail |
| ----- | ------ | ------ |
| **Engine** | ✅ Done | Phase 3 base + Phase 26 Oaken layers + Phase 27 benchmark dimensions on every `cost` export. |
| **Documentation** | ✅ Done | `METHODOLOGY.md` §6.5; Phases 26–27 sections below. |
| **Paper** | 📝 Pending | [§Evaluation Protocol → COST](#evaluation-protocol-l255285-labelsubseceval_protocol); Oaken Discussion paragraph; calibration table appendix. |

### Paper change log — section by section (`conference_101719.tex`)

| When | Section | Why | What to change |
| ---- | ------- | --- | -------------- |
| **Rewrite pass 2, step 11** | **NEW §COST** (after SYSTEM, ~L285) | Paper omits cost entirely | Mirror Phase 3 tree: compression / offline / online; cross-ref **Phase 26** Oaken layers + **Phase 27** benchmark dimensions table |
| **Same pass** | **§Plug-in Interface** (L153–155) | Hooks list incomplete | Add `offline_cost_metadata()`, `theoretical_compression_ratio()` hooks |
| **Same pass** | **§Case-Study Methods — TurboQuant** (L181) | Lloyd-Max calibration mentioned in prose only | Point to `cost.offline` fields (`calibration_time_ms`, `calibration_tokens`); compare QJL/RocketKV calibration-free in Phase 27 table |
| **Same pass** | **§Discussion Implications** (L623) | Practitioner checklist | Add: report offline preprocessing cost separately from FIDELITY metrics (Phase 26) |
| **Do not** | New GPU jobs | Cost fields in existing JSON | Export table from bundles — no re-sweep required for structure |

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

### Paper change log — section by section (`conference_101719.tex`)

| When | Section | Why | What to change |
| ---- | ------- | --- | -------------- |
| **Rewrite pass 1** | **NEW §Compression taxonomy** (after `\label{subsec:plugins}`, before L165) | Paper lists three families in prose only | Table: categories A–E → mechanism → plug-in (TQ=B, QJL=B+E, RocketKV=D+E, SnapKV=A, Palu=C+E); **empirical rows = TQ/QJL/RocketKV only** |
| **Same pass** | **§Case-Study Methods** (L165–167) | Implies exhaustive method survey | Cross-ref taxonomy; disclaimer: case studies demonstrate protocol, not full literature coverage |
| **Same pass** | **Related Work** (L65–78) | Palu/SnapKV cited as algorithms, not evaluated | Keep citations in Related Work §1–2; **do not** add SnapKV/Palu result rows without re-sweep |
| **Do not** | SnapKV/Palu empirical claims | Plug-ins exist (`compressors/snapkv.py`, `palu.py`) | Taxonomy + methodology only until sweeps run |

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

### Paper change log — section by section (`conference_101719.tex`)

| When | Section | Why | What to change |
| ---- | ------- | --- | -------------- |
| **Rewrite pass 1** | **Abstract + Intro** (L45, L58) | “benchmarking framework” undersells contribution | Lead with **controlled interception-and-transformation engine**; emphasize same model/input/decode loop, only KV transformation varies |
| **Same pass** | **§Design Principles** (L88–96) | Missing controlled-comparison principle | New bullet: **Controlled comparison** — matched inference path; only compressor/budget changes (diagram in this Phase 6 section) |
| **Same pass** | **Fig. pipeline** (L98–102) | Caption still dual-metric | Regenerate or recaption: interception at decode boundary → plug-in → three-branch evaluation |
| **Same pass** | **§Discussion opening** (L595–598) | Contribution framed as method ranking | Reframe: contribution is **how to evaluate** under controlled conditions (Phase 15/22) |
| **Do not** | Claim new compressors | Engine contribution is methodological | Case studies remain TQ/QJL/RocketKV unless re-sweep expands |

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

### Paper change log — section by section (`conference_101719.tex`)

| When | Section | Why | What to change |
| ---- | ------- | --- | -------------- |
| **Rewrite pass 1** (with Phase 14) | **§Experimental Configuration** (L220–229) | Setup in bullets, not causal contract | **Table: Controlled experimental conditions** — model, tokenizer, WikiText-2 split, ctx 128/256/512, gen 64, batch 1, FP16, greedy, A10G, PPL stride 512; footnote: only `compression_method`/budget varies |
| **Same pass** | **§Experiments opening** (L216–218) | “dual Section A/B” | Three branches + per-job `controlled_conditions` JSON sentence |
| **Same pass** | **§Evaluation Protocol intro** (L255) | Implicit fairness | Explicit: *“Same model + same input + same decode loop + same hardware + different KV transformation.”* |
| **At re-sweep** | Results table captions | Reproducibility | Cite `controlled_conditions.variable.compression_budget` fields (bitwidth, stage, RocketKV r256/r512/r1024, QJL seed 42) |
| **Do not** | Expand ctx/batch grid | Phase 12 deferred | Keep 128–512 / batch 1 / 64 tok — document as fixed axes |

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

**Code:** `eval/reproducibility/manifest.py` (`extract_phase14_manifest`, `validate_phase14_manifest`) · `eval/controlled_conditions.py` · `eval/runner.py` · `docs/reproducibility/REPRODUCIBILITY.md` · **Tests:** `tests/test_reproducibility_harness.py`, `tests/test_controlled_conditions.py`

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

# Phase 15: Redefine the Main Research Question 📝 **Paper only**

> **Status (2026-08-20):** **Paper-writeup phase only** — no engine or test changes required. The codebase already implements the *answer* to the reframed question (controlled interception + FIDELITY/BEHAVIOR/SYSTEM + `controlled_conditions`). This section is the **rewrite specification** for `docs/research_paper_writeup/conference_101719.tex` so the manuscript states the question explicitly instead of implying a compressor horse-race.

This is probably the **single most important conceptual change**.

### Research question shift

| | Current paper (implicit) | Target paper (explicit) |
| --- | ------------------------ | ----------------------- |
| **Primary question** | Which KV-cache compression method performs best? | **How should KV-cache transformations be evaluated under controlled and realistic inference conditions?** |
| **Role of KVBench** | “Benchmarking framework” / “yardstick for comparing methods” | **Instrument for answering the evaluation question** — controlled interception environment, not a winner-picker |
| **Case studies (TQ/QJL/RocketKV)** | Read as the main contribution | **Demonstrations** of what the instrument reveals when conditions are matched |
| **Empirical headline** | Rankings and trade-offs (still valid) | Rankings **illustrate** why multi-branch, controlled evaluation is necessary — not the end goal |

This turns your work from:

**"another KV compression comparison"**

into:

**"an inference-aware methodology for evaluating KV transformations."**

### Phrasing guide — retire vs. adopt

| Retire (horse-race framing) | Adopt (methodology framing) |
| --------------------------- | --------------------------- |
| “benchmarking framework” (standalone noun) | “controlled interception-and-transformation **evaluation environment**” |
| “yardstick for comparing KV compression **methods**” | “**instrument** for evaluating KV transformations under matched conditions” |
| “which compressor wins” / “best method” (implicit) | “what each branch measures” / “when rankings flip under controlled comparison” |
| “dual Section A/B” as the contribution | “three-branch FIDELITY / BEHAVIOR / SYSTEM protocol” (Phase 1 terminology) |
| Contribution (2) = “27-job study” as co-equal with (1) | Contribution (1) = protocol + engine; (2) = **case-study demonstrations** on two SLMs |

**Keep unchanged:** empirical numbers, Pareto figure, offline≠online finding, GQA vs MHA narrative — only the *framing* around them changes.

### When to apply

| Timing | Rationale |
| ------ | --------- |
| **First pass of paper rewrite** (same session as Phase 1 terminology + Phase 6/7 controlled-conditions table) | Question framing is structural — do **before** polishing result tables |
| **No new experiments required** | Reframing is prose-only; existing Phase-5 bundles support the case-study role |
| **After Phase 15, before Phase 16/17** | Phase 15 sets the *question*; Phases 16–17 refine *problem statement* and *novelty claim* (separate sections below) |

### Paper change log — section by section (`conference_101719.tex`)

| When | Section (label, lines) | Why | What to change |
| ---- | ------------------------ | --- | -------------- |
| **Rewrite pass 1** | **Title** (L31) | Title currently emphasizes “Bridging Offline Fidelity and Online Inference” — comparison framing | Consider: *KVBench: A Controlled Evaluation Instrument for KV-Cache Transformations in Small Language Models* (or keep subtitle + add “controlled evaluation methodology”). See also [Title](#title-l31) in paper alignment guide. |
| **Rewrite pass 1** | **Abstract** (L44–46) | Opens with “hard to compare methods”; closes with “yardstick for comparing KV compression methods” | **Opening:** state the evaluation problem (“protocols incompatible; metrics siloed”). **Middle:** “We present KVBench, a controlled interception engine and three-branch evaluation protocol (FIDELITY, BEHAVIOR, SYSTEM) for KV transformations on SLMs.” **End:** replace “yardstick for comparing methods” with “demonstrates that controlled multi-branch evaluation is necessary — case studies on TurboQuant, QJL, and RocketKV show offline fidelity does not predict behavior or system outcomes.” Keep numeric claims. |
| **Rewrite pass 1** | **Keywords** (L48–50) | “benchmarking” centers horse-race | Add: `controlled evaluation`, `KV interception`; keep method names as case-study tags |
| **Rewrite pass 1** | **§Introduction** opening (L52–57) | Literature survey OK; gap paragraph should lead to *evaluation* gap | After L56, add explicit sentence: *“The open question is not only which compressor to deploy, but how to evaluate KV transformations under incremental decode when representation, behavior, and system metrics disagree.”* |
| **Rewrite pass 1** | **§Introduction** KVBench paragraph (L58) | Already says “interception-and-transformation engine” — good | Reframe **What** axis: three branches not Section A/B. Add: *“KVBench is the instrument; plug-ins are case studies.”* De-emphasize “comparing compressors” as the goal. |
| **Rewrite pass 1** | **Contributions** (L60) | (1) engine+protocol, (2) 27-job study, (3) offline≠online — (2) reads as co-primary | **Reorder/reword:** (1)~Controlled interception engine + three-branch evaluation protocol + reproducible configuration export. (2)~Empirical demonstrations on Qwen3 + OLMo~2 that FIDELITY, BEHAVIOR, and SYSTEM diverge under matched conditions. (3)~Evidence that architecture (GQA vs MHA) changes conclusions — motivating controlled replication, not winner claims. |
| **Rewrite pass 1** | **§Related Work → Positioning** (L80–81) | “missing shared yardstick” = comparison frame | Replace with: *“missing shared **evaluation methodology** under controlled incremental decode.”* Case-study methods are **coverage**, not the research output. |
| **Rewrite pass 1** | **§Methodology opening** (L83–86) | “benchmarking harness rather than a new compressor” — close but passive | Lead with: *“This section specifies **how** KV transformations should be evaluated inside KVBench.”* Case-study families = plug-in **coverage** of quantization / sketching / eviction. |
| **Rewrite pass 1** | **§Design Principles** (`\label{subsec:design}`, L88–96) | Dual evaluation bullet | Add opening principle: **Evaluation question.** *Only the compressor varies; model, input, decode loop, hardware, and metric definitions are fixed* (cross-ref Phase 6/7 controlled-conditions table). |
| **Rewrite pass 1** | **§Experiments opening** (L215–218) | “This section describes … full Phase-5 **results**” | Reframe: *“This section **demonstrates** the evaluation instrument on two SLMs and three compressor families; results illustrate branch divergence, not a definitive ranking.”* |
| **Rewrite pass 1** | **§Discussion opening** (L598) | Already says “contribution is the shared protocol” — **best anchor in current `.tex`** | Strengthen first sentence: *“KVBench addresses **how** to evaluate KV-cache transformations, not which algorithm wins outright.”* Keep “four patterns” paragraph; rename Section A/B → FIDELITY/BEHAVIOR in body text. |
| **Rewrite pass 1** | **§Discussion → Implications** (L623) | Practitioner list is methodology-aligned | Preface with: *“These implications follow from treating KVBench as an evaluation instrument, not a leaderboard.”* |
| **Rewrite pass 1** | **§Conclusion** (L627–629) | Closes with “yardstick for comparing KV compression methods” | **First sentence:** restate research question answered. **Middle:** list protocol contributions (engine, three branches, cost, reproducibility export). **Last sentence:** replace “yardstick for comparing methods” with *“a controlled methodology for evaluating KV transformations before deployment on resource-constrained models.”* Case-study rankings = **examples**, not the contribution claim. |
| **Optional** | **Fig. pipeline caption** (L98–102) | Visual may still say Section A/B | Caption should show FIDELITY / BEHAVIOR / SYSTEM boxes; subtitle: “only the plug-in varies.” |
| **Do not** | Results tables / figures (L319+, Pareto L475+) | Empirical content stays | Do **not** rewrite numbers or drop Pareto — interpret in Discussion as “instrument output,” not “winner chart” |

### Cross-references (apply in same rewrite pass)

| Related phase | Overlap with Phase 15 |
| ------------- | --------------------- |
| **Phase 1** | Section A/B → FIDELITY/BEHAVIOR/SYSTEM terminology (required for consistent question framing) |
| **Phase 6–7** | Controlled comparison principle + conditions table (supports “matched conditions” claim) |
| **Phase 14** | Reproducibility export (supports “instrument” credibility) |
| **Phase 16** | Problem-statement diagram — apply **after** Phase 15 question sentence is in Intro |
| **Phase 17** | Novelty claim refinement — apply **after** Phase 15; avoids defending “first benchmark ever” |

### Completeness record

| Track | Status | Detail |
| ----- | ------ | ------ |
| **Engine** | — | No changes; existing engine is the instrument Phase 15 describes. |
| **Documentation** | ✅ Done | This section + paper alignment guide [Introduction](#introduction-l5260), [Abstract](#abstract-l4446), [§Discussion](#discussion-l595623-labelsecdiscussion), [§Conclusion](#conclusion-l625629-labelsecconclusion). |
| **Paper** | 📝 Pending | Full pass on sections in table above. **No new GPU jobs.** Apply at start of `.tex` rewrite. |

---

# Phase 16: Reframe the Core Problem Statement 📝 **Paper only**

> **Status (2026-08-20):** **Paper-writeup phase only** — no engine changes. The empirical case studies already *demonstrate* the problem cascade below (e.g., QJL moderate FIDELITY / catastrophic BEHAVIOR; TurboQuant high compression / low SYSTEM throughput). Phase 16 makes that problem **explicit in prose and optionally as a figure** in the `.tex` rewrite.

This is the heart of the revised paper.

### Problem cascade (target narrative)

```text
Existing KV-cache research
        ↓
Many different algorithms
        ↓
Different implementations · models · workloads · metrics · hardware
        ↓
Results are difficult to compare
        ↓
Compression ratio ≠ memory savings
Memory savings ≠ speedup
Tensor fidelity (FIDELITY) ≠ task behavior (BEHAVIOR)
Offline proxies ≠ incremental-decode outcomes
        ↓
Need controlled, multi-branch evaluation under matched conditions
        ↓
KVBench (instrument)
```

### Mapping cascade → paper evidence (already in `.tex`)

| Problem link | Where paper already shows it | Phase 16 action |
| ------------ | ---------------------------- | --------------- |
| Metric silos / incompatible protocols | Abstract L45; Intro L56 | State explicitly in **problem paragraph** |
| FIDELITY ≠ BEHAVIOR | Discussion L608–609 (QJL, RocketKV) | Cite as **motivation**, not surprise finding only |
| Memory ≠ speed | Discussion L617 (TurboQuant tok/s vs compression) | Add SYSTEM branch name; Pareto figure |
| Rankings flip across models | Discussion L621 (Qwen3 GQA vs OLMo~2 MHA) | Tie to “heterogeneous conditions” bullet |
| Need controlled evaluation | Design principles L90–95 (partial) | Expand + cross-ref controlled-conditions table (Phase 7) |

### When to apply

| Timing | Rationale |
| ------ | --------- |
| **Rewrite pass 1, immediately after Phase 15** | Problem statement follows research question in Intro |
| **No new experiments** | Cascade is illustrated by existing Phase-5 results |
| **Optional figure** | Only if page budget allows — otherwise prose + Discussion cross-refs |

### Paper change log — section by section (`conference_101719.tex`)

| When | Section (label, lines) | Why | What to change |
| ---- | ------------------------ | --- | -------------- |
| **Rewrite pass 1** | **§Introduction** gap paragraph (L56–57) | Currently lists fragmentation; does not state metric **decoupling** chain | Replace/extend with 3–4 sentences walking the cascade: incompatible protocols → non-comparable results → **compression ratio ≠ memory savings ≠ speedup ≠ tensor fidelity ≠ task quality** under incremental decode. End with: *“Controlled evaluation under matched conditions is therefore a prerequisite, not an optional appendix.”* |
| **Rewrite pass 1** | **NEW §Problem statement** (insert after Intro, before `\label{sec:related}`, ~L61) **or** opening of `\label{sec:methodology}` | Readers need one canonical problem block | Short subsection (½ column) or bullet list mirroring cascade diagram above. Optional **Fig. problem** (TikZ) — same content as ASCII block in this doc. |
| **Rewrite pass 1** | **§Design Principles** (L88–96) | Principles list mechanisms, not the *problem* | Add first bullet: **Problem.** *KV-cache literature mixes algorithms, implementations, and metrics; without fixed model/input/decode/hardware, cross-paper comparison is misleading.* |
| **Rewrite pass 1** | **§Discussion** (L608–621) | Findings already prove cascade | Add bridging sentence at L608: *“These patterns instantiate the evaluation problem motivating KVBench: no single metric axis is sufficient.”* Rename offline/online → FIDELITY/BEHAVIOR/SYSTEM when editing. |
| **Rewrite pass 1** | **§Conclusion** (L627) | Conclusion lists findings but not problem solved | One clause: *“The study demonstrates why uncontrolled or single-metric KV evaluation is insufficient for deployment decisions.”* |
| **Do not** | Results tables | Numbers stand | Do not add new claims beyond what Phase-5 data supports |

### Completeness record

| Track | Status | Detail |
| ----- | ------ | ------ |
| **Engine** | — | Engine implements the *solution*; problem is narrative only. |
| **Documentation** | ✅ Done | This section; empirical mapping in `METHODOLOGY.md` §6 rationale. |
| **Paper** | 📝 Pending | Intro problem paragraph + optional figure/subsection. Apply after Phase 15. |

---

# Phase 17: Reframe the Novelty 📝 **Paper only**

> **Status (2026-08-20):** **Paper-writeup phase only** — no engine changes. Novelty is **controlled interception + multi-branch evaluation under matched conditions**, not “first KV benchmark” or new compression algorithms.

### Claims to retire vs. adopt

| Do **not** claim | Safer replacement |
| ---------------- | ----------------- |
| “KV-cache compression has never been benchmarked.” | “Existing studies use **heterogeneous** implementations and experimental conditions.” |
| “First open-source KV benchmark.” | “Controlled **interception-and-transformation** environment with plug-in compressors.” |
| “We compare all major methods.” | “We host **representative** families (quantization, sketching, eviction) as **case studies**.” |
| Algorithm novelty for TQ/QJL/RocketKV | “KVBench does **not** claim algorithmic novelty for plug-ins” (already L167 — keep and repeat in Intro) |

### Canonical novelty sentence (use in Abstract / Intro / Conclusion)

> **Existing KV-cache studies evaluate individual compression mechanisms under heterogeneous implementations and experimental conditions. KVBench provides a controlled interception-and-transformation environment in which different KV transformations execute through a common incremental autoregressive decode loop, enabling representation-level (FIDELITY), behavioral (BEHAVIOR), and system-level (SYSTEM) comparisons under matched conditions.**

Shorter variant for Abstract (if space-limited):

> **KVBench is a controlled evaluation instrument — not a new compressor — that pairs FIDELITY, BEHAVIOR, and SYSTEM metrics under fixed model, data, and decode conditions.**

### When to apply

| Timing | Rationale |
| ------ | --------- |
| **Rewrite pass 1, after Phases 15–16** | Novelty must align with reframed question and problem statement |
| **Related Work pass** | Differentiate from `kvbench2026serving` and method papers without overclaiming |
| **No new experiments** | Novelty is positioning only |

### Paper change log — section by section (`conference_101719.tex`)

| When | Section (label, lines) | Why | What to change |
| ---- | ------------------------ | --- | -------------- |
| **Rewrite pass 1** | **Abstract** (L45) | “reproducible benchmarking framework” underspecifies novelty | After presenting KVBench, insert controlled-environment clause (canonical sentence above, shortened). Avoid “first” or “only.” |
| **Rewrite pass 1** | **§Introduction** (L58–60) | L58 “comparing KV compressors” sounds like novelty = comparison | Reframe: novelty = **protocol + engine**. L60 contribution (1) = controlled interception + three-branch protocol; cite Phase 15 reorder. |
| **Rewrite pass 1** | **§Related Work → KV Cache Compression** (L66) | Good start (“does not add another algorithm”) | Keep; add: *“Novelty is methodological — shared incremental loop, not algorithmic.”* |
| **Rewrite pass 1** | **§Related Work → Benchmarking** (L77–78) | Must distinguish from serving benchmarks | Explicit contrast: *“Concurrent serving benchmarks~\cite{kvbench2026serving} optimize deployment stacks; KVBench fixes a **controlled SLM factorial** for plug-in comparison before serving claims.”* Do not imply KVBench replaces serving eval. |
| **Rewrite pass 1** | **§Positioning of KVBench** (L80–81) | “missing shared yardstick” is weak vs 2026 literature | Replace with canonical novelty paragraph. Emphasize **matched conditions** + **three branches**, not leaderboard. |
| **Rewrite pass 1** | **§Case-Study Methods** (L165–167) | Already disclaims algorithm novelty | Keep L167 verbatim spirit; cross-ref in Intro contributions. |
| **Rewrite pass 1** | **§Conclusion** (L627–629) | “community benchmark” can imply horse-race | Reframe “grow as community benchmark” → *“extensible evaluation instrument”*; novelty = protocol export (`controlled_conditions`, artifacts), not winning compressor. |
| **Do not** | Related Work citations | Do not dismiss prior benchmarks | Acknowledge `chen2026pitfalls`, `kvbench2026serving` as **complementary** (Phase 13 serving path stays out of scope) |

### Completeness record

| Track | Status | Detail |
| ----- | ------ | ------ |
| **Engine** | — | Code already matches safe novelty (plug-in API, controlled export). |
| **Documentation** | ✅ Done | This section; `README.md` lead; Phase 6 principle. |
| **Paper** | 📝 Pending | Abstract, Intro, Related Work positioning, Conclusion. Apply after Phases 15–16. |

---

# Phase 18: Clarify Exactly What KVBench Is 📝 **Paper only**

> **Status (2026-08-20):** **Paper-writeup phase only** — terminology boundary. The codebase name aligns with **interception-and-transformation engine** + **evaluation layer**; it is **not** a serving engine (vLLM/SGLang — Phase 13 out of scope).

### Terminology decision table

| Use (preferred) | Avoid | Why |
| --------------- | ----- | --- |
| **Controlled interception-and-transformation evaluation environment** | “Full inference engine” | No scheduling, batching, continuous serving, or PD disaggregation |
| **Inference-time KV transformation and evaluation layer** | “Serving engine” / “vLLM-like” | Phase 13 explicitly not planned |
| **Extensible plug-in evaluation engine** (for `KVCacheEngine` + runner) | “Unified inference optimization framework” (too broad) | Scope = KV boundary + eval branches |
| **KVBench** (proper noun) | Generic “the framework” without definition | Define once in Intro |

**Pick one primary descriptor for Abstract/Intro** (recommended):

> **An extensible inference-time KV-cache compression evaluation engine** — or shorter: **a controlled KV interception-and-transformation engine for evaluation.**

Secondary acceptable (Title/subtitle):

> *A unified KV-cache inference benchmarking and transformation framework* — only if “benchmarking” is paired with “controlled evaluation,” not “method ranking.”

### Code ↔ paper alignment (no code changes)

| Code artifact | What it is | Paper should say |
| ------------- | ---------- | ---------------- |
| `framework/kv_engine.py` (`KVCacheEngine`) | Incremental decode loop + compress/decompress at cache boundary | “KV cache **engine**” = decode-step orchestrator, **not** full LM serving stack |
| `eval/runner.py` | FIDELITY / BEHAVIOR / SYSTEM orchestrator | “Evaluation orchestrator” or “benchmark driver” |
| `modal_app/worker.py` | One-shot CUDA eval jobs | “Reference hardware path,” not production serving |
| vLLM / SGLang | Not in repo | Mention only in Related Work / future work if at all (Phase 13) |

### When to apply

| Timing | Rationale |
| ------ | --------- |
| **Same rewrite pass as Phases 15–17** | Terminology must be consistent across Abstract → Methodology |
| **Early in Methodology** | Define KVBench once before `\label{subsec:engine}` |
| **No new experiments** | Naming only |

### Paper change log — section by section (`conference_101719.tex`)

| When | Section (label, lines) | Why | What to change |
| ---- | ------------------------ | --- | -------------- |
| **Rewrite pass 1** | **Abstract** (L45) | “benchmarking framework” is ambiguous | Use primary descriptor + parenthetical: *“…evaluation engine (not a production serving stack).”* |
| **Rewrite pass 1** | **§Introduction** (L58) | Already “interception-and-transformation engine” — **keep** | Add boundary sentence: *“KVBench is an evaluation layer at the KV-cache boundary; it is complementary to serving systems such as vLLM and SGLang, which optimize deployment rather than controlled plug-in comparison.”* One sentence only. |
| **Rewrite pass 1** | **§Related Work → Benchmarking** (L78) | Says “not a full serving stack” — **good anchor** | Keep; strengthen: *“KVBench does not implement continuous batching, request scheduling, or distributed serving.”* |
| **Rewrite pass 1** | **§Methodology opening** (L83–86) | “benchmarking harness” undersells interception | Replace lead: *“KVBench is a controlled interception-and-transformation **evaluation environment** for KV-cache plug-ins on fixed causal SLMs.”* Retain “not a new compressor.” |
| **Rewrite pass 1** | **§KV cache engine** (`\label{subsec:engine}`, L106–114) | “KV cache engine” can confuse with vLLM | Opening footnote or sentence: *“Engine here means the incremental decode orchestrator (Algorithm~\ref{alg:engine}), not a production inference server.”* |
| **Rewrite pass 1** | **Fig. pipeline caption** (L101) | “incremental decode engine” OK | Add: *“Evaluation branches (FIDELITY / BEHAVIOR / SYSTEM) attach to the same engine; only the plug-in varies.”* |
| **Rewrite pass 1** | **§Conclusion** (L629) | “community benchmark” + “yardstick” blurs boundary | Close with evaluation-layer framing (Phase 15) + explicit: *“not a replacement for serving-engine benchmarks.”* |
| **Do not** | Title alone | Do not call it “Serving Engine” or “Inference Engine for Deployment” | Keep SLM + KV + evaluation in title |

### Completeness record

| Track | Status | Detail |
| ----- | ------ | ------ |
| **Engine** | ✅ Aligned | `README.md`, `SYSTEM_DESIGN.md` already use interception/evaluation framing. |
| **Documentation** | ✅ Done | This section; Phase 13 marked not planned for serving integration. |
| **Paper** | 📝 Pending | Terminology pass on Abstract, Intro, Methodology opening, engine subsection, Conclusion. |

---

# Phase 19: Reframe the Domain Positioning 📝 **Paper only**

> **Status (2026-08-20):** **Paper-writeup phase only** — no engine changes. Positions KVBench within **LLM inference systems / inference-engineering**, not generic ML benchmarking or algorithm-only KV compression. The paper already touches this (L54 serving pressure, L58 SLM testbed, SYSTEM metrics) but does not state the **domain ladder** explicitly.

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

### What this is / is not (reviewer-facing)

| Position as | Not as |
| ----------- | ------ |
| Inference-systems contribution (controlled eval at KV boundary) | Pure quantization / algorithms paper |
| SLM-scale inference-engineering study (${\approx}$1B, Modal A10G, decode metrics) | Large-model SOTA compression claim |
| Evaluation infrastructure for KV transformations | Production serving system (vLLM/SGLang — Phase 13) |
| Complements serving benchmarks~\cite{kvbench2026serving} | Replacement for deployment-stack evaluation |

### When to apply

| Timing | Rationale |
| ------ | --------- |
| **Rewrite pass 1, after Phases 15–18** | Domain positioning synthesizes question + problem + novelty + terminology |
| **Before venue submission cover letter** | Same ladder can appear in 2–3 sentences for editors/reviewers |
| **No new experiments** | Positioning is prose (+ optional figure) only |

### Paper change log — section by section (`conference_101719.tex`)

| When | Section (label, lines) | Why | What to change |
| ---- | ------------------------ | --- | -------------- |
| **Rewrite pass 1** | **Abstract** (L45) | Abstract reads as generic “benchmarking framework” | One clause anchoring domain: *“…for **inference-time KV-cache optimization** on SLMs within an LLM **systems** evaluation setting.”* Mention decode-time memory/bandwidth pressure (L54 theme). |
| **Rewrite pass 1** | **Keywords** (L48–50) | Missing systems/inference-engineering tags | Add: `LLM inference systems`, `KV-cache optimization`, `inference engineering` (keep existing method names). Consider demoting standalone `benchmarking` or pair with `evaluation infrastructure`. |
| **Rewrite pass 1** | **§Introduction** opening (L54–55) | Strong systems hook already (“long-context serving”, KV footprint) | Extend L54 with explicit ladder sentence: *“We situate KVBench at the intersection of KV-cache optimization and **inference evaluation infrastructure** for resource-constrained SLM deployment.”* Keep citation to `kvbench2026serving`, `yuan2026shortrl`. |
| **Rewrite pass 1** | **§Introduction** SLM paragraph (L58) | SLM testbed rationale exists | Tie to domain: *“Exhaustive method$\times$context factorial on single-GPU SLMs is an **inference-engineering** testbed, not a claim about 70B-scale deployment.”* |
| **Rewrite pass 1** | **NEW optional figure** (Intro or Methodology, after `\label{subsec:design}`) | Visual helps reviewers place contribution | Small taxonomy figure (TikZ) — same tree as ASCII block above. Caption: *“Domain positioning: KVBench is inference evaluation infrastructure for KV-cache compression/transformation, not a serving engine.”* Skip if page-limited. |
| **Rewrite pass 1** | **§Related Work opening** (L65–66) | Currently algorithm-family survey first | Opening sentence: *“We review KV-cache optimization from an **LLM inference systems** perspective: eviction, representation compression, and evaluation methodology.”* Then existing subsections. |
| **Rewrite pass 1** | **§Related Work → Benchmarking** (L77–78) | Distinguish systems sub-community | Explicit: KVBench = **pre-deployment controlled factorial** on SLMs; serving benchmarks = **post-integration** quality+system under realistic stacks. Both are “systems” papers but different layers. |
| **Rewrite pass 1** | **§Experiments opening** (L215–218) | “Experiments and Results” sounds pure ML | Subtitle or first sentence: *“Case-study demonstrations of the evaluation instrument (inference-engineering evidence on two SLM architectures).”* |
| **Rewrite pass 1** | **§Discussion → Implications** (L623) | Practitioner list is systems-oriented | Reframe bullets as **inference-engineering checklist** (report FIDELITY+BEHAVIOR+SYSTEM, incremental decode, multi-architecture replication). |
| **Rewrite pass 1** | **§Conclusion** (L629) | “resource-constrained models” is close but vague | Final scope sentence: *“We scope claims to **SLM inference engineering**: controlled KV transformation evaluation before deployment, not datacenter-scale serving optimization.”* |
| **Optional** | **Cover letter / submission metadata** | Venues differ (systems vs ML) | Lead with: inference-systems + evaluation infrastructure + SLM KV-cache; case studies as evidence. |
| **Do not** | Algorithm Related Work subsections | Content stays | Do not shrink method survey — only add **framing** sentences at section boundaries |

### Cross-references

| Phase | Link |
| ----- | ---- |
| **15** | Research question — *how* to evaluate (instrument) |
| **16** | Problem cascade — why inference-systems evaluation matters |
| **17** | Novelty — controlled env, not “first benchmark” |
| **18** | Boundary — evaluation layer vs serving engine |
| **10** | SYSTEM + A10G evidence supports inference-engineering claims |
| **13** | Serving validation explicitly out of scope |

### Completeness record

| Track | Status | Detail |
| ----- | ------ | ------ |
| **Engine** | ✅ Aligned | SYSTEM branch, Modal CUDA path, SLM configs match inference-engineering scope. |
| **Documentation** | ✅ Done | This section; `SYSTEM_DESIGN.md` deployment context; Phase 13 serving out of scope. |
| **Paper** | 📝 Pending | Intro/Keywords/Related Work framing + optional domain figure + Conclusion scope. Apply after Phases 15–18. |

---

# Phase 20: Completely Restructure Related Work 📝 **Paper only**

> **Status (2026-08-20):** **Paper-writeup phase only** — no engine changes. The `.tex` Related Work (L62–81) already surveys many methods but uses **algorithm-family subsections** (Quantization / Low-rank / Eviction / Benchmarking / Positioning) and buries the **evaluation gap → KVBench** narrative in a short positioning paragraph. Phase 20 restructures into **four thematic sections** ending with *What is still missing?* so Related Work mirrors the problem cascade (Phases 16, 21) and domain positioning (Phase 19).

### Current vs target structure (`conference_101719.tex` L62–81)

| Current `\subsection` (L) | Target section | Action |
| ------------------------- | -------------- | ------ |
| **KV Cache Compression** (L65–66) | — (opening only) | Replace with **1–2 sentence systems framing** (Phase 19); drop generic catch-all |
| **Quantization-based Methods** (L68–69) | **§2 KV Representation Compression** | Move + merge with low-rank |
| **Low-rank and Projection-based Methods** (L71–72) | **§2 KV Representation Compression** | Merge with quantization |
| **Eviction-based Methods** (L74–75) | **§1 KV-Cache Eviction** | Keep core content; add Scissorhands/MiniCache/Ada-KV if not already cited |
| **Benchmarking and Evaluation Frameworks** (L77–78) | **§4 KV-Cache Evaluation and Benchmarking** | Expand — this becomes the **critical** section |
| **Positioning of KVBench** (L80–81) | **§4 closing** (*What is still missing?*) | **Delete standalone subsection**; fold into §4 closing + Phase 17 canonical novelty |

### Target four sections (content map)

#### §1. KV-Cache Eviction

**Purpose:** Token-budget policies — what is retained vs dropped.

| Paper | Role in narrative | In `.tex` today? |
| ----- | ----------------- | ---------------- |
| H2O~\cite{zhang2023h2o} | Heavy-hitter retention baseline | ✅ L74–75 |
| Scissorhands~\cite{liu2023scissorhands} | Importance persistence at test time | ✅ cited in L81; add prose in §1 |
| StreamingLLM~\cite{xiao2024streamingllm} | Attention-sink + sliding window | ✅ L74 |
| SnapKV~\cite{li2024snapkv} | Observation-window selection | ✅ L74; plug-in exists, no results |
| PyramidKV~\cite{cai2024pyramidkv} | Layer-wise pyramid budgets | ✅ L74 |
| Ada-KV~\cite{feng2024adakv} | Adaptive budget allocation | ✅ L75 (via MiniCache line) |

**Closing sentence for §1:** eviction reduces memory by **dropping tokens**, not by changing representation — motivates separate evaluation from §2.

#### §2. KV Representation Compression

**Purpose:** Low-bit / sketch / projection methods that **transform** cached tensors.

| Paper | Role | In `.tex` today? |
| ----- | ---- | ---------------- |
| MiniCache~\cite{liu2024minicache} | Depth-dimension compression | ✅ L75 |
| QJL~\cite{zandieh2025qjl,qjlcs2025} | 1-bit JL sketching | ✅ L71–72; case study |
| Palu~\cite{chang2025palu} | Low-rank projection | ✅ L71–72; plug-in, no results |
| Outlier Tokens~\cite{su2025outlier} | Heavy-tail handling | ✅ L69 |
| KVSink~\cite{su2025kvsink} | Evolving attention sinks | ✅ L69 |
| AsymKV~\cite{tao2025asymkv} | Asymmetric K/V bit allocation | ✅ L69 |
| XQuant~\cite{yang2025xquant} | Cross-layer ultra-low-bit | ✅ L69 |
| TurboQuant~\cite{zandieh2026turboquant,hu2026patternkv} | Rotation + Lloyd–Max (+ PatternKV) | ✅ L69; case study |

**Closing sentence for §2:** representation methods optimize **tensor fidelity** — but fidelity alone does not guarantee behavioral or system benefit (bridge to §4).

#### §3. Architecture- and Serving-Aware KV Optimization

**Purpose:** Layout, paging, and hybrid policies that tie compression to **inference structure**.

| Paper | Role | In `.tex` today? |
| ----- | ---- | ---------------- |
| MHA → GQA~\cite{jin2025mha2gqa,costoptgqa2025} | Head alignment / cost under GQA | ✅ L75–76 |
| PagedEviction~\cite{wang2026pagedeviction} | Block-wise eviction for paged memory | ✅ L75 |
| HqeKV~\cite{wang2026hqekv} | Hybrid quant + eviction | ✅ L75 |
| RocketKV~\cite{rocketkv} | Two-stage sparse attention | ✅ L75–76; case study |
| HybridKV | Heterogeneous adaptive compression | 📝 **Add bib entry**; cite in §3 |
| CompressKV~\cite{compresskv2026} | GQA-aware semantic retrieval heads | ✅ L75–76 |

**Closing sentence for §3:** architecture and serving constraints change **which tokens/heads matter** — rankings are not portable without controlled replication (motivates KVBench GQA vs MHA split).

#### §4. KV-Cache Evaluation and Benchmarking *(new critical section)*

**Purpose:** Survey **how** KV methods are evaluated — then expose the gap KVBench fills.

| Paper | Role in narrative | In `.tex` today? |
| ----- | ----------------- | ---------------- |
| **The Pitfalls of KV Cache Compression**~\cite{chen2026pitfalls} | Offline prefix proxies ≠ decode behavior | ✅ L56, L77 |
| **Benchmarking KV-Cache Optimizations…**~\cite{kvbench2026serving} | Closest serving benchmark — **must discuss explicitly** (Phase 30) | ✅ L54, L77–78 |
| **Oaken** (ISCA 2025) | Offline prep vs online inference **cost** | 📝 **Add bib**; cross-ref Phase 3 cost, Phase 26 |
| **SCOPE** (ACL 2025) | Prefill/decode split; long-context generation eval | 📝 **Add bib** |
| **CacheBlend** (EuroSys 2025) | Serving / RAG workload effects | 📝 **Add bib** |
| **KVCache Cache in the Wild** (USENIX ATC 2025) | Real workload non-uniformity | 📝 **Add bib**; cross-ref Phase 11 (deferred) |
| **yuan2026shortrl** | Reasoning rollouts intensify cache pressure | ✅ L54, L78 |

**§4 closing — *What is still missing?* (replace L80–81):**

> Prior work introduces diverse compressors and increasingly sophisticated **evaluation analyses**, yet no shared environment fixes model, incremental decode loop, and multidimensional metrics while swapping plug-ins under matched conditions. Serving benchmarks~\cite{kvbench2026serving} optimize deployment stacks; method papers optimize algorithms — **controlled pre-deployment factorial evaluation** on SLMs remains underspecified.

Then introduce KVBench with Phase 17 canonical novelty sentence.

### Logical chain (use in §4 prose or optional figure)

```text
Many compression techniques          (§1–§3)
          ↓
Fragmented evaluation                (§4 opening)
          ↓
Behavioral failures discovered       (chen2026pitfalls, SCOPE)
          ↓
Serving/runtime effects discovered   (Oaken, kvbench2026serving, Cache in the Wild)
          ↓
Need controlled evaluation           (Phase 16 cascade)
          ↓
KVBench                              (instrument — Phase 15)
```

### Bibliography work (`.bib` / `reference`)

Add before rewrite if not already in `reference.bib`:

| Key (suggested) | Paper | Priority |
| --------------- | ----- | -------- |
| `oaken2025` | Oaken — ISCA 2025 | High (cost narrative) |
| `scope2025` | SCOPE — ACL 2025 | High (decode eval) |
| `cacheblend2025` | CacheBlend — EuroSys 2025 | Medium (workload) |
| `kvcachewild2025` | KVCache Cache in the Wild — USENIX ATC 2025 | Medium (workload realism) |
| `hybridkv2026` | HybridKV — ACL 2026 | Low (§3 coverage) |

**Already cited — redistribute only:** `chen2026pitfalls`, `kvbench2026serving`, `yuan2026shortrl`, `jin2025mha2gqa`, `compresskv2026`.

### When to apply

| Timing | Rationale |
| ------ | --------- |
| **Rewrite pass 1, after Phase 21 Intro draft** | Intro ¶4–5 should foreshadow §4 evaluation gap |
| **Same pass as Phases 15–19** | Terminology (FIDELITY/BEHAVIOR/SYSTEM), novelty, domain framing must align |
| **Before result-table edits** | Related Work restructure is prose-only; no new GPU jobs |
| **Page budget** | Four subsections may exceed current length — trim §2 method descriptions to 1–2 sentences each; keep §4 at full width |

### Paper change log — section by section (`conference_101719.tex`)

| When | Section (label, lines) | Why | What to change |
| ---- | ------------------------ | --- | -------------- |
| **Rewrite pass 1** | **§Related Work opening** (before L65) | L65–66 is algorithm catch-all | **New opening (2 sentences):** LLM inference-systems survey of KV optimization — eviction (§1), representation compression (§2), architecture/serving-aware hybrids (§3), evaluation methodology (§4). Cross-ref Phase 19. |
| **Rewrite pass 1** | **DELETE** `\subsection{KV Cache Compression}` (L65–66) | Absorbed into four-section structure | Remove; content moves to §1–§3 openings |
| **Rewrite pass 1** | **NEW** `\subsection{KV-Cache Eviction}` | Phase 20 §1 | Migrate L74–76 eviction prose; ensure H2O, Scissorhands, StreamingLLM, SnapKV, PyramidKV, Ada-KV cited. Mention RocketKV as hybrid — pointer to §3. |
| **Rewrite pass 1** | **NEW** `\subsection{KV Representation Compression}` | Phase 20 §2 | Merge L68–72 (quantization + low-rank). Cover MiniCache, QJL, Palu, Outlier, KVSink, AsymKV, XQuant, TurboQuant. End with fidelity ≠ behavior bridge. |
| **Rewrite pass 1** | **NEW** `\subsection{Architecture- and Serving-Aware KV Optimization}` | Phase 20 §3 | Migrate GQA/MHA, PagedEviction, HqeKV, CompressKV, RocketKV from L74–76. Add HybridKV if bib ready. Tie to Qwen3 vs OLMo~2 replication. |
| **Rewrite pass 1** | **REPLACE** `\subsection{Benchmarking and Evaluation Frameworks}` (L77–78) | Becomes §4 — expanded | Retitle `\subsection{KV-Cache Evaluation and Benchmarking}`. Add Oaken, SCOPE, CacheBlend, Cache in the Wild (with new cites). Structure: (i) behavioral pitfalls, (ii) serving/system benchmarks, (iii) workload realism, (iv) offline/online cost separation. Use FIDELITY/BEHAVIOR/SYSTEM names. |
| **Rewrite pass 1** | **DELETE** `\subsection{Positioning of KVBench}` (L80–81) | Fold into §4 | Replace with **`\paragraph{What is still missing?}`** + Phase 17 canonical novelty + explicit contrast with `kvbench2026serving` (Phase 30 substance): *controlled interception layer vs serving-stack benchmark*. |
| **Rewrite pass 1** | **§Introduction** forward refs (L60) | Section list outdated after restructure | Update roadmap sentence: *“Section~\ref{sec:related} reviews KV optimization and evaluation methodology; …”* |
| **Do not** | Methodology / Results | Unchanged scope | Do not claim SnapKV/Palu **results** unless re-swept; Related Work may cite them as implemented plug-ins |
| **Do not** | Dismiss `kvbench2026serving` | Reviewer trap | Acknowledge as complementary — Phase 30 explicit contrast |

### Cross-references

| Phase | Link |
| ----- | ---- |
| **15** | KVBench = instrument answering *how* to evaluate |
| **16** | §4 closing = problem cascade terminus |
| **17** | Canonical novelty in *What is still missing?* |
| **18** | §4 distinguishes evaluation layer vs serving engine |
| **19** | Related Work opening = inference-systems lens |
| **21** | Intro ¶2–4 foreshadow §1–§4; Intro ¶6 points to Methodology |
| **30** | Explicit `kvbench2026serving` contrast paragraph in §4 |

### Completeness record

| Track | Status | Detail |
| ----- | ------ | ------ |
| **Engine** | — | Taxonomy A–E and plug-ins already cover §1–§3 methods; no code changes. |
| **Documentation** | ✅ Done | This section; [Related Work in paper alignment guide](#related-work-l6281); Phase 29 bib priority list. |
| **Paper** | 📝 Pending | Restructure L62–81 into four subsections + §4 gap closing. **Add 4–5 bib entries.** Apply after Phase 21 Intro draft. **No new GPU jobs.** |

---

# Phase 21: Rewrite the Introduction Around This Story 📝 **Paper only**

> **Status (2026-08-20):** **Paper-writeup phase only** — no engine changes. The current Intro (L52–60) already contains strong material (KV bottleneck L54, fragmentation L56, SLM testbed L58, contributions L60) but reads as **“compression is useful → we built a comparator.”** Phase 21 reorders into a **seven-paragraph methodology/systems narrative** so the paper opens as an **inference evaluation** contribution, not a compressor horse-race.

### Narrative shift

| | Current paper (L52–60) | Target paper (seven paragraphs) |
| --- | ---------------------- | -------------------------------- |
| **Opening hook** | KV cache grows; long-context serving pressure | Same — **¶1 bottleneck** |
| **Literature** | One dense citation paragraph listing methods | **¶2** many approaches exist (brief, by category) |
| **Gap** | Fragmentation + offline/online mismatch | **¶3** compression ratio ≠ inference benefit; **¶4** recent evidence of decoupling |
| **Solution** | KVBench as comparator with Section A/B | **¶5** need controlled multidimensional environment; **¶6** KVBench as instrument |
| **Contributions** | Engine + 27-job study + offline≠online | **¶7** protocol-first contributions (Phase 15 reorder) |

This makes the paper read as a **research methodology / LLM inference-systems** paper (Phase 19), not “another KV compression comparison.”

### Seven-paragraph outline → `.tex` mapping

| ¶ | Content | Source / action on current `.tex` |
| --- | ------- | ----------------------------------- |
| **1** | KV cache is a major LLM **inference bottleneck** (memory, bandwidth, serving pressure) | Keep L54 opening; trim method citations to footnote or defer to Related Work |
| **2** | Many compression/transformation approaches now exist (eviction, quant, sketch, hybrid) | Split citation list from L54 — high-level categories only; detail → Related Work §1–§3 |
| **3** | However, **compression ratio is not equivalent to inference benefit** | **New** — Phase 16 metric decoupling: ratio ≠ memory ≠ speed ≠ fidelity ≠ behavior |
| **4** | Recent work demonstrates: behavioral degradation; workload dependence; hardware/runtime effects; serving-specific effects | Expand L56 with cites: `chen2026pitfalls`, `kvbench2026serving`, Oaken, Cache in the Wild (add bibs); bullet list OK in prose |
| **5** | Therefore, evaluating KV transformations requires a **controlled, multidimensional inference environment** | Merge L56–57 gap + Phase 15 question: *how* to evaluate under incremental decode |
| **6** | Introduce **KVBench** — interception engine, FIDELITY/BEHAVIOR/SYSTEM, plug-in API, SLM testbed | Refactor L58: keep Where/What/How but replace Section A/B with three branches; Phase 18 one-sentence serving boundary; **one** empirical preview sentence (rankings flip) — details → Experiments |
| **7** | State **contributions** (protocol first, case studies second) | Rewrite L60 per **[Phase 22](#phase-22-rewrite-the-contributions--paper-only)** minimal (3) or standard (4) packaging |

### Retire vs adopt (Intro-specific)

| Retire | Adopt |
| ------ | ----- |
| Opening paragraph = method bibliography dump | ¶1–2 separate **problem** from **landscape** |
| “Comparing KV compressors” as implicit goal | “Evaluating KV transformations under matched conditions” |
| Section A / Section B naming | FIDELITY / BEHAVIOR / SYSTEM |
| Full empirical results in Intro (L58 second half) | One-sentence preview; tables/figures in Experiments |
| Contribution (2) = “27-job study” as co-equal headline | (2) = **demonstrations** of branch divergence |

### When to apply

| Timing | Rationale |
| ------ | --------- |
| **Rewrite pass 1, step 6** (before Phase 20 Related Work) | Intro sets up evaluation-gap story that §4 closes |
| **Same pass as Phases 15–19** | Question (15), problem (16), novelty (17), definition (18), domain (19) all touch Intro |
| **No new experiments** | Prose restructure only; keep numeric preview until re-sweep |
| **After Abstract draft** | Abstract and Intro must agree on instrument framing |

### Paper change log — section by section (`conference_101719.tex`)

| When | Section (label, lines) | Why | What to change |
| ---- | ------------------------ | --- | -------------- |
| **Rewrite pass 1** | **¶1 Bottleneck** (replace L54 first sentence block) | Current L54 mixes bottleneck + method survey | 3–4 sentences: autoregressive KV growth; decode-time memory/bandwidth dominance; long-context serving + reasoning rollouts~\cite{yuan2026shortrl,kvbench2026serving}. **Phase 19:** inference-systems hook. |
| **Rewrite pass 1** | **¶2 Landscape** (new, after ¶1) | Methods belong after problem | 2–3 sentences: eviction, representation compression, hybrids — “many transformations, heterogeneous eval.” Minimal cites or “see §Related Work.” |
| **Rewrite pass 1** | **¶3 Decoupling** (new) | Missing explicit thesis | 2–3 sentences: compression ratio ≠ memory savings ≠ throughput ≠ tensor fidelity ≠ task quality. **Phase 16** cascade in prose. |
| **Rewrite pass 1** | **¶4 Recent evidence** (extend/replace L56) | L56 good start but narrow | Add bullets as prose: behavioral failures~\cite{chen2026pitfalls}; serving benchmarks~\cite{kvbench2026serving}; offline/online cost (Oaken); workload realism (Cache in the Wild); GQA layout effects~\cite{jin2025mha2gqa,compresskv2026}. |
| **Rewrite pass 1** | **¶5 Need** (merge L56–57) | Bridge to solution | Explicit research question (Phase 15): *how should KV transformations be evaluated under controlled incremental decode?* |
| **Rewrite pass 1** | **¶6 KVBench** (refactor L58) | L58 too long + Section A/B | Define KVBench once (Phase 18 descriptor). Three axes: Where / What (FIDELITY, BEHAVIOR, SYSTEM) / How (plug-in). SLM testbed sentence. **One** preview result. Serving-boundary sentence. Case-study families = coverage, not novelty. |
| **Rewrite pass 1** | **¶7 Contributions** (rewrite L60) | Horse-race ordering | Apply **[Phase 22](#phase-22-rewrite-the-contributions--paper-only)** minimal (3-bullet) or standard (4-bullet) packaging. Section roadmap sentence. |
| **Rewrite pass 1** | **Abstract** (L44–46) | Must match Intro story | Apply same narrative arc in compressed form — see Phase 15 Abstract row. |
| **Optional** | **Fig. domain taxonomy** | Visual anchor | Same ladder as Phase 19 optional figure — only if page budget |
| **Do not** | Experiments / Results | Scope unchanged | Do not move tables into Intro |

### Cross-references

| Phase | Link |
| ----- | ---- |
| **15** | ¶5 research question; ¶7 contribution reorder |
| **16** | ¶3–4 problem cascade |
| **17** | ¶6–7 safe novelty; no “first benchmark” |
| **18** | ¶6 KVBench definition + serving boundary |
| **19** | ¶1 inference-systems hook; ¶6 SLM testbed scope |
| **20** | ¶2 points to Related Work; ¶4 foreshadows §4 gap |
| **22** | ¶7 = authoritative contribution bullet text |

### Completeness record

| Track | Status | Detail |
| ----- | ------ | ------ |
| **Engine** | — | Intro describes existing engine; no code changes. |
| **Documentation** | ✅ Done | This section; [Introduction in paper alignment guide](#introduction-l5260); rewrite-order step 6. |
| **Paper** | 📝 Pending | Restructure L52–60 into seven paragraphs. Apply **before** Phase 20 Related Work restructure. **No new GPU jobs.** |

---

# Phase 22: Rewrite the Contributions 📝 **Paper only**

> **Status (2026-08-20):** **Paper-writeup phase only** — no engine changes. The current contributions block (L60) lists three items centered on **dual Section A/B protocol** and a **$2{\times}27$-job study** as co-primary outputs. Phase 22 replaces this with a **five-contribution taxonomy** (methodology-first) and specifies **how many bullets to show in Intro** vs Conclusion given page budget and experimental support.

Phase 22 **implements** the contribution reorder sketched in Phase 15 and **lands in** Phase 21 ¶7 — it is the detailed spec for the `\textbf{Contributions.}` paragraph, not a separate section in the paper.

### Current vs target (`conference_101719.tex` L60, L627–629)

| Current (L60) | Problem | Target (Phase 22) |
| ------------- | ------- | ----------------- |
| (1) “open incremental interception engine and **dual Section A/B** evaluation protocol” | Undersells three branches; horse-race framing | **C1 + C5:** unified interception engine + extensible plug-in API |
| (2) “reproducible $2{\times}27$-job **study**” | Reads as primary empirical contribution | **C4:** cross-method **demonstrations** (not definitive ranking) |
| (3) “offline metrics mis-rank; GQA vs MHA” | Valid finding but buried as (3) | Part of **C4** + architecture-replication clause |
| — | Missing | **C2:** FIDELITY / BEHAVIOR / SYSTEM methodology |
| — | Missing | **C3:** controlled comparison protocol (`controlled_conditions`, Phase 7/14) |

### Five-contribution taxonomy (canonical wording)

Use these as **building blocks**; merge for Intro if page-limited (see packaging below).

#### Contribution 1 — Unified interception framework

> **A unified KV interception and transformation framework** that allows heterogeneous KV optimization methods to operate inside the same autoregressive incremental decode loop (`KVCacheEngine`, no re-compression of frozen prior payloads).

**Engine support:** ✅ `framework/kv_engine.py`, `compressors/base.py`  
**Paper anchor:** `\label{subsec:engine}` L106–114, Algorithm `\ref{alg:engine}`

#### Contribution 2 — Multidimensional evaluation methodology

> **A multidimensional evaluation methodology** separating representation fidelity (**FIDELITY**), behavioral quality under incremental decode (**BEHAVIOR**), and system/runtime performance (**SYSTEM**).

**Engine support:** ✅ `eval/runner.py`, `eval/{fidelity,behavior,system}/`  
**Paper anchor:** `\label{subsec:eval_protocol}`; retire Section A/B naming  
**Full tier add-on:** explicit **cost accounting** dimension (Phase 3) as sub-bullet, not separate numbered contribution

#### Contribution 3 — Controlled comparison protocol

> **A controlled comparison protocol** that normalizes model, workload, decode configuration, hardware, and compression budget — only the plug-in varies (`controlled_conditions` export per job).

**Engine support:** ✅ `eval/controlled_conditions.py`, `eval/reproducibility/manifest.py` (Phase 14)  
**Paper anchor:** Design Principles `\label{subsec:design}`; **Controlled conditions** table (Phase 7)

#### Contribution 4 — Cross-method empirical demonstrations

> **A cross-method empirical study** revealing where compression ratio, fidelity, behavioral quality, and runtime efficiency **diverge** under matched conditions on two SLM architectures (Qwen3 GQA, OLMo~2 MHA).

**Engine support:** ✅ Phase-5 bundles; Pareto export (Phase 9); three case-study plug-ins only in **results**  
**Paper anchor:** Experiments + Discussion L608–621; Pareto figure  
**Wording guard (Phase 17):** say *“demonstrates”* / *“illustrates”*, not *“determines the best method”* or *“comprehensive comparison of all KV methods”*

#### Contribution 5 — Extensible plug-in architecture

> **An extensible plug-in architecture** supporting quantization, eviction, projection, and hybrid KV transformations via a shared compressor interface and mechanism taxonomy (categories A–E).

**Engine support:** ✅ `compressors/taxonomy.py`; TQ/QJL/RocketKV + SnapKV/Palu registered  
**Paper anchor:** `\label{subsec:plugins}`, taxonomy table (Phase 4)  
**Do not overclaim:** general **adaptive/stateful policy API** is Phase 5 **not planned** — adaptive behavior may exist **inside** individual plug-ins (RocketKV, SnapKV), not as a framework-level contribution

### Experimental support matrix — what to claim

| Contribution | Claim in Intro? | Claim in Conclusion? | Requires new GPU jobs? |
| ------------ | --------------- | -------------------- | ---------------------- |
| **C1** Framework | ✅ Always | ✅ Always | No |
| **C2** Three-branch methodology | ✅ Always | ✅ Always | No |
| **C3** Controlled protocol | ✅ Always (minimal); detail in Methodology | ✅ Always | No |
| **4** Empirical demonstrations | ✅ Always — **secondary** to C1–C3 | ✅ Always — summarize **Findings 1–7** (Phase 23), not winners | No for existing TQ/QJL/RocketKV; yes to add SnapKV/Palu **results** |
| **C5** Plug-in architecture | ⚠️ Merge into C1 if tight; else ✅ | ✅ Always | No |
| **Cost accounting** (optional) | Full tier only | Full tier | No |
| **Reproducibility export** (optional) | Fold into C3 | ✅ Full tier | No |

### Intro packaging (choose at rewrite time)

| Tier | Intro `\textbf{Contributions.}` format | Maps from |
| ---- | -------------------------------------- | --------- |
| **Minimal (3 bullets)** — recommended for IEEE column | (1)~**Engine + plug-ins:** controlled interception framework with extensible compressor API (C1+C5). (2)~**Protocol:** FIDELITY/BEHAVIOR/SYSTEM evaluation under matched conditions with per-job configuration export (C2+C3). (3)~**Demonstrations:** case studies on Qwen3 + OLMo~2 showing metric divergence and architecture-dependent rankings (C4). | Phases 15, 17 |
| **Standard (4 bullets)** | Split C2 and C3; keep C1+C5 merged; C4 separate | Full paper update tier |
| **Expanded (5 bullets)** | One bullet per C1–C5 | Only if page budget allows; risk of looking like a feature list |

**Retire in all tiers:**

- “dual Section A/B evaluation protocol”
- “$2{\times}27$-job study” as a **contribution headline** (keep job count in Experiments setup, not contribution claim)
- “yardstick for comparing methods” (Phase 15/17)

### When to apply

| Timing | Rationale |
| ------ | --------- |
| **Rewrite pass 1, step 7** — immediately after Phase 21 Intro draft | ¶7 is the contribution paragraph; Phase 22 is the authoritative bullet text |
| **Same pass as Phases 15, 17, 21** | Ordering and wording must match research question and safe novelty |
| **Mirror in Abstract + Conclusion** | Abstract: one-sentence protocol summary + one-sentence demonstration finding. Conclusion L627–629: enumerate C1–C3 structurally; C4 as evidence, not headline |
| **No new experiments** | Prose-only for current three case studies |

### Paper change log — section by section (`conference_101719.tex`)

| When | Section (label, lines) | Why | What to change |
| ---- | ------------------------ | --- | -------------- |
| **Rewrite pass 1** | **§Introduction `\textbf{Contributions.}`** (L60) | Horse-race ordering; Section A/B | Replace with **minimal (3)** or **standard (4)** packaging from table above. End with section roadmap sentence (unchanged structure). Cross-ref `\ref{sec:methodology}`, `\ref{sec:experiments}`. |
| **Rewrite pass 1** | **Abstract** (L45 closing clause) | Implied contribution = “yardstick” | List protocol contributions in one clause; one clause for demonstration finding (offline ≠ online; rankings flip). No numbered list in Abstract. |
| **Rewrite pass 1** | **§Methodology opening** (L83–86) | Should reflect contribution claims | First sentence: *“This section instantiates Contributions (1–2)”* or equivalent — engine + protocol, not case-study methods. |
| **Rewrite pass 1** | **§Design Principles** (L88–96) | C3 evidentiary home | Ensure **Controlled comparison** bullet matches C3 wording verbatim where possible. |
| **Rewrite pass 1** | **§Experiments opening** (L215–218) | C4 must not read as contribution headline | Open with: *“We use three compressor families as **demonstrations** of the evaluation protocol (Contribution 4), not as an exhaustive method survey.”* |
| **Rewrite pass 1** | **§Discussion opening** (L598) | Contribution framing anchor | First sentence ties to C2: multidimensional evaluation necessary. Do not reopen contribution list — interpret C4 patterns. |
| **Rewrite pass 1** | **§Conclusion** (L627–629) | L627 repeats “benchmarking framework” + dual metrics | Restate C1–C3 in prose (engine, three branches, controlled export). L628 case-study sentence = C4 evidence. Remove “yardstick for comparing methods”; close with methodology scope (Phase 15/19). |
| **Optional** | **Cover letter** | Reviewer routing | Bullet C1–C3 for systems/methodology venues; C4 as supporting evidence on SLMs |
| **Do not** | Claim all five as **equal-weight** contributions if merged in Intro | Reviewer clarity | If Intro uses 3 bullets, Conclusion may spell out 5 themes in prose without re-numbering |
| **Do not** | C5 “adaptive/stateful transformations” without Phase 5 | Overclaim | Say “plug-ins may implement stateful policies internally”; framework API remains one-plug-in / one-config |

### Cross-references

| Phase | Link |
| ----- | ---- |
| **15** | Research question — contributions answer *how* to evaluate, not *who wins* |
| **16** | C4 demonstrations instantiate problem cascade |
| **17** | Safe novelty — C1–C3 are methodological; C4 is illustrative |
| **18** | C1 wording — evaluation layer, not serving engine |
| **19** | C4 scoped to SLM inference-engineering testbed |
| **21** | ¶7 = landing zone for Phase 22 bullets |
| **3** | Cost as optional sub-bullet under C2 or C3 (full tier) |
| **4** | C5 taxonomy table cross-ref |
| **7, 14** | C3 controlled conditions + reproducibility export |
| **23** | C4 empirical demonstrations expressed as **Findings 1–7** |

### Completeness record

| Track | Status | Detail |
| ----- | ------ | ------ |
| **Engine** | ✅ Aligned | All five contributions map to shipped code paths; C5 adaptive API explicitly **not** claimed (Phase 5 not planned). |
| **Documentation** | ✅ Done | This section; Phase 15 contribution reorder; [Introduction in paper alignment guide](#introduction-l5260). |
| **Paper** | 📝 Pending | Rewrite L60 + mirror Abstract/Conclusion. Apply at rewrite step **7** after Phase 21. **No new GPU jobs.** |

---

# Phase 23: Change the Results Narrative 📝 **Paper only**

> **Status (2026-08-20):** **Paper-writeup phase only** — no engine changes. The Experiments section (L319+) reports results **method-by-method** (“TurboQuant achieved…”, “QJL achieved…”), which reads as a **leaderboard**. Discussion (L608–621) already states the right **decoupling** claims but uses method-centric paragraph titles. Phase 23 reframes both sections around **seven research questions (findings)** that the case studies **answer**, supporting Phase 15/22’s “demonstrations, not horse-race” framing.

### Narrative shift

| | Current paper | Target paper |
| --- | ------------- | -------------- |
| **Results structure** | `\paragraph{TurboQuant results}` → `\paragraph{QJL results}` → `\paragraph{RocketKV results}` | Tables stay; prose shortened. Optional `\subsection{Summary of findings}` with F1–F7 before detail tables |
| **Discussion structure** | “Offline fidelity does not predict…” + mechanism/context/architecture paragraphs | **Same content**, reorganized under `\textbf{Finding N: …?}` headers |
| **Sentence pattern** | “Method X achieved the lowest Y” | “**Finding N:** Under matched conditions, … **therefore** FIDELITY alone is insufficient” |
| **Reader takeaway** | Implicit ranking (TurboQuant “wins” on Qwen3) | Explicit **evaluation lessons** — no deployment recommendation |

### Seven findings — questions, evidence, `.tex` anchors

| ID | Research question | Answer in current data (summary) | Primary `.tex` / figure | Claim in paper? |
| -- | ----------------- | -------------------------------- | ----------------------- | --------------- |
| **F1** | Does better KV reconstruction imply better model quality? | **No** — QJL moderate attention metrics, catastrophic PPL; TQ 4-bit inverse pattern | L608, L394; `fig:offon` (attention RMSE vs PPL) | ✅ Yes |
| **F2** | Does higher compression imply lower memory? | **Mostly yes**, with metadata/shared-projection caveats (QJL mem ratio modest despite 1-bit keys) | FIDELITY memory columns in tables; `tab:cross` | ✅ Yes |
| **F3** | Does lower memory imply higher throughput? | **No** — TQ trades memory for speed (decompress overhead); RocketKV high tok/s but quality collapse on Qwen3 GQA | L617, SYSTEM tok/s columns | ✅ Yes |
| **F4** | Does offline (FIDELITY) quality predict online (BEHAVIOR) generation quality? | **No** — weak Pearson coupling; central claim | L608–609, `fig:offon`; cite `chen2026pitfalls` | ✅ Yes — headline |
| **F5** | Does the best method change with context length? | **Yes** — TQ 4-bit stable across T; 2/3-bit degrade at T=512; RocketKV only compresses meaningfully when B < T | L619 | ✅ Yes |
| **F6** | Does the best method change with workload? | **Not evaluated** in paper (WikiText-2 PPL only) | — | ⏸ **Future work** (Phase 11); mention as open question in Discussion closing, **do not answer** |
| **F7** | What is the quality–memory–speed Pareto frontier? | **No single winner** — TQ quality/memory, RocketKV speed, QJL neither; empirical front at T=512 | `fig:pareto` L475–481, L617; `scripts/analyze_pareto.py` | ✅ Yes |

### Retire vs adopt (results prose)

| Retire (leaderboard) | Adopt (findings) |
| -------------------- | ---------------- |
| “TurboQuant achieved the lowest reconstruction error” (standalone headline) | “**Finding 1:** tensor/attention reconstruction does not rank methods for BEHAVIOR (e.g., QJL vs TQ 4-bit)” |
| “QJL achieved X tok/s” as a success metric | “**Finding 3:** memory reduction and SYSTEM throughput decouple (mechanism-dependent online cost)” |
| “TurboQuant is the best method on Qwen3” | “TurboQuant 4-bit is the **most stable BEHAVIOR/memory trade-off in this SLM grid** — not a universal winner (F7)” |
| Per-method results paragraphs as primary narrative | Finding-led Discussion + compact method tables as **evidence** |
| “Section A / Section B” in finding text | FIDELITY / BEHAVIOR / SYSTEM |

### Recommended `.tex` structure (choose at rewrite time)

| Tier | Experiments (`\label{sec:experiments}`) | Discussion (`\label{sec:discussion}`) |
| ---- | --------------------------------------- | ------------------------------------- |
| **Minimal** | Keep method subsections + tables; add 1 paragraph at L218 listing F1–F7 as “questions this section answers”; reframe Discussion L608–621 with `\textbf{Finding N:}` headers only | Lowest diff |
| **Standard (recommended)** | Add `\subsection{Research Findings}` after setup (before L319) with 7 short bullets (2–3 sentences each, no new numbers); method tables move to `\subsection{Detailed case-study tables}` | Clearest for reviewers |
| **Full** | Findings subsection + `\subsection{Cross-dimensional analysis}` stub pointing to Phase 24 appendix | Only if page budget |

### When to apply

| Timing | Rationale |
| ------ | --------- |
| **Rewrite pass 2, step 9** — after framing pass (steps 1–8) and **with** result table/figure update | Findings cite numbers; restructure can use existing Phase-5 data if numbers unchanged |
| **Same pass as Phase 22 C4 wording** | Findings = contribution (4) demonstrations |
| **After Phase 1 terminology** | FIDELITY/BEHAVIOR/SYSTEM in finding headers and figure captions |
| **No new GPU jobs** for F1–F5, F7 with existing bundles | Replot Pareto + offline-vs-online from JSON (Phase 9) |
| **F6 explicitly out of scope** | Phase 11 deferred — one sentence in Discussion future work only |

### Paper change log — section by section (`conference_101719.tex`)

| When | Section (label, lines) | Why | What to change |
| ---- | ------------------------ | --- | -------------- |
| **Rewrite pass 2** | **§Experiments opening** (L215–218) | “full Phase-5 **results**” = leaderboard frame | Replace with: *“This section **answers seven evaluation questions** (Findings 1–7) using three compressor families as controlled demonstrations; detailed tables follow.”* Cross-ref `\ref{sec:discussion}`. **Phase 19/22:** demonstrations, not exhaustive survey. |
| **Rewrite pass 2** | **NEW `\subsection{Research Findings}`** (insert ~L318, before `\label{sec:qwen3}`) **or** Discussion-only restructure | Readers need finding list before tables | Seven `\textbf{Finding N: …?}` bullets (2–3 sentences each). F6 = *“left to future multi-workload evaluation (Phase 11).”* |
| **Rewrite pass 2** | **§Qwen3 / §OLMo2 paragraph leads** (L343+, L393+, RocketKV paragraphs) | Method-first prose | Shorten to: *“Table~\ref{…} provides FIDELITY/BEHAVIOR/SYSTEM evidence for Findings 1–5, 7; see Discussion.”* Keep 1 illustrative sentence per method max. |
| **Rewrite pass 2** | **Table / figure captions** (L347, L375, `fig:offon`, `fig:pareto`) | Section A/B labels | Rename axes/branches. **`fig:offon` caption:** “Evidence for **Finding 4** (FIDELITY vs BEHAVIOR decoupling).” **`fig:pareto` caption:** “**Finding 7:** empirical Pareto front at T=512; no config dominates all axes.” |
| **Rewrite pass 2** | **§Discussion opening** (L598) | “four patterns” is good but unnamed | Open with: *“The case studies answer seven evaluation questions; we organize the discussion as Findings 1–7.”* Keep “contribution is protocol” sentence. |
| **Rewrite pass 2** | **§Discussion body** (L608–621) | Content maps to findings; titles don't | Rename paragraphs: **Finding 4** (L608), **Finding 3** + **F7** (L617), **Finding 5** (L619), **Finding 1** + architecture (L621 → split F1 vs GQA replication). Merge L621 architecture content as sub-finding: *rankings are not architecture-invariant*. |
| **Rewrite pass 2** | **§Discussion Implications** (L623) | Should follow finding frame | Preface: *“These implications follow from Findings 1–7, not from a single winning compressor.”* Replace Section A/B → three branches. |
| **Rewrite pass 2** | **§Conclusion** (L627) | Repeats method ranking tone | Lead with 2–3 findings (F4, F7, architecture), not “TurboQuant remains most stable” as headline — relegate to parenthetical evidence. **Phase 22** C4 wording. |
| **Rewrite pass 2** | **Abstract** (L45) | Empirical clause is leaderboard-like | One clause: *“Case studies show FIDELITY, BEHAVIOR, and SYSTEM decouple (Findings 1–4, 7).”* |
| **Do not** | Claim F6 answer | No multi-workload data | WikiText only — future work sentence |
| **Do not** | Delete result tables | Evidence required | Tables support findings; shrink prose not data |

### Cross-references

| Phase | Link |
| ----- | ---- |
| **15** | Findings answer *how to evaluate*, not *who wins* |
| **16** | F1–F4 instantiate metric decoupling cascade |
| **17** | Avoid “best method” deployment claims; F7 = no universal winner |
| **22** | C4 = cross-method demonstrations framed as **Findings 1–7** (Phase 23) |
| **23** | ¶7 / C4 land as finding-led Discussion |
| **9** | F7 Pareto — regenerate via `scripts/analyze_pareto.py` |
| **11** | F6 deferred — do not claim |
| **24** | Optional appendix: correlation matrix supporting F1–F3 (full tier) — **`correlations_ctx512.json`** |
| **25** | **`plot_tradeoff_ctx512.pdf`** visualizes F7; complements Phase 9 Pareto |

### Completeness record

| Track | Status | Detail |
| ----- | ------ | ------ |
| **Engine** | ✅ Aligned | Phase-5 bundles + Pareto export supply all F1–F5, F7 evidence; no code changes. |
| **Documentation** | ✅ Done | This section; [Results](#results-qwen3--olmo2-l319-labelsecqwen3-labelsecolmo2) + [Discussion](#discussion-l595623-labelsecdiscussion) in paper alignment guide. |
| **Paper** | 📝 Pending | Restructure Experiments + Discussion around F1–F7. Apply at rewrite **step 9**. **No new GPU jobs** for current grid. |

---

# Phase 24: Add Cross-Dimensional Analysis ✅ **Done**

> **Status (2026-08-20):** **Implementation complete** — `eval/cross_dim/` computes Pearson correlations across six predefined metric pairs from job JSON or `EvaluationResult` lists. CLI: `scripts/analyze_cross_dim.py`. Reporter: `ResultReporter.save_cross_dim()`. Tests: `tests/test_cross_dim_analysis.py`. Paper appendix / Discussion citation deferred.

Answers the Phase 23 question: **which metrics actually predict real inference performance?** — by quantifying decoupling (weak |r|) rather than asserting it in prose only.

### Metric pairs (engine defaults)

| Pair | Metrics | Phase 23 finding |
| ---- | ------- | ---------------- |
| Compression ratio ↔ memory reduction | `theoretical_compression_ratio` ↔ `compression_ratio` | F2 |
| Compression ratio ↔ PPL | `compression_ratio` ↔ `perplexity_ratio` | F2 |
| Reconstruction error ↔ PPL | `attention_rmse` ↔ `perplexity_ratio` | F1, F4 |
| Reconstruction error ↔ task accuracy | `attention_rmse` ↔ `retrieval_accuracy` | F1 (requires BEHAVIOR retrieval in bundle) |
| Memory reduction ↔ throughput | `compression_ratio` ↔ `tokens_per_second` | F3 |
| Online overhead ↔ throughput | `online_overhead_ms` ↔ `tokens_per_second` | F3 |

**Summary question** (exported in JSON): *Which metrics actually predict real inference performance?*

**Interpretation guide for paper:**

| \|r\| range | Write in Discussion |
| ---------- | ------------------- |
| **< 0.5** | Metric X is a **weak predictor** of Y under this grid — supports decoupling claim |
| **≥ 0.5** | Partial coupling — cite with caution; do not overgeneralize beyond SLM grid |
| **n < 3** | Omit pair or mark “insufficient sample” (legacy bundles without cost/retrieval) |

### Code ↔ paper mapping

| Artifact | Path | Paper use |
| -------- | ---- | --------- |
| Correlation export | `results/cross_dim/correlations_ctx512.json` | Appendix table or Discussion footnote |
| Pearson bar chart | `plot_correlation_ctx512.pdf` | Optional appendix figure |
| CLI | `scripts/analyze_cross_dim.py` | Reproducibility appendix command |
| Reporter hook | `ResultReporter.save_cross_dim(results)` | Post-sweep automation |

**Point-ID fix (shared with Phase 9):** job `label` (e.g. `rocketkv_r256`, `turboquant` bitwidth via `stage`+`bN`) prevents config collapse when merging bundles.

### Paper change log — section by section (`conference_101719.tex`)

| When | Section | Why | What to change |
| ---- | ------- | --- | -------------- |
| **Rewrite pass 2, step 10** | **§Discussion Finding 1 / 4** (L608) | Currently cites Pearson r from offline-vs-online figure only | Add: *“Cross-dimensional analysis (`correlations_ctx512.json`) shows attention RMSE ↔ PPL ratio \|r\| < 0.5 on the Phase-5 grid — reconstruction is not a reliable BEHAVIOR proxy.”* Use **actual r** from JSON after replot. |
| **Rewrite pass 2** | **§Discussion Finding 3** (L617) | Memory ↔ speed decoupling asserted in prose | Cite `compression_ratio` ↔ `tokens_per_second` and `online_overhead_ms` ↔ `tokens_per_second` pairs from JSON. |
| **Optional** | **Appendix: Cross-dimensional correlations** | Full tier only | Small table: pair label, n, Pearson r, linked Finding ID |
| **Optional** | **`fig:offon` caption** (L613) | Complements correlation export | Note figure r matches `analyze_cross_dim` attention_rmse ↔ perplexity_ratio pair |
| **Do not** | Claim retrieval ↔ RMSE correlation | Legacy Phase-5 bundles lack retrieval metrics | Pair appears with n=0 until BEHAVIOR retrieval sweeps run (Phase 11) |

### Cross-references

| Phase | Link |
| ----- | ---- |
| **23** | F1, F2, F3, F4 supported by correlation pairs |
| **9** | Pareto answers F7; correlations answer “why no single winner” |
| **25** | Trade-off figure visualizes same points |
| **3** | `theoretical_compression_ratio` pair requires cost in bundle |
| **11** | Retrieval accuracy pair deferred until external workloads |

### Completeness record

| Track | Status | Detail |
| ----- | ------ | ------ |
| **Engine** | ✅ Done | `eval/cross_dim/{points,correlation,plot}.py`, CLI, reporter, tests; point_id fix in `eval/pareto/analysis.py`. |
| **Documentation** | ✅ Done | This section; `METHODOLOGY.md` §6.8. |
| **Paper** | 📝 Pending | Discussion cites + optional appendix. Regenerate from Phase-5 JSON — **no new GPU jobs**. |

---

# Phase 25: Add a "Compression Trade-off" Figure ✅ **Done**

> **Status (2026-08-20):** **Implementation complete** — Phase 9 `plot_pareto.pdf` covers **F7** (memory vs log PPL, marker ∝ tok/s). Phase 25 adds a **reader-facing trade-off figure** with explicit Quality↔Memory and Quality↔Speed panels (`plot_tradeoff_ctx512.pdf`) plus optional 3D scatter (`--3d`). Same CLI as Phase 24: `scripts/analyze_cross_dim.py`.

### Figure map (choose at rewrite time)

| Figure | File | Axes | Phase / Finding |
| ------ | ---- | ---- | --------------- |
| **Pareto (existing)** | `plot_pareto_ctx512.pdf` | compression ratio × log₁₀ PPL ratio; size ∝ tok/s | Phase 9, **F7** |
| **Trade-off (new)** | `plot_tradeoff_ctx512.pdf` | Panel A: memory ratio × **quality score**; Panel B: tok/s × quality score | **Phase 25** — central thesis visual |
| **3D (optional)** | `plot_tradeoff_3d_ctx512.pdf` | compression ratio × tok/s × quality score | Appendix only |
| **Offline vs online** | `plot_offline_vs_online.pdf` | attention RMSE × log PPL ratio | **F4** |

**Quality score:** `1 / (1 + max(0, log10(PPL/baseline)))` — higher is better; matches commented dumbbell figure semantics in `.tex`.

### Target visual (Phase 25 panels)

```text
  Quality score ↑          Quality score ↑
       ● A                        ● A
                 ● B
  ● C                              ● C
       └─ Memory ratio →           └─ tok/s →
```

This communicates **no config dominates all three axes** more directly than a single Pareto plot for non-expert readers.

### Paper change log — section by section (`conference_101719.tex`)

| When | Section | Why | What to change |
| ---- | ------- | --- | -------------- |
| **Rewrite pass 2, step 10** | **NEW fig: trade-off** (Experiments or Discussion, near `fig:pareto`) | Phase 25 deliverable | Include `plot_tradeoff_ctx512.pdf`. Caption: *“Quality↔Memory and Quality↔Speed trade-offs at T=512; quality score from BEHAVIOR PPL ratio. Generated by `scripts/analyze_cross_dim.py`.”* Cross-ref **Finding 7**. |
| **Rewrite pass 2** | **`fig:pareto` caption** (L480) | Provenance | Footnote: regenerated via `scripts/analyze_pareto.py`; optimal set in `pareto_ctx512.json`. |
| **Rewrite pass 2** | **§Discussion L617** | F7 narrative | Reference both Pareto front **and** trade-off panels: *“No single point dominates quality, memory, and speed (Fig. pareto + Fig. tradeoff).”* |
| **Page-limited** | Keep Pareto only | Already in paper | Trade-off figure → appendix; still regenerate both from CLI |
| **Do not** | Replace result tables with figures only | Evidence requirement | Figures supplement tables |

### Cross-references

| Phase | Link |
| ----- | ---- |
| **9** | Pareto = F7 primary; Phase 25 extends visualization |
| **23** | F7 finding text references both figures |
| **24** | Same CLI / JSON bundle; generate figures together |
| **16** | Visual encodes metric decoupling cascade |

### Completeness record

| Track | Status | Detail |
| ----- | ------ | ------ |
| **Engine** | ✅ Done | `eval/cross_dim/plot.py` (`save_tradeoff_figure`, `save_tradeoff_3d_figure`); wired in CLI + reporter. |
| **Documentation** | ✅ Done | This section; `METHODOLOGY.md` §6.8; [Figures row](#figures-offline-vs-online-pareto-l475-l608615) in paper alignment guide. |
| **Paper** | 📝 Pending | Add trade-off figure at rewrite step **10**; regenerate Pareto. **No new GPU jobs** for current grid. |

---

# Phase 26: Add an Explicit "Offline Does Not Mean Cheap" Discussion ✅ **Done (engine) · 📝 Paper pending**

> **Status (2026-08-20):** **Engine complete** — `eval/cost/oaken_taxonomy.py` exports five Oaken-inspired layers on every `EvaluationResult.cost` as `oaken_layers`. **Paper deferred** — `.tex` still conflates “Section A offline” with zero-cost offline; no Oaken citation or five-layer Discussion paragraph yet.

### Problem (paper today vs target)

| | Paper (`conference_101719.tex`) | Codebase (now) |
| --- | ----------------- | ---------------- |
| **Terminology** | “Section A offline fidelity” vs “Section B online” (L45, L95, L268) | **FIDELITY** branch + **`cost.oaken_layers`** five-way split |
| **Implicit claim** | “Offline” reads as **free** preprocessing | Layer 1 = FIDELITY **metrics** (not $); Layer 2 = **calibration cost** (TurboQuant Lloyd-Max) |
| **Online cost** | TurboQuant tok/s penalty in prose (L344, L617) | Layers 3–5: transform / attention / e2e in `cost.online` + SYSTEM |
| **Oaken cite** | Not in `.tex` | Cross-ref in Phase 20 Related Work §4 (bib pending) |

### Five Oaken layers (engine mapping)

| Layer | Meaning | Engine source | “Measured” when |
| ----- | ------- | ------------- | --------------- |
| **1. Offline evaluation** | Static FIDELITY before/around decode | `fidelity.*` → `oaken_layers[0].metrics` | FIDELITY branch ran |
| **2. Offline preprocessing** | Calibration / codebooks / rank search | `cost.offline.*` | `calibration_required` or `calibration_time_ms` set |
| **3. Online transformation** | Per-step compress/decompress | `cost.online.compress*` | `--kernel-cost` or aggregate CD time |
| **4. Online attention** | Attention with transformed cache | `cost.online.attention_cost_ms` | `--kernel-cost` |
| **5. End-to-end serving** | User-visible decode latency / tok/s | `system.throughput.*` | Default SYSTEM branch |

**Key distinction for paper:** Layer **1** is **not** Layer **2**. QJL can show moderate FIDELITY (Layer 1) with **zero** calibration (Layer 2) yet catastrophic BEHAVIOR — and high Layer 5 latency.

### Code artifacts

| Artifact | Path |
| -------- | ---- |
| Taxonomy builder | `eval/cost/oaken_taxonomy.py` — `OakenCostLayer`, `build_oaken_layers()` |
| JSON export | `EvaluationResult.cost.to_dict()["oaken_layers"]` (5 entries) |
| Tests | `tests/test_oaken_benchmark_dimensions.py` — offline eval ≠ preprocessing |

### Paper change log — section by section (`conference_101719.tex`)

| When | Section | Why | What to change |
| ---- | ------- | --- | -------------- |
| **Rewrite pass 2** | **§Methodology → COST** (new after SYSTEM, ~L285) | Retire “offline = free” | Add **Oaken-inspired cost taxonomy** subsection: five layers above; map to FIDELITY / `cost.offline` / `cost.online` / SYSTEM. Cite Oaken (bib from Phase 20). |
| **Rewrite pass 2** | **§Design Principles** (L95) | “Dual evaluation” conflates layers 1 and 5 | Rename to three-branch + **“Report FIDELITY metrics separately from offline preprocessing cost and online serving cost.”** |
| **Rewrite pass 2** | **§Discussion → Finding 3** (L617) | TurboQuant speed story | Frame as **Layers 3–5 decoupling**: high compression (FIDELITY memory) + heavy Layer 3 transform → low Layer 5 tok/s despite acceptable Layer 1 scores. |
| **Rewrite pass 2** | **§Discussion Implications** (L623) | Practitioner checklist incomplete | Add: *(v)~ distinguish offline **evaluation** from offline **preprocessing** cost; (vi)~ report online transform + attention + e2e separately when `--kernel-cost` enabled.* |
| **Do not** | Claim Oaken numbers | No Oaken implementation | Cite for **taxonomy** only; KVBench evidence from TQ/QJL/RocketKV |

### Cross-references

| Phase | Link |
| ----- | ---- |
| **3** | Phase 26 extends Phase 3 cost tree with explicit layers |
| **18** | Terminology: FIDELITY ≠ “offline cost” |
| **20** | Oaken bib in Related Work §4 |
| **27** | Layer 2 columns overlap calibration table |
| **23** | Finding 3 (memory ≠ throughput) uses Layers 3–5 |

### Completeness record

| Track | Status | Detail |
| ----- | ------ | ------ |
| **Engine** | ✅ Done | `oaken_layers` on every `evaluate_cost()`; robust to partial FIDELITY test fakes. |
| **Documentation** | ✅ Done | `METHODOLOGY.md` §6.5; this section. |
| **Paper** | 📝 Pending | COST subsection + Discussion paragraph. **No new GPU jobs.** |

---

# Phase 27: Add Calibration as a Benchmark Dimension ✅ **Done (engine) · 📝 Paper pending**

> **Status (2026-08-20):** **Engine complete** — Phase 27 columns exported as `cost.benchmark_dimensions` on every job; CSV via `ResultReporter` + `scripts/export_method_benchmark_table.py`. **Paper deferred** — `.tex` mentions TurboQuant Lloyd-Max “offline on Gaussian samples” (L181) but no cross-method calibration comparison table.

### Phase 27 column spec ↔ engine

| Property | Engine field | TQ (case study) | QJL | RocketKV |
| -------- | ------------ | --------------- | --- | -------- |
| Calibration required | `benchmark_dimensions.calibration_required` | **Yes** | No | No |
| Calibration data | `calibration_dataset` | `gaussian_synthetic` | `fixed_seed_projection` | `online_token_selection` |
| Calibration tokens | `calibration_tokens` | 1,000,000 | — | — |
| Calibration time | `calibration_time_ms` | measured at init | — | — |
| Calibration memory | `calibration_memory_bytes` | optional | — | — |
| Stateful | `stateful` (`reset_state` hook) | No | **Yes** | **Yes** |
| Online overhead | `online_overhead_ms_per_token` | from SYSTEM throughput | high (ms/tok in results) | moderate |

**Logical rule (tested):** `online_overhead_ms_per_token` = `throughput.latency_ms_per_token` when available; else `end_to_end_decode_cost_ms / generated_tokens`.

### Code ↔ paper alignment

| | Paper today | Code today |
| --- | ----------- | ---------- |
| **Calibration prose** | TurboQuant Lloyd-Max on Gaussian samples (L181) | `TurboQuantCompressor.offline_cost_metadata()` + export |
| **QJL/RocketKV** | “calibration-free” implied | Explicit `calibration_required=False` in JSON |
| **Stateful** | Not discussed | `stateful=True` when plug-in has `reset_state` |
| **Comparison table** | **Missing** | `results/method_benchmark_dimensions.csv` (static plug-ins) or per-job JSON |

### Paper change log — section by section (`conference_101719.tex`)

| When | Section | Why | What to change |
| ---- | ------- | --- | -------------- |
| **Rewrite pass 2** | **NEW table: Method benchmark dimensions** (Methodology after COST or appendix) | Fair comparison requires calibration + statefulness | Table from `method_benchmark_dimensions.csv` for TQ/QJL/RocketKV (+ identity). Columns match Phase 27 spec. |
| **Rewrite pass 2** | **§Case-Study Methods** (L165–167) | Per-method calibration differs | One sentence each: TQ offline Lloyd-Max; QJL/RocketKV calibration-free; RocketKV/QJL **stateful**. |
| **Rewrite pass 2** | **§Plug-in Interface** (L155) | Hooks exist but not named | List `offline_cost_metadata()`, `reset_state()`, `theoretical_compression_ratio`. |
| **Rewrite pass 2** | **§Discussion** | Calibration as hidden variable | Short paragraph: *“Calibration-free methods are not comparable to calibration-heavy quantizers on deployment cost alone — KVBench exports both.”* |
| **Re-sweep optional** | Per-job table | Legacy Phase-5 JSON lacks `benchmark_dimensions` | Re-run eval or merge static plug-in table for paper; **static table sufficient** for three case studies |
| **Do not** | Claim Palu/SnapKV calibration numbers in results | Not in Phase-5 sweeps | May cite from plug-in metadata in Methodology only |

### Cross-references

| Phase | Link |
| ----- | ---- |
| **3** | Phase 27 formalizes Phase 3 offline block as benchmark column |
| **26** | Layer 2 = calibration; Layer 5 = `online_overhead_ms_per_token` |
| **14** | `calibration` in reproducibility manifest |
| **4** | Taxonomy `calibration_free` flag aligns with `calibration_required` |

### Completeness record

| Track | Status | Detail |
| ----- | ------ | ------ |
| **Engine** | ✅ Done | `benchmark_dimensions` on `cost`; CSV export script; reporter columns `stateful`, `online_overhead_ms_per_token`. |
| **Documentation** | ✅ Done | `METHODOLOGY.md` §6.5; `tests/test_oaken_benchmark_dimensions.py`. |
| **Paper** | 📝 Pending | Calibration dimension table + Discussion sentence. Static CSV OK for current submission. |

---

# Phase 28: Add a Workload-Aware Discussion 📝 **Paper only · engine scoped**

> **Status (2026-08-20):** **Paper-writeup + scope documentation only** — no new workload integration (Phase **11 deferred**). The engine already runs synthetic BEHAVIOR tasks (`eval/behavior/`) but the **paper correctly reports WikiText-2 PPL + throughput only**. Phase 28 adds the **Discussion / future-work framing** so reviewers do not treat WikiText as universal.

### Codebase vs paper alignment

| Workload type | Engine | Paper results | Phase 28 claim |
| ------------- | ------ | ------------- | -------------- |
| WikiText-2 PPL | ✅ `eval/behavior/task_quality.py` | ✅ Primary BEHAVIOR metric | **In scope** — controlled SLM case study |
| Throughput / latency | ✅ SYSTEM | ✅ Reported | **In scope** |
| Retrieval (needle) | ✅ synthetic `eval/behavior/retrieval.py` | ❌ Not in `.tex` | **Future work** (Phase 11) |
| Instruction following | ✅ synthetic | ❌ Not in `.tex` | **Future work** |
| Reasoning | ✅ opt-in `--reasoning` | ❌ Not in `.tex` | **Future work** |
| Long-context 2K–32K | ❌ capped 512 (Phase 12 deferred) | ctx 128–512 | **Future work** |
| RAG / multi-turn / serving | ❌ Phase 13 not planned | ❌ | **Future work** — cite Cache in the Wild, CacheBlend in Related Work |

**Phase 23 Finding 6** (*best method vs workload*) must remain **unanswered** — one explicit sentence in Discussion.

### Target Discussion paragraph (canonical)

> KV reuse and compression interact with **workload shape** (prompt length distribution, retrieval-heavy vs chat, long generation vs single-shot scoring). Our controlled factorial uses WikiText-2 sliding-window perplexity and greedy 64-token generation as an **SLM inference-engineering testbed**, not a claim about datacenter serving mixes. Extending KVBench to LongBench/RULER-style probes, multi-turn dialogs, and longer contexts is important future work (Phases 11–12).

### Paper change log — section by section (`conference_101719.tex`)

| When | Section | Why | What to change |
| ---- | ------- | --- | -------------- |
| **Rewrite pass 2** | **§Experiments setup** (L222) | WikiText stated but scope unstated | Add: *“We scope BEHAVIOR to WikiText-2 PPL under incremental decode; broader workload types are future work.”* |
| **Rewrite pass 2** | **§Discussion → Finding 6** | F6 must not be answered | Explicit: *“Finding 6 (workload dependence) is not evaluated in this study.”* |
| **Rewrite pass 2** | **NEW `\paragraph{Workload scope and limits.}`** (Discussion, before Implications) | Phase 28 deliverable | Canonical paragraph above + bullets: long-context, RAG, multi-turn, reasoning rollouts → Phase 11–12 / cite `kvcachewild2025`, `cacheblend2025`, `yuan2026shortrl`. |
| **Rewrite pass 2** | **§Conclusion future work** (L629) | Already mentions “alternative online metrics” | Align with Phase 28 list; cross-ref Phase 11 deferred explicitly |
| **Rewrite pass 2** | **§Related Work §4** (Phase 20) | Workload realism literature | Cache in the Wild + CacheBlend support workload-aware **motivation**, not empirical claims |
| **Do not** | Add LongBench/RULER results | Phase 11 not planned | Prose + future work only |
| **Do not** | Soften WikiText findings | Case study valid | Frame as **controlled testbed**, not universal |

### Cross-references

| Phase | Link |
| ----- | ---- |
| **11** | Workload extension — **deferred**; Phase 28 cites as future |
| **12** | Long-context scaling — deferred |
| **13** | Serving stacks — out of scope |
| **19** | SLM inference-engineering testbed framing |
| **23** | F6 left open |

### Completeness record

| Track | Status | Detail |
| ----- | ------ | ------ |
| **Engine** | ✅ Scoped | BEHAVIOR modules exist; default paper grid = WikiText PPL only (`METHODOLOGY.md`, Phase 11 flagged). |
| **Documentation** | ✅ Done | This section; `METHODOLOGY.md` §6.5 workload note; Phase 11 section unchanged (deferred). |
| **Paper** | 📝 Pending | Discussion workload paragraph + F6 disclaimer + Experiments scope sentence. **No engine changes required.** |

---

# Phase 29: Use Recent Literature to Strengthen, Not Replace, Your Existing Work 📝 **Paper only · docs staged**

> **Status (2026-08-20):** **Paper-writeup + documentation phase** — no new compressor implementations. The engine already embodies the *response* to this literature (three branches, cost/Oaken layers, Pareto/cross-dim, controlled conditions). Phase 29 adds **citations and narrative hooks** at rewrite time. Staging BibTeX and alignment table live in-repo; **do not merge into `reference.bib` until metadata verified**.

These papers **shape methodology and positioning** — not every paper requires implementation.

### Literature ↔ engine ↔ paper map

| # | Paper | Key (target) | Engine touchpoint | Paper use | In `.tex` today? |
| - | ----- | ------------ | ----------------- | --------- | ---------------- |
| 1 | **Oaken** — ISCA 2025 | `oaken2025` | `eval/cost/oaken_taxonomy.py` (Phase 26) | Related Work §4; COST subsection | ❌ — staging only |
| 2 | **SCOPE** — ACL 2025 | `scope2025` | SYSTEM TTFT/ITL vs BEHAVIOR PPL | Related Work §4; prefill/decode eval gap | ❌ |
| 3 | **RocketKV** — ICML 2025 | `rocketkv` | `compressors/rocketkv.py` | Case study + §3 hybrid | ✅ |
| 4 | **TurboAttention** — MLSys 2025 | `turboattention2025` | `eval/system/kernel_cost.py` | Related Work §3–§4 (attention execution cost) | ❌ |
| 5 | **R-KV** — NeurIPS 2025 | `rkv2025` | BEHAVIOR `reasoning.py` (opt-in) | Future work / F6 motivation | ❌ |
| 6 | **Pitfalls** — ACL 2026 | `chen2026pitfalls` | BEHAVIOR stack | Related Work §4; F1/F4 | ✅ L56, L77 |
| 7 | **OjaKV** — ACL 2026 | `ojakv2026` | `cost.benchmark_dimensions.stateful` | Related Work §3; Phase 27 table | ❌ |
| 8 | **HybridKV** — ACL 2026 | `hybridkv2026` | Taxonomy D+E (Phase 5 deferred) | Related Work §3 | ❌ |
| 9 | **Benchmarking KV-Cache Optimizations…** — 2026 | `kvbench2026serving` | Phase 30 contrast | Intro + Related Work §4 — **explicit differentiation** | ✅ L54, L77–78 |
| 10 | **KVCache Cache in the Wild** — USENIX ATC 2025 | `kvcachewild2025` | Phase 28 workload scope | Related Work §4; Discussion F6 | ❌ |
| 11 | **CacheBlend** — EuroSys 2025 | `cacheblend2025` | Phase 28 RAG/serving future | Related Work §4 | ❌ |

**Repo artifacts:**

| Artifact | Path |
| -------- | ---- |
| Alignment table + narrative roles | `docs/literature/LITERATURE_ALIGNMENT.md` |
| Staging BibTeX (merge at rewrite) | `docs/literature/staging_entries.bib` |
| Overlap with Phase 20 bib list | [Phase 20 bibliography work](#bibliography-work-bib--reference) |

### Paper change log — section by section (`conference_101719.tex`)

| When | Section | Why | What to change |
| ---- | ------- | --- | -------------- |
| **Rewrite pass 1** (with Phase 20 §4) | **Related Work §4** | Missing cost/workload/decode-eval literature | Add Oaken, SCOPE, Cache in the Wild, CacheBlend cites from merged `staging_entries.bib` |
| **Same pass** | **Intro ¶4** (extend L56) | Recent evidence list incomplete | Add Oaken (offline/online cost), SCOPE (decode eval), workload papers (Cache in the Wild) — **do not** claim KVBench implements them |
| **Rewrite pass 2, step 11** | **§COST** (Phase 26) | Oaken motivates five-layer taxonomy | Cite `oaken2025`; contrast KVBench software instrumentation vs Oaken hardware co-design |
| **Rewrite pass 2, step 12** | **Discussion workload** (Phase 28) | Workload realism motivation | Cite `kvcachewild2025`, `cacheblend2025`; F6 remains unanswered |
| **Do not** | Methods / Results | No new algorithms | Do **not** add OjaKV/HybridKV/R-KV **results** without implementation sweeps |

### Completeness record

| Track | Status | Detail |
| ----- | ------ | ------ |
| **Engine** | — | No new implementations; existing engine is the methodological response. |
| **Documentation** | ✅ Done | `docs/literature/LITERATURE_ALIGNMENT.md`, `staging_entries.bib`, this section. |
| **Paper** | 📝 Pending | Merge staging bibs + Related Work / Intro cites at rewrite. |

---

# Phase 30: Explicitly Address the Closest Competing Benchmark 📝 **Paper only**

> **Status (2026-08-20):** **Paper-writeup phase only**. The paper **already cites** `kvbench2026serving` (L54, L56, L66, L78, L81) but treats it as complementary background — **not** as the closest benchmark requiring an explicit contrast paragraph. Phase 30 makes the distinction **prominent in Intro + Related Work §4** (not a footnote).

### Paper today vs target

| | `conference_101719.tex` today | Target after rewrite |
| --- | ----------------------------- | -------------------- |
| **Citation** | ✅ `kvbench2026serving` in Intro + Related Work | Keep — do not remove |
| **Framing** | “Concurrent serving benchmarks stress joint quality…” (L78) | **Explicit contrast paragraph** (below) |
| **Risk** | Names collide (“KVBench” vs their title) | Clarify: *this repo* = controlled interception instrument; *their work* = serving-stack benchmark across workloads |
| **Positioning subsection** | L80–81 “shared yardstick” | Replace with Phase 17 novelty + Phase 30 contrast (folded into §4 closing per Phase 20) |

### Canonical contrast (use in Intro + Related Work §4)

> Agrawal and Mayer evaluate existing KV-cache optimizations across **task quality and system performance for long-context serving**. KVBench differs by providing a **controlled KV interception and transformation layer** in which different transformations execute through a **common incremental autoregressive decode path**, enabling matched **representation-level (FIDELITY), behavioral, and runtime (SYSTEM)** analysis under fixed model, input, and decode conditions — a pre-deployment factorial instrument on SLMs, not a serving-engine leaderboard.

### Codebase alignment (why the contrast is accurate)

| Their emphasis (serving benchmark) | KVBench emphasis (this repo) | Evidence |
| ---------------------------------- | ------------------------------ | -------- |
| Workloads + system metrics in serving context | Controlled plug-in factorial on fixed decode loop | `eval/controlled_conditions.py`, `framework/kv_engine.py` |
| Existing optimizations as deployed | TQ / QJL / RocketKV as **case studies** under same engine | `compressors/`, Phase-5 bundles |
| Long-context serving | SLM ctx 128–512, batch 1 (Phase 12 deferred) | `configs/model.yaml`, Phase 28 scope |
| Joint quality + efficiency | FIDELITY + BEHAVIOR + SYSTEM + cost export | `eval/runner.py`, `eval/cost/` |

### Paper change log — section by section (`conference_101719.tex`)

| When | Section | Why | What to change |
| ---- | ------- | --- | -------------- |
| **Rewrite pass 1, step 8** (with Phase 20) | **Related Work §4 closing** | Reviewer will search for this paper | Add **dedicated `\paragraph{Relation to serving benchmarks.}`** with canonical contrast above |
| **Rewrite pass 1** | **Intro ¶6** (KVBench description, L58) | Name collision + contribution clarity | One sentence: *“Unlike serving-stack benchmarks that score deployed optimizations~\cite{kvbench2026serving}, KVBench fixes the incremental decode path and swaps compressors as plug-ins.”* |
| **Rewrite pass 1** | **DELETE L80–81 positioning** | Absorbed into §4 | Fold contrast into *What is still missing?* — do **not** dismiss their work |
| **Do not** | Abstract empirical claims | Different scopes | Do **not** claim superiority on serving workloads; claim **complementary controlled evaluation** |

### Cross-references

| Phase | Link |
| ----- | ---- |
| **17** | Safe novelty — instrument vs serving benchmark |
| **18** | KVBench is not vLLM/SGLang |
| **20** | §4 evaluation section hosts contrast |
| **29** | `kvbench2026serving` row in literature map |

### Completeness record

| Track | Status | Detail |
| ----- | ------ | ------ |
| **Engine** | — | No changes. |
| **Documentation** | ✅ Done | This section; `LITERATURE_ALIGNMENT.md` §Closest competing benchmark. |
| **Paper** | 📝 Pending | Intro + Related Work §4 contrast paragraph. |

---

# Phase 31: Clean the Bibliography 📝 **Paper only · audit tooling done**

> **Status (2026-08-20):** **Audit tooling complete** — `scripts/audit_bibliography.py` + `tests/test_bibliography_audit.py` validate `reference.bib` against `conference_101719.tex` and Phase 31 rules. **Paper `.bib` cleanup deferred** — apply at rewrite; do **not** edit `reference.bib` until cite keys are finalized with Phase 20/29 merges.

### Audit snapshot (2026-08-20)

| Check | Status | Detail |
| ----- | ------ | ------ |
| All `\cite{}` keys resolve in `reference.bib` | ✅ Pass | 27 unique cite keys in `.tex` |
| Phase 31 **keep** list | ✅ Pass | H2O, Scissorhands, StreamingLLM, SnapKV, … RocketKV, Pitfalls, KVBench Serving, WikiText |
| **SnapKV venue** | ✅ Already NeurIPS 2024 | `li2024snapkv` → `Advances in Neural Information Processing Systems`, 2024 — **no ACL fix needed** (commented L641 in `.tex` is stale) |
| **Anonymous entries** | ⚠️ Flagged | `expectedattn2026`, `qjlcs2025` — replace with verified authors or remove cites at rewrite |
| **Remove candidates still cited** | ⚠️ Action at rewrite | `costoptgqa2025`, `qjlcs2025`, `expectedattn2026`, `yuan2026shortrl` — drop or replace per row below |
| **Phase 29 staging** | 📝 Not merged | 8 keys in `docs/literature/staging_entries.bib` — merge after metadata verification |

### Definitely keep (must remain in `reference.bib`)

H2O, Scissorhands, StreamingLLM, SnapKV, PyramidKV, MiniCache, QJL, Palu, Outlier Tokens, KVSink, AsymKV, XQuant, Qwen3, OLMo 2, TurboQuant, HqeKV, The Pitfalls, KVBench Serving (`kvbench2026serving`), WikiText, RocketKV — plus cites already in `.tex` for CompressKV, PatternKV, MHA→GQA, PagedEviction, etc.

### Verify / fix at rewrite

| Key | Issue | Action |
| --- | ----- | ------ |
| `feng2024adakv` | NeurIPS 2025 + arXiv note | Verify proceedings pages |
| `su2025kvsink` | arXiv only | Promote to venue or keep as arXiv with note |
| `rocketkv` | ICML 2025 PMLR 267 | ✅ metadata present — spot-check pages |
| `compresskv2026` | arXiv 2026 | Update if published |
| `jin2025mha2gqa` | Findings EMNLP 2025 | Keep; distinct from `costoptgqa2025` |
| `li2024snapkv` | Comment L641 says ACL 2024 | **Delete stale comment**; bib is already NeurIPS |

### Remove unless specifically needed (rewrite pass)

| Key | Why remove | Replacement strategy |
| --- | ---------- | -------------------- |
| `costoptgqa2025` | Anonymous-adjacent; overlaps `jin2025mha2gqa` / `compresskv2026` | Keep `jin2025mha2gqa` + `compresskv2026` for GQA narrative |
| `qjlcs2025` | Anonymous OpenReview | Drop cite or replace with peer-reviewed QJL follow-up if available |
| `expectedattn2026` | Anonymous ICLR entry | Drop cite or replace when authors public |
| `yuan2026shortrl` | Reasoning motivation only; arXiv 2025 | Move to future work / Phase 28; cite `rkv2025` staging if reasoning scope expanded later |

### Engine tooling

```bash
python scripts/audit_bibliography.py
python scripts/audit_bibliography.py --json results/bibliography_audit.json
```

**Tests:** `tests/test_bibliography_audit.py`

### Paper change log — section by section (`conference_101719.tex`)

| When | Section | Why | What to change |
| ---- | ------- | --- | -------------- |
| **Rewrite pass 1** | **`reference.bib`** | Anonymous + duplicate GQA cites | Apply remove/replace table; merge `staging_entries.bib`; re-run audit CLI |
| **Same pass** | **Intro L54** | `yuan2026shortrl` + dense cite list | Trim remove candidates; keep Pitfalls + kvbench2026serving |
| **Same pass** | **Related Work L75** | `expectedattn2026` anonymous | Remove or footnote “under review” — prefer drop |
| **Same pass** | **Comment block L641–642** | Stale SnapKV ACL comment | Delete commented `\bibitem{li2024snapkv}` ACL line |
| **Do not** | Break build | All cites must resolve | Run `audit_bibliography.py` after every bib edit |

### Completeness record

| Track | Status | Detail |
| ----- | ------ | ------ |
| **Engine** | ✅ Done | `scripts/audit_bibliography.py`, tests, staging bib. |
| **Documentation** | ✅ Done | This section; audit snapshot above. |
| **Paper** | 📝 Pending | Bib cleanup at rewrite — **no `.tex` / `.bib` edit until then**. |

---

# Phase 32: The Final Conceptual Model of KVBench ✅ **Docs done · 📝 Paper figure pending**

> **Status (2026-08-20):** **Engine implements the full pipeline** below (interception → taxonomy → three branches → Pareto/cross-dim/findings). **Paper figure + narrative deferred** — `.tex` pipeline figure (L98–102) still shows dual Section A/B boxes.

### Target conceptual model

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
    Attention            Retrieval*          ITL
    Similarity           Instruction*        Throughput
    Memory               Reasoning*          VRAM*
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ↓
                 QUALITY / MEMORY / SPEED
                            ↓
                  Pareto + Cross-Dim Analysis
                            ↓
                    Research Findings F1–F7

    * engine default/opt-in; paper grid = WikiText PPL + tok/s (Phase 28)
```

### Code ↔ diagram mapping

| Diagram block | Code / docs |
| ------------- | ----------- |
| KV interception layer | `framework/kv_engine.py`, `eval/controlled_conditions.py` |
| Eviction / Quantization / Projection | `compressors/taxonomy.py` categories A–C (+ D/E hybrids) |
| Same decode loop | `KVCacheEngine.generate()`, Phase 6–7 fixed axes |
| FIDELITY / BEHAVIOR / SYSTEM | `eval/fidelity/`, `eval/behavior/`, `eval/system/` |
| Quality / memory / speed | `eval/cost/`, FIDELITY.memory, SYSTEM.throughput |
| Pareto + cross-dim | `eval/pareto/`, `eval/cross_dim/` (Phases 9, 24–25) |
| Research findings | Phase 23 F1–F7 narrative spec |

**Canonical doc:** [`docs/architecture/SYSTEM_DESIGN.md` §Phase 32](../architecture/SYSTEM_DESIGN.md#phase-32-conceptual-model-end-to-end-story)

### Paper change log — section by section (`conference_101719.tex`)

| When | Section | Why | What to change |
| ---- | ------- | --- | -------------- |
| **Rewrite pass 2** | **Fig. pipeline** (`\label{fig:pipeline}`, L98–102) | Dual Section A/B diagram | Regenerate asset: interception layer → taxonomy fork → three branches → analysis layer; caption lists FIDELITY/BEHAVIOR/SYSTEM + cost |
| **Same pass** | **§Methodology opening** (L83–86) | Narrative doesn't match figure | One paragraph walking top-to-bottom through diagram |
| **Same pass** | **Abstract** (L45) | Implied two-metric story | Mention three branches + controlled interception (Phase 1) |
| **Optional** | **Discussion** before findings | Reader orientation | Small inline version of diagram or pointer to Fig. pipeline |
| **Do not** | Claim full BEHAVIOR grid in results | Phase 28 scope | Asterisk metrics in figure caption as “engine capabilities; paper reports WikiText PPL + SYSTEM” |

### Completeness record

| Track | Status | Detail |
| ----- | ------ | ------ |
| **Engine** | ✅ Done | Full pipeline implemented (Phases 1–10, 14, 24–27). |
| **Documentation** | ✅ Done | `SYSTEM_DESIGN.md` §Phase 32; this section. |
| **Paper** | 📝 Pending | Regenerate pipeline figure + caption. |

---

# Phase 33: The new paper narrative in one sentence 📝 **Paper only**

> **Status (2026-08-20):** **Paper-writeup phase only** — canonical identity sentence for Abstract, Conclusion, and reviewer elevator pitch. Aligns with Phases 15 (research question), 17 (novelty), 22 (contributions), and 32 (conceptual model).

### Canonical sentence

> **KVBench is not primarily a benchmark asking which KV-cache compression method wins; it is a controlled inference-time experimentation framework for understanding how different KV transformations trade representation fidelity, model behavior, memory efficiency, and actual generation performance under matched conditions.**

### Where to place

| Location | Current problem | Phase 33 action |
| -------- | --------------- | --------------- |
| **Abstract closing** (L45) | “yardstick for comparing methods” tone | Replace closing clause with canonical sentence (trim for length if needed) |
| **Conclusion opening** (L627) | “benchmarking framework that bridges offline/online” | Open with canonical sentence; then enumerate Phase 22 contributions |
| **Intro contributions** (L60) | Horse-race framing | Optional epigraph before `\textbf{Contributions.}` — one sentence max |
| **Title subtitle** (optional) | “Bridging offline/online” | Consider “controlled inference-time experimentation framework” phrase |

### Paper change log — section by section (`conference_101719.tex`)

| When | Section | Why | What to change |
| ---- | ------- | --- | -------------- |
| **Rewrite pass 1, step 7** (Phase 22) | **Conclusion L627–629** | Old identity persists | First sentence = Phase 33 canonical; remove “yardstick for comparing KV compression methods” |
| **Same pass** | **Abstract L45** | Dual-metric + comparison framing | Lead with controlled experimentation; close with methodology-for-evaluating-KV-transformations clause |
| **Same pass** | **Intro L60** | Contributions sound like rankings | Contributions follow identity sentence — protocol first, case studies second (Phase 22) |
| **Do not** | Results tables | Numbers unchanged | Identity is framing only |

### Cross-references

| Phase | Link |
| ----- | ---- |
| **15** | Research question asks *how to evaluate*, not *who wins* |
| **17** | Novelty = protocol + engine |
| **22** | Contributions packaging mirrors this sentence |
| **32** | Diagram is the visual expansion of this sentence |

### Completeness record

| Track | Status | Detail |
| ----- | ------ | ------ |
| **Engine** | — | Identity describes existing instrument. |
| **Documentation** | ✅ Done | `LITERATURE_ALIGNMENT.md`; this section; cross-refs Phases 15, 22, 32. |
| **Paper** | 📝 Pending | Abstract + Conclusion + optional Intro epigraph. |

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
