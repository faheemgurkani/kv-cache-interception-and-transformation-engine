# KVBench: Complete Research Improvement Roadmap

## Completeness record — Phases 1–7 (verified 2026-08-19)

Tracks **engine** (code + tests), **documentation** (in-repo docs), and **paper** (`docs/research_paper_writeup/conference_101719.tex`). Phases **5** and **8** are flagged **not planned** — design reference only.

**Executive verdict:** Phases **1–4**, **6** (code/docs), and **7** are **complete in the engine and documentation**. The paper still reflects the **pre-redesign** framing (Section A/B, three case-study methods, no cost/taxonomy/controlled-conditions export). Paper updates for Phases 1–7 are **deferred as one coordinated rewrite** — see per-phase **Paper** rows below for exact locations.

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

**Cross-cutting tests:** `tests/test_eval_runner.py`, `tests/test_controlled_conditions.py`, `tests/test_cost_accounting.py`, `tests/test_taxonomy.py`, `tests/test_behavior_modules.py`, `tests/test_system_modules.py`, `tests/test_*_reference.py`.

**Paper rewrite hub:** all pending paper work targets [`docs/research_paper_writeup/conference_101719.tex`](research_paper_writeup/conference_101719.tex). Export live contracts from any job JSON: `result.to_dict()["controlled_conditions"]`.

**Intentional engine gaps (documented, not paper blockers):**

- BEHAVIOR retrieval/instruction/reasoning use **synthetic in-repo generators**, not LongBench/RULER (`CURRENT_STATE.md`).
- SYSTEM peak VRAM / GPU util require CUDA (`--peak-memory`, `--gpu-utilization`).
- Reasoning is opt-in (`--reasoning`); skip flags: `--skip-retrieval`, `--skip-instruction-following`.
- Hybrid FIDELITY/`recurrent` extension: `eval/fidelity/recurrent.py` (Falcon-H1).

Authoritative metric definitions: [`docs/methodology/METHODOLOGY.md`](methodology/METHODOLOGY.md) §1.1, §6.

### Paper rewrite checklist (Phases 1–7, coordinated pass)

When updating [`conference_101719.tex`](research_paper_writeup/conference_101719.tex), apply in this order:

1. **Global rename:** Section A → **FIDELITY**; Section B → split into **BEHAVIOR** (quality/tasks) and **SYSTEM** (latency/throughput/memory).
2. **Abstract + Introduction** (L45–60): three-branch protocol; controlled interception framing (Phases 1, 6).
3. **Methodology — Design Principles + Fig. pipeline** (L86–102): three branches; interception diagram (Phase 6).
4. **Methodology — Evaluation Protocol** (L255–315): three subsubsections FIDELITY / BEHAVIOR / SYSTEM; add Cost subsection (Phase 3); add Controlled conditions table (Phase 7).
5. **Methodology — Taxonomy** (new, after plug-ins): categories A–E table (Phase 4); case studies remain TQ/QJL/RocketKV unless new sweeps added.
6. **Results tables** (L321+): column/ caption renames only unless re-running eval.
7. **Discussion + Conclusion** (L598+, L627+): fidelity–behavior gap; SYSTEM tradeoffs; engine-as-contribution wording.

Phases **5** and **8** require **no paper changes**.

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
| **Paper** | 📝 Pending | **Abstract** (L45): replace “dual Section A/B” with **FIDELITY / BEHAVIOR / SYSTEM**. **Introduction** (L58–60): three axes under “What”. **§Design Principles** (L90–96): bullet “Dual evaluation” → three-branch contract. **Fig. pipeline caption** (L101): rename Section A/B in caption. **§Evaluation Protocol** (L255–257): restructure as three subsubsections. **Discussion/Conclusion** (L598+, L627+): “offline–online” → “fidelity–behavior gap”; SYSTEM as third lens. |

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
| **Paper** | 📝 Pending | **§Section A: Offline Fidelity** (L267–274) → **§FIDELITY**. Add: relative reconstruction error, attention-output RMSE, attention-distribution KL (already in code). Table headers “Section A” → “FIDELITY” throughout Results (e.g. L347, L417). Optional footnote: recurrent sub-metric for hybrid models only. |

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
| **Paper** | 📝 Pending | **§Section B: Online Inference** (L276–284) → split: **BEHAVIOR** (PPL + task probes) vs legacy “online” wording. Paper currently reports **PPL only** in results tables — add short paragraph on synthetic retrieval/instruction-following protocol (or defer to appendix with “future work: LongBench/RULER”). Rename table captions “Section B PPL” → “BEHAVIOR / perplexity”. |

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
| **Paper** | 📝 Pending | Throughput appears under Section B (L284, results tables L375+). Add **§SYSTEM** subsection: TTFT, ITL, decode latency — not just tok/s. Discussion (L608+): cite TurboQuant ~0.08 tok/s as SYSTEM–FIDELITY tradeoff. Optional appendix table for `--peak-memory` / `--kernel-cost` CUDA runs. |

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
| **Paper** | 📝 Pending | **Not in paper today.** Add **§Cost accounting** under Methodology (after Evaluation Protocol, ~L315): compression / offline / online cost tree (mirror plan diagram above). One summary table column or appendix row: calibration required?, theoretical ratio, decode ms/token. Cite TurboQuant Lloyd-Max calibration vs identity/QJL calibration-free. |

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
| **Paper** | 📝 Pending | **§Case-Study Methods** (L165–167): still **three** families (TQ, QJL, RocketKV). Related Work cites Palu/SnapKV (L72, L75) but no results. Options: (a) add taxonomy table (A–E) in Methodology; (b) list SnapKV/Palu as **engine extensions** in Conclusion future work (L629+) without new result tables; (c) optional appendix plug-in inventory. Do **not** claim SnapKV/Palu empirical results unless new sweeps are run. |

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
| **Paper** | 📝 Pending | Reframe contribution — **not** “another benchmark.” **Title/Abstract** (L31, L45): “controlled interception-and-transformation engine” over “benchmarking framework.” **Contributions** (L60): item (1) → interception engine + three-branch protocol. **§Design Principles** (L90–95): lead with plug-in isolation + controlled path. **Fig. pipeline** (L98–102): caption → FIDELITY/BEHAVIOR/SYSTEM + interception diagram (Phase 6 ASCII). **Discussion** (L598): “benchmarking study” → “controlled experimentation framework.” |

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
| **Paper** | 📝 Pending | **§Experimental setup** (L218–229): add **Controlled conditions** table — model, tokenizer, WikiText-2, ctx 128/256/512, batch 1, greedy decode, A10G, generation 64, PPL stride 512; state explicitly that **only compressor + budget vary**. Reference reproducibility: “per-job JSON includes `controlled_conditions`.” Can paste a sanitized example from `results/*/jobs/*.json`. Align with Phase 6 narrative rewrite. |

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

# Phase 9: Add Pareto Analysis

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



---

# Phase 10: Add Hardware-Aware Evaluation

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

---

# Phase 11: Add a Realistic Workload Dimension

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

---

# Phase 12: Add Workload Scaling

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

---

# Phase 13: Add a Serving-Engine Validation Path

You don't need to turn KVBench into vLLM.

Instead:

```text
KVBench
   │
   ├── Controlled research environment
   │
   └── Optional serving integration
           │
           ├── vLLM
           └── SGLang
```

The idea is:

> First establish controlled results inside KVBench, then validate selected findings inside a real serving engine.

This would make your systems claim much stronger.

The recent literature increasingly connects compression to actual serving systems and memory-management architectures. 

---

# Phase 14: Add Reproducibility as a First-Class Feature

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

This is particularly important because one of the central problems in the literature is that different papers use different:

* models
* tasks
* budgets
* serving stacks

making direct comparison difficult. 

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
