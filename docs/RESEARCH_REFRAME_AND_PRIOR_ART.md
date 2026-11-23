# KVBench Research Reframe and Prior-Art Survey

**Previously:** `docs/MISC_Research.md` (renamed and restructured 2026-11-24).

| | |
|---|---|
| **Status** | ✅ **Adopted 2026-11-24.** This is the authoritative research framing for the project. |
| **What this is** | The research framing for the KVBench paper (question, contributions, hypotheses, evidence requirements), the prior-art survey behind it, and the verdict on publication level. Work-package status is **not** tracked here: see [`RESEARCH_REDESIGN_PLAN.md` → Part 0](RESEARCH_REDESIGN_PLAN.md#part-0--reframe-adoption-and-master-status-2026-11-24). |
| **Original survey window** | Jan 2025 – 1 Oct 2026 (peer-reviewed work at top venues, plus key preprints). |
| **Updated with** | [`literature/FRAMING_VALIDATION_2026-11-24.md`](literature/FRAMING_VALIDATION_2026-11-24.md): a re-validation against literature posted up to 24 Nov 2026 Later reading extended this to 46 downloaded papers in `literature/papers/` (local only; gitignored, not in the repository). |
| **Project status it depends on** | [`audits/DOCS_PLANS_RESULTS_ALIGNMENT_AUDIT.md`](audits/DOCS_PLANS_RESULTS_ALIGNMENT_AUDIT.md) and [`audits/GEMMA3_POST_OVERHAUL_AUDIT.md`](audits/GEMMA3_POST_OVERHAUL_AUDIT.md). |
| **Related plans** | [`RESEARCH_REDESIGN_PLAN.md`](RESEARCH_REDESIGN_PLAN.md), [`ENGINE_AND_EVALUATION_FRAMEWORKS_REDESIGN_PLAN.md`](ENGINE_AND_EVALUATION_FRAMEWORKS_REDESIGN_PLAN.md). Both carry a Part 0 (2026-11-24) that implements this document: the research plan holds master status (R0–R16) and the per-phase re-evaluation; the engine plan holds engine items (E1–E8). |

Markers used below: **[Oct 2026]** = finding of the original survey; **[Nov 2026 update]** = change from the 24 Nov re-validation.

## Contents

1. [Executive summary](#1-executive-summary)
2. [The reframe: from "benchmark" to "transfer and accounting study"](#2-the-reframe-from-benchmark-to-transfer-and-accounting-study)
3. [Three contributions, each with a measurable claim](#3-three-contributions-each-with-a-measurable-claim)
4. [Repositioning against existing work](#4-repositioning-against-existing-work)
5. [What the evidence needs](#5-what-the-evidence-needs)
   - [5.1 Hypotheses (draft)](#51-hypotheses-draft)
6. [Publication-level verdict](#6-publication-level-verdict)
7. [Prior-art survey (Jan 2025 – 1 Oct 2026)](#7-prior-art-survey-jan-2025--1-oct-2026)
8. [Post-cut-off update (1 Oct – 24 Nov 2026)](#8-post-cut-off-update-1-oct--24-nov-2026)
9. [Perishability](#9-perishability)
10. [Related Work outline](#10-related-work-outline)
11. [Recommendations](#11-recommendations)
12. [Caveats](#12-caveats)
13. [Sources](#13-sources)

---

## 1. Executive summary

**The gap.** The survey showed that "a better way to compare KV compressors" is crowded, but "do KV-compression results survive a change of attention architecture?" is not. No paper found, published or posted up to 24 Nov 2026, does what KVBench can do as a whole:
- a single controlled decode loop hosting heterogeneous plug-in compressors;
- run on architecturally diverse small models (MHA / GQA / MQA+sliding / MLA / Mamba2-hybrid);
- with a three-branch FIDELITY / BEHAVIOR / SYSTEM protocol and explicit offline-vs-online cost accounting.

**But several individual claims are already published** (harness, taxonomy, "ratio ≠ speedup", "no single winner"). KVBench should therefore present itself as an architecture-aware, cost-accounted measurement instrument, not as "the first unified KV benchmark".

**[Nov 2026 update] Where things stand now:**
1. **The reframe still holds and is still unclaimed as a whole.** Its window is short: since 1 Oct at least 12 new preprints have each taken a piece of the story. Estimated window: about one submission cycle (3–6 months).
2. **Ranking transfer (C2) is now the strongest, best-motivated contribution.** Published side results show architecture-induced ranking flips (RIFT, JSSA), and a published claim says trends transfer (Agrawal & Mayer, tested on two GQA models only).
3. **The hybrid "claimed vs realized" gap (C1) is real but no longer new as a phenomenon.** vLLM RFC #55196 measures it on Falcon-H1 directly, and Shin (2026) runs a joint KV + recurrent-state quantization study with Falcon-H1 as its control. What remains ours is a cross-mechanism × cross-architecture realized-savings table.
4. **"Fidelity ≠ behavior" is contradicted in its blanket form** (KV-CoRE, KVDiagnosis, Hold Onto That Thought). The defensible claim is that proxy validity is conditional on mechanism and architecture, reported with confidence intervals.
5. **The scale objection is stronger than first stated.** At ≤ 1K tokens the hybrid effect is essentially zero (vLLM measures 1.00×), and Gemma3's sliding window is 512, so the current 128–512 grid makes the architecture claim trivial.
6. **Our only hybrid is the older kind.** The 2026 hybrid frontier is Gated DeltaNet (GDN; Qwen3.5 / 3.6 / 3.8, including sub-2B checkpoints), and Mamba-2 and GDN states behave oppositely under quantization.
7. **MLA expanded-vs-native is still open**, but needs native-latent interception (our Gate C).
8. **Project state:** current measured results are not submittable (most BEHAVIOR/SYSTEM numbers invalid; see the alignment audit). All conclusions here assume a clean re-run.

---

## 2. The reframe: from "benchmark" to "transfer and accounting study"

### 2.1 Central research question

> Do published KV-cache compression results transfer across attention architectures (MHA, GQA, MQA + sliding window, MLA, attention-Mamba hybrid), and what does a compression ratio actually buy once the full inference state and offline cost are counted?

Nearly all prior work (KVPress, Rethinking, SCBench, Agrawal & Mayer, KVDiagnosis) evaluates dense GQA transformers at 7–8B. They implicitly assume the answer is "yes, it transfers". This study tests that assumption. KVBench stops being the headline and becomes the instrument that makes the test valid, because everything else is held fixed.

**[Nov 2026 update]** The assumption is no longer only implicit. Agrawal & Mayer state it explicitly: trends "are transferrable across model families" (Section 5), based on Llama-3.1-8B and Mistral-7B, both GQA. That gives the paper a concrete published claim to test.

### 2.2 Why this is a stronger pitch

- It makes a falsifiable claim that other people can cite or refute, instead of describing a tool.
- It fills the gap the survey found: no peer-reviewed compressor evaluation on Mamba2 hybrids with state accounting, or on native vs expanded MLA.
- It turns the small-scale setup from a weakness into a design choice. Small models are where these architectures actually exist as open checkpoints, and controlled comparison across them is only feasible at this scale.

If the gaps and ranking flips turn out small, the honest framing shifts to "transfers, with these exceptions", which is still a paper.

### 2.3 Title direction

- *Do KV-Cache Compression Results Transfer Across Attention Architectures? A Controlled Study with Full-State Accounting*
- *Beyond Dense Transformers: What KV-Cache Compression Actually Saves on MHA, GQA, MQA, MLA and Hybrid Models*

Avoid "KVBench: A Unified Benchmark…" as the title. (The current draft title is "KVBench: Bridging Offline Fidelity and Online Inference Evaluation…", which falls under this.)

---

## 3. Three contributions, each with a measurable claim

The engine, the controlled-conditions export and the taxonomy become a **supporting artifact contribution**, not the lead.

### C1. Claimed vs realized savings, per architecture

**Claim [Oct 2026].** For each compressor × architecture, report the nominal KV compression ratio next to the reduction in *total* inference state:
- On Falcon-H1 the Mamba state is untouched, so the realized reduction is smaller.
- On MLA, compressing the expanded K/V is not the same as compressing the native latent.
- On Gemma 3, sliding layers are already bounded.

This gives a "claimed vs realized" gap table that nobody has published. The vLLM hybrid fp8 observation is practitioner evidence that the gap exists.

**[Nov 2026 update]**
- **Hybrid gap confirmed, but documented by others.**
  - vLLM RFC #55196 (verified) measures fp8 KV capacity at 2.00× on attention-only models. On Falcon-H1-1.5B it gives 1.00× at 512 and 1024 tokens, 1.51× at 2K, 1.76× at 8K and 1.84× at 32K. The RFC says the state-vs-KV asymmetry is "not claimed as novel" (Marconi, MLSys 2025).
  - Nemotron 3 Ultra quantizes its Mamba SSM cache separately, because it dominates memory.
  - Shin (2026) claims the first joint KV + recurrent-state bit-allocation study: Qwen3.5 0.8B–9B (GDN) with **Falcon-H1 as a Mamba-2 control**. It derives a memory crossover T\* = 768 tokens on Qwen3.5-4B, close to our ≈ 800-token estimate for Falcon-H1. Quantization only; hybrids only.
  - RIFT: "KV cache compression methods are rarely evaluated on hybrid-attention models."
- **MLA:** RAQ-MLA states that "existing KV-cache quantization methods … do not account for what MLA actually caches". RoLaQ, NVFP4-MLA and KVarN all compress the native latent. JSSA notes that "most existing baselines do not define a direct mapping to MLA's shared latent cache". No one compares expanded vs native caches for third-party compressors.
- **Sliding window:** JSSA, KVzip, AnchorKV and the vLLM fp8 blog all compress only global layers or skip sliding-window layers. This is now standard practice: a disclosure, not a contribution.
- **What remains novel:** a **cross-mechanism** realized-savings table (eviction, low-rank, quantization, sketching × five architecture classes, with quality attached), and MLA expanded-vs-native for off-the-shelf compressors.
- **Our own numbers** (arithmetic from configs and probe bytes, not measurements; see the alignment audit §4):
  - Falcon-H1: attention KV 18,432 B/token vs a constant ~14.7 MB Mamba state. Attention overtakes Mamba at ≈ 800 tokens. A nominal 4× KV compression gives ≈ 1.42× total-state reduction at 512 tokens and ≈ 2.69× at 4K.
  - Gemma3: `sliding_window = 512`, so at ctx ≤ 512 the sliding layers barely evict.
  - TinyDeepSeek: the expanded cache is 384 elements per token per layer vs 288 native latent (`kv_lora_rank` 256 + RoPE 32), so ratios on the expanded cache overstate savings by ≈ 1.33×.

### C2. Ranking transferability

**Claim [Oct 2026].** Do compressor rankings agree across architectures? Report Kendall's τ between architecture pairs, with bootstrap confidence intervals. Either result is publishable: if rankings flip, methods are being tuned to GQA; if they hold, that is a useful reassurance.

**[Nov 2026 update]** Now the best-motivated contribution:
- **RIFT** (Oct 2026): on full-attention Llama 3.1, SnapKV beats a sliding window throughout generation. On Nemotron3 (hybrid SSM) and GPT-OSS (interleaved sliding window), the sliding window beats SnapKV.
- **JSSA** (Oct 2026): competing methods show "architecture-dependent failures" across GQA, hybrid local/global and MLA models. FASA's gap is largest on Gemma4 and GLM (MLA).
- **Pitfalls** (ACL 2026): "both eviction policy and model play a role" in degradation.
- **Agrawal & Mayer**: the published counterclaim of cross-family transfer (two GQA models only).
- No paper computes rank agreement with CIs across ≥ 4 attention classes.

### C3. Proxy validity, with cost

**Claim [Oct 2026].** Test whether cheap FIDELITY metrics predict BEHAVIOR and SYSTEM outcomes:
- per architecture, with rank correlations, confidence intervals and held-out prediction, not 27 points and a Pearson r;
- add an offline-cost break-even analysis, e.g. how many tokens it takes before a calibration-heavy method's offline step pays for itself versus a calibration-free one (TurboQuant being the calibration-free anchor);
- engage the contested KV-CoRE and attention-loss results directly.

**How "linking tensor fidelity to behavior and system metrics" works.** Every job runs one compressor config through the same model, input, decode loop and GPU, and records three kinds of numbers for that same run:

| Branch | Example metric | Cost to measure |
|---|---|---|
| FIDELITY | attention-output RMSE, cosine, KL, reconstruction error | cheap: one forward pass, no generation |
| BEHAVIOR | perplexity ratio, retrieval accuracy | medium |
| SYSTEM | tokens/s, TTFT, ITL, VRAM | needs real decode on real hardware |

Because everything else is held fixed, each config becomes one row with all three. The cross-dimensional analysis (Research plan Phase 24) then asks: **does the cheap FIDELITY number predict the expensive BEHAVIOR and SYSTEM numbers?** For example, if attention RMSE ranks methods one way but perplexity ranks them another, fidelity is a poor proxy. That is the "link": a measured, per-architecture relationship, not just three separate tables. It matters because many compression papers justify their method with fidelity-style proxies, and the field disagrees on whether those proxies work. With roughly 27 jobs per model, Pearson correlations are weak evidence; the link only counts as a contribution if tested properly (§5, §11).

**[Nov 2026 update]**
- **Proxies that do predict:**
  - KV-CoRE: NER vs perplexity Pearson r = 0.88 (values) and 0.64 (keys), Qwen3-4B, low-rank truncation only.
  - Hold Onto That Thought: "higher performance correlates with less attention loss".
  - KVDiagnosis: all ten diagnostics separate failed from successful compression, AUROC 0.684–0.871.
- **Proxies that do not:**
  - *Does Accuracy Equal Evidence?* (arXiv 2608.01631; Scholar lists Findings of ACL 2026): eviction preserves final-answer accuracy while reasoning support degrades (the "answer-evidence gap"); quantization is much less affected.
  - Pitfalls: benchmark averages hide instruction-level failures (measured with Spearman).
- **Consequence:** the current paper headline ("offline fidelity does not predict online quality") is contradicted in its blanket form. Defensible version: *proxy validity is conditional on mechanism and architecture; here is where it holds and where it breaks, with CIs.*
- **Protocol novelty is gone.** EdgeLLM-Nano (Oct 2026) shows single-shot perplexity "gives the same 5.461 score to all methods" because it never reads the compressed cache, and introduces LCDF (teacher-forced decode through the cache under test). This is the same pitfall our audit found (B1) and the same fix as our new PPL protocol. Cite it; do not claim it.
- **Break-even:** no paper treats calibration cost as a measured, amortized axis. Agrawal & Mayer contrast data-aware vs data-oblivious methods only qualitatively. Still open, but a secondary result.

---

## 4. Repositioning against existing work

**"Not the first unified KV benchmark."** Several existing works already do "one harness, many compressors, compare them": NVIDIA KVPress, the MLSys 2025 "Rethinking" paper, SCBench and the Agrawal & Mayer preprint. If KVBench's headline is "a unified benchmark for comparing KV methods", a reviewer will say that already exists. The pitch has to rest on what those works lack.

**"Demote claims 1, 4 and 5."** These are the plug-in harness, the taxonomy, and the finding that compression ratio doesn't equal speedup. Others have done or shown all three. Keep them in the paper, but present them as "we reproduce this under tighter control, across architectures", not as discoveries.

**[Nov 2026 update]** The overlap has grown:
- KVDiagnosis adds a 25-method taxonomy and a common record format.
- *Ablation, Statistical Inference, and Validation for KV-Cache Compression* (Jun 2026) already studies TurboQuant/QJL fidelity with KS significance tests and seed effects, overlapping our FIDELITY branch.
- KVPress now lists Qwen3 and Gemma3 support (no MLA support found).
- Agrawal & Mayer find TurboQuant has the worst throughput and TTFT overhead of their methods, consistent with our ≈ 0.08 tok/s.

---

## 5. What the evidence needs

Otherwise the reframe collapses.

**[Oct 2026] requirements:**
1. **Run all five models with at least 4–5 compressor configs each**, including SnapKV and Palu, with several budgets and seeds. → R11, R13
2. **State hypotheses up front**, e.g. "realized total-state reduction on hybrids is less than the nominal KV ratio", so findings aren't fished for afterwards. → R4 (§5.1)
3. **Some scale and a real task**, e.g. 2–4K context and a needle-style retrieval task, so reviewers can't dismiss it as WikiText-only. → R5, R14
4. **Disclose per-architecture caveats**, such as which compressors run on expanded vs native MLA caches and which parts of Falcon-H1 are compressed. → R7, R16

**[Nov 2026] additions:**
5. **Context ≥ 2K, and ≥ 4K for Falcon-H1 and Gemma3.** Below that the hybrid and sliding-window effects are degenerate by construction (vLLM 1.00× at ≤ 1K; Falcon-H1 crossover ≈ 800 tokens; Gemma3 window 512; Self-Pruning Transformer, COLM 2026: at short lengths hybrids can use *more* memory than full attention). OLMo2-1B's 4096-token limit makes ~4K the common ceiling. → R5
6. **Add a GDN hybrid** (e.g. a Qwen3.5 sub-2B checkpoint) next to Falcon-H1. Shin finds a 4-bit state costs about 1 nat on GDN but nothing on Mamba-2, so a Mamba-2 vs GDN contrast is itself a transfer result. → R6
7. **Rank statistics with CIs** (Spearman/Kendall, bootstrap) for both C2 and C3. → R8
8. **A clean re-run on one commit.** The current measured data is invalid (see the alignment audit). → R2, R13


### 5.1 Hypotheses (draft)

Stated before any grid run, per requirement 2. **Status: draft (R4).** Thresholds and the exact compressor/budget sets must be frozen, and recorded here with a date, before R12/R13 run. Each hypothesis states what would count against it.

| ID | Contribution | Hypothesis | Test | Counts against it |
|---|---|---|---|---|
| **H1** | C2 | Compressor rankings by BEHAVIOR quality at matched nominal budget are **not** invariant across attention classes. | Kendall τ between per-architecture rankings for every architecture pair, 95% bootstrap CI over seeds/prompts. Proposed threshold (to freeze): at least one pair with CI upper bound < 0.6. | All pairwise CIs include τ ≥ 0.8 → report "rankings transfer" (still publishable). |
| **H2** | C1 | On attention-SSM hybrids, realized total-state reduction is **smaller** than the nominal KV ratio, and the gap closes as context grows, crossing over near the analytically predicted length (≈ 800 tokens for Falcon-H1). | Claimed-vs-realized table (E6) at ≥ 3 context lengths spanning the crossover; compare with the analytic prediction. | Realized ≈ nominal at all lengths, or crossover far from prediction. |
| **H3** | C1 | Compression ratios measured on MLA's **expanded** K/V overstate savings relative to a native-latent deployment (≈ 1.33× for TinyDeepSeek), and compressor rankings differ between the two caches. | Same compressors on expanded vs native latent caches (needs E4). | No ranking difference and ratio gap explained fully by the 384/288 element ratio. If E4 is not built, H3 is dropped and reported as a disclosure. |
| **H4** | C3 | FIDELITY → BEHAVIOR proxy validity is **conditional**: rank correlation is higher within a compression mechanism than across mechanisms, and varies by architecture. | Spearman ρ per architecture and per mechanism with bootstrap CIs; held-out prediction of BEHAVIOR from FIDELITY. | Uniformly high ρ across mechanisms and architectures (would support KV-CoRE's view; report as such). |
| **H5** | C3 (cost) | Calibration-heavy methods (e.g. Palu's offline SVD) need a **finite** number of generated tokens to amortize their offline cost against a calibration-free method (TurboQuant) at equal quality. | Break-even calculation (E7) from `cost.offline` and `cost.online`. | Break-even never reached within realistic deployment lengths. |
| **H6** | SYSTEM | "Higher compression ratio ≠ faster decode" holds at short context on small models and **weakens** as context grows. | tokens/s and ITL vs context (≥ 3 lengths) per compressor; locate the crossover where it exists. | No change with context (then the finding is regime-independent; report as such). |

---

## 6. Publication-level verdict

### 6.1 As it stood [Oct 2026]

Judged from the plan documents and the survey, not from result tables or the `.tex` draft, so provisional.

**Workshop level, or at best Findings-tier / a lower-tier journal.** Reasons:
- **Scale.** Context 128–512, batch 1, 64 generated tokens, WikiText-2 perplexity, one A10G. Reviewers at top venues will call this a toy regime.
- **Narrow empirics.** Results cover three compressors on two models. The five-model matrix is *engine support*, and the plan says results stay on TurboQuant/QJL/RocketKV unless re-swept.
- **Behavior tasks are synthetic** in-repo generators, not LongBench/RULER.
- **Findings are mostly known** (no single winner, ratio ≠ speed).
- **No statistical rigor described:** seeds, confidence intervals, significance.
- **A lot of the plan is framing.** Phases 15–33 are about how to word the paper, which doesn't add evidence.

**What is genuinely good:** the engineering is thorough, the controlled-conditions export is a real reproducibility asset, and the architecture matrix (MLA, Mamba2 hybrid, sliding window) plus offline/online cost accounting fill gaps not found in published work.

### 6.2 As it stands [Nov 2026 update]

| Scenario | Level |
|---|---|
| Current data | **Not submittable.** Most BEHAVIOR/SYSTEM numbers are invalid, including those behind the draft abstract (alignment audit). |
| With the §5 evidence requirements met | **ACL/EMNLP Findings or TMLR is credible.** |
| Main track at a top venue | Realistic only if the paper leads with C2, adds a GDN hybrid, runs ≥ 2–4K context, and reports rank statistics with CIs that come out clearly positive or clearly negative. The small-models-only constraint is the main ceiling. |

### 6.3 Venue options

- **TMLR** or a systems/ML journal: realistic with the fixes, since TMLR values rigor over scale.
- **NeurIPS Datasets & Benchmarks, or MLSys:** possible only with added scale, real workloads, and a strong architecture result.
- **EMNLP/ACL Findings:** a plausible fallback.
- **IEEE Transactions-style journal:** fits a thorough instrument paper, but aim higher first.
- Without new experiments, it stays workshop level.

### 6.4 What to cut from the current plan

The Phase 15–33 effort on wording (identity sentences, ladder diagrams, the five-contribution taxonomy) won't add evidence, so keep that text short. Put the effort into the experiments in §5, because those are what make the reframe credible.

---

## 7. Prior-art survey (Jan 2025 – 1 Oct 2026)

Scope: peer-reviewed work from 2025 through 1 Oct 2026 at top venues, plus key preprints, in five groups: KV-cache compression benchmarks and evaluation frameworks; surveys and tutorials on KV-cache management and LLM inference; controlled-comparison and multi-branch evaluation work; cost accounting and calibration work; evaluation on hybrid, MLA and small-model architectures. For each paper, venue and year were checked and overlap with KVBench's claims rated.

### 7.1 TL;DR

- **Not scooped as a whole, but claims 1, 4, 5 and 8 are individually crowded.** Unified plug-in harnesses already exist: NVIDIA KVPress (its companion Expected Attention paper says it already includes "more than 20 techniques"), the MLSys 2025 "Rethinking KV Cache Compression" toolkit, and SCBench (ICLR 2025). The "compression ratio ≠ speedup / no single winner" finding has also been published (Rethinking, Pitfalls ACL 2026, Agrawal & Mayer). KVBench cannot claim these as headline novelties.
- **KVBench's real novelty sits in claims 2, 3, 6 and 7**: the fidelity branch tied to behavior; offline/calibration vs online cost as a benchmark dimension; the per-job controlled-conditions export; and the five-architecture matrix including MLA and Mamba2-hybrid with state passthrough. No peer-reviewed 2025–2026 study evaluates KV compressors on Mamba/SSM hybrids with total-inference-state accounting, or on native-MLA small models. SCBench is the only partial exception: it includes Jamba-1.5-Mini and Codestral-Mamba as *alternatives* to compression, not as compression targets.
- **Closest overall competitor:** the unpublished May 2026 preprint by Agrawal & Mayer, which already reports that "compression ratio alone is a poor predictor of end-to-end performance" using TurboQuant and SnapKV.
- **Main risks:**
  - (a) The scale (0.27–1.7B, 128–512 tokens, WikiText-2, batch 1) is far below where the field evaluates (LongBench/RULER, 8B models, 4K–128K). Reviewers may call it a toy setting unless small-model/edge deployment is the explicit framing. In that regime "memory savings ≠ speedup" is almost guaranteed, so it can look like a side effect of the setup.
  - (b) Preprints like KVDiagnosis (Aug 2026), MetaKV (Sept 2026) and Agrawal & Mayer (May 2026) could be published first and absorb the "evaluation methodology" story.
  - (c) "Fidelity ≠ behavior" is disputed: KV-CoRE and Hold Onto That Thought report that offline measures do predict quality.

**[Nov 2026 update]** Claims 2 and 7 remain partly novel, but each has been narrowed (§3). Claim 3 (calibration cost as a measured axis) is still unclaimed. See §8 for what changed.

### 7.2 Verdict per KVBench claim

| # | KVBench claim | Closest published prior work | Overlap | Where KVBench is still novel | [Nov 2026 update] |
|---|---|---|---|---|---|
| 1 | Common interception layer / single decode loop hosting heterogeneous compressors as plug-ins | NVIDIA KVPress (library; paper arXiv-only); Gao et al. "Rethinking" MLSys 2025 (LMDeploy-based toolkit); SCBench ICLR 2025 | **High** | KVPress is eviction-centric and hooks into HF prefill. KVBench hosts quantization (TurboQuant/QJL), low-rank (Palu), eviction (SnapKV) and two-stage sparse (RocketKV) in the *same incremental decode loop* | Unchanged; KVPress now lists Qwen3/Gemma3 |
| 2 | Three-branch protocol: fidelity (KV/attention-output RMSE, KL, cosine, metadata overhead) + behavior + system | Rethinking (throughput, output length, per-sample accuracy); Agrawal & Mayer (quality + TTFT + throughput + realized ratio); KVDiagnosis (cache/logit/attention diagnostics) | Medium | No published work combines tensor-level fidelity, behavior and p50/p99 ITL/kernel/bandwidth in one protocol | Narrowed: Ablation/Statistical-Inference covers TurboQuant/QJL fidelity with significance tests; EdgeLLM-Nano covers compressed-cache decode fidelity |
| 3 | Offline preprocessing/calibration cost vs online per-token cost; stateful vs stateless | Oaken (ISCA 2025: offline threshold profiling + online scaling); TurboQuant (ICLR 2026: data-oblivious, calibration-free) | Medium (design principle); Low (benchmark dimension) | No benchmark treats calibration cost as a measured axis | Still unclaimed |
| 4 | Taxonomy A–E (eviction / quantization / low-rank / hybrid / modified attention) | TMLR 2025 survey (token/model/system levels); Agrawal & Mayer (quantization/pruning/merging/low-rank); KVDiagnosis (25 methods, 5 families); Yuan et al. Findings EMNLP 2024 | **High** | Only the "E: compression + modified attention" class and its tie to the cost taxonomy are distinctive | Unchanged |
| 5 | Pareto + cross-metric correlation: ratio ≠ memory ≠ speedup; fidelity ≠ behavior | Rethinking (memory savings don't yield throughput under FlashAttention/PagedAttention; longer outputs raise E2E latency); Agrawal & Mayer (ratio poor predictor); Pitfalls ACL 2026 | **High** for ratio ≠ speed; Medium for fidelity ≠ behavior | Quantified correlation between *offline tensor proxies* and *online decode quality* is not established in published work | Blanket "fidelity ≠ behavior" contradicted (KV-CoRE, KVDiagnosis, Hold Onto); reframe as conditional |
| 6 | Reproducibility harness with per-job controlled_conditions JSON | Rethinking artifact; KVDiagnosis "common record format" (preprint); KVPress leaderboard | Low–Medium | Machine-readable controlled-conditions provenance per job is not standard | Unchanged |
| 7 | Architecture matrix: MHA, GQA, MQA+sliding, MLA (expanded vs latent), Mamba2 hybrid | SCBench (Jamba, Codestral-Mamba as alternatives); KVzip (Gemma 3: compresses only global layers); MHA2MLA ACL 2025, TransMLA NeurIPS 2025, X-EcoMLA COLM 2025 (conversion, not compressor evaluation) | Low | **Strongest novelty.** No peer-reviewed compressor evaluation on Mamba2 hybrids with recurrent-state accounting, or on native vs expanded MLA caches | Hybrid accounting narrowed (vLLM #55196, Shin, Nemotron 3); MLA expanded-vs-native still open; GDN hybrid missing from our matrix |
| 8 | Rankings flip across architectures; no single winner; best method depends on context length | Hold Onto That Thought ("no singular strategy fits all"); Agrawal & Mayer (workload-dependent); Pitfalls ACL 2026; Rethinking | **High** for "no single winner"; Low for "flip GQA vs MHA" | The *architecture-induced* ranking flip is new; "no single winner" by itself is not | Flips now reported as side results (RIFT, JSSA); a controlled statistical test is still missing |

### 7.3 Main published table (2025 – 1 Oct 2026, venue verified unless marked)

Overlap columns: claims 1–8 (H/M/L/–).

| Paper | Authors | Venue / year | Link | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Rethinking Key-Value Cache Compression Techniques for LLM Serving | Wei Gao, Xinyu Zhou, Peng Sun, Tianwei Zhang, Yonggang Wen | MLSys 2025 | proceedings.mlsys.org/paper_files/paper/2025/hash/26289c647c6828e862e271ca3c490486-Abstract-Conference.html | H | M | L | M | H | M | – | M |
| SCBench: A KV Cache-Centric Analysis of Long-Context Methods | Yucheng Li, Huiqiang Jiang, Qianhui Wu, … Lili Qiu | ICLR 2025 | openreview.net/pdf?id=gkUyYcY1W9 | M | L | – | H | M | L | M | M |
| The Pitfalls of KV Cache Compression | Alex Chen, Renato Geh, Aditya Grover, Guy Van den Broeck, Daniel Israel | ACL 2026 (long) | aclanthology.org/2026.acl-long.1926 | L | L | – | L | M | – | – | H |
| R-KV: Redundancy-aware KV Cache Compression for Reasoning Models | Zefan Cai, Wen Xiao, Hanshi Sun, … Junjie Hu | NeurIPS 2025 | proceedings.neurips.cc (hash 57e0358a…) | – | – | – | L | L | – | – | L |
| Hold Onto That Thought: Assessing KV Cache Compression on Reasoning | Minghui Liu, Aadi Palnitkar, Tahseen Rabbani, … | NeurIPS 2025 Workshop on Efficient Reasoning (ICLR 2026 main **unverified**) | neurips.cc/virtual/2025/126664; arXiv 2512.12008 | M | L | – | L | L | – | – | H |
| SCOPE: Optimizing KV Cache Compression in Long-context Generation | Wu et al. | ACL 2025 (long) | aclanthology.org/2025.acl-long.529 | L | – | – | L | L | – | – | L |
| EvolKV: Evolutionary KV Cache Compression for LLM Inference | — | Findings of EMNLP 2025 | aclanthology.org/2025.findings-emnlp.88 | – | – | M (offline search cost) | – | – | – | – | L |
| RocketKV | Payman Behnam, Yaosheng Fu, Ritchie Zhao, … Alexey Tumanov | ICML 2025 (PMLR 267) | openreview.net/forum?id=RyOpooIxDF | – | – | L | M (hybrid class) | L | – | – | – |
| Oaken: Online-Offline Hybrid KV Cache Quantization | Minsu Kim, Seongmin Hong, Ryeo Wook Ko, … Jongse Park | ISCA 2025 | doi.org/10.1145/3695053.3731019 | – | – | **H** | – | L | – | – | – |
| TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate | Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni | ICLR 2026 | iclr.cc/virtual/2026/poster/10006985 | – | M (distortion metrics) | M (calibration-free) | – | – | – | – | – |
| Palu: KV-Cache Compression with Low-Rank Projection | Chi-Chih Chang, Wei-Cheng Lin, Chien-Yu Lin, … | ICLR 2025 | iclr.cc/virtual/2025/poster/29993 | – | L | M (offline SVD) | – | – | – | – | – |
| TurboAttention | Hao Kang, Srikant Bharadwaj, James Hensman, … | MLSys 2025 | proceedings.mlsys.org (f4f55846…) | – | – | L | M (class E) | – | – | – | – |
| CacheBlend | Jiayi Yao, Hanchen Li, Yuhan Liu, … | EuroSys 2025 | doi.org/10.1145/3689031.3696098 | – | – | – | – | – | – | – | – |
| KVCache Cache in the Wild | Jiahao Wang, Jinbo Han, Xingda Wei, … | USENIX ATC 2025 | usenix.org/conference/atc25/presentation/wang-jiahao | – | – | – | – | L | – | – | – |
| Towards Economical Inference: MHA2MLA | Tao Ji et al. | ACL 2025 (long) | aclanthology.org/2025.acl-long.1597 | – | – | – | – | – | – | M | – |
| TransMLA | Fanxu Meng, Pingzhi Tang, Zengwei Yao, … Muhan Zhang | NeurIPS 2025 (spotlight, per authors' repo/mlanthology) | mlanthology.org/neurips/2025/meng2025neurips-transmla | – | – | – | – | – | – | M | – |
| X-EcoMLA | Guihong Li, Mehdi Rezagholizadeh, Mingyu Yang, … | COLM 2025 | openreview.net/forum?id=CPJ9EAeYfd | – | – | – | – | – | – | M (1B small model → MLA) | – |
| Multi-head Temporal Latent Attention | Keqi Deng, Phil Woodland | NeurIPS 2025 | neurips.cc/virtual/2025/poster/116809 | – | – | – | – | – | – | L | – |
| LaCache: Ladder-Shaped KV Caching | Dachuan Shi, Yonggan Fu, Xiangchi Yuan, … | ICML 2025 | icml.cc/virtual/2025/poster/45236 | – | – | – | – | – | – | L (SmolLM2-1.7B) | – |
| KVzip: Query-Agnostic KV Cache Compression | Kim et al. | NeurIPS 2025 | proceedings.neurips.cc (f4eaa4b8…) | – | – | – | – | – | – | M (Gemma 3 local/global) | – |
| Hybrid Transformer-Mamba LMs (Jamba) | AI21 | ICLR 2025 | proceedings.iclr.cc (a9ed43fa…) | – | – | – | – | – | – | L (architecture ref only) | – |
| Marconi: Prefix Caching for the Era of Hybrid LLMs | — | MLSys 2025 | mlsys.org/virtual/2025 (session listing) | – | – | – | – | – | – | M (hybrid state accounting in serving) | – |

**Pre-window baselines to cite:** Yuan et al., "KV Cache Compression, But What Must We Give in Return?", Findings of EMNLP 2024 (aclanthology.org/2024.findings-emnlp.266). It already provides a taxonomy and evaluates 10+ methods, including linear-time and hybrid architectures, on LongBench/NIAH. It is the canonical "benchmark of KV compression" reference. **[Nov 2026 update]** Read: its linear-time/hybrid models (Mamba-2.8B, Mamba2-2.7B, RWKV-5, RecurrentGemma-2b/9b) are evaluated as *alternatives* to compressing Llama-3-8B, not as hosts for compressors, like SCBench. Similarly, *Key, Value, Compress* (CICC 2025, read) treats MQA/GQA/MLA as compression *methods* in its taxonomy, not as architectures to evaluate compressors on. Neither closes the C2 gap. SnapKV (NeurIPS 2024) and QJL (venue reported inconsistently across citing papers; verify before citing) are also pre-window.

### 7.4 Closest papers in detail

- **Rethinking KV Cache Compression (MLSys 2025)** is the single closest *published* competitor. It reviews existing benchmark studies and catalogs their "missing pieces". It shows that memory reduction does not translate into throughput under FlashAttention/PagedAttention, and that compression can lengthen outputs and raise end-to-end latency. It ships open-source tools. Overlap: claims 1, 5, 8. It does **not** do tensor fidelity, calibration-cost accounting, small models, or non-GQA architectures. KVBench's "memory reduction ≠ throughput gain" finding reproduces this result and should be framed as *confirmation at small-model scale across architectures*.
- **SCBench (ICLR 2025)** organizes methods by KV lifecycle (generation, compression, retrieval, loading). It evaluates eight categories, including Codestral-Mamba and Jamba-1.5-Mini, and finds compression methods underperform in shared multi-turn contexts. Overlap: claims 4, 7 (partially), 8. It is quality-only (no system metrics, no fidelity), uses 8 long-context LLMs, and treats SSM/hybrid models as competitors to compression rather than as hosts for it.
- **Pitfalls (ACL 2026)** evaluates five eviction methods on Llama3.1-8B and Qwen2.5-14B with IFEval multi-instruction prompts. Certain instructions are effectively dropped and system prompts leak. It supports KVBench's behavior branch (instruction following) and claim 8, but is eviction-only with no system or fidelity metrics.
- **Oaken (ISCA 2025)** is the origin of the offline-profiling / online-quantization split that KVBench's cost taxonomy builds on. It is an accelerator paper, not an evaluation framework. Cite it as the inspiration for claim 3 and say explicitly that KVBench turns the split into a measured benchmark axis.
- **TurboQuant (ICLR 2026)** reports "absolute quality neutrality with 3.5 bits per channel and marginal quality degradation with 2.5 bits per channel", with distortion within a small constant (≈ 2.7) factor of the information-theoretic lower bound. It is data-oblivious/online, the natural "zero-calibration" anchor on KVBench's cost axis. Its distortion-rate framing (MSE and inner-product distortion) overlaps KVBench's fidelity branch.
- **RocketKV (ICML 2025)**, final version, reports up to 400× compression ratio but only up to 3.7× end-to-end speedup and 32.6% peak memory reduction on A100. arXiv v1 reported "up to 3×" speedup and "up to 31%" peak memory reduction on H100; cite the ICML numbers. Either way it is an author-reported example of "compression ratio ≠ memory savings ≠ speedup" that KVBench can cite and test at small-model scale.
- **[Nov 2026 update] Agrawal & Mayer** (read in full): KIVI, TurboQuant, SnapKV and CaM (via KVPress) on Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.3 (both GQA; Mistral adds sliding window), A100-40GB, FP16 + FlashAttention2, LongBench (six datasets for quality; NarrativeQA, Qasper, GovReport for system metrics). KIVI4 is the most stable; SnapKV gives the best throughput; TurboQuant has the highest TTFT and throughput overhead (one GovReport TurboQuant run takes 3–4 days); CaM's compression ranges from 1.0× to 14.2× depending on workload. Lessons learned include the explicit claim that relative method performance is "consistent for both" models and "transferrable across model families". No fidelity, no calibration cost, no non-GQA architectures, no seeds or CIs.

### 7.5 Surveys, tutorials and position papers

**Published (verified):**
- **"A Survey on Large Language Model Acceleration based on KV Cache Management"**, Haoyang Li, Yiming Li, Anxin Tian, … Lei Chen, *TMLR* 2025 (openreview.net/forum?id=z3JZzu9EA3). Taxonomy: token-level (selection, budget allocation, merging, quantization, low-rank), model-level, system-level. The best single citation for the taxonomy section.
- **"Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems"**, *ACM Computing Surveys* 58(1), 2025. Venue confirmed only via citing papers; verify the DOI before citing.
- **Rethinking KV Cache Compression (MLSys 2025)** works as a de facto SoK/position paper on how KV methods *should be evaluated*: the closest thing to an evaluation-methodology position paper at a top venue.

**Lower-tier or unverified venue (cite cautiously):**
- "KV Cache Compression for Inference Efficiency in LLMs: A Review", ACM DL (doi 10.1145/3778534.3778567). Proceedings venue not verified; appears to be a smaller conference.
- "Key, Value, Compress: A Systematic Exploration of KV Cache Compression Techniques", Javidnia et al., CICC 2025. **[Nov 2026 update]** Scholar lists it as 2025 IEEE Custom Integrated Circuits Conference (39 citations); it also discusses Mamba2-Hybrid and MLA.
- A survey of small language models (arXiv 2411.03350) covers KV compression for edge deployment; venue not verified.

No 2025–2026 tutorial at ACL/EMNLP/NeurIPS/MLSys/ISCA specifically on KV-cache compression evaluation was found. Treat this as "not found", not "does not exist".

### 7.6 Preprints to monitor (not peer-reviewed as of 1 Oct 2026)

| Preprint | arXiv | Why it matters for KVBench | [Nov 2026 update] |
|---|---|---|---|
| Agrawal & Mayer, "Benchmarking KV-Cache Optimizations across Task Quality and System Performance for Long-Context Serving [Experiment, Analysis & Benchmark]" | 2607.05399 (v1, 3 May 2026) | **Closest scoop risk.** TurboQuant and SnapKV plus KIVI and CaM on LongBench-style multi-doc QA, single-doc QA, few-shot and summarization. Quality + TTFT + throughput + realized compression ratio; ratio is a poor predictor of end-to-end performance. Per the abstract: "KIVI4 provides the most stable quality across models, SnapKV delivers the strongest long-context throughput," and CaM "exhibits substantial workload sensitivity." PVLDB-formatted (acceptance unverified). Llama-3.1-8B and Mistral-7B only; no fidelity, calibration, or non-GQA architectures | Still arXiv v1; explicit cross-family transfer claim (§7.4) |
| KVDiagnosis (Chen Qiu, … Panos Kalnis, KAUST) | 2608.09412 (Aug 2026) | 25-method taxonomy in five families; per-method FullCache-paired controls; common record format linking outputs to cache/likelihood/attention/decoding diagnostics; 59,800 compressed runs on Qwen3-8B. Overlaps claims 2, 4, 6 | Verified; diagnostics AUROC 0.684–0.871; one listing says KDD '27 (unverified) |
| MetaKV: Adaptive KV Cache Compression for Constrained LLM Inference | 2609.07966 (Sept 2026) | Uses KVQuant, H2O and **RocketKV**. Reports KVQuant at ~8.0× H2O's latency with ~29% less peak KV memory: direct support for "memory ≠ speed" | Downloaded |
| Expected Attention / KVPress (Devoto, Jeblick, Jégou, NVIDIA) | 2510.00636 | KVPress is the de facto plug-in harness; "more than 20 techniques". ICLR 2026 submission; acceptance not found | Downloaded |
| Can LLMs Maintain Fundamental Abilities under KV Cache Compression? (KVFundaBench) | 2502.01941 | Task-dependent degradation of 1–40%; ICLR 2026 submission **withdrawn** | — |
| Ablation, Statistical Inference, and Validation for KV-Cache Compression | 2607.09683 | Statistical rigor for KV evaluation; relevant to claim 6 | TurboQuant/QJL fidelity with KS significance; seed is an explicit variable; HIP/C++ on AMD |
| Statistical inference and quality measures of KV cache quantisations inspired by TurboQuant (D'Alberto) | 2605.08114 | Quality-measure methodology for TurboQuant-style quantizers (fidelity branch) | — |
| From Tensor Buffer to Distributed Memory Hierarchy: A Survey of KV Cache Management for LLM Serving | 2607.02574 | Systems-side survey | — |
| KV-CoRE (Normalized Effective Rank as a compressibility indicator) | 2602.05929 | Offline proxy that "correlates strongly with performance degradation": a direct counterpoint to "offline proxies don't predict online quality" | Scholar: arXiv only (6 citations); Pearson r = 0.88 / 0.64 |
| Understanding the Physics of KV Cache Compression through Attention Dynamics | 2603.01426 | Theory of why benchmark claims of 80–90% savings fail | Downloaded |
| Don't Waste Bits! Adaptive KV-Cache Quantization for On-Device LLMs | 2604.04722 | SmolLM-135M/360M/1.7B; accuracy + ms/token. Closest small-model KV quantization study | — |

### 7.7 Details by research question

**Non-standard architectures (MLA, hybrids, sliding window).**
- MLA work in 2025 is about *converting* models to MLA: MHA2MLA (ACL 2025), TransMLA (NeurIPS 2025), X-EcoMLA (COLM 2025, which converts Llama3.2-1B-Instruct with 6.4× KV compression at equal average score). None evaluates third-party compressors on an MLA cache or compares compressing the expanded K/V against the native latent.
- For hybrids, the only peer-reviewed KV-centric evaluation touching Mamba-attention models is SCBench, and it does not compress the hybrid's KV. Marconi (MLSys 2025) addresses prefix caching for hybrids and is the right citation for "total inference state" accounting.
- The strongest practical evidence that KV compression under-delivers on hybrids is a GitHub issue: vLLM RFC #55196, "fp8 KV cache gives little to no memory benefit on Mamba/GDN hybrid models". fp8 KV gives 2.00× capacity on attention-only models but "1.84x at 32k context and 1.00x (no gain) at short context" on Mamba/GDN hybrids, because "the Mamba state is not quantized by --kv-cache-dtype and all groups share one uniform page". This is exactly KVBench's total-inference-state argument.
- For sliding/global models, KVzip (NeurIPS 2025) applies eviction only to Gemma 3's global layers: a precedent for per-layer-type handling.
- **[Nov 2026 update]** RFC verified: Falcon-H1-1.5B per context = 1.00× (512), 1.00× (1024), 1.51× (2048), 1.76× (8192), 1.84× (32768); one anchor hybrid plus two attention-only controls; the RFC explicitly does not claim the asymmetry as novel. The MLA and hybrid spaces have since filled with method papers (§8).

**Frameworks and toolkits.** KVPress (NVIDIA, Apache-2.0) is the dominant harness. Its "press" abstraction hooks into HF transformers, and the Hold Onto That Thought fork added decoding-phase compression. The Rethinking artifact builds on LMDeploy. KVBench's differentiator is hosting quantizers and low-rank projectors next to evictors in one decode loop, with system telemetry; KVPress is predominantly eviction/pruning. **[Nov 2026 update]** KVPress 0.3.0 lists Llama, Mistral, Phi3, Qwen2, Qwen3 and Gemma3; some presses (e.g. SnapKVPress, ExpectedAttentionPress) "might not work with all models"; no MLA support found.

**Metric decoupling.**
- Published evidence: Rethinking (memory savings that don't translate into throughput; longer outputs), Pitfalls (benchmark averages hide instruction-level failure), SCBench (single-turn results don't predict multi-turn).
- Preprint evidence: Agrawal & Mayer (ratio vs end-to-end), KVDiagnosis (aggregate scores hide which executions fail).
- Counter-evidence KVBench must address: KV-CoRE claims an offline rank proxy correlates strongly with degradation, and Hold Onto That Thought observes that lower attention loss correlates with higher performance. "Fidelity ≠ behavior" is therefore *contested*, not settled. Report correlation coefficients with confidence intervals, not a binary verdict.
- **[Nov 2026 update]** Add KVDiagnosis AUROC 0.684–0.871 (proxies partly predict), *Does Accuracy Equal Evidence?* (accuracy hides lost reasoning support under eviction), and EdgeLLM-Nano (perplexity that never reads the compressed cache cannot separate methods at all).

---

## 8. Post-cut-off update (1 Oct – 24 Nov 2026)

Method: Google Scholar (2025+ filter) and arXiv in the browser, plus web search. Agrawal & Mayer and Shin were read in full. For the rest: abstract, contributions, setup (model and architecture list), conclusion, and every sentence matching the claim under test. Full detail: [`literature/FRAMING_VALIDATION_2026-11-24.md`](literature/FRAMING_VALIDATION_2026-11-24.md).

### 8.1 New competitors

| Paper | Date / venue | Threat to | Note |
|---|---|---|---|
| Shin, *Joint KV-Cache and Recurrent-State Quantization in Hybrid Linear-Attention LLMs* | 2026, ResearchGate (no arXiv found); read in full | C1 (hybrid) | Claims the "first" joint KV + state bit-allocation study; Qwen3.5 0.8–9B GDN plus **Falcon-H1 control**; crossover T\* = 768 tokens on Qwen3.5-4B; Pareto across context, scale and task; 4-bit KV costs ≤ 0.03 nats, 4-bit GDN state ≈ 1 nat, Mamba-2 state at 4 bits costs nothing; limiting memory is task-dependent (retrieval → KV, reasoning → state; 8-bit state drops GSM8K 0.600 → 0.345 without error feedback); cites vLLM #55196. **Closest direct overlap**, but quantization only and hybrids only |
| RIFT (2610.31098) | Oct 2026, arXiv | C2 | SnapKV ↔ sliding-window ranking flips on hybrid and SWA models; "decode staleness" worse on hybrids |
| JSSA (2610.19026) | Oct 2026, arXiv | C2, MLA | Qwen3-8B (GQA), Gemma4-26B-A4B (local/global), GLM-4.7-Flash (MLA); "architecture-dependent failures" of baselines; applies only to Gemma4 global layers |
| EdgeLLM-Nano (2610.14475) | Oct 2026, arXiv | Protocol novelty | Perplexity-without-compressed-cache pitfall; LCDF; seeds and effect sizes; 8.39M-parameter model on Raspberry Pi 5 |
| *Does Accuracy Equal Evidence?* (2608.01631) | Aug/Oct 2026; Scholar lists Findings of ACL 2026 | C3 | Answer-evidence gap: 10 eviction methods + 1 quantization, 3 models; eviction preserves accuracy but loses reasoning support |
| GateSight (2610.14754) | Oct 2026, arXiv | C1 hybrid (method) | Query-agnostic eviction for GDN hybrids, Qwen3.8-27B, up to 262K tokens |
| DCKV (2610.36950) | Oct 2026, arXiv | C1 hybrid (method) | Learned KV codec on Qwen3.5-2B hybrid; > 55× fewer bytes than bf16 at ≤ 1.8-point loss |
| RAQ-MLA (2610.33625), RoLaQ (2610.20686), NVFP4-MLA (2611.18972) | Oct–Nov 2026, arXiv | MLA | Native-latent compression; RAQ states generic quantizers mis-fit MLA; NVFP4-MLA speedup grows with KV length (1.3–1.5× at 2K to 2.2–2.8× at 128K) |
| Low-Bit Recurrent States (2609.30950), When Dynamic Scaling Fails (2610.15139, NeurIPS 2026 workshop, non-archival), Quantizing Recurrent LMs Is Easier (2611.01864), STEPQuant (2609.38169), LeapQuant (2609.38166), DAMP (2608.27513) | Aug–Nov 2026 | C1 hybrid | Recurrent-state quantization is now its own crowded subfield |
| Why Gated DeltaNet Survives 4-Bit Quantization (2609.04098) | Sep 2026, arXiv | C1 hybrid | NVFP4 W4A4 on a 27B GDN hybrid matches BF16; FP8 attention cache essentially free |
| LinearKV (2608.11231) | Jul 2026, arXiv | C2 (hybrid) | State composition is "architecture-fragile": harmful on Mamba-2, unnecessary on GDN |
| Nemotron 3 Ultra (2606.15007) | Jun 2026, arXiv | C1 | Quantizes KV to FP8 and the Mamba SSM cache separately (FP32 → FP16 with stochastic rounding) |
| Self-Pruning Transformer (2610.09051) | **COLM 2026** | Scale / hybrid | At short lengths hybrids can use more memory than full attention |
| *Beyond Proxy Metrics: … Generative Faithfulness* | ICLR 2026 submission (PDF header "Under review"; acceptance unverified) | C3 | Replaces PPL/QA with direct output fidelity on ShareGPT prompts; finds KV-cache dropping "can be over-optimized to achieve favorable Perplexity scores" while outputs diverge, and that INT4 quantization and KV dropping degrade as context grows. Hierarchy: low-precision attention > INT4 > 50% pruning > KV dropping |
| AnchorKV (2611.00117), Spend Bytes on Breadth (2610.05685) | Oct 2026, arXiv | Background | Gemma-3 local/global handling; decode-time KV compression for long CoT |

### 8.2 Re-checked citations from the original survey

| Item | Original survey said | Now |
|---|---|---|
| vLLM RFC #55196 | practitioner evidence | **Verified**, with Falcon-H1 per-context numbers (§7.7) |
| Agrawal & Mayer | May 2026 preprint, PVLDB-formatted | Still arXiv v1; Llama-3.1-8B and Mistral-7B (both GQA); explicit cross-family transfer claim |
| KVDiagnosis | Aug 2026 preprint | Verified (Qwen3-8B, 59,800 runs, AUROC 0.684–0.871) |
| Hold Onto That Thought | NeurIPS 2025 workshop; ICLR 2026 unverified | Scholar still lists arXiv only |
| KV-CoRE | OpenReview submission | Scholar: arXiv 2602.05929 only |
| Pitfalls | ACL 2026 | Verified (ACL Anthology); uses Spearman |
| Rethinking | MLSys 2025 | Verified |
| SCBench | ICLR 2025 | Verified |

---

## 9. Perishability

| Sub-claim | Open on 2026-11-24? | Pressure | Half-life |
|---|---|---|---|
| Whole study (transfer + full-state accounting, five architecture classes, controlled) | **Yes** | High: every component is being touched | ~1 submission cycle |
| C2 ranking transfer with statistics | Yes | Medium–high (RIFT, JSSA report flips as side results) | ~1 cycle |
| C1 hybrid realized-savings phenomenon | **No**: documented (vLLM #55196, Nemotron 3, Shin) | — | Already spent; usable only as confirmation |
| C1 cross-mechanism realized-savings table | Yes | Medium | 1–2 cycles |
| C1 MLA expanded vs native for third-party compressors | Yes | Medium (native-latent methods appearing monthly) | 1–2 cycles; needs Gate C |
| C3 proxy validity (conditional, with CIs) | Partly; debate is live | Medium | Durable if framed as conditional |
| C3 calibration-cost break-even | Yes | Low | Durable, but minor |
| Compressed-cache-in-the-loop PPL protocol | **No** (EdgeLLM-Nano LCDF) | — | Cite, don't claim |

---

## 10. Related Work outline

Four sections, and what to cite in each.

1. **Eviction / token selection:** SnapKV (NeurIPS 2024), SCOPE (ACL 2025), R-KV (NeurIPS 2025), EvolKV (Findings EMNLP 2025), LaCache (ICML 2025), KVzip (NeurIPS 2025), RocketKV (ICML 2025, as eviction + dynamic sparse). *[Nov 2026: add RIFT, GateSight, JSSA as preprints.]* Differentiate: these are *methods* evaluated in their own harnesses; KVBench hosts them under matched conditions.
2. **Representation compression (quantization / low-rank):** TurboQuant (ICLR 2026), QJL (verify venue), Palu (ICLR 2025), Oaken (ISCA 2025), TurboAttention (MLSys 2025). *[Nov 2026: add RAQ-MLA, RoLaQ, NVFP4-MLA for MLA latents; Shin, Low-Bit Recurrent States, DAMP, LeapQuant, STEPQuant for recurrent state.]* Differentiate: each reports its own offline cost inconsistently or not at all; KVBench measures calibration and per-token cost on one axis.
3. **Architecture- and serving-aware:** MHA2MLA (ACL 2025), TransMLA (NeurIPS 2025), X-EcoMLA (COLM 2025), MTLA (NeurIPS 2025), Jamba (ICLR 2025), Marconi (MLSys 2025), CacheBlend (EuroSys 2025), KVCache Cache in the Wild (USENIX ATC 2025). *[Nov 2026: add vLLM RFC #55196 (as practitioner motivation, not a paper), Nemotron 3, Self-Pruning Transformer (COLM 2026), LinearKV.]* Differentiate: these change the architecture or the serving system; KVBench holds both fixed and asks how *existing* compressors behave across MHA/GQA/MQA/MLA/hybrid.
4. **Evaluation and benchmarking:** Yuan et al. (Findings EMNLP 2024), SCBench (ICLR 2025), Rethinking (MLSys 2025), Pitfalls (ACL 2026), Hold Onto That Thought (NeurIPS 2025 workshop), TMLR 2025 survey; preprints Agrawal & Mayer, KVDiagnosis, KVPress/Expected Attention, KVFundaBench, MetaKV. *[Nov 2026: add EdgeLLM-Nano, Does Accuracy Equal Evidence?, KV-CoRE, Ablation/Statistical Inference.]* Differentiate on the three axes none of them covers together: tensor fidelity, offline/online cost, non-GQA architectures.

---

## 11. Recommendations

These are the research recommendations. Their execution, as work packages with status, is tracked in [`RESEARCH_REDESIGN_PLAN.md` → Part 0](RESEARCH_REDESIGN_PLAN.md#part-0--reframe-adoption-and-master-status-2026-11-24).

Highest impact first. Items 1–3 move the paper from "framework description" to "an empirical finding other people can build on": the difference between workshop and journal or conference level.

1. **Run the full five-model matrix in the results.** The MLA and Mamba2-hybrid results are the most novel evidence, and right now they exist as engine support, not results.
2. **Make ranking transfer (C2) the headline experiment** *[Nov 2026: was "claim 7 as headline"]*. Test Agrawal & Mayer's transfer claim directly; explain the RIFT/JSSA flips. The GQA-vs-MHA flip, MLA expanded-vs-latent comparison and hybrid total-state accounting are the supporting evidence. Cite vLLM RFC #55196 as practitioner motivation (not as a paper).
3. **Add scale and real tasks:** contexts up to 2–4K where the models allow (≥ 4K for Falcon-H1 and Gemma3), a RULER/NIAH-style task, and at least one downstream benchmark beyond WikiText-2 perplexity. Alternatively, frame explicitly as an *edge/small-model regime* where, as Rethinking shows, KV is a small share of memory and overheads dominate. Batch 1 at 128–512 tokens is where "memory reduction ≠ speedup" is almost guaranteed, so show where it crosses over.
4. **Make the fidelity-to-behavior link rigorous and conditional.** Many configs (bit-widths, budgets, seeds); Spearman/Kendall rank correlations with CIs, per architecture; test whether fidelity metrics *predict* behavior on held-out configs. Frame the result as "where proxies hold and where they break", not a binary verdict.
5. **Add more compressors to the results**, including SnapKV and Palu, which are already implemented.
6. **Add a GDN hybrid** (e.g. a Qwen3.5 sub-2B checkpoint) next to Falcon-H1 *[Nov 2026]*.
7. **Add seeds and confidence intervals everywhere**, and show where "memory ≠ speed" stops holding as batch size or context grows.
8. **Engage the counter-evidence directly:** KV-CoRE, the attention-loss correlation in Hold Onto That Thought, KVDiagnosis *[Nov 2026]*.
9. **Cite EdgeLLM-Nano** for the perplexity pitfall, and disclose that our own pre-fix results had it (audit B1) *[Nov 2026]*.
10. **Reposition the contribution.** Lead with: "a controlled, architecture-aware measurement instrument that separates offline from online cost and links tensor fidelity to behavior and system metrics on small models." Demote claims 1, 4 and 5 and present them as *replications under tighter control*, citing Rethinking and Agrawal & Mayer.
11. **Fix the title and abstract** so they promise a measurement methodology, not "which method wins" (§2.3).

---

## 12. Caveats

- **Provisional basis [Oct 2026]:** the original verdict was judged from plan documents and the survey, not from result tables or the `.tex` draft. [Nov 2026] The alignment audit has since checked the results: they are largely invalid pending a clean re-run.
- **Venue status** was verified at authoritative pages for most entries. Exceptions:
  - Hold Onto That Thought: ICLR 2026 main track unverified; only the NeurIPS 2025 workshop is verified.
  - ACM Computing Surveys serving survey: confirmed only via citations.
  - TransMLA: NeurIPS 2025 per the authors' repo and mlanthology, not checked on neurips.cc.
  - Marconi: MLSys 2025 session listing only.
  - QJL: citing papers disagree on the venue.
  - [Nov 2026] *Does Accuracy Equal Evidence?*: Findings of ACL 2026 per Scholar; not checked on the ACL Anthology.
  - [Nov 2026] KVDiagnosis "KDD '27": one listing only.
- **Unverified names:** HybridKV, OjaKV and PatternKV were not verified. The only "HybridKV" found (arXiv 2604.05887; also ACL 2026 long) targets multimodal LLMs. Search for the others separately before citing.
- **Absence findings** ("no peer-reviewed hybrid-SSM compressor evaluation"; "no KV-evaluation tutorial"; "no cross-architecture rank-transfer study") come from a bounded search and should be re-checked right before submission. The field produces several KV preprints per week.
- **Reading depth [Nov 2026]:** Agrawal & Mayer and Shin were read in full; other papers were read for the abstract, contributions, setup, conclusion and claim-relevant sentences.

---

## 13. Sources

Local copies: [`literature/papers/`](literature/papers/) (index in its `README.md`).

**Benchmarks and evaluation**
- Agrawal & Mayer — https://arxiv.org/abs/2607.05399
- KVDiagnosis — https://arxiv.org/abs/2608.09412
- KV-CoRE — https://arxiv.org/abs/2602.05929
- Hold Onto That Thought — https://arxiv.org/abs/2512.12008
- Rethinking KV Cache Compression (MLSys 2025) — https://arxiv.org/abs/2503.24000
- SCBench (ICLR 2025) — https://arxiv.org/abs/2412.10319
- The Pitfalls of KV Cache Compression (ACL 2026) — https://aclanthology.org/2026.acl-long.1926
- Expected Attention / KVPress — https://arxiv.org/abs/2510.00636
- MetaKV — https://arxiv.org/abs/2609.07966
- Ablation, Statistical Inference, and Validation for KV-Cache Compression — https://arxiv.org/abs/2607.09683
- Understanding the Physics of KV Cache Compression — https://arxiv.org/abs/2603.01426
- EdgeLLM-Nano — https://arxiv.org/abs/2610.14475
- Does Accuracy Equal Evidence? — https://arxiv.org/abs/2608.01631

**Hybrid models and recurrent state**
- vLLM RFC #55196 — https://github.com/vllm-project/vllm/issues/55196
- vLLM fp8 KV-cache blog — https://vllm.ai/blog/fp8-kvcache
- Shin, Joint KV-Cache and Recurrent-State Quantization — https://www.researchgate.net/publication/415823391
- RIFT — https://arxiv.org/abs/2610.31098
- GateSight — https://arxiv.org/abs/2610.14754
- DCKV — https://arxiv.org/abs/2610.36950
- Self-Pruning Transformer (COLM 2026) — https://arxiv.org/abs/2610.09051
- Low-Bit Recurrent States in Hybrid LMs — https://arxiv.org/abs/2609.30950
- When Dynamic Scaling Fails — https://arxiv.org/abs/2610.15139
- Quantizing Recurrent LMs Is Easier Than You Think — https://arxiv.org/abs/2611.01864
- STEPQuant — https://arxiv.org/abs/2609.38169
- LeapQuant — https://arxiv.org/abs/2609.38166
- DAMP — https://arxiv.org/abs/2608.27513
- LinearKV — https://arxiv.org/abs/2608.11231
- Why Gated DeltaNet Survives 4-Bit Quantization — https://arxiv.org/abs/2609.04098
- Nemotron 3 Ultra — https://arxiv.org/abs/2606.15007

**MLA and sliding-window**
- JSSA — https://arxiv.org/abs/2610.19026
- RAQ-MLA — https://arxiv.org/abs/2610.33625
- RoLaQ — https://arxiv.org/abs/2610.20686
- NVFP4-MLA — https://arxiv.org/abs/2611.18972
- AnchorKV — https://arxiv.org/abs/2611.00117
- Spend Bytes on Breadth — https://arxiv.org/abs/2610.05685

**Original-survey venues** (see the link column of §7.3 for the remaining entries).
