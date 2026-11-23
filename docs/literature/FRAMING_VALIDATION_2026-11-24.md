# Framing Validation vs. the Research Market (2026-11-24)

Validates the reframe and verdict in [`../RESEARCH_REFRAME_AND_PRIOR_ART.md`](../RESEARCH_REFRAME_AND_PRIOR_ART.md) (formerly `MISC_Research.md`) (survey cut-off 1 Oct 2026)
against literature published or posted **2025 – 24 Nov 2026**, and against our own state in
[`../audits/DOCS_PLANS_RESULTS_ALIGNMENT_AUDIT.md`](../audits/DOCS_PLANS_RESULTS_ALIGNMENT_AUDIT.md).

Method: Google Scholar (2025+ filter) and arXiv in the browser, plus web search. 31 PDFs were
downloaded to [`papers/`](papers/) (index: [`papers/README.md`](papers/README.md)) and converted to
text. Agrawal & Mayer was read in full. For the others: the abstract, contributions, setup (model
and architecture list), conclusion, and every sentence matching the claim under test were read.
Venue status is as printed on the paper or Scholar on 2026-11-24.

> **Outcome (2026-11-24):** the reframe was **adopted**. Framing and hypotheses now live in [`../RESEARCH_REFRAME_AND_PRIOR_ART.md`](../RESEARCH_REFRAME_AND_PRIOR_ART.md); execution status in [`../RESEARCH_REDESIGN_PLAN.md` → Part 0](../RESEARCH_REDESIGN_PLAN.md#part-0--reframe-adoption-and-master-status-2026-11-24). This file is the dated evidence record and is not updated further. Later reading (Shin in full; 14 more PDFs, now 46 in total) is reflected in the reframe doc §7–8.

---

## Bottom line

1. **The reframe is still the right one, and it is still unclaimed as a whole.** No paper found
   runs a fixed set of standard compressors across MHA / GQA / MQA+sliding / MLA / attention-SSM
   hybrid and reports (a) ranking transfer with statistics and (b) claimed-vs-realized total-state
   savings. That is the same gap MISC identified.
2. **But it is highly perishable.** Between 1 Oct and 24 Nov 2026 at least 12 relevant preprints
   appeared. Each takes one piece of the story: hybrid state accounting, architecture-dependent
   ranking flips, MLA-native compression, and compressed-cache-in-the-loop evaluation. The *whole*
   is still open. Most *parts* are now either published or being tested by others. Estimated
   window: one submission cycle (roughly the next 3–6 months).
3. **Three of MISC's supporting claims need correcting:**
   - "Fidelity ≠ behavior" is contested *in the opposite direction* from the paper's current
     headline. Several sources report proxies that *do* predict.
   - The hybrid "claimed vs realized" gap is real but is **no longer novel as a phenomenon**.
   - Handling only the global layers of sliding-window models is now standard practice, not a
     finding.
4. **The scale objection is stronger than MISC said.** At ≤ 1K tokens the hybrid gap is
   degenerate: vLLM measures **1.00×** fp8 gain on Falcon-H1 at 512 and 1024 tokens. Our current
   128–512 grid would therefore produce a trivial result for the headline architecture claim.
5. **Our hybrid model is the "old" kind.** The 2026 hybrid frontier is Gated-DeltaNet
   (Qwen3.5 / 3.6 / 3.8), including sub-2B checkpoints. One study finds Mamba-2 and GDN states
   behave oppositely under quantization. A Falcon-H1-only hybrid row will be flagged.

**Conference-level verdict:**
- **With current data:** not submittable. Every BEHAVIOR/SYSTEM number is invalid (see the
  alignment audit).
- **With MISC's experiments plus the adjustments below:** the paper is a credible
  Findings-tier (ACL/EMNLP), TMLR, or Datasets & Benchmarks–style submission.
- **Main track at a top venue:** realistic only with ≥ 2–4K context, a GDN hybrid, and
  statistically tested ranking-transfer results that are clearly positive or clearly negative.
  The "small models only" constraint is the main ceiling.

---

## 1. Claim-by-claim validation of MISC

| MISC claim | Evidence found (2025–Nov 2026) | Verdict |
|---|---|---|
| **C1. Claimed vs realized savings per architecture** (hybrid Mamba state untouched; MLA expanded ≠ latent; Gemma sliding layers bounded) | **Hybrid:** vLLM RFC #55196 (verified): fp8 KV gives 2.00× on attention-only models, but on Falcon-H1-1.5B gives 1.00× at 512/1024 tokens, 1.51× at 2K, 1.76× at 8K, 1.84× at 32K; the RFC itself says the state-vs-KV asymmetry is "not claimed as novel" (Marconi, MLSys 2025). Nemotron 3 Ultra: the Mamba SSM cache is quantized separately because it dominates. Shin (2026, ResearchGate): "first joint study" of KV + recurrent-state quantization on GDN hybrids (Qwen3.5 0.8B–9B) **with Falcon-H1 as control**. RIFT (Oct 2026): "KV cache compression methods are rarely evaluated on hybrid-attention models." **MLA:** RAQ-MLA (Oct 2026): "Existing KV-cache quantization methods … do not account for what MLA actually caches"; RoLaQ, NVFP4-MLA and KVarN all compress the *native* latent. JSSA: "Most existing baselines do not define a direct mapping to MLA's shared latent cache." **Sliding:** JSSA, KVzip, AnchorKV and the vLLM fp8 blog all compress only the global layers or skip sliding-window layers. | **Phenomenon confirmed, novelty reduced.** The hybrid gap is now practitioner-documented *and* has a dedicated quantization study (which also uses Falcon-H1). What remains new is a **cross-mechanism** realized-savings table (eviction, low-rank, quantization, sketching × 5 architectures, with quality attached). MLA expanded-vs-native for *third-party* compressors is still unclaimed, but needs our Gate C. The sliding-layer point is a disclosure, not a contribution. |
| **C2. Ranking transferability (Kendall τ across architectures)** | RIFT: on full-attention Llama 3.1, SnapKV > SWA; on Nemotron3 (hybrid SSM) and GPT-OSS (SWA interleave), **SWA > SnapKV**, an architecture-induced flip. JSSA (Oct 2026): competing methods show "architecture-dependent failures" across GQA / hybrid local-global / MLA; FASA's gap is largest on Gemma4 and GLM (MLA). Agrawal & Mayer explicitly claim trends "are transferrable across model families", but test only two GQA models. Pitfalls (ACL 2026): "both eviction policy and model play a role". No paper computes rank agreement with CIs across ≥ 4 attention classes. | **Strongest remaining contribution, and now better motivated than in October.** There is published evidence of flips (method papers, side results) and a published counterclaim of transfer (Agrawal & Mayer). A controlled study answering it with statistics is exactly the missing piece. |
| **C3. Proxy validity (FIDELITY → BEHAVIOR/SYSTEM) with rank stats, CIs, held-out prediction** | KV-CoRE (arXiv, Feb 2026): NER vs PPL **Pearson r = 0.88 (values), 0.64 (keys)** on Qwen3-4B, low-rank truncation only. Hold Onto That Thought: "higher performance correlates with less attention loss". KVDiagnosis (Aug 2026): all ten diagnostics separate failed from successful compression, **AUROC 0.684–0.871**. Pitfalls already uses Spearman rank correlation. EdgeLLM-Nano (Oct 2026): perplexity gives **the same 5.461 score to every method** because it never reads the compressed cache; they introduce LCDF (teacher-forced decode through the compressed cache). | **The paper's current headline ("offline fidelity does not predict online quality") is contradicted** by several 2026 results. Defensible version: *proxy validity is conditional on mechanism and architecture; here is where it holds and where it breaks, with CIs.* EdgeLLM-Nano also publishes the exact pitfall our audit found (B1), and its fix (LCDF ≈ our new PPL protocol), so we cannot claim that protocol as novel. Cite it. |
| **Offline-cost break-even** (calibration-heavy vs calibration-free) | No paper found that treats calibration cost as a measured, amortized axis. Agrawal & Mayer contrast "data-aware vs data-oblivious" only qualitatively. | **Still open, but small.** A good secondary result, not a headline. |
| **"Not the first unified benchmark"; demote claims 1, 4, 5** | Confirmed and strengthened. Agrawal & Mayer (PVLDB-formatted, arXiv v1, 3 May 2026); KVDiagnosis (25-method taxonomy, common record format, Qwen3-8B, 59,800 runs); Ablation/Statistical-Inference (Jun 2026, TurboQuant/QJL fidelity with KS significance and seed effects); KVPress (supports Qwen3/Gemma3; no MLA support found). Rethinking (MLSys 2025) and MetaKV both confirm "ratio ≠ speed". Agrawal & Mayer find TurboQuant has the **worst throughput and TTFT overhead**, consistent with our 0.08 tok/s. | **Agree.** The harness, the taxonomy, and "ratio ≠ speedup" are reproductions. Note: Ablation/Statistical-Inference already studies TurboQuant/QJL fidelity with significance testing, which overlaps with our FIDELITY branch. |
| **Scale critique** (128–512 tokens, batch 1, WikiText) | vLLM RFC: Falcon-H1 fp8 gain is exactly 1.00× at 512/1024 tokens. Self-Pruning Transformer (COLM 2026): at short lengths Mamba2/GDN hybrids use *more* memory than full attention, so compression gains look smaller. Agrawal & Mayer use LongBench 1K–64K; JSSA up to 128K; GateSight up to 262K. | **Stronger than MISC stated.** At our grid the hybrid and sliding-window effects are near-degenerate, so the architecture claim needs ≥ 2K (Falcon-H1 crossover ≈ 800 tokens; Gemma3 window 512). |

## 2. New competitors since MISC's cut-off (1 Oct 2026)

| Paper | Date / venue | Threat to | Note |
|---|---|---|---|
| Shin, *Joint KV-Cache and Recurrent-State Quantization in Hybrid Linear-Attention LLMs* | 2026, ResearchGate (no arXiv found); read in full | C1 (hybrid) | Claims the "first" joint KV + state bit-allocation study; Qwen3.5 0.8–9B GDN plus **Falcon-H1 control**; derives the memory **crossover T\* (768 tokens on Qwen3.5-4B)**, close to our ≈ 800-token estimate for Falcon-H1; Pareto across context, scale and task; 4-bit state costs ~1 nat on GDN but nothing on Mamba-2; limiting memory is task-dependent; cites vLLM #55196. **Closest direct overlap**, but quantization only (no eviction / low-rank / sketching) and hybrids only (no MHA / GQA / MLA comparison). |
| RIFT (2610.31098) | Oct 2026, arXiv | C2 | SnapKV↔SWA ranking flips on hybrid and SWA models |
| JSSA (2610.19026) | Oct 2026, arXiv | C2, MLA | GQA / hybrid local-global / MLA; "architecture-dependent failures" of baselines |
| EdgeLLM-Nano (2610.14475) | Oct 2026, arXiv | Protocol novelty | PPL-without-compressed-cache pitfall; LCDF; seeds and effect sizes |
| GateSight (2610.14754) | Oct 2026, arXiv | C1 hybrid (method) | Hybrid GDN eviction, Qwen3.8-27B |
| DCKV (2610.36950) | Oct 2026, arXiv | C1 hybrid (method) | Learned KV codec on Qwen3.5-2B hybrid |
| RAQ-MLA (2610.33625), RoLaQ (2610.20686), NVFP4-MLA (2611.18972) | Oct–Nov 2026, arXiv | MLA | Native-latent compression; RAQ states generic quantizers mis-fit MLA |
| Low-Bit Recurrent States (2609.30950), When Dynamic Scaling Fails (2610.15139, NeurIPS 2026 workshop), Quantizing Recurrent LMs Is Easier (2611.01864), STEPQuant, LeapQuant, DAMP | Aug–Nov 2026 | C1 hybrid | Recurrent-state quantization is now its own crowded subfield |
| Self-Pruning Transformer (2610.09051) | **COLM 2026** | Scale / hybrid | Short-context hybrid memory caveat |
| *Does accuracy equal evidence? Reasoning faithfulness under KV cache compression* (arXiv 2608.01631) | **Findings of ACL 2026** (Scholar) | C3 | Answer-evidence gap: eviction keeps final-answer accuracy while reasoning support degrades; quantization much less affected (downloaded) |
| *Beyond Proxy Metrics: … Generative Faithfulness* | OpenReview (venue unverified) | C3 | Proxy metrics correlate poorly; covers KV-cache dropping (abstract only) |

## 3. Status of MISC's own citations (re-checked)

| Item | MISC said | Now |
|---|---|---|
| vLLM RFC #55196 | practitioner evidence | **Verified**; contains Falcon-H1 per-context numbers (above) |
| Agrawal & Mayer | May 2026 preprint, PVLDB-formatted | Still arXiv v1; models Llama-3.1-8B, Mistral-7B (both GQA); **explicit cross-family transfer claim** (Section 5) |
| KVDiagnosis | Aug 2026 preprint | Verified (Qwen3-8B, 59,800 runs, AUROC 0.684–0.871) |
| Hold Onto That Thought | NeurIPS 2025 workshop; ICLR 2026 unverified | Scholar still lists arXiv only |
| KV-CoRE | OpenReview submission | Scholar: arXiv 2602.05929 only (6 citations) |
| Pitfalls | ACL 2026 | Verified (ACL Anthology); uses Spearman |
| Rethinking | MLSys 2025 | Verified |
| SCBench | ICLR 2025 | Verified |

## 4. Perishability

| Sub-claim | Open on 2026-11-24? | Pressure | Half-life |
|---|---|---|---|
| Whole study (transfer + full-state accounting, 5 architecture classes, controlled) | **Yes** | High: every component is being touched | ~1 cycle |
| Ranking transfer with statistics | Yes | Medium–high (RIFT, JSSA report flips as side results) | ~1 cycle |
| Hybrid realized-savings phenomenon | **No**, documented (vLLM RFC, Nemotron, Shin) | — | Already spent; usable only as confirmation |
| Cross-mechanism realized-savings table | Yes | Medium | 1–2 cycles |
| MLA expanded vs native for third-party compressors | Yes | Medium (native-latent methods appearing monthly) | 1–2 cycles; needs Gate C |
| Proxy validity (conditional, with CIs) | Partly; debate is live | Medium | Durable if framed as conditional |
| Calibration-cost break-even | Yes | Low | Durable, but minor |
| Compressed-cache-in-the-loop PPL protocol | **No** (EdgeLLM-Nano LCDF) | — | Cite, don't claim |

## 5. What this implies for the framing (no changes made yet)

1. **Lead with C2 (ranking transfer).** It has the clearest gap, a published counterclaim to test
   (Agrawal & Mayer), and published anecdotal flips to explain (RIFT, JSSA).
2. **Reposition C1** from "nobody has shown the hybrid gap" to "a cross-mechanism, cross-
   architecture realized-savings table, with quality". Cite vLLM #55196, Marconi, Nemotron 3 and
   Shin as prior evidence.
3. **Add a GDN hybrid** (e.g. a Qwen3.5 sub-2B checkpoint) next to Falcon-H1. A Mamba-2 vs GDN
   contrast is itself an architecture-transfer result.
4. **Recast the proxy claim as conditional**, with Spearman/Kendall and bootstrap CIs per
   architecture, and engage KV-CoRE, Hold Onto That Thought, KVDiagnosis and Pitfalls directly.
5. **Run ≥ 2K context** (≥ 4K for Falcon-H1 and Gemma3); below that the architecture effects are
   degenerate by construction.
6. **Cite EdgeLLM-Nano** for the perplexity pitfall, and disclose that our own pre-fix results had
   it (audit B1).

## Sources

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
- vLLM RFC #55196 — https://github.com/vllm-project/vllm/issues/55196
- vLLM fp8 KV-cache blog — https://vllm.ai/blog/fp8-kvcache
- Shin, Joint KV-Cache and Recurrent-State Quantization — https://www.researchgate.net/publication/415823391
- RIFT — https://arxiv.org/abs/2610.31098
- JSSA — https://arxiv.org/abs/2610.19026
- EdgeLLM-Nano — https://arxiv.org/abs/2610.14475
- GateSight — https://arxiv.org/abs/2610.14754
- DCKV — https://arxiv.org/abs/2610.36950
- Self-Pruning Transformer (COLM 2026) — https://arxiv.org/abs/2610.09051
- RAQ-MLA — https://arxiv.org/abs/2610.33625
- RoLaQ — https://arxiv.org/abs/2610.20686
- NVFP4-MLA — https://arxiv.org/abs/2611.18972
- AnchorKV — https://arxiv.org/abs/2611.00117
- Low-Bit Recurrent States in Hybrid LMs — https://arxiv.org/abs/2609.30950
- When Dynamic Scaling Fails — https://arxiv.org/abs/2610.15139
- Quantizing Recurrent LMs Is Easier Than You Think — https://arxiv.org/abs/2611.01864
- STEPQuant — https://arxiv.org/abs/2609.38169
- LeapQuant — https://arxiv.org/abs/2609.38166
- DAMP — https://arxiv.org/abs/2608.27513
- LinearKV — https://arxiv.org/abs/2608.11231
- Why Gated DeltaNet Survives 4-Bit Quantization — https://arxiv.org/abs/2609.04098
- Nemotron 3 Ultra — https://arxiv.org/abs/2606.15007
- Spend Bytes on Breadth — https://arxiv.org/abs/2610.05685
