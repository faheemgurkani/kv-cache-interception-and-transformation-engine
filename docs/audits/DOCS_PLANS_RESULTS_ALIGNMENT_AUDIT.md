# Docs, Plans and Results — Alignment Audit vs `MISC_Research.md`

Assessment date: 2026-11-24 (after commits #201–#204).
Scope: current codebase, evaluation framework and measured evaluations, checked against
[`../RESEARCH_REFRAME_AND_PRIOR_ART.md`](../RESEARCH_REFRAME_AND_PRIOR_ART.md) (formerly `MISC_Research.md`),
[`../ENGINE_AND_EVALUATION_FRAMEWORKS_REDESIGN_PLAN.md`](../ENGINE_AND_EVALUATION_FRAMEWORKS_REDESIGN_PLAN.md) and
[`../RESEARCH_REDESIGN_PLAN.md`](../RESEARCH_REDESIGN_PLAN.md).
Companion: [`GEMMA3_POST_OVERHAUL_AUDIT.md`](GEMMA3_POST_OVERHAUL_AUDIT.md).

---

## Bottom line

The docs are not aligned. `MISC_Research.md` and the two redesign plans disagree on the research question, the headline contribution, and what evidence the paper needs. The measured results are in worse shape than any of the three documents assumes: the October 2026 Gemma3 audit invalidated most BEHAVIOR and SYSTEM numbers, including the ones behind the paper's abstract.

Neither redesign plan mentions `MISC_Research.md`. Both were last marked "complete" between Aug 19 and Sept 1, before the audit.

## 1. Where things stand

**Engine and framework: mostly ready.** Four of five shortlist models pass all compatibility gates. TinyDeepSeek still fails Gate C. There are 63 test files. FIDELITY, BEHAVIOR, SYSTEM, cost, Pareto, cross-dimensional analysis and the controlled-conditions export are all implemented. All audit fixes (B1–B8, S1–S4, P1, P2, F1, R1) are now committed (#201–#204).

**Measured results: very thin, and largely invalid.**

| Model | What was actually measured | Status |
|---|---|---|
| Qwen3-1.7B (legacy) | 27 jobs: identity, TurboQuant, QJL, RocketKV | TQ `full`-stage value/PPL invalid (B8). QJL invalid (B3, B8; the Aug ProdQJL re-run came after the bug landed). RocketKV throughput likely affected (B6). Old SYSTEM baselines invalid (S1). Only TQ keys and the `wht_quant` ablation hold. |
| OLMo-2-1B | Same 27-job grid | Same contamination. Not re-measured (audit item L1 open). |
| Gemma3-270M | 18 jobs: 6 methods × ctx 128/256/512, one config each, one seed | Audit verdict: "not usable for the paper". Every PPL number is invalid. |
| Qwen3-0.6B | One identity smoke run at ctx 128 | Plumbing check only |
| Falcon-H1 / TinyDeepSeek | Nothing; the reference tests pass, but there are zero result files | — |

- **Modal state:** no compute has run since the fixes. The workspace was paused in the last log entry (2026-10-02).
- **Paper headline is directly affected:** the abstract's "TurboQuant 4-bit ≈1.3× PPL" is the `tq_full_b4` row, which B8 invalidates. Its "QJL collapses" claim rests on QJL runs B3 invalidates.
- **Partly salvageable:** the unaffected `wht_quant` b4 row gives a similar number (18.84 vs 18.64 at ctx 512).

"All five models run full eval" (the engine plan, `CURRENT_STATE.md`, `CLAUDE.md`) means the reference tests pass, not that results exist.

## 2. What MISC_Research needs vs current status

| MISC requirement | Status |
|---|---|
| All 5 models × 4–5 compressor configs, several budgets and seeds | ❌ 0 valid models. Gemma3 has one config per method. Multiple budgets are **Phase 8: "Not planned"**. |
| Hypotheses stated up front | ❌ None in any plan |
| 2–4K context plus a needle/RULER-style task | ❌ Every config caps at 512. **Phases 11 and 12: "do not implement"**. The needle task exists but is synthetic. |
| Rank correlations with CIs, per architecture, held-out prediction | ❌ `eval/cross_dim/correlation.py` computes Pearson only. There is no Spearman, Kendall, bootstrap or CI anywhere in the code. |
| Kendall's τ for ranking transfer across architectures | ❌ Not implemented |
| Claimed vs realized savings (contribution 1) | ⚠️ Half-built. `memory.py` reports both `kv_compression_ratio` and total-state ratio for hybrids, but there is no table and no data. |
| MLA expanded vs native | ❌ Needs Gate C (native latent), which the engine plan lists as optional future work |
| Offline-cost break-even | ❌ Cost accounting exists; no break-even or amortization calculation exists |
| Engage KV-CoRE, Hold Onto That Thought, Rethinking, SCBench, KVPress, Marconi, KVzip | ❌ None are in `reference.bib` or `docs/literature/`. Agrawal & Mayer is in the bib (`kvbench2026serving`). |

## 3. Where the documents contradict each other

1. **Research question.** Research plan Phase 15: "How should KV-cache transformations be evaluated?", with KVBench as the headline. MISC: do results transfer across architectures, with KVBench demoted to a supporting artifact. The paper title is still "KVBench: Bridging…", which MISC says to avoid.
2. **Contributions.** Phase 22 makes C1 the "unified interception framework" and C4 "two SLM architectures". MISC says demote claims 1, 4 and 5 and make the five-architecture matrix the headline.
3. **Evidence scope.** Phases 8, 11 and 12 are frozen as "sufficient for the paper". These are exactly MISC's evidence items 1 and 3.
4. **Fidelity vs behavior.** Phase 23 makes F4 ("offline does not predict online") the headline, judged by a `|r| < 0.5` Pearson rule. MISC calls the claim contested and asks for correlations with CIs, not a yes/no verdict.
5. **Effort allocation.** Phases 15–33 are mostly wording specs. MISC explicitly says to cut them.
6. **"No new GPU jobs".** Phases 15, 23 and 24 assume the existing bundles can be reused. The audit says the whole grid must be re-run on one clean commit.

**The engine plan is the most compatible with MISC.** Hybrid state accounting, per-layer Gemma handling and the expanded-KV disclosure are what contribution 1 needs. But its "Bottom line: hardening complete" predates eight critical or high-severity bugs, and it leaves native MLA as optional.

## 4. Numbers I worked out from the repo

These bear directly on MISC's contribution 1. They are my arithmetic from model configs and probe bytes, not measured results.

- **Falcon-H1:**
  - Attention KV is 18,432 B/token. The Mamba state is a constant ~14.7 MB.
  - Attention only overtakes Mamba at about 800 tokens.
  - A nominal 4× KV compression gives about 1.42× total-state reduction at ctx 512, and about 2.69× at 4K.
  - So the claimed-vs-realized gap depends on context length, and the 128–512 grid shows only the extreme end of it.
- **Gemma3:** `sliding_window = 512`. With ctx ≤ 512, sliding layers start evicting only during the 64-token decode tail of the ctx-512 jobs. The "sliding layers are already bounded" effect is essentially invisible in the current grid.
- **TinyDeepSeek:** the expanded cache is 384 elements per token per layer, versus 288 for the native latent (`kv_lora_rank` 256 + RoPE 32). Any ratio measured on the expanded cache overstates savings by about 1.33× relative to a native-MLA deployment.

The first two points mean MISC's scale requirement is necessary for the architecture claim itself, not only for reviewer optics.

## 5. Other inconsistencies

- **`CLAUDE.md` names the wrong legacy model.** It says the legacy results ran on `models/legacy/olmo1b` (OLMo-1). The results JSON says `allenai/OLMo-2-0425-1B`, which is the same checkpoint as shortlist `models/olmo2_1b`. I can fix this; it's a local file.
- **`CURRENT_STATE.md` is out of date.** It still presents the August ProdQJL sweeps as canonical, which the audit now invalidates.

## Recommendation

If you adopt MISC's framing, the critical path is:

1. Decide on the reframe. Then rewrite Phase 15/17/22 and un-freeze Phases 8 and 12, at least ctx 1–4K for Falcon and Gemma3.
2. Resume Modal and run the `audit_verify` preset to confirm the fixes on GPU.
3. Add Spearman/Kendall with bootstrap CIs to `eval/cross_dim/`, plus a claimed-vs-realized table and a break-even calculation.
4. Write the hypotheses down, then run one clean-commit grid: 5 models × about 5 configs × budgets × seeds.

I can write this up as a shareable page, or turn it into a reconciliation section at the top of `RESEARCH_REDESIGN_PLAN.md`, if you want.

---

## Appendix — Resolution status (2026-11-24)

The body above is the verbatim 2026-11-24 snapshot. On the same day the project adopted the reframe and aligned all plans and docs. Status of each item:

| Item from this audit | Status | Where resolved |
|---|---|---|
| §3.1 Research question conflict | ✅ Resolved | Reframe adopted; Phase 15 marked superseded ([Part 0](../RESEARCH_REDESIGN_PLAN.md#part-0--reframe-adoption-and-master-status-2026-11-24) §0.3) |
| §3.2 Contributions conflict | ✅ Resolved | Phase 22 superseded by reframe C1–C3 |
| §3.3 Phases 8/11/12 frozen | ✅ Resolved | Un-frozen / partly un-frozen (R11, R14, R5) |
| §3.4 Fidelity-vs-behavior headline | ✅ Resolved in docs | F4 retired in blanket form; restated as conditional hypothesis H4 (README, CURRENT_STATE, METHODOLOGY, REPRODUCIBILITY updated) |
| §3.5 Phases 15–33 wording effort | ✅ Resolved | Superseded/revised per §0.3; effort moves to experiments |
| §3.6 "No new GPU jobs" | ✅ Resolved | Declared void in Part 0 and phase banners |
| §2 Hypotheses stated up front | 🟡 Drafted | H1–H6 in [`../RESEARCH_REFRAME_AND_PRIOR_ART.md` §5.1](../RESEARCH_REFRAME_AND_PRIOR_ART.md#51-hypotheses-draft); freeze = R4 |
| §2 Five models × configs × budgets × seeds | ❌ Open | R11, R13 |
| §2 2–4K context + needle/RULER task | ❌ Open | R5, R14 |
| §2 Rank correlations with CIs / Kendall τ | ❌ Open | R8 (engine E5) |
| §2 Claimed vs realized savings | 🟡 Open | R9 (engine E6) |
| §2 MLA expanded vs native | ❌ Open | R7 (engine E4), now required for H3 |
| §2 Offline-cost break-even | ❌ Open | R10 (engine E7) |
| §2 Missing literature in bib | 🟡 Survey done, bib open | R15 |
| §1 Modal verification of fixes | ⛔ Blocked | R2 (workspace paused as of last log) |
| §5 `CLAUDE.md` wrong legacy model | ✅ Fixed | Also corrected in `architecture/MODEL_ARCHITECTURE_MATRIX.md` |
| §5 `CURRENT_STATE.md` presents ProdQJL sweeps as canonical | ✅ Fixed | Marked invalid |
| §1 "All five models run full eval" wording | ✅ Clarified | Engine plan Part 0, CURRENT_STATE, MODEL_ARCHITECTURE_MATRIX, shortlist results page |
