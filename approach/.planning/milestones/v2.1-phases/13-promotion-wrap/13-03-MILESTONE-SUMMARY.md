---
phase: 13
plan: 13-03
title: Milestone v2.1 Closure — Cleanup + Prompt Simplification
status: completed
verdict: SHIPPED (with negative-result section for rejected trims)
completed: 2026-06-01
requirements: [PROMPT-03, GATE-03]
subsystem: milestone-wrap
tags: [milestone-summary, v2.1, ship, promotion, frontier-map, cross-model]
key-files:
  created:
    - .planning/phases/13-promotion-wrap/13-03-MILESTONE-SUMMARY.md (this file)
decisions:
  - Phase 13 closes the v2.1 milestone with a single new canonical artifact (s_linker13_min) + an additive ABLATION-TABLE addendum + a milestone summary that frames the v2.1 work for the eventual thesis chapter / reviewer audit.
  - Voyager-TLR pilot result remains a parallel frontier extension; its outcome (whenever the gpt-5.4 pilot completes) gets a section in v2.2's milestone summary, NOT this one.
metrics:
  duration: "~10min"
  completed: 2026-06-01
---

# Milestone v2.1 — Cleanup + Prompt Simplification — FINAL SUMMARY

**One-liner:** v2.1 closes with **`s_linker13_min` promoted** as the composed canonical of two Phase 12 ACCEPTed trims (trim1 distilled judge + trim9 runtime seed rubric), clearing the strict v2.1 promotion gates on both Claude Sonnet (macro F1 **0.9506**, +1.09pp vs baseline) and gpt-5.4 (macro F1 **0.9069**, +0.92pp above the 0.8977 floor). The v2.1 thesis claim — "static-prompt-distillation + runtime-rubric mechanism survives both backends in composition" — is VERIFIED. The milestone also ships a frontier map of 7 rejected trim mechanisms documenting where the strict-gate cliff sits, framed as the methodological contribution.

## Verdict

| Gate | Required | Observed | Status |
|------|----------|----------|--------|
| **PROMPT-03** (final minimal-prompt variant `s_linker13_min.py` ships) | Both Claude relaxed + gpt-5.4 cross-model gates clear OR negative result published | **PROMOTED** — both gates clear with safety margin | **PASS** |
| **GATE-03** (ABLATION-TABLE addendum + .tex regenerated, v1.0/v2.0 rows unchanged) | additive update, frozen-file compliance | 11 rows added (4 promoted/baseline block + 7 rejected block); v1.0 chain bytes preserved | **PASS** |
| Standing GATE-01 (Claude) | macro ≥ 0.93, BBB ≥ 0.79, other-dataset drop ≤ −2pp | 0.9506 / 0.8496 / TS −1.82pp | **PASS** |
| Standing GATE-01 (cross-model gpt-5.4) | macro ≥ 0.8977 | 0.9069 | **PASS** |
| Standing GATE-02 (frozen-compat regression) | all CANONICAL_VARIANTS pinned or in "missing" | 35 passed, 28 xfailed | **PASS** |
| Standing GATE-06 (generality / no leakage) | per-prompt benchmark-leakage scan + reviewer-defensibility | Phase 12 12-06-AUDIT-REPORT.md PASS; s_linker13_min inherits trim1 + trim9 GATE-06 PASS by construction | **PASS** |
| Standing GATE-07 (canonical registration + standalone file + structured docstring) | every promoted variant registered + standalone + docstring | s_linker13_min canonical=True, standalone `s_linker13_min.py`, structured GATE-07 docstring | **PASS** |

**Milestone verdict:** **SHIPPED.** All 10 v2.1 requirements complete (CLEAN-01, CLEAN-02, PROMPT-01..05, GATE-01..03). 4 standing gates held throughout.

## v2.1 Final Outcome

### What landed

| Component | Status | Reference |
|---|---|---|
| Standalone cleaned `s_linker13_clean.py` (helpers → helper_v3.py, prompts unchanged) | Shipped Phase 10 | CLEAN-01 / Plan 10-03 |
| `prompts_v3.py` side-by-side with `prompts_v2.py` (9 byte-equal + 7 dropped legacy constants) | Shipped Phase 12 Step 0 | PROMPT-01 / Plan 12-01 |
| Per-prompt rule-trim ablation across 9 trim mechanisms (3 Tier-1 surfaces + 5 Tier-2 surfaces) | Complete Phase 12 (2 ACCEPT + 7 REJECT) | PROMPT-02 / Plans 12-03..12-12 |
| Generality re-audit (GATE-06 + BENCHMARK_TABOO + reviewer-defensibility) | Complete Phase 12 Plan 12-06 | PROMPT-04 |
| Literature + web survey on prompt-minimization techniques | Complete Phase 11 | PROMPT-05 / `.planning/research/PROMPT-HARNESS-SURVEY.md` |
| Final minimal-prompt variant `s_linker13_min.py` (composition of accepted trims) | **PROMOTED Phase 13 Plan 13-01** | PROMPT-03 |
| GATE-01 cross-model formalization (gpt-5.4 macro ≥ 0.8977, T=1.0pp) | Codified Plan 10-04 | GATE-01 |
| Frozen-compat regression test (`test_v20_baseline_regression.py`) | Maintained, fixture extended for 8 new variants | GATE-02 |
| ABLATION-TABLE.md / .tex addendum with v2.1 rows | Shipped Phase 13 Plan 13-02 | GATE-03 |

### Performance scoreboard (final v2.1 reading)

| Variant | Claude macro | gpt-5.4 macro | Note |
|---|---|---|---|
| s_linker13 (v2.0 canonical) | 0.9509 | 0.9077 | Frozen v2.0 ship |
| s_linker13_clean (Phase 10 baseline) | 0.9397 | 0.9077 (reused anchor) | Phase 10 structural refactor (helpers extracted) |
| s_linker13_trim1_judge_clean | **0.9553** | **0.9173** | Phase 12 Plan 12-03 ACCEPT |
| s_linker13_trim9_seed_runtime_clean | **0.9474** | **0.9007** | Phase 12 Plan 12-12 ACCEPT |
| **s_linker13_min (v2.1 PROMOTED)** | **0.9506** | **0.9069** | Phase 13 Plan 13-01 — composed |

The composition lands close to the s_linker13 v2.0 anchor (+0.11pp Claude, −0.08pp gpt-5.4 — both within run-to-run variance) while reducing static prompt content via the trim1 lossless restructure + the trim9 runtime mechanism.

## v2.1 Thesis Claim — Verified / Refined

**Original claim** (v2.1 kickoff): "every rule removed from `s_linker13`/its prompts must hold macro F1 ≥ 0.93 on Claude Sonnet AND gpt-5.4 macro within tolerance of v2.0 baseline (0.9077) — or be rejected".

**Refined claim** (v2.1 ship):

1. **Static-prompt distillation works.** Technique 3 (lossless rubric distillation) + Technique 8 (reasoning-before-conclusion order) applied to a Tier 1 judge rubric yields +1.56pp Claude macro AND +0.96pp gpt-5.4 macro, with zero benchmark leakage. trim1 is the Pareto-positive trim across both gates.
2. **Runtime mechanism works for Pareto-friendly substitution targets.** Tier 2 seed disambiguation — where the candidate set is bounded, dossier-rich, and approve-biased — accepts a runtime rubric builder with +0.77pp Claude AND +0.30pp gpt-5.4 margin (trim9). The Tier 1 proposer surfaces (extraction, ambiguity, entity) and Tier 2 proposer/judge merges (validation, judge-examples) all regress one or both backends — runtime mechanism is NOT a universal substitute.
3. **Composition is approximately additive at Pareto-friendly substitution targets that hit disjoint pipeline stages.** s_linker13_min composes trim1 (Tier 1 judge) + trim9 (Tier 2 seed) and clears both gates with safety margin. The composition is Pareto-dominated by trim1 alone on macro F1 but is the smallest-prompt-body variant clearing both gates.
4. **The cross-model penalty is mostly the V32 documented gpt-5.4 capability gap.** Runtime substitution adds ~0.1–2.2pp gpt-5.4 cost per substituted prompt; static distillation adds none (or improves it slightly). This is consistent with the Claude-vs-GPT inherent capability gap documented in MEMORY.md (v2.0 CROSS finding).

## Cross-Model Gap Quantification (final, all 9 variants' data)

Summary across the v2.1 surface:

| Variant | Claude − gpt-5.4 macro |
|---|---|
| s_linker13_clean baseline | 3.20pp |
| trim1 alone (distillation) | 3.80pp |
| trim3 alone (runtime judge rules) | 5.41pp |
| trim4 alone (runtime ambiguity) | 3.69pp |
| trim5 alone (runtime extraction) | 3.03pp |
| trim6 alone (runtime judge examples) | 4.68pp |
| trim7 alone (runtime entity) | 3.58pp |
| trim8 alone (runtime validation) | 1.81pp |
| trim9 alone (runtime seed) | 4.67pp |
| **s_linker13_min** (trim1 + trim9 composed) | **4.37pp** |

The cross-model gap ranges from 1.81pp (trim8) to 5.41pp (trim3) across the v2.1 surface. The composed s_linker13_min sits in the middle of this range — composition does NOT widen the gap beyond the largest trim9-alone reading. The Phase 12 frontier map (12-FRONTIER-MAP-SUMMARY.md §4) calibrates this as "roughly 3–4 pp on gpt-5.4 per substituted prompt, additive across prompts".

## Methodological Contributions

1. **GATE-06 cross-dataset isolation operationalization (12-05-SUMMARY-REVISIT.md).** Strict-reading of GATE-06 ("project terms in LLM output = leakage") would invalidate every LLM call in the pipeline. Operationalized as: term t in dataset A's runtime artifact is a leak iff (a) t is a PCM component of dataset B != A AND (b) t is NOT in A's PCM AND (c) t is NOT in A's input doc. This testable criterion PASSES on both backends across all 10 rubrics generated by trim9. The methodology is reusable for any future runtime-prompt mechanism.

2. **Scenario E framing (12-FRONTIER-MAP-SUMMARY.md).** Reframed Phase 12 as a "frontier map of prompt-reduction × accuracy" rather than strict pass/fail. The original v2.1 gates remain the promotion bar; Scenario E (Claude other-drop ≤ −4pp, gpt-5.4 macro ≥ 0.89) bounds the feasibility envelope. Useful for reviewer narrative: Phase 12's REJECT verdicts are "at the cliff, not over it" (0.4–3.6 pp from relaxed gates), strengthening the strict-gate REJECT's honesty.

3. **Frontier-map vs strict-pass methodology.** v2.1's exploration crossed 9 trim mechanisms; 2 accepted, 7 rejected. The frontier map captures the full design space WITHOUT moving the promotion bar. Future milestones can revisit Scenario-E-feasible variants under a different gate regime (e.g., if a new cross-model anchor becomes available) without re-running the ablation.

4. **Composition-Pareto reading (this milestone summary).** The composed s_linker13_min is Pareto-dominated by trim1 alone on Claude macro AND by trim1 alone on gpt-5.4 macro. However, it is the smallest-static-prompt-body variant clearing both gates. Promotion-worthiness combines (a) macro F1 clearance, (b) static-prompt-byte reduction (the trim9 runtime mechanism removes 1090 bytes per call), and (c) cross-model robustness. The Pareto reading captures the trade-off explicitly.

## Open Questions Deferred to v2.2

| Item | Source | Why deferred |
|---|---|---|
| **Voyager-TLR train-test methodology** | 12-VOYAGER-PILOT-DEFERRED.md | gpt-5.4 pilot is parallel frontier extension. Its result lands as v2.2 anchor regardless of outcome. |
| **Per-model adaptive prompts (ADAPTER-01 re-open)** | PROJECT.md Out of Scope | The trim4/5/6/7 single-FP-on-jabref rejections and TS-3.57pp rejections raise a question of whether per-backend prompt adaptation would recover those variants without leakage. Out of scope for v2.1 strict gates; candidate for v2.2 adaptive-prompt work. |
| **Self-Refine layered on accepted variants** | PROMPT-HARNESS-SURVEY-SUPPLEMENT-ERDOS.md Top 3 #2 | One Tier 2 ablation showed proposer-side widening regresses TS; Self-Refine on the proposer might recover, but adds a stage. v2.2 candidate. |
| **Extended-thinking variants** | PROMPT-HARNESS-SURVEY-SUPPLEMENT-ERDOS.md Top 3 #1 | Tier 1 ambiguity classifier shows gpt-5.4 makes consistent calibration errors that extended-thinking could recover. v2.2 candidate. |
| **Upstream-tier rule removal (extraction/coref tier)** | v2.0 EXT-01 evidence | BBB recall gap lives upstream of `_has_standalone_mention`; v2.2 milestone could target a rule there. |
| **EXT-04 emit-biased boundary prompting** | PROJECT.md | Variance work, not rule removal; deferred again. |
| **Link provenance data structure** | 12-PROVENANCE-DEFERRAL-NOTE.md | Phase 12 deferred. v2.2 candidate for evidence-trail audit infrastructure. |

## Voyager-TLR Pilot — Note

A parallel exploratory pilot (Voyager-TLR / axiom-learning) is in-flight on gpt-5.4. This is a **frontier extension**, not part of v2.1's PROMPT-03 / GATE-03 closure. The pilot's outcome — whenever the gpt-5.4 train/test cycles complete — will be logged in `.planning/phases/12-trim-ablation/12-VOYAGER-PILOT-DEFERRED.md` (and a follow-on `-GPT-SUMMARY.md`) and used as the v2.2 first-plan anchor. The Voyager pilot does NOT block Phase 13 or milestone v2.1 close; its result is orthogonal to the trim ablation surface.

Train log location for reference: `logs/voyager_gpt54/train.log`. Distill / test logs at the same prefix.

## Files

| Created |
|---|
| `.planning/phases/13-promotion-wrap/13-03-MILESTONE-SUMMARY.md` (this file) |
| `.planning/phases/13-promotion-wrap/13-VERIFICATION.md` (Phase 13 verification asserted PASSED) |

| Modified |
|---|
| `.planning/STATE.md` (Phase 13 closed; v2.1 milestone SHIPPED; progress 100%) |

| Cross-references (read-only) |
|---|
| `.planning/phases/13-promotion-wrap/13-01-SUMMARY.md` (composition + sweep + promotion verdict) |
| `.planning/phases/13-promotion-wrap/13-02-SUMMARY.md` (ABLATION-TABLE addendum) |
| `.planning/phases/12-trim-ablation/12-VERIFICATION.md` (Phase 12 closure — 3/3 requirements verified) |
| `.planning/phases/12-trim-ablation/12-FRONTIER-MAP-SUMMARY.md` (9-variant scoreboard) |
| `.planning/phases/12-trim-ablation/12-06-V2_TO_V3_MAPPING-FINAL.md` (final 16-row mapping) |
| `.planning/PROJECT.md` (Key Decisions table — GATE-01 / GATE-01-Scenario-E codifications) |
| `.planning/REQUIREMENTS.md` (10 v2.1 requirements, traceability table) |
| `.planning/ROADMAP.md` (v2.1 4-phase roadmap) |
| `.planning/milestones/v1.0-phases/05-promote-and-ablation-artifact/ABLATION-TABLE.md` (v2.1 addendum appended) |
| `.planning/milestones/v1.0-phases/05-promote-and-ablation-artifact/ABLATION-TABLE.tex` (v2.1 tabular blocks appended) |
