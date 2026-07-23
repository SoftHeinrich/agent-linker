---
phase: 12-trim-ablation
plan: 12-06
title: GATE-06 Defensibility Audit + Final Trim Disposition — PROMPT-04 closure
status: completed
verdict: ACCEPT (audit PASS on every retained surface; Phase 12 close signal asserted)
completed: 2026-06-01
requirements: [PROMPT-01, PROMPT-04]
subsystem: prompts + linkers/experimental + planning
tags: [gate-06, taboo-audit, reviewer-defensibility, prompt-engineering, phase-12-close]
key-files:
  created:
    - .planning/phases/12-trim-ablation/12-06-AUDIT-REPORT.md
    - .planning/phases/12-trim-ablation/12-06-V2_TO_V3_MAPPING-FINAL.md
    - .planning/phases/12-trim-ablation/12-06-SUMMARY.md (this file)
    - .planning/phases/12-trim-ablation/12-VERIFICATION.md
  modified:
    - .planning/STATE.md (Phase 12 close — progress 25 → 75%, milestone status, decision log)
decisions:
  - Audit covered the FULL BENCHMARK_TABOO surface (100 distinct case-insensitive whole-word terms across all 5 benchmark projects + Universal Taboo), not the narrow 9-name probe used during Wave 2 trim plans.
  - Reviewer adjudication is applied per-hit; 4 hits surfaced across 12 files and 17 constants, all dispositioned **safe** (English vocabulary in textbook-SE context).
  - Mapping conflict on DOC_KNOWLEDGE_JUDGE_RULES (Plans 12-03 + 12-05 both targeted it) is resolved by GATE-01 outcomes alone — trim1 ACCEPTs both arms, trim3 fails gpt-5.4 cross-model. trim1 is the v3 status.
  - Rejected trim variants (trim2/3/4/5/6/7/8) are kept in the repo for negative-result traceability; they are NOT carried to Plan 13-01 but are part of the Phase 12 frontier-map narrative.
---

# Phase 12 Plan 12-06 — SUMMARY

**One-liner:** Full BENCHMARK_TABOO + reviewer-defensibility re-audit on every Phase-12 retained surface (prompts_v3 + s_linker13_clean_v3 + helper_v3 + trim1 + trim9 + all 7 frontier variants) PASSES. Closes PROMPT-04 (generality re-audit) and finalizes PROMPT-01 (v2→v3 mapping with trim outcomes). Phase 12 is now ready to close — Plan 13-01 receives the unambiguous {trim1, trim9} carry-forward set.

## Outcomes

- **Variants audited:** 9 trim variants + prompts_v3.py + s_linker13_clean_v3.py + helper_v3.py = **12 files, 17 module-level prompt-body constants**.
- **GATE-06 lexical sweep:** 4 hits total across 17 constants (`layer`, `order`, `common`, `validation` — all Universal Taboo English words in textbook-SE contexts). **Zero leaked. Zero borderline. All 4 safe** under reviewer adjudication.
- **GATE-06 reviewer-defensibility:** **PASS** on all 12 files (every prompt body uses Safe SE Textbook allow-list examples — compilers, OS, e-commerce middleware, game engines).
- **Trims accepted (carry to Plan 13-01):** **2** — trim1 (judge-rules distillation, Plan 12-03) + trim9 (runtime seed disambiguation, Plan 12-12).
- **Trims rejected:** **7** — trim2 / trim3 / trim4 / trim5 / trim6 / trim7 / trim8. All REJECT verdicts are on **GATE-01** (Claude per-dataset drop or gpt-5.4 cross-model gap), **NOT** on GATE-06. Every rejected variant is GATE-06 compliant.
- **Final v2→v3 mapping:** committed at `12-06-V2_TO_V3_MAPPING-FINAL.md`, superseding Plan 12-01's initial split.

## Per-trim Final Disposition

### Plan 12-03 trim1 — Judge-rules distillation
- **Mechanism:** Technique 3 (lossless rubric distillation) + Technique 8 (reasoning-before-conclusion ordering). `DOC_KNOWLEDGE_JUDGE_RULES` 773 → 888 bytes prose form.
- **GATE-01 Claude:** PASS (macro 0.9553, BBB +2.54pp).
- **GATE-01 cross-model gpt-5.4:** PASS (macro 0.9173 ≥ 0.8977 floor).
- **GATE-06 lexical sweep:** PASS (0 hits on DOC_KNOWLEDGE_JUDGE_RUBRIC_V3; DOC_KNOWLEDGE_JUDGE_EXAMPLES_V3 byte-equal to v2, audited under v2.0).
- **GATE-06 reviewer-defensibility:** PASS.
- **Carry to Plan 13-01: YES.**

### Plan 12-04 trim2 — ENT+VAL merge
- **Mechanism:** ENTITY_EXTRACTION_RULES + VALIDATION_RULES merged (14 → 10 rules; shared core + 2 role-specific headers).
- **GATE-01 Claude:** FAIL (macro 0.9235 < 0.93; BBB −6.59pp). Pre-empts cross-model.
- **GATE-06 lexical sweep:** PASS (0 hits).
- **GATE-06 reviewer-defensibility:** PASS.
- **Failing arm:** Claude / bigbluebutton.
- **Carry to Plan 13-01: NO** (frozen in repo for negative-result traceability).

### Plan 12-05 / 12-05-REVISIT trim3 — Runtime judge-rules rubric
- **Mechanism:** `DOC_KNOWLEDGE_JUDGE_RULES` replaced by inference-time rubric builder; compiler-style seed example + document → 4–6 item rubric.
- **GATE-01 Claude:** PASS under relaxed Scenario E (macro 0.9396, BBB +0.72pp; TM −1.93pp).
- **GATE-01 cross-model gpt-5.4:** FAIL (macro 0.8855 < 0.8977 floor by 1.22pp; teammates 0.8130, bigbluebutton 0.7636).
- **GATE-06 lexical sweep:** PASS (0 hits on static surface; runtime cross-dataset isolation PASS — see 12-05-SUMMARY-REVISIT.md).
- **GATE-06 reviewer-defensibility:** PASS. Prior REJECT (which applied strict-reading of GATE-06) overturned in 12-05-REVISIT.
- **Failing arm:** gpt-5.4 / teammates + bigbluebutton (model-capability gap, not methodological flaw).
- **Carry to Plan 13-01: NO** (case study of the runtime-rubric mechanism under cross-dataset isolation).

### Plan 12-07 trim4 — Runtime ambiguity rubric
- **Mechanism:** `AMBIGUITY_FEW_SHOT + AMBIGUITY_RULES` (Tier 1 classifier) → runtime rubric builder.
- **GATE-01 Claude:** FAIL (JAB −2.56pp > 2pp tolerance; 1 FP on 19-link surface).
- **GATE-01 cross-model gpt-5.4:** PASS (macro 0.9005).
- **GATE-06:** PASS lexical (1 safe hit: "common" in "common abbreviations") + PASS defensibility.
- **Failing arm:** Claude / jabref.
- **Carry to Plan 13-01: NO** (frontier-only under Scenario E).

### Plan 12-08 trim5 — Runtime extraction rubric
- **Mechanism:** `DOC_KNOWLEDGE_EXTRACTION_RULES` (Tier 1 alias extractor) → runtime rubric builder.
- **GATE-01 Claude:** FAIL (TS −3.57pp > 2pp tolerance).
- **GATE-01 cross-model gpt-5.4:** PASS (macro 0.9056).
- **GATE-06:** PASS lexical (0 hits) + PASS defensibility.
- **Failing arm:** Claude / teastore.
- **Carry to Plan 13-01: NO** (frontier-only).

### Plan 12-09 trim6 — Runtime judge-examples (+ trim1 rules)
- **Mechanism:** `DOC_KNOWLEDGE_JUDGE_EXAMPLES` regenerated at runtime; inherits trim1's distilled RULES.
- **GATE-01 Claude:** PASS (macro 0.9406; TM −1.77pp).
- **GATE-01 cross-model gpt-5.4:** FAIL (macro 0.8938 < 0.8977 floor by 0.39pp).
- **GATE-06:** PASS lexical (0 hits) + PASS defensibility.
- **Failing arm:** gpt-5.4 / cross-model gap.
- **Carry to Plan 13-01: NO** (frontier-only).

### Plan 12-10 trim7 — Runtime entity rubric
- **Mechanism:** `ENTITY_EXTRACTION_RULES` (Tier 2 entity proposer) → runtime rubric builder.
- **GATE-01 Claude:** FAIL (JAB −2.56pp).
- **GATE-01 cross-model gpt-5.4:** PASS (macro 0.9007).
- **GATE-06:** PASS lexical (0 hits) + PASS defensibility.
- **Failing arm:** Claude / jabref.
- **Carry to Plan 13-01: NO** (frontier-only).

### Plan 12-11 trim8 — Runtime validation rubric
- **Mechanism:** `VALIDATION_RULES` (Tier 2 judge, both passes) → runtime rubric builder.
- **GATE-01 Claude:** FAIL (TS −3.57pp + JAB −2.56pp; two-dataset violation).
- **GATE-01 cross-model gpt-5.4:** PASS (macro 0.9070).
- **GATE-06:** PASS lexical (1 safe hit: "validation" in "validation rubric" — structural task label) + PASS defensibility.
- **Failing arm:** Claude / teastore + jabref.
- **Carry to Plan 13-01: NO** (frontier-only).

### Plan 12-12 trim9 — Runtime seed disambiguation
- **Mechanism:** `SEED_DISAMBIGUATION_RULES` replaced by inference-time rubric builder; per-document rubric built once and reused across per-component dossiers. No static fallback (RuntimeError on empty rubric).
- **GATE-01 Claude:** PASS (macro 0.9474, BBB +4.04pp; worst TS −1.82pp within 2pp tolerance).
- **GATE-01 cross-model gpt-5.4:** PASS (macro 0.9007 ≥ 0.8977 floor by 0.30pp).
- **GATE-06 lexical sweep:** PASS (0 hits on `SEED_RUBRIC_BUILDER_SEED_EXAMPLE` + `SEED_RUBRIC_BUILDER_PROMPT`).
- **GATE-06 reviewer-defensibility:** PASS (compiler-textbook static surface; runtime rubrics inherit Plan 12-05-REVISIT's cross-dataset isolation guarantee).
- **Carry to Plan 13-01: YES.**

## Rejected Trims Register (milestone-level)

| trim_id | source_plan | failing_gate | failing_arm | datasets | mitigation_signal |
|---------|-------------|--------------|-------------|----------|-------------------|
| trim2_entval_clean | 12-04 | GATE-01 Claude (macro 0.9235 < 0.93; BBB −6.59pp) | Claude | bigbluebutton | carry as negative result — extraction-vs-validation boundary erasure regresses BBB |
| trim3_runtime_rubric_clean | 12-05 / 12-05-REVISIT | GATE-01 cross-model gpt-5.4 (macro 0.8855 < 0.8977 floor) | gpt-5.4 | teammates, bigbluebutton | carry as negative result + methodology case study — model-capability gap, GATE-06 compliant |
| trim4_ambiguity_runtime_clean | 12-07 | GATE-01 Claude per-dataset (JAB −2.56pp > 2pp tolerance) | Claude | jabref | carry as negative result — proposer-side widening on small-surface dataset |
| trim5_extraction_runtime_clean | 12-08 | GATE-01 Claude per-dataset (TS −3.57pp) | Claude | teastore | carry as negative result — runtime extraction rubric over-widens recall |
| trim6_judge_examples_runtime_clean | 12-09 | GATE-01 cross-model gpt-5.4 (0.39pp short) | gpt-5.4 | macro | carry as negative result — at the edge under original gate; PASS under Scenario E |
| trim7_entity_runtime_clean | 12-10 | GATE-01 Claude per-dataset (JAB −2.56pp) | Claude | jabref | carry as negative result — proposer-side single-FP on jabref |
| trim8_validation_runtime_clean | 12-11 | GATE-01 Claude per-dataset (TS −3.57pp + JAB −2.56pp) | Claude | teastore, jabref | carry as negative result — two-dataset violation under original gate |

All 7 rejected variants are GATE-06 compliant. Their rejection is on GATE-01, not on leakage. The repo retains all variant source files for milestone reproducibility.

## Artifact Index

| Path | Purpose |
|------|---------|
| `.planning/phases/12-trim-ablation/12-06-AUDIT-REPORT.md` | Per-trim GATE-06 audit; full TABOO sweep table; reviewer-defensibility narrative per trim. |
| `.planning/phases/12-trim-ablation/12-06-V2_TO_V3_MAPPING-FINAL.md` | Final v2→v3 mapping with trim outcomes; supersedes Plan 12-01. |
| `.planning/phases/12-trim-ablation/12-06-SUMMARY.md` | This file. |
| `.planning/phases/12-trim-ablation/12-VERIFICATION.md` | Phase-12 verification artifact (created in this same plan). |
| `.planning/phases/12-trim-ablation/12-FRONTIER-MAP-SUMMARY.md` | Pareto frontier of 9 trim variants under original + Scenario E gates. |
| `.planning/phases/12-trim-ablation/12-05-SUMMARY-REVISIT.md` | Methodological correction on Plan 12-05; GATE-06 leakage REJECT overturned. |
| `src/llm_sad_sam/linkers/experimental/prompts_v3.py` | Step 0 dead-code drop — 9 byte-equal constants. |
| `src/llm_sad_sam/linkers/experimental/s_linker13_trim1_judge_clean.py` | Shipped — carries to Plan 13-01. |
| `src/llm_sad_sam/linkers/experimental/s_linker13_trim9_seed_runtime_clean.py` | Shipped — carries to Plan 13-01. |
| `src/llm_sad_sam/linkers/experimental/s_linker13_clean_v3.py` | Shipped infrastructure. |
| `src/llm_sad_sam/linkers/experimental/helper_v3.py` | Shipped infrastructure. |
| `src/llm_sad_sam/linkers/experimental/s_linker13_trim{2,3,4,5,6,7,8}_*_clean.py` | Rejected variants — retained for negative-result traceability. |
| `results/ablation_results/12_03_trim1_judge/verdict.json` | trim1 ACCEPT evidence (immutable). |
| `results/ablation_results/12_04_trim2_entval/verdict.json` | trim2 REJECT evidence. |
| `results/ablation_results/12_05_trim3_runtime_rubric/verdict.json` | trim3 REJECT evidence. |
| `results/ablation_results/12_extension_runtime_variants/scoreboard.json` | trim4–9 verdicts. |
| `BENCHMARK_TABOO.md` | Audit term surface (snapshot 2026-05-31, 100 terms). |

## Plan 13-01 Hand-off

**Accepted trims (carry into `s_linker13_min`):**
- **trim1** — distilled `DOC_KNOWLEDGE_JUDGE_RULES` (subclass `SLinker13Trim1JudgeClean`, file `s_linker13_trim1_judge_clean.py`).
- **trim9** — runtime `SEED_DISAMBIGUATION_RULES` rubric builder (subclass `SLinker13Trim9SeedRuntimeClean`, file `s_linker13_trim9_seed_runtime_clean.py`).

**Rejected trims (NOT carried):** trim2, trim3, trim4, trim5, trim6, trim7, trim8.

**Mapping conflict resolution:** Plans 12-03 + 12-05 both targeted `DOC_KNOWLEDGE_JUDGE_RULES`. Resolved unambiguously by GATE-01 outcomes — trim1 wins (PASS both arms); trim3 retired (gpt-5.4 cross-model FAIL).

**Composition risk for Plan 13-01:** trim1 and trim9 target disjoint pipeline stages (Tier 1 alias judge vs Tier 2 seed validation). Interaction effects are expected to be small but are NOT yet measured. Plan 13-01 must run the composed variant through both gates before promoting `s_linker13_min`.

**Rephrase candidates (future Phase 12 revision):** none — no borderline TABOO hits required user adjudication (Task 3 checkpoint was vestigial at the lexical layer).

**Voyager-TLR pilot (Phase 12 EXTENSION):** in-flight gpt-5.4 axiom-learning experiment is **not** part of Phase 12's PROMPT-01/02/04 closure. Documented separately in `12-VOYAGER-PILOT-DEFERRED.md`. Its outcome will be logged when the run completes but does NOT block Phase 12 close.

## Self-Check: PASSED

- `12-06-AUDIT-REPORT.md` exists and contains required sections (Audit Surface, Full TABOO Sweep Results, Reviewer-Defensibility Per Trim, Final Trim Disposition).
- `12-06-V2_TO_V3_MAPPING-FINAL.md` exists with the 16-constant table updated with trim-outcome column.
- `12-VERIFICATION.md` exists with per-requirement verification and final Phase 12 verdict.
- No edits to frozen v2.0 files: `prompts_v2.py`, `s_linker13.py`, `s_linker13_clean.py`, `data_types_v2.py`, `document_loader_v2.py`, `pcm_parser_v2.py` are unchanged.
- Zero LLM calls during this plan (pure documentation closure).

---
*Phase 12 close signal asserted 2026-06-01. PROMPT-01 + PROMPT-04 closed. Hand-off to Plan 13-01.*
