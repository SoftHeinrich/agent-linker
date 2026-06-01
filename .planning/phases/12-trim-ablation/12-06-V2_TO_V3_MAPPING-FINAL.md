---
phase: 12-trim-ablation
plan: 12-06
artifact: v2_to_v3_prompt_mapping_FINAL
supersedes: .planning/phases/12-trim-ablation/12-01-V2_TO_V3_MAPPING.md
audited: "2026-06-01"
requirements: [PROMPT-01, PROMPT-02, PROMPT-04]
---

# prompts_v2 → prompts_v3 Mapping (FINAL — Phase 12 close)

Per-constant disposition for every top-level prompt constant in
`src/llm_sad_sam/linkers/experimental/prompts_v2.py` after all of Phase 12 is
complete. Supersedes Plan 12-01's initial mapping by adding the **trim
outcomes** (kept / dropped / merged / replaced-by-runtime-rubric / distilled /
trim-attempted-rejected) and the per-row **carry-to-Plan-13-01** signal.

The audit input is the Phase 12 trim verdict register:
- Plan 12-03 (trim1, judge-rules distillation) → **ACCEPT**
- Plan 12-04 (trim2, ENT+VAL merge) → **REJECT** (Claude macro 0.9235 < 0.93; BBB −6.59pp)
- Plan 12-05 / 12-05-REVISIT (trim3, runtime judge rubric) → **REJECT** (gpt-5.4 cross-model 0.8855 < 0.8977 floor)
- Plan 12-07 (trim4, runtime ambiguity rubric) → **REJECT** (Claude JAB −2.56pp)
- Plan 12-08 (trim5, runtime extraction rubric) → **REJECT** (Claude TS −3.57pp)
- Plan 12-09 (trim6, runtime judge-examples) → **REJECT** (gpt-5.4 cross-model 0.8938 < 0.8977 floor by 0.39pp)
- Plan 12-10 (trim7, runtime entity rubric) → **REJECT** (Claude JAB −2.56pp)
- Plan 12-11 (trim8, runtime validation rubric) → **REJECT** (Claude TS −3.57pp + JAB −2.56pp)
- Plan 12-12 (trim9, runtime seed disambiguation) → **ACCEPT**

The full GATE-06 BENCHMARK_TABOO + reviewer-defensibility re-audit on every
audited surface above PASSES (see `12-06-AUDIT-REPORT.md`). Therefore the v3
status of each constant is determined by GATE-01 outcomes alone.

## Table — All 16 prompts_v2 Top-Level Constants (FINAL)

| constant_name                                                  | v2_lines | v3_status (FINAL)                                                                | trim_plan_that_modified_it | carry_to_13-01 | reviewer_defensibility_note                                                                                                                                                                       |
| -------------------------------------------------------------- | -------- | -------------------------------------------------------------------------------- | -------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| AMBIGUITY_FEW_SHOT                                             | 14-47    | **kept (byte-equal)**; trim attempted (trim4) → REJECTED → reverted to v2 form   | 12-07 (trim4)              | YES (as v2)    | trim4 ran a runtime rubric replacement; failed Claude JAB by 2.56pp (1 FP on a 19-link surface). v3 keeps v2's 4 calibration examples. GATE-06 audit PASS.                                       |
| AMBIGUITY_RULES                                                | 50-64    | **kept (byte-equal)**; trim attempted (trim4) → REJECTED → reverted to v2 form   | 12-07 (trim4)              | YES (as v2)    | bundled with AMBIGUITY_FEW_SHOT in trim4. Same outcome. v3 keeps v2 verbatim.                                                                                                                    |
| DOC_KNOWLEDGE_EXTRACTION_RULES                                 | 71-84    | **kept (byte-equal)**; trim attempted (trim5) → REJECTED → reverted to v2 form   | 12-08 (trim5)              | YES (as v2)    | trim5 runtime extraction rubric regressed Claude TS by 3.57pp (proposer-side widening). v3 keeps v2 verbatim. GATE-06 audit PASS.                                                                |
| DOC_KNOWLEDGE_JUDGE_EXAMPLES                                   | 87-121   | **kept (byte-equal)**; trim attempted (trim6) → REJECTED → reverted to v2 form   | 12-09 (trim6)              | YES (as v2)    | V35a guard transferred — trim6 (runtime regeneration of these 7 examples) crossed gpt-5.4 floor by 0.39pp. Distilled rules of trim1 still preserve these examples byte-equal. GATE-06 PASS.       |
| DOC_KNOWLEDGE_JUDGE_RULES                                      | 124-139  | **distilled via trim1** (Technique 3 + Technique 8; 773 → 888 bytes prose form)  | 12-03 (trim1, ACCEPT) [+ 12-05 trim3 REJECT, conflict resolved] | YES (as trim1) | Lossless restructure: "When in doubt APPROVE" emitted before decision (Technique 8). Three numbered rules collapsed into prose rubric. All semantic content preserved. GATE-06 PASS (0 hits).    |
| WORD_USAGE_PROMPT                                              | 146-172  | **dropped**                                                                      | (none — dead in s_linker13_clean) | n/a            | Legacy ≤ s_linker12c word-usage classifier; not imported by s_linker13_clean. Survives in prompts_v2.py for older variants.                                                                       |
| ENTITY_EXTRACTION_RULES                                        | 179-191  | **kept (byte-equal)**; trim attempted (trim2 merge + trim7 runtime) → BOTH REJECTED → reverted to v2 form | 12-04 (trim2), 12-10 (trim7) | YES (as v2)    | trim2 merged with VALIDATION_RULES (regressed BBB −6.59pp); trim7 runtime ran independently (regressed JAB −2.56pp). v3 keeps v2 verbatim. GATE-06 audit PASS on both attempts.                  |
| VALIDATION_RULES                                               | 194-205  | **kept (byte-equal)**; trim attempted (trim2 merge + trim8 runtime) → BOTH REJECTED → reverted to v2 form | 12-04 (trim2), 12-11 (trim8) | YES (as v2)    | trim2 merged with ENTITY_EXTRACTION_RULES (FAIL); trim8 runtime ran independently (regressed TS −3.57pp + JAB −2.56pp, two-dataset violation). v3 keeps v2 verbatim. GATE-06 audit PASS.        |
| COREF_RULES                                                    | 212-222  | **kept (byte-equal)**                                                            | (no trim attempted)        | YES (as v2)    | No trim plan targeted COREF. Stable in v2.0 and v2.1; carries forward verbatim.                                                                                                                  |
| STANDALONE_MENTION_RULES_PRE_FILTERED                          | 229-238  | **dropped**                                                                      | (none — EXT-01 legacy)     | n/a            | EXT-01 sub-variant; used only by s_linker13g_pre. Deferred to v2.2+; not imported by s_linker13_clean.                                                                                            |
| STANDALONE_MENTION_RULES_LLM_ONLY                              | 241-255  | **dropped**                                                                      | (none — EXT-01 legacy)     | n/a            | EXT-01 sub-variant; used only by s_linker13g_sem. Deferred.                                                                                                                                       |
| STANDALONE_MENTION_RULES_PRE_FILTERED_ALIAS_AWARE              | 271-286  | **dropped**                                                                      | (none — EXT-01 legacy)     | n/a            | EXT-01 alias-aware; used only by s_linker13g_pre_alias. Deferred.                                                                                                                                 |
| STANDALONE_MENTION_RULES_LLM_ONLY_ALIAS_AWARE                  | 289-310  | **dropped**                                                                      | (none — EXT-01 legacy)     | n/a            | EXT-01 alias-aware; used only by s_linker13g_sem_alias. Deferred.                                                                                                                                 |
| STANDALONE_MENTION_RULES_PRE_FILTERED_FULL_KNOWLEDGE           | 313-334  | **dropped**                                                                      | (none — EXT-01 legacy)     | n/a            | EXT-01 full-knowledge; used only by s_linker13g_pre_full. Deferred.                                                                                                                               |
| STANDALONE_MENTION_RULES_LLM_ONLY_FULL_KNOWLEDGE               | 337-365  | **dropped**                                                                      | (none — EXT-01 legacy)     | n/a            | EXT-01 full-knowledge; used only by s_linker13g_sem_full. Deferred.                                                                                                                               |
| SEED_DISAMBIGUATION_RULES                                      | 372-390  | **replaced by inference-time rubric builder** (trim9, runtime)                   | 12-12 (trim9, ACCEPT)      | YES (as trim9) | The only runtime-rubric variant accepted on BOTH arms. Per-document rubric built once and reused across per-component dossiers; no static fallback (RuntimeError on empty rubric). GATE-06 PASS. |

**Total:** 16 prompts_v2 constants →
- **7 kept byte-equal** (no trim or trim REJECTED → reverted): AMBIGUITY_FEW_SHOT, AMBIGUITY_RULES, DOC_KNOWLEDGE_EXTRACTION_RULES, DOC_KNOWLEDGE_JUDGE_EXAMPLES, ENTITY_EXTRACTION_RULES, VALIDATION_RULES, COREF_RULES
- **1 distilled (trim1)**: DOC_KNOWLEDGE_JUDGE_RULES
- **1 replaced by runtime rubric (trim9)**: SEED_DISAMBIGUATION_RULES
- **7 dropped** (dead in s_linker13_clean): WORD_USAGE_PROMPT + 6 EXT-01 STANDALONE_MENTION_RULES variants

## Mapping Conflict Resolution

**Plans 12-03 (trim1) and 12-05 (trim3) both target `DOC_KNOWLEDGE_JUDGE_RULES`.** The resolution is unambiguous:

- trim1 (static distillation, Technique 3 + Technique 8) → ACCEPT on both Claude and gpt-5.4 arms (macro 0.9553 / 0.9173).
- trim3 (runtime rubric regeneration) → REJECT on gpt-5.4 cross-model arm (macro 0.8855 < 0.8977 floor by 1.22pp).

**→ trim1 is the v3 status for `DOC_KNOWLEDGE_JUDGE_RULES`.** trim3 stays in the repo for negative-result traceability and as a case study of the runtime-rubric mechanism's cross-model behaviour (REVISIT methodology in `12-05-SUMMARY-REVISIT.md` shows GATE-06 PASSES under cross-dataset isolation; the rejection is on model-capability gap, not leakage).

**Plans 12-03 (trim1) and 12-09 (trim6) jointly modify the judge surface (RULES + EXAMPLES).** trim6 inherits trim1's distilled RULES and additionally regenerates EXAMPLES at runtime. trim6 → REJECT (cross-model 0.39pp short). The v3 surface for this region is trim1's distilled RULES + byte-equal EXAMPLES.

**No outstanding mapping conflicts.** All 16 constants have a single, unambiguous v3 disposition.

## Carry-Forward Set for Plan 13-01

The minimal-prompt variant `s_linker13_min.py` (Plan 13-01) composes:

1. **prompts_v3.py** (Step 0 dead-code drop — 9 byte-equal constants from v2; 7 dropped legacy constants).
2. **trim1** — distilled `DOC_KNOWLEDGE_JUDGE_RULES` override (`s_linker13_trim1_judge_clean.py`).
3. **trim9** — runtime `SEED_DISAMBIGUATION_RULES` rubric builder (`s_linker13_trim9_seed_runtime_clean.py`).
4. **s_linker13_clean_v3.py** + **helper_v3.py** — the cleaned shared infrastructure already shipped (Phase 10).

Plan 13-01 must verify that the trim1 + trim9 composition holds under both gates (Claude relaxed GATE-01 + gpt-5.4 cross-model GATE-01); interaction effects across the two trims are not yet measured (the two prompts target disjoint pipeline stages — judge vs seed — so cross-talk is expected to be small).

## Acceptance

- **PROMPT-01** (REQUIREMENTS.md): `prompts_v3.py` ships side-by-side with `prompts_v2.py`. This mapping table documents the v3 status — kept / dropped / merged / replaced-by-runtime / distilled / trim-rejected-reverted — for every constant.
- **PROMPT-02** (REQUIREMENTS.md): Per-prompt rule-trim ablation completed; 9 trim variants evaluated; 2 accepted (trim1, trim9); 7 rejected; rejected trims documented in this table with the failing arm + dataset (cross-references `12-FRONTIER-MAP-SUMMARY.md`).
- **PROMPT-04** (REQUIREMENTS.md): Generality re-audit (GATE-06 + BENCHMARK_TABOO + reviewer-defensibility) completed on every retained surface. **PASS** for all 9 audited files (4 shipped + 5 frontier — see `12-06-AUDIT-REPORT.md`). No retained surface introduces benchmark-derived phrasing.

## Cross-References

- `.planning/phases/12-trim-ablation/12-01-V2_TO_V3_MAPPING.md` — initial mapping (16 rows). This file supersedes it with the trim-outcome column.
- `.planning/phases/12-trim-ablation/12-06-AUDIT-REPORT.md` — per-constant TABOO-sweep results.
- `.planning/phases/12-trim-ablation/12-FRONTIER-MAP-SUMMARY.md` — Pareto frontier of 9 trim variants under original + Scenario E gates.
- `results/ablation_results/12_03_trim1_judge/verdict.json` — trim1 ACCEPT evidence.
- `results/ablation_results/12_extension_runtime_variants/scoreboard.json` — trim4–9 verdicts.
- `results/ablation_results/12_05_trim3_runtime_rubric/verdict.json` — trim3 REJECT evidence.
- `results/ablation_results/12_04_trim2_entval/verdict.json` — trim2 REJECT evidence.

---
*Final mapping authored 2026-06-01. Supersedes Plan 12-01's initial split.*
