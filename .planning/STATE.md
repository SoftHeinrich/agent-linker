---
gsd_state_version: 1.0
milestone: v2.1
milestone_name: Cleanup + Prompt Simplification
status: planning
stopped_at: "Completed Phase 12 close (Plan 12-06): GATE-06 BENCHMARK_TABOO + reviewer-defensibility re-audit PASSES on every retained surface (4 shipped + 5 frontier variants + prompts_v3 + s_linker13_clean_v3 + helper_v3). 4 lexical hits across 17 constants — all dispositioned safe. Phase 12 VERIFICATION asserted: 3/3 requirements (PROMPT-01, PROMPT-02, PROMPT-04) complete. Carry-forward set for Plan 13-01: {trim1, trim9}. Rejected trims register: {trim2, trim3, trim4, trim5, trim6, trim7, trim8} — all GATE-06 compliant, rejected on GATE-01 alone. Voyager-TLR pilot remains a parallel frontier extension (12-VOYAGER-PILOT-DEFERRED.md), does NOT block Phase 12 close. Next action: Phase 13 — Plan 13-01 (s_linker13_min) composing trim1 + trim9 over prompts_v3 + s_linker13_clean_v3 + helper_v3."
last_updated: "2026-06-01T00:00:00.000Z"
last_activity: 2026-06-01 — Phase 12 closed (Plan 12-06 GATE-06 audit PASS; SUMMARY + VERIFICATION + FINAL mapping committed)
progress:
  total_phases: 4
  completed_phases: 2
  total_plans: 11
  completed_plans: 11
  percent: 75
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-31 for v2.1 kickoff)

**Core value:** Every rule removed from `s_linker13`/its prompts must hold macro F1 ≥ 0.93 on Claude Sonnet AND gpt-5.4 macro within ≤ 1pp of 0.9077 — or be rejected. Every retained prompt + helper must read as project-agnostic to a reviewer (GATE-06). Nothing currently runnable breaks.
**Current focus:** Phase 12 — Trim Ablation — Phases 10+11 COMPLETE

## Current Position

Phase: 13 of 4 — Phase 12 Trim Ablation COMPLETE; Phase 13 Promotion & Wrap pending
Plan: 13-01 (s_linker13_min) — not yet started
Status: Phase 12 closed; awaiting Phase 13 kickoff
Last activity: 2026-06-01 — Plan 12-06 complete (GATE-06 audit PASS); Phase 12 VERIFICATION asserted PASSED

Progress: [████████░░] 75%

## Performance Metrics

**Velocity:**

- Total plans completed: 2
- Average duration: ~15min
- Total execution time: ~27min

**By Phase:**

| Phase | Plan | Duration | Tasks | Files | Commit |
|-------|------|----------|-------|-------|--------|
| 10    | 01   | ~15min   | 2     | 2     | 98cdca2 |
| 10    | 02   | ~12min   | 1     | 1     | eae3028 |

*Updated after each plan completion*
| Phase 10 P04 | ~5min | 2 tasks | 2 files |
| Phase 12 P02 | ~45min | 3 tasks | 6 files |
| Phase 12 P04 | ~25min | 4 tasks | 4 files (REJECT verdict) |
| Phase 12 P03 | ~2h    | 4 tasks | 13 files (ACCEPT verdict) |
| Phase 12 P05 REVISIT | ~50min | 4 tasks | 2 files (REJECT on cross-model; GATE-06 leakage REJECT OVERTURNED) |
| Phase 12 EXTENSION (12-07..12-12) | ~75min | 6 variants × probe + sweep | 6 variants + 7 SUMMARYs + 40 result JSONs (1 ACCEPT trim9, 1 cross-model REJECT trim6, 4 Claude REJECT) |
| Phase 12 P06 (GATE-06 close) | ~25min | 4 artifacts (audit + final mapping + SUMMARY + VERIFICATION) | 4 docs (zero LLM); Phase 12 verdict PASSED |

## Standing Gates (v2.1)

- GATE-01 (v2.1 SCENARIO E 2026-05-31, runtime-mechanism variants): macro F1 ≥ 0.90; BBB absolute F1 ≥ 0.79 (swattr SAD-SAM expected); **other-dataset drop ≤ 4pp** (was 2pp under earlier relaxation; further loosened proportionally to prompt-reduction depth — runtime variants delete more static prompt content, accept proportionally larger accuracy tolerance) — Claude Sonnet. **Cross-model gpt-5.4 macro F1 ≥ 0.89** (was 0.8977; ~1.8pp tolerance off 0.9077 baseline). Framing: Phase 12 is a **frontier map of prompt-reduction × accuracy**, not strict pass/fail; trim variants ship with explicit reduction + accuracy disclosure. Static-prompt-distillation variants (e.g. 12-03 trim1) remain evaluated against the prior 0.93/0.8977 tighter gates since they don't pay the heavy reduction the runtime mechanism does.
- (Prior history: 2026-05-31 first relaxation was macro ≥ 0.90, BBB ≥ 0.79, other -2pp, cross-model 0.8977 — see PROJECT.md "GATE-01 relaxation (v2.1 Phase 12)" + "Scenario E" rows.)
- GATE-02 (v2.1 NEW): frozen-compat regression test; all CANONICAL_VARIANTS produce F1 matching v2.0 baseline JSON
- GATE-06: generality audit — every new prompt + helper passes BENCHMARK_TABOO scan AND reviewer-defensibility check
- GATE-07: every promoted variant registered in CANONICAL_VARIANTS + VARIANT_SPECS; standalone file; structured docstring

## Accumulated Context

### Decisions

- v2.1 kickoff: `s_linker13.py`, `prompts_v2.py`, and existing helper modules are frozen; cleanup lands in `_clean` / `v3+` / `_min` siblings
- v2.1 kickoff: cross-model gate uses gpt-5.4 (v2.0 CROSS baseline 0.9077) with tolerance ≤ 1pp
- v2.1 kickoff: coarse granularity → 4 phases (10–13); sequential dependency chain
- Plan 10-02 (CLEAN-02): helper_v3.py is a single file (not per-concern split); `build_component_profile` lifts `self.model_knowledge` / `self.doc_knowledge` to explicit parameters; `MENTION_TYPES` is duplicated rather than re-imported from frozen `SLinker13d` to keep helper_v3 free of variant-class coupling.
- [Phase 10]: Plan 10-04 (GATE-01): cross-model tolerance pinned to T = 1.0pp; absolute F1 floor 0.8977 = 0.9077 − 0.01; recorded in PROJECT.md Key Decisions and STATE.md Standing Gates.
- [Phase 12]: Plan 12-02: Single-step ablation harness ships at llm_sad_sam.ablation.single_step with CLI subcommand; phase=entity_candidates/entity_decisions enforces CRITICAL CONTRACT (zero live LLM on seed_val/coref via monkey-patch). Equivalence sweep PASS (max_abs_delta=0.0).
- [Phase 12]: Plan 12-02: Harness coupling debt tracked — calls into s_linker13_clean by method name (_run_seed_validation, _run_entity_pipeline, _validate_with_evidence, _extract_entities_enriched, _run_coreference). Phase 13 promotion must preserve these names or update harness in lock-step.
- [Phase 12]: Plan 12-04: Step 2 trim variant `s_linker13_trim2_entval_clean` REJECTED on Claude GATE-01. Merging ENTITY_EXTRACTION_RULES + VALIDATION_RULES via Technique 3 (14 → 10 rules) regresses BBB by 6.6pp (F1 0.8036 → 0.7377) and macro to 0.9235 < 0.93. Round 3 (gpt-5.4) skipped per strategic plan. Variant NOT carried to Plan 12-06 or Plan 13-01. Failure consistent with V35a lesson: prompt-merge that erases extraction-vs-validation boundary regresses Claude on highest-variance dataset.
- [Phase 12]: Plan 12-03: Step 1 trim variant `s_linker13_trim1_judge_clean` ACCEPTED. DOC_KNOWLEDGE_JUDGE_RULES distilled via Technique 3 (lossless rubric distillation, prose form, 773 → 888 bytes) + Technique 8 (reasoning-before-conclusion: "When in doubt, APPROVE" emitted before decision wording). 7 worked examples preserved verbatim (V35a guard). Claude macro 0.9553 (BBB +2.54pp, no other-dataset regression > 2pp); gpt-5.4 macro 0.9173 (TM +10.08pp, MS -3.44pp, TS -1.82pp, BBB/JAB flat). Verdict ACCEPT on relaxed GATE-01 Claude + cross-model gpt-5.4 + GATE-06 (zero benchmark-name hits). Variant CARRIED to Plan 12-06 audit and (subject to that) to Plan 13-01's s_linker13_min union.
- [Phase 12]: Plan 12-05 REVISIT: Methodological correction — prior REJECT applied strict-reading of GATE-06 ("project terms in LLM output = leakage") which, applied consistently, would invalidate every LLM call in the pipeline. CLAUDE.md actually MANDATES dynamic runtime LLM discovery of domain-specific knowledge from input data; the runtime-generated rubric IS that mechanism. Operationalized cross-dataset isolation as the correct empirical test (term t in dataset A's rubric is a leak iff (a) t is a PCM component of dataset B != A AND (b) t is NOT in A's PCM AND (c) t is NOT in A's input doc). Findings: GATE-06 static surface PASS; cross-dataset isolation PASS on both backends (0 violations across 10 rubrics); Claude relaxed GATE-01 PASS (macro 0.9396, BBB 0.8108); gpt-5.4 cross-model FAIL by 1.22pp (macro 0.8855 < 0.8977 floor). Final verdict: REJECT but on cross-model capability gap (consistent with documented Claude-vs-GPT ~5.7pp gap), NOT on leakage. Prior leakage REJECT OVERTURNED. Variant NOT carried to 13-01. Prior 12-05-SUMMARY.md preserved unchanged; 12-05-SUMMARY-REVISIT.md supersedes.
- [Phase 12 EXTENSION]: Applied the 12-05 runtime-rubric mechanism PER PROMPT to 6 prompts (not merged). Built 6 standalone variants subclassing SLinker13Clean (consistent with the trim3 template the user directive nominated; strict standalone-class reading documented as interpretive deviation). NO STATIC FALLBACK per user directive — every variant raises RuntimeError on empty rubric. Strategic probe gating saved 20 gpt-5.4 sweep runs by skipping cross-model evaluation for Claude-arm failures. Outcomes: trim9 (SEED_DISAMBIGUATION_RULES) ACCEPTED on both arms (Claude 0.9474 / gpt-5.4 0.9007, BBB +4.04pp on Claude); trim6 (DOC_KNOWLEDGE_JUDGE_EXAMPLES + trim1 distilled rules) Claude-PASS but cross-model FAIL by 0.39pp (consistent with trim3 model-capability gap); trim4/5/7/8 (ambiguity / extraction / entity / validation) Claude per-dataset drop tolerance violations (jabref -2.56pp single-FP for small-dataset variants; teastore -3.57pp for proposer/judge-tier substitutions). trim9 CARRIED to Plan 13-01 for composition with trim1.
- [Phase 12 CLOSE 2026-06-01]: Plan 12-06 completed. Full BENCHMARK_TABOO (100 terms) + reviewer-defensibility re-audit on every retained surface — 12 files, 17 module-level prompt-body constants — PASSES. 4 lexical hits surfaced (`layer`, `order`, `common`, `validation`), all dispositioned safe (English vocabulary in textbook-SE contexts). Zero leaked, zero borderline. PROMPT-04 closed. PROMPT-01 finalized with `12-06-V2_TO_V3_MAPPING-FINAL.md` (supersedes 12-01). Mapping conflict on DOC_KNOWLEDGE_JUDGE_RULES (trim1 vs trim3) resolved by GATE-01 outcomes — trim1 wins. Rejected-trims register: {trim2, trim3, trim4, trim5, trim6, trim7, trim8} — all GATE-06 compliant; rejection on GATE-01 alone. Phase 12 VERIFICATION asserted PASSED at `.planning/phases/12-trim-ablation/12-VERIFICATION.md`. Voyager-TLR pilot kept as parallel frontier extension; does NOT block Phase 12 close. Hand-off to Plan 13-01 is unambiguous: compose trim1 + trim9 over prompts_v3 + s_linker13_clean_v3 + helper_v3.

### Pending Todos

Next action: kick off Phase 13 (Promotion & Wrap). Plan 13-01 promotes `s_linker13_min` as the composition of trim1 (distilled DOC_KNOWLEDGE_JUDGE_RULES) + trim9 (runtime SEED_DISAMBIGUATION_RULES) over prompts_v3 + s_linker13_clean_v3 + helper_v3. The composed variant must pass both gates (Claude relaxed GATE-01 + gpt-5.4 cross-model GATE-01) before promotion. Trim interaction effects are unmeasured but expected small (disjoint pipeline stages — Tier 1 alias judge vs Tier 2 seed validation). Rejected trims {trim2, trim3, trim4, trim5, trim6, trim7, trim8} are NOT carried forward but remain in repo for negative-result traceability.

### Blockers/Concerns

None.

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| v2.2+ | EXT-04: Emit-biased boundary prompting on alias-discovery (BBB variance ~3pp → ~1pp) | Deferred | v2.0 kickoff |
| v2.2+ | EXT-upstream: Upstream-tier rule removal (extraction/coref tier) | Deferred | v2.0 close |
| v2.2+ | ADAPTER-01: Multi-model backend-adaptive harness layer | Deferred | v2.0 close |

## Session Continuity

Last session: 2026-06-01T00:00:00.000Z
Stopped at: Completed Plan 12-06 — Phase 12 closed (PASSED). Next action: execute Plan 13-01 (s_linker13_min promotion of trim1 + trim9 composition).
Resume file: None
