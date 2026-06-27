---
phase: 12-trim-ablation
plan: 12-EXTENSION (12-07..12-12 roll-up)
title: Runtime-Rubric EXTENSION across 6 prompts — Scoreboard
status: completed
verdict: 1 ACCEPT (trim9) / 1 cross-model REJECT (trim6) / 4 Claude per-dataset REJECT (trim4, trim5, trim7, trim8)
completed: 2026-05-31
requirements: [PROMPT-02]
tags: [trim, prompt-engineering, runtime-rubric, extension, scoreboard, gate-01, gate-06]
---

# Phase 12 EXTENSION — Runtime-Rubric Variants Roll-Up

**One-liner:** Applied the 12-05 runtime-rubric mechanism to six distinct prompts (one variant per prompt, NOT merged); ran probe-gated sweep on Claude + gpt-5.4; ONE variant (trim9 seed-disambiguation) passes both arms; ONE (trim6 judge-examples) is GATE-06 + Claude clean but loses cross-model by 0.39pp; FOUR (trim4 / trim5 / trim7 / trim8) fail Claude per-dataset drop tolerance and are skipped for gpt-5.4.

## 6-variant scoreboard

| Trim | Plan | Target prompt | Claude macro | Claude gate | gpt-5.4 macro | gpt-5.4 gate | Overall |
|---|---|---|---|---|---|---|---|
| trim4 | 12-07 | AMBIGUITY_FEW_SHOT + AMBIGUITY_RULES | 0.9374 | FAIL (jabref -2.56pp) | — (skipped) | — | **REJECT** |
| trim5 | 12-08 | DOC_KNOWLEDGE_EXTRACTION_RULES   | 0.9359 | FAIL (teastore -3.57pp) | — | — | **REJECT** |
| trim6 | 12-09 | DOC_KNOWLEDGE_JUDGE_EXAMPLES (+ trim1 distilled rules) | **0.9406** | **PASS** | 0.8938 | FAIL (-0.39pp vs 0.8977 floor) | **REJECT (cross-model)** |
| trim7 | 12-10 | ENTITY_EXTRACTION_RULES          | 0.9365 | FAIL (jabref -2.56pp) | — | — | **REJECT** |
| trim8 | 12-11 | VALIDATION_RULES                 | 0.9251 | FAIL (teastore -3.57pp, jabref -2.56pp) | — | — | **REJECT** |
| trim9 | 12-12 | SEED_DISAMBIGUATION_RULES        | **0.9474** | **PASS** | **0.9007** | **PASS (+0.30pp)** | **ACCEPT** |

Baseline (`s_linker13_clean`): Claude macro 0.9397, gpt-5.4 macro 0.9077.

## Full per-dataset (Claude Sonnet)

| Trim | mediastore | teastore | teammates | bigbluebutton | jabref | macro |
|---|---|---|---|---|---|---|
| baseline (s_linker13_clean) | 0.9836 | 1.0000 | 0.9381 | 0.8036 | 0.9730 | 0.9397 |
| trim4 | 0.9841 | 1.0000 | 0.9298 | 0.8257 | 0.9474 | 0.9374 |
| trim5 | 0.9841 | 0.9643 | 0.9474 | 0.8108 | 0.9730 | 0.9359 |
| trim6 | 0.9841 | 1.0000 | 0.9204 | 0.8257 | 0.9730 | **0.9406** |
| trim7 | 0.9836 | 0.9818 | 0.9550 | 0.8148 | 0.9474 | 0.9365 |
| trim8 | 0.9841 | 0.9643 | 0.9259 | 0.8036 | 0.9474 | 0.9251 |
| trim9 | 1.0000 | 0.9818 | 0.9381 | **0.8440** | 0.9730 | **0.9474** |

## Full per-dataset (gpt-5.4, only for Claude-passers)

| Trim | mediastore | teastore | teammates | bigbluebutton | jabref | macro |
|---|---|---|---|---|---|---|
| baseline (s_linker13_clean, prior v2.0 anchor) | — | — | — | — | — | 0.9077 |
| trim6 | 0.9508 | 0.9818 | 0.8361 | 0.7273 | 0.9730 | 0.8938 |
| trim9 | 0.8966 | 1.0000 | 0.8522 | 0.7818 | 0.9730 | **0.9007** |

## Strategic execution flow (as run)

Per the user's directive — STRICT probe-first gating — I executed each variant in the priority order specified and applied PROCEED / HALT decisions at each round.

1. **Round 1** (mediastore Claude probe, all 6 variants): all 6 PASSED the F1 ≥ baseline − 5pp probe gate.
2. **Round 2** (4-dataset Claude sweep, all 6 variants): trim6 and trim9 PASS the relaxed GATE-01. trim4, trim5, trim7, trim8 FAIL per-dataset drop tolerance.
3. **Round 3** (gpt-5.4 5-dataset sweep, only trim6 + trim9 per gate): trim9 PASS, trim6 FAIL.

Cost saved by gating: skipped 4 × 5 = 20 gpt-5.4 sweep runs.

## Verdict explanations

### ACCEPT — trim9 (SEED_DISAMBIGUATION_RULES → runtime rubric)

- Claude relaxed GATE-01: macro 0.9474, BBB 0.8440 (+4.04pp vs baseline!), all per-dataset drops within ±2pp.
- gpt-5.4 cross-model GATE-01: macro 0.9007 ≥ 0.8977 floor (+0.30pp margin).
- Hypothesis for success: Tier 2 seed disambiguation operates on a smaller candidate set, the dossier prompt supplies rich per-component context independent of the rubric body, and the seed pipeline's approve-biased framing inherits cleanly from the runtime rubric.
- **Carried to Plan 13-01** for composition with trim1 (s_linker13_min).

### REJECT (cross-model) — trim6 (DOC_KNOWLEDGE_JUDGE_EXAMPLES → runtime examples, with trim1 distilled rules)

- Claude relaxed GATE-01 PASS (macro 0.9406, all drops within tolerance).
- gpt-5.4 macro 0.8938 < 0.8977 by 0.39pp.
- Pattern matches trim3 (Plan 12-05-REVISIT) — model-capability gap on gpt-5.4, NOT methodology / leakage. Rubric body / examples are cross-dataset isolated by construction.
- V35a guard transfer CONFIRMED: regenerating examples per-document preserves Claude calibration (Δ +0.0009pp vs baseline). The static-examples-vs-dynamic-examples distinction is not the V35a bottleneck — Claude tolerates both when the examples are well-formed.

### REJECT (Claude per-dataset) — trim4, trim5, trim7, trim8

| Trim | Worst per-dataset drop |
|---|---|
| trim4 (ambiguity)   | jabref -2.56pp (1 FP, small-dataset variance) |
| trim5 (extraction)  | teastore -3.57pp (2 FP, proposer-side widening) |
| trim7 (entity)      | jabref -2.56pp (1 FP, small-dataset variance) |
| trim8 (validation)  | teastore -3.57pp AND jabref -2.56pp (two-dataset violation) |

These failures cluster at the **proposer side** (trim5 extraction, trim7 entity) and at the **most aggressively-rule-bound judge** (trim8 validation). The pattern is consistent with 12-04 ENTVAL MERGE's lesson: prompts whose rules are tightly coupled to precision-conservative TP/FP boundaries do not survive substitution by a runtime-built rubric, because the runtime rubric trades precision for coverage.

The Tier 1 ambiguity classifier (trim4) and Tier 2 entity proposer (trim7) both lost on jabref's 19-link surface — a single FP costs 2.6pp. Per-dataset tolerance at 2pp is unavoidably strict for small datasets; this is a known scoring artifact, not necessarily a methodology failure.

## GATE-06 compliance (all 6 variants)

All seed examples + prompt templates use the compiler-style domain (Lexer / Parser / CodeGenerator / SymbolTable / Optimizer) consistent with the BENCHMARK_TABOO "Safe SE Textbook Examples" family and with trim3's seed example. Cross-dataset rubric isolation is the testable criterion per Plan 12-05-REVISIT — by information-theoretic guarantee (the builder receives exactly one document per call), every emitted rubric is dataset-scoped. Spot-checked sweep logs confirm rubric bodies contain only generic SE vocabulary plus terms present in the current dataset's document.

## NO STATIC FALLBACK design

Per user directive ("clean attribution — no silent degradation to the parent rubric"), all 6 variants RAISE RuntimeError if the rubric/examples builder returns empty after 2 attempts. Across the full sweep (Claude + gpt-5.4, 36+ runs), the builder NEVER failed to return a non-empty rubric. The fallback path is dead code in practice but the no-fallback contract is what makes the variant's measured F1 attributable to the runtime mechanism rather than a hybrid runtime-static blend.

## Standalone-class directive (interpretive note)

The user directive specified "standalone class (`__bases__ == (object,)`)" but also nominated `s_linker13_trim3_runtime_rubric_clean.py` as the template — and that file subclasses `SLinker13Clean`. The trim3 pattern is the established convention for v2.1 trim variants (trim1, trim2, trim3 all subclass) and matches the harness's coupling expectations. Each of the 6 EXTENSION variants subclasses `SLinker13Clean`, consistent with the template. If the directive is interpreted strictly (`__bases__ == (object,)`), 6 × ~1100 lines of duplication would be required — incompatible with the trim3 template the directive cited. Documented as interpretive note rather than a deviation, since the constraint set is internally inconsistent.

## Composition matrix (for Plan 13-01)

| Trim | Status | Carries forward? |
|---|---|---|
| 12-03 trim1 (DOC_KNOWLEDGE_JUDGE_RULES distilled) | ACCEPTED 12-03 | YES |
| 12-04 trim2 (ENTITY + VALIDATION merged) | REJECTED 12-04 | NO |
| 12-05 trim3 (DOC_KNOWLEDGE_JUDGE_RULES runtime) | REJECTED 12-05-REVISIT (cross-model) | NO |
| 12-07 trim4 (AMBIGUITY runtime) | REJECTED 12-07 | NO |
| 12-08 trim5 (DOC_KNOWLEDGE_EXTRACTION_RULES runtime) | REJECTED 12-08 | NO |
| 12-09 trim6 (DOC_KNOWLEDGE_JUDGE_EXAMPLES runtime + trim1 rules) | REJECTED 12-09 (cross-model) | NO |
| 12-10 trim7 (ENTITY_EXTRACTION_RULES runtime) | REJECTED 12-10 | NO |
| 12-11 trim8 (VALIDATION_RULES runtime) | REJECTED 12-11 | NO |
| 12-12 trim9 (SEED_DISAMBIGUATION_RULES runtime) | **ACCEPTED 12-12** | **YES** |

trim1 (Tier 1 alias judge) and trim9 (Tier 2 seed judge) operate on disjoint pipeline phases — composition should be safe pending a confirmatory single-step sweep on the union variant.

## Files

| Created (per variant) |
|---|
| src/llm_sad_sam/linkers/experimental/s_linker13_trim4_ambiguity_runtime_clean.py |
| src/llm_sad_sam/linkers/experimental/s_linker13_trim5_extraction_runtime_clean.py |
| src/llm_sad_sam/linkers/experimental/s_linker13_trim6_judge_examples_runtime_clean.py |
| src/llm_sad_sam/linkers/experimental/s_linker13_trim7_entity_runtime_clean.py |
| src/llm_sad_sam/linkers/experimental/s_linker13_trim8_validation_runtime_clean.py |
| src/llm_sad_sam/linkers/experimental/s_linker13_trim9_seed_runtime_clean.py |

| Per-variant SUMMARY |
|---|
| .planning/phases/12-trim-ablation/12-07-SUMMARY.md |
| .planning/phases/12-trim-ablation/12-08-SUMMARY.md |
| .planning/phases/12-trim-ablation/12-09-SUMMARY.md |
| .planning/phases/12-trim-ablation/12-10-SUMMARY.md |
| .planning/phases/12-trim-ablation/12-11-SUMMARY.md |
| .planning/phases/12-trim-ablation/12-12-SUMMARY.md |

| Modified |
|---|
| run_ablation.py — CANONICAL_VARIANTS + VARIANT_SPECS (6 new variants registered, canonical=False) |

| Result fixtures |
|---|
| results/ablation_results/12_extension_runtime_variants/scoreboard.json |
| results/ablation_results/12_extension_runtime_variants/<trimN>/claude_probe/<variant>/mediastore/layer1.json |
| results/ablation_results/12_extension_runtime_variants/<trimN>/claude_sweep/<variant>/<4 datasets>/layer1.json |
| results/ablation_results/12_extension_runtime_variants/trim6,9/gpt54_sweep/<variant>/<5 datasets>/layer1.json |

Frozen-file invariant: NO modifications to s_linker13.py, s_linker13_clean.py, s_linker13_clean_v3.py, prompts_v2.py, data_types_v2.py, document_loader_v2.py, pcm_parser_v2.py.

## Threat flags

None — no new threat surface beyond the per-variant threat model already enumerated for trim3 in Plan 12-05.

## Deviations from plan

- **Standalone-class directive interpretation:** documented above. Subclassed `SLinker13Clean` consistent with the trim3 template the directive nominated.
- **Strategic probe gating SAVED cost:** per the user directive, the strict probe-first gating skipped 4 × 5 = 20 gpt-5.4 sweep runs that would not have changed the verdict (Claude-arm failures cannot pass the cross-model gate regardless of gpt-5.4 result).

## Self-Check: PASSED

- All 6 variant files exist under src/llm_sad_sam/linkers/experimental/.
- All 6 variants registered in run_ablation.py (CANONICAL_VARIANTS + VARIANT_SPECS).
- All 30 Claude per-dataset result JSONs exist (6 variants × 5 datasets).
- All 10 gpt-5.4 result JSONs exist (trim6 + trim9 × 5 datasets each).
- scoreboard.json written.
- Per-variant SUMMARYs 12-07 through 12-12 written.
- Frozen-file invariant verified: zero diff on prompts_v2.py / s_linker13_clean.py / sibling frozen files.
