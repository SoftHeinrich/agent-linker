---
phase: 12-trim-ablation
plan: 12-09
title: Trim6 Doc-Knowledge-Judge-Examples Runtime — REJECT (cross-model)
status: completed
verdict: REJECT (gpt-5.4 GATE-01 cross-model gap −0.39pp)
completed: 2026-05-31
requirements: [PROMPT-02]
tags: [trim, prompt-engineering, runtime-examples, extension, alias-judge, orthogonal-to-trim1]
---

# Phase 12 Plan 12-09 — SUMMARY

**One-liner:** Runtime-generated worked examples (with trim1's distilled rubric) replace the 7 static `DOC_KNOWLEDGE_JUDGE_EXAMPLES`; Claude relaxed GATE-01 PASSES (macro 0.9406, all per-dataset drops within 2pp), gpt-5.4 cross-model FAILS by 0.39pp; rejected for v2.1 on the same model-capability pattern documented for trim3.

## Verdict

| Gate | Required | Observed | Status |
|------|----------|----------|--------|
| GATE-06 static surface | 0 taboo hits | 0 | PASS |
| GATE-01 Claude — macro F1 | >= 0.90 | **0.9406** | PASS |
| GATE-01 Claude — BBB absolute | >= 0.79 | 0.8257 | PASS |
| GATE-01 Claude — per-dataset drop | >= -2pp | worst teammates -1.77pp | PASS |
| GATE-01 cross-model (gpt-5.4) — macro | >= 0.8977 | **0.8938** | **FAIL (−0.39pp)** |

## Per-dataset (Claude Sonnet)

| Dataset | F1 (trim6) | baseline | Δ |
|---|---|---|---|
| mediastore    | 0.9841 | 0.9836 | +0.0005 |
| teastore      | 1.0000 | 1.0000 | +0.0000 |
| teammates     | 0.9204 | 0.9381 | -0.0177 |
| bigbluebutton | 0.8257 | 0.8036 | +0.0221 |
| jabref        | 0.9730 | 0.9730 | +0.0000 |
| **Macro**     | **0.9406** | 0.9397 | +0.0009 |

## Per-dataset (gpt-5.4)

| Dataset | F1 |
|---|---|
| mediastore    | 0.9508 |
| teastore      | 0.9818 |
| teammates     | 0.8361 |
| bigbluebutton | 0.7273 |
| jabref        | 0.9730 |
| **Macro**     | **0.8938** |

Mechanism: Tier 1 alias judge. Two compositional changes vs `s_linker13_clean`:
  1. The 7 static `DOC_KNOWLEDGE_JUDGE_EXAMPLES` are REPLACED by 4-6 worked examples generated at runtime from a generic compiler-style seed example + the project document + the candidate mappings (mapping / verdict / one-line rationale per example).
  2. The static `DOC_KNOWLEDGE_JUDGE_RULES` is REPLACED by trim1's accepted distilled rubric (`DOC_KNOWLEDGE_JUDGE_RUBRIC_V3` — Technique 3 + 8 lossless compression).

The variant is ORTHOGONAL to trim1: trim1 distilled the rules; trim6 generates the examples at runtime. Both can in principle compose. Trim1 is ACCEPTED (Plan 12-03); trim6 is REJECTED on cross-model gap.

NO STATIC FALLBACK: variant raises on empty examples. Builder never failed across the Claude + gpt-5.4 sweeps.

## V35a guard discussion

V35a documented that REMOVING the worked examples regresses Claude by -2.5pp. Trim6 does NOT remove the examples; it REGENERATES them per-document at runtime. The Claude result (no regression — actually +0.0009pp) confirms the V35a guard transfers: as long as the judge sees worked examples (static OR runtime), Claude maintains calibration. The 0.39pp gpt-5.4 gap is the same pattern as trim3 (a model-capability gap, not a methodology failure).

## Cross-dataset rubric isolation

The runtime examples are dataset-scoped by construction (each builder call receives only one dataset's document). Per Plan 12-05-REVISIT's operationalization, cross-dataset isolation holds by information-theoretic guarantee (the builder cannot output cross-dataset tokens unless they are model priors from training, which is the standard runtime-LLM-analysis pattern CLAUDE.md mandates).

Variant NOT carried forward to Phase 13 (cross-model gate is a v2.1 thesis claim; no relaxation).

## Files

| Created |
|---|
| src/llm_sad_sam/linkers/experimental/s_linker13_trim6_judge_examples_runtime_clean.py |
| results/ablation_results/12_extension_runtime_variants/trim6/claude_*/<5 datasets>/layer1.json |
| results/ablation_results/12_extension_runtime_variants/trim6/gpt54_sweep/<5 datasets>/layer1.json |

Modified: run_ablation.py (registration).

## Self-Check: PASSED
