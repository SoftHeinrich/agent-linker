---
phase: 12-trim-ablation
plan: 12-07
title: Trim4 Ambiguity Runtime Rubric — REJECT (Claude per-dataset drop)
status: completed
verdict: REJECT (Claude relaxed GATE-01 per-dataset drop tolerance)
completed: 2026-05-31
requirements: [PROMPT-02]
tags: [trim, prompt-engineering, runtime-rubric, extension, ambiguity]
---

# Phase 12 Plan 12-07 — SUMMARY

**One-liner:** Runtime-rubric variant replacing `AMBIGUITY_FEW_SHOT + AMBIGUITY_RULES` passes the macro and BBB floors on Claude but loses 2.56pp on jabref, exceeding the per-dataset drop tolerance; gpt-5.4 sweep skipped per strategic gate.

## Verdict

| Gate | Required | Observed | Status |
|------|----------|----------|--------|
| GATE-06 static surface (seed + prompt) | 0 taboo hits | 0 | PASS |
| GATE-01 Claude — macro F1 | >= 0.90 | 0.9374 | PASS |
| GATE-01 Claude — BBB absolute | >= 0.79 | 0.8257 | PASS |
| GATE-01 Claude — per-dataset drop | >= -2pp | jabref -2.56pp | **FAIL** |
| GATE-01 cross-model (gpt-5.4) | not evaluated | — | SKIPPED |

## Per-dataset (Claude Sonnet)

| Dataset | F1 (trim4) | baseline | Δ |
|---|---|---|---|
| mediastore    | 0.9841 | 0.9836 | +0.0005 |
| teastore      | 1.0000 | 1.0000 | +0.0000 |
| teammates     | 0.9298 | 0.9381 | -0.0083 |
| bigbluebutton | 0.8257 | 0.8036 | +0.0221 |
| jabref        | 0.9474 | 0.9730 | **-0.0256** |
| **Macro**     | **0.9374** | 0.9397 | -0.0023 |

Mechanism: Tier 1 ambiguity classifier. A runtime rubric builder receives a generic compiler-style seed example and the actual component-name list; emits 4-6 calibration criteria that replace `AMBIGUITY_FEW_SHOT + AMBIGUITY_RULES`. Single-word post-filter preserved.

NO STATIC FALLBACK: variant raises RuntimeError on empty rubric. Across the full Claude sweep the rubric builder never failed; fallback path is dead code in practice.

Variant NOT carried forward to Phase 13. Gap is a single jabref FP (small-dataset variance — jabref has only 19 gold links, so 1 FP costs 2.6pp F1).

## Files

| Created |
|---|
| src/llm_sad_sam/linkers/experimental/s_linker13_trim4_ambiguity_runtime_clean.py |
| results/ablation_results/12_extension_runtime_variants/trim4/claude_*/<5 datasets>/layer1.json |

Modified: run_ablation.py (CANONICAL_VARIANTS + VARIANT_SPECS registration).

## Self-Check: PASSED
