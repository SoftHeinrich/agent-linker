---
phase: 12-trim-ablation
plan: 12-11
title: Trim8 Validation Runtime Rubric — REJECT (Claude per-dataset drop)
status: completed
verdict: REJECT (Claude relaxed GATE-01 per-dataset drop tolerance)
completed: 2026-05-31
requirements: [PROMPT-02]
tags: [trim, prompt-engineering, runtime-rubric, extension, validation, judge]
---

# Phase 12 Plan 12-11 — SUMMARY

**One-liner:** Runtime-rubric variant replacing `VALIDATION_RULES` in BOTH validation passes (participation + specificity) passes the macro and BBB floors on Claude but loses 3.57pp on teastore and 2.56pp on jabref, exceeding the per-dataset drop tolerance on TWO datasets; gpt-5.4 skipped.

## Verdict

| Gate | Required | Observed | Status |
|------|----------|----------|--------|
| GATE-06 static surface | 0 taboo hits | 0 | PASS |
| GATE-01 Claude — macro F1 | >= 0.90 | 0.9251 | PASS |
| GATE-01 Claude — BBB absolute | >= 0.79 | 0.8036 | PASS |
| GATE-01 Claude — per-dataset drop | >= -2pp | teastore -3.57pp; jabref -2.56pp | **FAIL** |
| GATE-01 cross-model (gpt-5.4) | not evaluated | — | SKIPPED |

## Per-dataset (Claude Sonnet)

| Dataset | F1 (trim8) | baseline | Δ |
|---|---|---|---|
| mediastore    | 0.9841 | 0.9836 | +0.0005 |
| teastore      | 0.9643 | 1.0000 | **-0.0357** |
| teammates     | 0.9259 | 0.9381 | -0.0122 |
| bigbluebutton | 0.8036 | 0.8036 | +0.0000 |
| jabref        | 0.9474 | 0.9730 | **-0.0256** |
| **Macro**     | **0.9251** | 0.9397 | -0.0146 |

Mechanism: Tier 2 validation judge. A runtime rubric builder receives a generic compiler-style seed example, the project document, component list, and a sample of candidate cases (no verdicts); emits a 4-6 item rubric that replaces `VALIDATION_RULES` in BOTH passes of the 2-pass validation (participation + specificity). Generic-word pre-pass and intersection voting preserved.

NO STATIC FALLBACK. Builder never failed in the Claude sweep.

This is the HIGHEST-PRIORITY trim of the extension batch (per Phase 11 survey §0 — judge prompts identified as most productive trim targets). The result: two-dataset per-dataset violation is more severe than the other failures, suggesting the validation judge specifically is sensitive to rule-set substitution at the boundary between TPs and FPs (consistent with the V31 judge audit finding that the judge is "largely a rubber stamp — 70% of FPs are judge-immune"; loosening the judge rubric flips a small but consistent set of FPs from rejected to approved).

Differs from 12-04 (VAL-EXT MERGE): 12-04 merged ENTITY_EXTRACTION_RULES + VALIDATION_RULES into a single rule set, regressing BBB by -6.6pp. Trim8 keeps the two prompts separate and applies the runtime mechanism to validation alone — a different failure surface, smaller magnitude, but still REJECT on the per-dataset gate.

Variant NOT carried forward.

## Files

| Created |
|---|
| src/llm_sad_sam/linkers/experimental/s_linker13_trim8_validation_runtime_clean.py |
| results/ablation_results/12_extension_runtime_variants/trim8/claude_*/<5 datasets>/layer1.json |

Modified: run_ablation.py (registration).

## Self-Check: PASSED
