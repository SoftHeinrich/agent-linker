---
phase: 12-trim-ablation
plan: 12-08
title: Trim5 Doc-Knowledge-Extraction Runtime Rubric — REJECT (Claude per-dataset drop)
status: completed
verdict: REJECT (Claude relaxed GATE-01 per-dataset drop tolerance)
completed: 2026-05-31
requirements: [PROMPT-02]
tags: [trim, prompt-engineering, runtime-rubric, extension, alias-extraction]
---

# Phase 12 Plan 12-08 — SUMMARY

**One-liner:** Runtime-rubric variant replacing `DOC_KNOWLEDGE_EXTRACTION_RULES` passes the macro and BBB floors on Claude but loses 3.57pp on teastore, exceeding the per-dataset drop tolerance; gpt-5.4 sweep skipped per strategic gate.

## Verdict

| Gate | Required | Observed | Status |
|------|----------|----------|--------|
| GATE-06 static surface | 0 taboo hits | 0 | PASS |
| GATE-01 Claude — macro F1 | >= 0.90 | 0.9359 | PASS |
| GATE-01 Claude — BBB absolute | >= 0.79 | 0.8108 | PASS |
| GATE-01 Claude — per-dataset drop | >= -2pp | teastore -3.57pp | **FAIL** |
| GATE-01 cross-model (gpt-5.4) | not evaluated | — | SKIPPED |

## Per-dataset (Claude Sonnet)

| Dataset | F1 (trim5) | baseline | Δ |
|---|---|---|---|
| mediastore    | 0.9841 | 0.9836 | +0.0005 |
| teastore      | 0.9643 | 1.0000 | **-0.0357** |
| teammates     | 0.9474 | 0.9381 | +0.0093 |
| bigbluebutton | 0.8108 | 0.8036 | +0.0072 |
| jabref        | 0.9730 | 0.9730 | +0.0000 |
| **Macro**     | **0.9359** | 0.9397 | -0.0038 |

Mechanism: Tier 1 alias extraction. A runtime rubric builder receives a generic compiler-style seed example, the project document, and the component list; emits 4-6 extraction criteria that replace `DOC_KNOWLEDGE_EXTRACTION_RULES`. ALIAS_SCOPE_SCHEMA and the downstream judge are unchanged.

NO STATIC FALLBACK: variant raises on empty rubric. Builder never failed in the Claude sweep.

The teastore regression (2 FP) indicates the LLM-generated extraction criteria are less precision-conservative than the hand-crafted rules at the EXTRACTION (proposer) stage, where loosened criteria cascade into entity/coref via the alias dictionary.

Variant NOT carried forward to Phase 13.

## Files

| Created |
|---|
| src/llm_sad_sam/linkers/experimental/s_linker13_trim5_extraction_runtime_clean.py |
| results/ablation_results/12_extension_runtime_variants/trim5/claude_*/<5 datasets>/layer1.json |

Modified: run_ablation.py (registration).

## Self-Check: PASSED
