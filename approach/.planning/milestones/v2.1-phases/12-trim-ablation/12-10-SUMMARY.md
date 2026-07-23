---
phase: 12-trim-ablation
plan: 12-10
title: Trim7 Entity-Extraction Runtime Rubric — REJECT (Claude per-dataset drop)
status: completed
verdict: REJECT (Claude relaxed GATE-01 per-dataset drop tolerance)
completed: 2026-05-31
requirements: [PROMPT-02]
tags: [trim, prompt-engineering, runtime-rubric, extension, entity-extraction]
---

# Phase 12 Plan 12-10 — SUMMARY

**One-liner:** Runtime-rubric variant replacing `ENTITY_EXTRACTION_RULES` passes the macro and BBB floors on Claude but loses 2.56pp on jabref; gpt-5.4 sweep skipped per strategic gate.

## Verdict

| Gate | Required | Observed | Status |
|------|----------|----------|--------|
| GATE-06 static surface | 0 taboo hits | 0 | PASS |
| GATE-01 Claude — macro F1 | >= 0.90 | 0.9365 | PASS |
| GATE-01 Claude — BBB absolute | >= 0.79 | 0.8148 | PASS |
| GATE-01 Claude — per-dataset drop | >= -2pp | jabref -2.56pp | **FAIL** |
| GATE-01 cross-model (gpt-5.4) | not evaluated | — | SKIPPED |

## Per-dataset (Claude Sonnet)

| Dataset | F1 (trim7) | baseline | Δ |
|---|---|---|---|
| mediastore    | 0.9836 | 0.9836 | +0.0000 |
| teastore      | 0.9818 | 1.0000 | -0.0182 |
| teammates     | 0.9550 | 0.9381 | +0.0169 |
| bigbluebutton | 0.8148 | 0.8036 | +0.0112 |
| jabref        | 0.9474 | 0.9730 | **-0.0256** |
| **Macro**     | **0.9365** | 0.9397 | -0.0032 |

Mechanism: Tier 2 entity proposer. A runtime rubric builder receives a generic compiler-style seed example, the project document, the component list, and the global-scope alias map; emits 4-6 extraction criteria that replace `ENTITY_EXTRACTION_RULES` in BOTH passes of the dual-pass extraction consensus. The rubric is built ONCE per document and reused across batches + passes (cost: +1 LLM call per dataset).

NO STATIC FALLBACK. Rubric builder never failed.

The jabref regression (1 FP) reflects small-dataset variance (jabref has 19 gold links — 1 link delta ≈ 2.6pp F1). Even so, the strict per-dataset tolerance is violated. 12-04 ENTVAL MERGE similarly identified proposer-side risks.

Variant NOT carried forward.

## Files

| Created |
|---|
| src/llm_sad_sam/linkers/experimental/s_linker13_trim7_entity_runtime_clean.py |
| results/ablation_results/12_extension_runtime_variants/trim7/claude_*/<5 datasets>/layer1.json |

Modified: run_ablation.py (registration).

## Self-Check: PASSED
