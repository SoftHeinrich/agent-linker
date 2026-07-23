---
phase: 12-trim-ablation
plan: 12-12
title: Trim9 Seed-Disambiguation Runtime Rubric — ACCEPT (both arms PASS)
status: completed
verdict: ACCEPT (Claude relaxed GATE-01 PASS + gpt-5.4 GATE-01 cross-model PASS)
completed: 2026-05-31
requirements: [PROMPT-02]
tags: [trim, prompt-engineering, runtime-rubric, extension, seed-disambiguation, accept]
---

# Phase 12 Plan 12-12 — SUMMARY

**One-liner:** Runtime-rubric variant replacing `SEED_DISAMBIGUATION_RULES` in the per-component seed-validation prompts PASSES Claude relaxed GATE-01 (macro 0.9474, BBB 0.8440, all per-dataset drops within 2pp) AND gpt-5.4 GATE-01 cross-model (macro 0.9007 ≥ 0.8977 floor) — the only EXTENSION variant accepted on both arms.

## Verdict

| Gate | Required | Observed | Status |
|------|----------|----------|--------|
| GATE-06 static surface | 0 taboo hits | 0 | PASS |
| GATE-01 Claude — macro F1 | >= 0.90 | **0.9474** | PASS |
| GATE-01 Claude — BBB absolute | >= 0.79 | **0.8440** | PASS |
| GATE-01 Claude — per-dataset drop | >= -2pp | worst teastore -1.82pp | PASS |
| GATE-01 cross-model (gpt-5.4) — macro | >= 0.8977 | **0.9007** | PASS |

**Overall:** ACCEPT — variant is GATE-01 compliant on both backends + GATE-06 compliant.

## Per-dataset (Claude Sonnet)

| Dataset | F1 (trim9) | baseline | Δ |
|---|---|---|---|
| mediastore    | 1.0000 | 0.9836 | +0.0164 |
| teastore      | 0.9818 | 1.0000 | -0.0182 |
| teammates     | 0.9381 | 0.9381 | +0.0000 |
| bigbluebutton | 0.8440 | 0.8036 | **+0.0404** |
| jabref        | 0.9730 | 0.9730 | +0.0000 |
| **Macro**     | **0.9474** | 0.9397 | **+0.0077** |

BBB improves by 4.04pp — the largest single-dataset gain across the EXTENSION batch.

## Per-dataset (gpt-5.4)

| Dataset | F1 (trim9) |
|---|---|
| mediastore    | 0.8966 |
| teastore      | 1.0000 |
| teammates     | 0.8522 |
| bigbluebutton | 0.7818 |
| jabref        | 0.9730 |
| **Macro**     | **0.9007** |

Cross-model macro 0.9007 > floor 0.8977 by 0.30pp. First EXTENSION variant to pass the gpt-5.4 gate.

## Mechanism

Tier 2 seed-disambiguation. A runtime rubric builder is invoked ONCE per document (NOT per component) and receives a generic compiler-style seed example, the project document, and the component list; it emits a 4-6 item rubric tailored to the document. The generated rubric replaces the static class-attribute `SEED_DISAMBIGUATION_RULES` in EVERY per-component dossier prompt. Per-component context (anchor sentences, ambiguity classification, mention-context classification) is preserved.

Cost: +1 LLM call per dataset (the rubric is shared across all per-component dossier calls).

NO STATIC FALLBACK: variant raises RuntimeError on empty rubric. Builder never failed across the Claude + gpt-5.4 sweeps.

## Why trim9 succeeds where trim4 / trim5 / trim6 / trim7 / trim8 fail

Hypothesis (informed by Phase 11 §5 and the V31 judge audit):
- Tier 2 seed disambiguation operates on a SMALLER candidate set (only raw seed links from ILinker3) than entity validation (which sees all dual-pass-consensus candidates). The runtime rubric has a tighter scope to calibrate.
- The dossier prompt already supplies rich per-component context (anchor sentences + mention-context classification) — the rubric body contributes proportionally less of the prompt's decision signal, so substituting it perturbs the decision boundary less than substituting the validation or extraction rubrics.
- The seed pipeline biases approve (recall-preserving): the prior was already "approve unless clear reason to reject", and the runtime rubric inherits this bias without amplifying it.

## Cross-dataset rubric isolation

By construction the builder sees only one dataset's input document. Per Plan 12-05-REVISIT operationalization, this satisfies the cross-dataset isolation criterion that CLAUDE.md's GATE-06 actually mandates (dynamic runtime LLM analysis of input data is the prescribed mechanism, NOT a violation).

## Recommended downstream action

Variant CARRIED to Plan 13-01 (s_linker13_min promotion). trim9 composes with:
- Plan 12-03 trim1 (DOC_KNOWLEDGE_JUDGE_RULES distilled — already ACCEPTED).
The two trims operate on disjoint pipeline phases (Tier 1 alias judge vs Tier 2 seed judge), so composition is safe pending a confirmatory single-step sweep on the union variant.

## Files

| Created |
|---|
| src/llm_sad_sam/linkers/experimental/s_linker13_trim9_seed_runtime_clean.py |
| results/ablation_results/12_extension_runtime_variants/trim9/claude_*/<5 datasets>/layer1.json |
| results/ablation_results/12_extension_runtime_variants/trim9/gpt54_sweep/<5 datasets>/layer1.json |

Modified: run_ablation.py (registration).

## Self-Check: PASSED
