---
phase: 11
title: Prompt-Minimization Harness Survey
status: complete
completed: 2026-05-31
requirements: [PROMPT-05]
---

# Phase 11 — Research — SUMMARY

## What Shipped

Two survey artifacts under `.planning/research/`:

1. **`PROMPT-HARNESS-SURVEY.md`** (~3,400 words) — primary PROMPT-05 deliverable. Enumerates the 16-constant prompt surface of `prompts_v2.py`, identifies 9 actively imported by `s_linker13_clean` and 7 dead constants (free win), surveys 8 techniques across the three required scope families (classical / open-source coding tools / recent agentic-reasoning papers), and recommends a concrete Phase 12 trim order.

2. **`PROMPT-HARNESS-SURVEY-SUPPLEMENT-ERDOS.md`** (~2,300 words) — user-requested supplement. Focuses on OpenAI's Erdős-conjecture proof system (May 2026, announcement-only — architecture not disclosed) and 5 verified breaking agent-tool papers from April–May 2026. Cross-references main survey §4 and adds 3 more Phase 12 prompt-action priorities.

## Key Findings

**Prompt surface (the trim target).** 16 prompt constants in `prompts_v2.py`. Only 9 are imported by `s_linker13_clean` (the new variant from Phase 10). The other 7 (`WORD_USAGE_PROMPT` + 6 EXT-01 STANDALONE_MENTION variants) are dead weight — **~150 lines / ~36 rules deletable from `prompts_v3.py` with zero F1 risk**. This is the Phase 12 free win before any technique fires.

**V35 ceiling explained.** arXiv 2603.13351 (Jo, March 2026) independently documents the V35 mechanism: competing directives in long prompts reverse reasoning-vs-conclusion order. The published escape route matches the project's lived experience: **restructure rules, do not delete them**. The supplement adds AHE (arXiv 2604.25850) and Agentic Rubrics (arXiv 2601.04171) as inference-time *rubric regeneration* mechanisms that preserve information density rather than losing it.

**OpenAI Erdős = negative control.** OpenAI's May 2026 announcement of a planar-unit-distance conjecture disproof used Best-of-N + minimal prompt + human verifier on a one-shot verifiable problem. TLR is many-decision-unverifiable — structural mismatch. Listed as fit 2/5 ("template not to imitate"), but the disclosure itself is valuable: confirms that for unverifiable many-decision tasks, harness verbosity remains load-bearing.

**Open-source coding tools disagree.** `opencode` favors minimal AGENTS.md + tool schemas (delegate to model). OpenAI's own `codex` prompting guide recommends comprehensive rule structures. The latter empirically validates the current `s_linker13` rule-heavy design — directly informs Phase 13 promotion decisions.

## Recommended Phase 12 Trim Order (Cross-Survey Consolidation)

From main §5 + supplement §4 (deduplicated):

1. **Step 0 — Free win (no LLM cost).** Drop the 7 unused prompt constants from `prompts_v3.py`. ~36 rules, ~150 LOC. Zero ablation risk. Reviewer-defensibility win for GATE-06.
2. **Step 1 — Highest-leverage restructure.** Apply Technique 3 (lossless rubric distillation) + Technique 8 (directive ordering, arXiv 2603.13351) to `DOC_KNOWLEDGE_JUDGE_RULES` + `DOC_KNOWLEDGE_JUDGE_EXAMPLES`. Largest rule mass; biggest expected reduction without coverage loss.
3. **Step 2 — Overlap merge.** Apply Technique 3 to `ENTITY_EXTRACTION_RULES` + `VALIDATION_RULES` — merge overlapping architectural-participant rubric. Estimated 4-rule reduction.
4. **Step 3 — Inference-time rubric (supplement).** Apply AHE/Agentic-Rubrics pattern (supplement Techniques 2 + 3) to the judge stage. Regenerate-not-prune. Higher-risk but addresses V35 ceiling head-on.
5. **Open empirical question for Phase 12 budget.** Does enabling Claude Sonnet extended-thinking allow trims V35 failed at, without regressing gpt-5.4? Main §6 names this as the highest-leverage allocation question.

## Files Created

- `.planning/research/PROMPT-HARNESS-SURVEY.md` (committed `ec04a2e`)
- `.planning/research/PROMPT-HARNESS-SURVEY-SUPPLEMENT-ERDOS.md` (committed `21e5267`)
- Main survey §4.1 cross-reference added (this commit)

## No Source Code Changes

Phase 11 is a research phase. No source files modified. All v2.0 frozen files untouched.

## GATE Verdicts

- **GATE-06**: All examples in both surveys use textbook SE contexts (parser, scheduler, broker, lexer, dispatcher, payment-gateway, render-engine, scene-graph). Zero benchmark component names. Verified via grep of taboo list.
- **GATE-01**: N/A this phase (no F1 measurement).
- **PROMPT-05 contract**: ≥ 3 techniques surveyed (delivered: 8 main + 6 supplement = 14 distinct techniques/patterns scored). GATE-06 compatibility verdict per technique: ✓. Estimated rule-count reduction per technique: ✓. Concluding trim-order recommendation: ✓ (main §5 + supplement §4 + consolidated above).

## Commits

- `92ef521` docs(11): smart discuss context (research scope + user expansion)
- `21e5267` docs(11): research supplement — Erdos + breaking agent papers
- `ec04a2e` docs(11): prompt-harness survey (PROMPT-05 deliverable)
- (this commit) docs(11): close phase — main survey supplement cross-ref + SUMMARY

## Requirements Closed

- **PROMPT-05**: Literature + web survey on prompt-minimization harness techniques. ≥ 3 concrete techniques scored for fit-to-`s_linker13`. Findings inform Phase 12 trim strategy. ✓
