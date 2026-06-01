---
plan: 12-03
phase: 12
title: Step 1 — Judge trim (alias-judge prompts via Technique 3 + Technique 8)
status: complete
verdict: ACCEPT
completed: 2026-05-31
requirements: [PROMPT-01, PROMPT-02]
subsystem: linkers/experimental + prompts + ablation
tags: [trim-ablation, prompt-engineering, technique-3, technique-8, judge, doc-knowledge]
key-files:
  created:
    - src/llm_sad_sam/linkers/experimental/s_linker13_trim1_judge_clean.py
    - tests/test_s_linker13_trim1_judge_registration.py
    - results/ablation_results/12_03_trim1_judge/claude/s_linker13_trim1_judge_clean/{ms,ts,tm,bbb,jab}/layer1.json
    - results/ablation_results/12_03_trim1_judge/gpt54/s_linker13_trim1_judge_clean/{ms,ts,tm,bbb,jab}/layer1.json
    - results/ablation_results/12_03_trim1_judge/verdict.json
    - .planning/phases/12-trim-ablation/12-03-SUMMARY.md (this file)
  modified:
    - run_ablation.py (CANONICAL_VARIANTS + VARIANT_SPECS — already staged by 12-04)
    - tests/fixtures/v2_0_baseline.json (variant added to `missing` list)
decisions:
  - Trim variant landed as a SUBCLASS of SLinker13Clean (not file-copy) — the surgical
    1-prompt override pattern is more reviewer-defensible than a 1000-line fork.
  - The override consumes its V3 constants via a `try/finally` monkey-patch of the parent
    module's name scope inside `_learn_document_knowledge_enriched`. Not thread-safe vs
    the parent module; documented in class docstring. Ablation harness runs variants
    sequentially per dataset, so no contention.
  - DOC_KNOWLEDGE_JUDGE_EXAMPLES preserved byte-equal (V35a guard — example removal
    regressed Claude by -2.5pp historically).
  - DOC_KNOWLEDGE_JUDGE_RUBRIC_V3 inflated from 773 to 888 bytes (114.9% of original)
    — within the 80-130% lossless-density window. Prose form expands explicit
    connective phrasing in exchange for dropping the numbered-rule shorthand.
metrics:
  duration: "~2h sweep (Claude full-pipeline × 5 + gpt-5.4 full-pipeline × 5; Tier 1 cascade)"
  completed: 2026-05-31
---

# Phase 12 Plan 12-03: Step 1 — Judge Trim — Summary

**Verdict: ACCEPT.** The trim variant `s_linker13_trim1_judge_clean` passes GATE-01
Claude (macro F1 0.9553), GATE-01 cross-model gpt-5.4 (macro F1 0.9173), and GATE-06
benchmark-leakage probe (zero hits). The variant carries forward into Plan 12-06
(reviewer-defensibility audit) and, subject to that audit, into Plan 13-01's
`s_linker13_min` union.

This closes PROMPT-02 for the highest-rule-mass prompt pair (#4
DOC_KNOWLEDGE_JUDGE_EXAMPLES + #5 DOC_KNOWLEDGE_JUDGE_RULES per Phase 11 survey
§5 row 1) and advances PROMPT-01 by adding `DOC_KNOWLEDGE_JUDGE_RUBRIC_V3` to the
v2→v3 mapping table maintained by Plan 12-01 + 12-06.

## What Built

**Trim variant** (subclass override, 1-method surgical scope):

- `src/llm_sad_sam/linkers/experimental/s_linker13_trim1_judge_clean.py`
  - `SLinker13Trim1JudgeClean(SLinker13Clean)` — overrides
    `_learn_document_knowledge_enriched` via parent-module monkey-patch with
    `try/finally`.
  - Exports `DOC_KNOWLEDGE_JUDGE_RUBRIC_V3` (Technique 3 + 8 distillation) and
    `DOC_KNOWLEDGE_JUDGE_EXAMPLES_V3` (byte-equal alias to v2 — V35a guard).
  - `_VARIANT_NAME = "s_linker13_trim1_judge_clean"` isolates the variant's
    checkpoint subtree from the parent's cache.

**Test suite** (14 registration / structural tests):

- `tests/test_s_linker13_trim1_judge_registration.py`
  - Importability + subclass relationship.
  - Technique 8 ordering: "When in doubt" precedes any verdict directive.
  - Technique 3 prose-form: zero numbered-rule markers.
  - Coverage preservation: 4 AUTO-APPROVE sub-categories named
    (abbreviations / trailing / camelcase / multi-word).
  - GATE-06 benchmark-component probe (21-name regex).
  - Length within 80-130% lossless-density window.
  - Frozen-file safety (`git diff --quiet` against prompts_v2 / s_linker13_clean / etc.).
  - Smoke instantiation with `LLMBackend.CHECKPOINT`.

**Ablation results** (10 per-dataset JSONs + verdict aggregate):

- `results/ablation_results/12_03_trim1_judge/claude/s_linker13_trim1_judge_clean/{ds}/layer1.json` × 5
- `results/ablation_results/12_03_trim1_judge/gpt54/s_linker13_trim1_judge_clean/{ds}/layer1.json` × 5
- `results/ablation_results/12_03_trim1_judge/verdict.json` — schema-validated
  PASS/REJECT aggregate against GATE-01 Claude + GATE-01 cross-model + GATE-06.

## The Trim — What Changed

Original `DOC_KNOWLEDGE_JUDGE_RULES` (prompts_v2.py:124-139, 773 bytes):

```
DECISION RULES (apply in order):

1. AUTO-APPROVE these — they are always valid mappings:
   - Abbreviations formed from the component name's initials or words
   - Trailing words of multi-word component names (if no other component shares that word)
   - CamelCase identifiers
   - Multi-word phrases that contain the component name

2. APPROVE if the term plausibly refers to exactly one component and is NOT
   a generic word like "system", "process", "utility", "component", "module".

3. REJECT only if the term is clearly generic and could refer to anything,
   or clearly refers to a different component or the system as a whole.

IMPORTANT: When in doubt, APPROVE. False approvals are filtered by later
pipeline stages; false rejections cause permanent recall loss.
```

V3 rubric (888 bytes — 114.9% of original; prose form, single block):

```
DECISION RUBRIC.

When in doubt, APPROVE — false approvals are filtered by later pipeline
stages, while false rejections cause permanent recall loss, so the bar to
reject sits above the bar to approve.

The following four shapes are always valid mappings and should be approved
on sight: abbreviations formed from the component name's initials or words,
trailing words of multi-word component names provided no other component
shares that word, CamelCase identifiers, and multi-word phrases that contain
the component name. Beyond these four shapes, approve any term that
plausibly refers to exactly one component and is not a bare generic word
such as "system", "process", "utility", "component", or "module". Reject
only when the term is clearly generic and could refer to anything, or when
it clearly refers to a different component or to the whole system rather
than the proposed one.
```

**Technique 3 (lossless rubric distillation, arXiv 2403.12968 family):** the
three numbered rules and the IMPORTANT closer merge into one prose block. All
4 AUTO-APPROVE sub-categories are retained inline (abbreviations, trailing-word,
CamelCase, multi-word). The generic-word exclusion is retained verbatim
("system / process / utility / component / module"). The REJECT clause is
preserved (clearly-generic OR different-component-or-system).

**Technique 8 (reasoning-before-conclusion directive order, arXiv 2603.13351):**
the "When in doubt, APPROVE" tie-breaker is emitted FIRST in the rubric body,
before any decision wording. The original v2 placed it as an IMPORTANT closer,
which arXiv 2603.13351 identifies as a competing-directive failure mode under
prompt complexity. Note that the verdict-format directive (`Return JSON:
{"approved": [...]}`) lives in the consumer method's prompt template, NOT in
the rubric body — so the rubric body itself has no verdict-format directive
to compete with the tie-breaker.

## Claude Sonnet — Per-Dataset Results

| Dataset | F1 trim | F1 baseline (s_linker13) | Delta | FP | FN | Tolerance | Verdict |
|---|---:|---:|---:|---:|---:|---|---|
| mediastore     | 1.0000 | 0.9841 | +0.0159 | 0 | 0  | delta ≥ -0.02 | ✓ |
| teastore       | 1.0000 | 1.0000 |  0.0000 | 0 | 0  | delta ≥ -0.02 | ✓ |
| teammates      | 0.9298 | 0.9474 | -0.0176 | 4 | 4  | delta ≥ -0.02 | ✓ |
| bigbluebutton  | 0.8468 | 0.8214 | +0.0254 | 2 | 15 | absolute ≥ 0.79 (swattr) | ✓ |
| jabref         | 1.0000 | 1.0000 |  0.0000 | 0 | 0  | delta ≥ -0.02 | ✓ |

- **Macro F1 trim:** 0.9553
- **Macro F1 baseline:** 0.9506
- **Macro delta:** +0.0047 (improvement vs Claude baseline)
- **GATE-01 Claude (relaxed v2.1 gate, commit 2b8226d):** PASS
  - Macro F1 0.9553 ≥ 0.90 (relaxed floor) ✓
  - BBB absolute F1 0.8468 ≥ 0.79 (Swattr SAD-SAM expected floor from
    `SwattrEvaluationProject.java`) ✓
  - All other per-dataset deltas ≥ -0.02 ✓
- **Also passes the pre-relaxation GATE-01** (macro ≥ 0.93 AND BBB delta ≥ -6pp) —
  trim is unconditionally accepted under both gate versions.

## gpt-5.4 — Per-Dataset Results

| Dataset | F1 trim | F1 baseline (s_linker13) | Delta | FP | FN | Verdict |
|---|---:|---:|---:|---:|---:|---|
| mediastore     | 0.9333 | 0.9677 | -0.0344 | 1 | 3  | local regression |
| teastore       | 0.9818 | 1.0000 | -0.0182 | 1 | 0  | local regression |
| teammates      | 0.8947 | 0.7939 | +0.1008 | 6 | 6  | strong improvement |
| bigbluebutton  | 0.8036 | 0.8037 | -0.0001 | 5 | 17 | essentially flat |
| jabref         | 0.9730 | 0.9730 |  0.0000 | 1 | 0  | flat |

- **Macro F1 trim:** 0.9173
- **Macro F1 anchor (s_linker13 gpt-5.4 v2.0 CROSS):** 0.9077
- **Macro delta:** +0.0096 (improvement over the v2.0 anchor)
- **GATE-01 cross-model:** PASS
  - Absolute floor 0.8977: 0.9173 ≥ 0.8977 ✓
  - Within 1.0pp of anchor 0.9077: |0.9173 - 0.9077| = 0.0096 ≤ 0.01 ✓

## GATE-06 Probe

The trimmed rubric body was scanned against a 21-term benchmark-component
probe (MediaStore + TeaStore + Teammates + BBB + JabRef components and aliases
from BENCHMARK_TABOO.md). **Zero hits.** Illustrative phrasing in the rubric
uses only generic terms (`system`, `module`, `utility`, `component`,
`process`) — all of which also appear in the v2 original and are part of the
permitted generic-word exclusion list.

## Cross-Model Behavior — Defensibility Note

The trim shifts behavior asymmetrically across backends:

- **Claude Sonnet** treats the prose rubric as denser information of the same
  rules. Macro improves by +0.47pp (within Claude's run-to-run variance band,
  so functionally tied with the baseline). BBB improves by +2.54pp — the
  judge's reordered tie-breaker preserves more partial-name aliases that the
  three-rule structure was rejecting.
- **gpt-5.4** treats the prose rubric as a stronger APPROVE prior than the
  numbered rules. Teammates improves by +10.08pp (recovering judge
  over-rejection observed in v2.0 CROSS). MediaStore and TeaStore each lose
  ~2-3pp from extra FPs (the same prior that helps TM lets through more FPs
  on small documents).
- Net cross-model gain is +0.96pp on macro, ABOVE the gpt-5.4 anchor by
  the margin required to satisfy GATE-01 cross-model.

This is the V35-escape mechanism that Phase 11 survey §5 row 1 predicted:
Technique 3 preserves the rule mass that Claude's information-density profile
demands, while Technique 8 fixes the directive-ordering failure mode that
gpt-5.4 was tripping on in v2.0.

## Deviations from Plan

None — plan executed exactly as written. The strategic 3-round gating
worked:
- Round 1 mediastore probe passed (F1 1.0 ≥ baseline - 3pp). Proceeded.
- Round 2 Claude sweep PASSED GATE-01. Proceeded.
- Round 3 gpt-5.4 sweep PASSED GATE-01 cross-model.

No deviations from the plan's Task 1 contract (Technique 3 + 8 distillation,
example preservation, length budget, GATE-06 probe). No frozen-file edits.
No retries needed on any per-dataset run.

## Threat Surface (Plan-Level Threat Register)

All 5 STRIDE threats from the plan are mitigated:
- **T-12-03-01** (tampering with prompts_v2.py): mitigated — git diff --quiet
  exits 0; the new rubric is a constant in the variant file, not an edit to v2.
- **T-12-03-02** (benchmark leakage in trimmed rubric body): mitigated — the
  21-name probe returns zero hits at both Task 1 (test) and Task 4 (verdict).
  Plan 12-06 will run the full TABOO sweep as defense in depth.
- **T-12-03-03** (monkey-patch leak via concurrency): mitigated — variant
  docstring documents the not-thread-safe constraint; ablation harness runs
  variants sequentially per dataset.
- **T-12-03-04** (verdict.json provenance): mitigated — verdict.json embeds
  variant name, ISO timestamp, both baseline source paths, and explicit
  gate-reason fields.
- **T-12-03-05** (downstream promotion of a leaky rubric): mitigated — this
  plan only ACCEPTs into the candidate pool; Plan 12-06 is the gate before
  the variant joins `s_linker13_min`.

## Downstream Lineage

- **Plan 12-06** (reviewer-defensibility audit): the trim variant + rubric
  are primary audit targets. Plan 12-06 runs the full BENCHMARK_TABOO sweep,
  checks reviewer-defensibility of the variant's docstring justifications,
  and either confirms ACCEPT or escalates to a Phase 12 revision.
- **Plan 13-01** (s_linker13_min union): subject to Plan 12-06's audit, the
  V3 rubric joins the union of accepted trims that constitutes the
  candidate canonical `s_linker13_min` for Phase 13 promotion.
- **Plan 12-01** (v2→v3 mapping table): adds the row
  `DOC_KNOWLEDGE_JUDGE_RULES → DOC_KNOWLEDGE_JUDGE_RUBRIC_V3` with the
  delta description above.

## Requirements Closed

- **PROMPT-02** (Step 1 / judge trim): ACCEPTED. Both gates clear. Trim
  variant exists, registered, ablated, recorded.
- **PROMPT-01** (Phase 12 v2→v3 mapping): progressed. The V3 rubric is the
  first prompt-fragment-level v2→v3 binding produced by Phase 12 (v3 sibling
  variant was Plan 12-01 Step 0, byte-equal — this is the first real
  distillation).

## Self-Check: PASSED

Created files verified present:
- `src/llm_sad_sam/linkers/experimental/s_linker13_trim1_judge_clean.py` — FOUND
- `tests/test_s_linker13_trim1_judge_registration.py` — FOUND
- `results/ablation_results/12_03_trim1_judge/verdict.json` — FOUND
- `results/ablation_results/12_03_trim1_judge/claude/s_linker13_trim1_judge_clean/{mediastore,teastore,teammates,bigbluebutton,jabref}/layer1.json` — all 5 FOUND
- `results/ablation_results/12_03_trim1_judge/gpt54/s_linker13_trim1_judge_clean/{mediastore,teastore,teammates,bigbluebutton,jabref}/layer1.json` — all 5 FOUND

Frozen files unchanged:
- `git diff --quiet src/llm_sad_sam/linkers/experimental/prompts_v2.py src/llm_sad_sam/linkers/experimental/s_linker13.py src/llm_sad_sam/linkers/experimental/s_linker13_clean.py src/llm_sad_sam/core/data_types_v2.py src/llm_sad_sam/core/document_loader_v2.py src/llm_sad_sam/pcm_parser_v2.py` → exit 0

Commit Task 1:
- `443460a feat(12-03): add s_linker13_trim1_judge_clean variant (Step 1 trim)` — FOUND
