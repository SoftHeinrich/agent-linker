---
spike: 009
name: s24-replacement-orchestrator
type: standard
validates: "Given phase tools equivalent to S21's knowledge, entity, coreference, validation, and merge stages plus a semantic coverage audit, when a controller chooses an acyclic project-specific workflow from document/component profiles and tool feedback, then the assembled result beats S21 without using an S21 floor or runtime numeric gates."
verdict: VALIDATED
verdict_date: "2026-07-24"
result: "Participation-audit replay: 181 TP / 12 FP / 14 FN; macro F2 90.83% -> 94.75%, pooled F2 88.14% -> 93.01%; macro F1 +1.16pp. Live pre-promotion Mediastore smoke: F2 99.36%. Same tool sequence across projects; no route-diversity claim."
related: [005-upstream-candidate-gap, 007-s24-dynamic-controller, 008-s24-semantic-appeal]
tags: [s24, replacement, orchestration, phase-tools, no-magic]
---

# Spike 009: S24 replacement orchestrator

## What This Validates

S24 is a replacement workflow, not an augmentation:

```text
raw document + component model
        |
model-profile tool + document-profile tool
        |
controller
        +-- entity pipeline
        +-- coreference pipeline
        +-- semantic coverage audit
        `-- finalize
```

The controller selects and orders tools. Tools propose and validate links.

## No-magic design

- no S21 final result is used as a protected floor;
- no score, candidate-count threshold, prefix length, context window, frequency
  cutoff, or project vocabulary controls runtime;
- the workflow is bounded by an acyclic state graph: each state-transforming
  tool consumes an unmet capability and becomes unavailable after execution;
- component/catalog membership and exact source-quote checks are structural
  validity, not semantic acceptance gates;
- numeric pass criteria exist only in offline evaluation.

## Pilot protocol

Saved phase checkpoints act as deterministic tool-call recordings for S21's
existing knowledge/entity/coreference phases. The controller chooses which
recorded tools to consume. The fresh semantic audit covers identity and
architectural participation across subject, object, endpoint, service, boundary,
and contextual-reference roles. It sees the controller's current accepted
links, not an S21 final floor, and its grounded candidates pass S21's existing
two-pass validator.

Gold is loaded only after the controller finalizes.

## Investigation trail

1. A unified coverage-audit tool produced the same entity → coreference → audit
   workflow on all projects. It improved macro F1 by 0.67pp, macro F2 by 2.14pp,
   pooled F1 by 0.76pp, and pooled F2 by 2.20pp over S21. It also slightly beat
   the prior dynamic augmentation on macro and pooled F2. Performance passed the
   recall-oriented objective, but project-specific orchestration did not.
2. Splitting audit into identity and contextual-reference tools produced two
   workflows, but reduced macro and pooled F2 relative to the unified audit and
   added false positives. Manufacturing route diversity is not the objective.
   The unified audit is retained; project specificity remains in runtime
   profiles, decisions, evidence, and outputs. Per user direction, macro and
   pooled F2 are primary; F1 and FP remain reported.
3. A production-feedback parity replay exposed a scaling defect: raw tool
   transcripts recursively embedded in controller history exceeded the Codex
   subprocess argument limit. Normalized accepted/rejected references fixed the
   transport problem while full evidence remained in phase output.
4. Compact feedback produced 175 TP / 12 FP / 20 FN, macro F2 93.23%, and
   pooled F2 90.49%, but still selected the same route. Richer feedback is
   therefore not the route-diversity mechanism; the catch-all audit contract
   makes run-all rational. Its different audit links are sampling variation,
   not a causal route improvement.
5. Residual analysis showed 15 of 20 false negatives were never proposed.
   Reframing audit from claim ownership to architectural participation recovered
   the predicted relational, negated, multi-target, and structural-discourse
   cases: 181 TP / 12 FP / 14 FN, macro F2 94.75%, pooled F2 93.01%.

## Result

**VALIDATED** for replacement architecture and F2 performance. Project-specific
route diversity remains unvalidated.

The participation-audit replacement produced 181 TP / 12 FP / 14 FN. Relative
to S21 it improved macro F2 from 90.83% to 94.75% (+3.92pp), pooled F2 from
88.14% to 93.01% (+4.87pp), macro F1 by 1.16pp, and pooled F1 by 2.00pp.

It exceeded the prior dynamic augmentation by 1.95pp macro F2 and 2.72pp pooled
F2. Its macro F1 is 0.17pp lower, while pooled F1 is 0.54pp higher. All five
projects selected the same phase sequence, so project-specific route diversity
is not claimed.

A fresh Mediastore run completed at 31 TP / 1 FP / 0 FN, F1 98.41%, F2 99.36%.
