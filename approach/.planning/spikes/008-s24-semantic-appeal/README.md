---
spike: 008
name: s24-semantic-appeal
type: standard
validates: "Given every grounded S21 candidate rejected from the fixed floor, when a semantic appeal judge reviews the original evidence without heuristic eligibility rules, then appeal-only S24 beats S21 and current dynamic S24 at high marginal precision."
verdict: INVALIDATED
verdict_date: "2026-07-24"
result: "Appeal-only iteration 1: 7 TP / 12 FP, macro F1 below S21. Structured identity/ownership iteration 2: 5 TP / 6 FP, macro F1 above S21 but below dynamic S24. User clarified that S24 must replace, not refine, S21; stop this architecture."
related: [005-upstream-candidate-gap, 007-s24-dynamic-controller]
tags: [s24, error-analysis, appeal, no-magic, fixed-floor]
---

# Spike 008: S24 semantic appeal

## What This Validates

Can S24 replace hand-built anchor/alias eligibility rules with a semantic appeal
over evidence S21 already produced?

## Error analysis

The saved fixed floors have 27 false negatives:

- 7 were proposed by entity extraction and rejected by Phase 4;
- 3 were proposed by coreference and rejected by Phase 5;
- 17 were never proposed.

The current dynamic S24 recovers some of both classes, but its anchor discovery
contains a local-context window, a prefix-length cutoff, uppercase/digit tests,
and a special sibling-name pattern. Those are precisely the magic gates this
spike is intended to avoid.

Prior S23 residual proposers already established that broad rediscovery can add
recall without improving F1. The first experiment therefore targets the smaller,
better-grounded rejected-candidate class.

## Hypothesis

Every residual rejected candidate is eligible. One semantic judge reviews:

- the source sentence and preceding sentence;
- exact matched span or coreference phrase;
- target component;
- original evidence bundle or antecedent;
- original validator outcome.

The judge must cite the architectural claim and explain reference resolution
before deciding. There are no runtime scores, thresholds, vocabulary lists,
window sizes, or project-specific prompt terms.

## Investigation trail

1. Iteration 1 used a direct semantic appeal contract. It recovered 7 true
   positives but admitted 12 false positives (36.84% marginal precision), lowering
   macro F1 below S21. The hypothesis that a single generic reconsideration
   instruction was sufficient is invalid.
2. Error analysis found two cross-project semantic failures: association was
   mistaken for referent identity, and claim ownership was inferred from nearby
   context rather than the grammatical/semantic subject. Iteration 2 asks the
   same single judge for a structured identity-and-ownership decision, including
   the strongest competing referent. No eligibility rule or extra vote is added.
3. Iteration 2 improved to 5 TP / 6 FP and lifted macro F1 above S21, but remained
   below current dynamic S24. The clean useful decisions were concentrated in
   entity appeals plus one clear singular coreference case; noisy coreference
   families dominated the remaining false positives. Iteration 3 gives the
   project-profile controller authority to select entity and coreference appeal
   tools, while the same semantic judge retains link authority.

## Result

**INVALIDATED.**

Iteration 1 produced 7 TP / 12 FP (36.84% marginal precision) and lowered macro
F1 below S21. Explicit referent-identity and claim-ownership fields improved
iteration 2 to 5 TP / 6 FP and macro F1 93.91%, but this remained below the
current dynamic S24 result.

More importantly, the user clarified the architectural requirement: the agent
must replace S21's fixed workflow, not refine an S21 result. An appeal controller
was therefore stopped during execution. The semantic error analysis carries
forward, but this architecture does not.

## Evaluation gate

Numeric gates are evaluation criteria only; they do not affect runtime:

- appeal-only macro F1 exceeds the fixed S21 floor;
- marginal precision is at least 0.95;
- appeal-only macro F1 exceeds the best preserved dynamic-S24 replay;
- no project identity or gold reaches the judge.

## Run

```bash
../.venv/bin/python pilot/s24_semantic_appeal_pilot.py
```
