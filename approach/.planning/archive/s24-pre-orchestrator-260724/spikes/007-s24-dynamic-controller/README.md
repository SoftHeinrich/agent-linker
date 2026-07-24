---
spike: 007
name: s24-dynamic-controller
type: standard
validates: "A sequential controller that profiles document/reference style, component ambiguity, floor coverage, and grounded phase evidence can improve the fixed S21 floor while invoking fewer recovery phases than run-all."
verdict: VALIDATED
verdict_date: "2026-07-24"
result: "Fixed-floor: 4 TP / 0 FP, macro F1 93.34% -> 94.48%, four workflows, five phase calls vs six run-all; adaptive zero-yield anchor -> alias route observed. Live Mediastore: same-run F1 96.77% -> 98.41%."
related: [006-s24-agentic-phase-tools]
tags: [agentic, controller, dynamic-workflow, oracle-analysis, s24]
---

# Spike 007: S24 dynamic controller

## Question

Can the S24 controller prepare a different workflow for each project from the
document/component profile, then revise that workflow from tool feedback,
instead of merely calling every phase whose candidate count is nonzero?

## Oracle-analysis boundary

The user explicitly requested reading ground truth before designing the ideal
controller. `ORACLE_ANALYSIS.md` therefore uses gold to discover generic failure
categories and feedback requirements. Gold is never passed to the runtime
controller or recovery tools.

Because all five benchmark projects informed that analysis, the subsequent
five-project score is oracle-informed method development, not held-out evidence.
The prompt is frozen before execution and contains no project names, benchmark
phrases, gold counts, or benchmark-derived vocabulary. Generalization requires
an additional unseen-project run.

## Falsifiable gate

- fixed-floor macro F1 strictly exceeds S21;
- marginal precision is at least 0.95 with no more than one false positive;
- at least three distinct ordered workflows;
- fewer recovery-phase executions than calling every eligible phase;
- every decision returns a structured, runtime-evidence-based assessment;
- controller and tools receive no gold or project identity.

## Run

```bash
../.venv/bin/python pilot/s24_dynamic_controller_pilot.py
```

## Result

**VALIDATED**, with an explicit external-validity caveat.

The final feedback iteration achieved 4 TP / 0 FP, 100% marginal precision,
macro F1 93.34% → 94.48% (+1.14pp), four distinct workflows, and five recovery
phase calls versus six for run-all. A live Mediastore run improved its freshly
sampled internal S21 floor from F1 96.77% to 98.41% with one true-positive and
zero false-positive additions.

The preceding frozen iteration also passed: 5 TP / 0 FP, macro F1 94.68%, and
four calls versus six. Resolver variance accounts for the difference.

The five-project scores are oracle-informed because gold was deliberately used
to design the generic reasoning protocol. Runtime is blind, but an unseen
project is still required for held-out validation.
