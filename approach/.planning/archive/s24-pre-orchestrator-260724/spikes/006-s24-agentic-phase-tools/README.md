---
spike: 006
name: s24-agentic-phase-tools
type: standard
validates: "Given a fixed S21 floor and only runtime project evidence, when a bounded controller selects existing knowledge/validation and S24 recovery tools, then the selected additions improve macro F1 with at least 95% marginal precision and project-dependent routes."
verdict: VALIDATED
verdict_date: "2026-07-24"
result: "Production replay: 5 TP / 0 FP, 100% marginal precision, fixed-floor macro F1 93.34% -> 94.68% (+1.34pp), four distinct plans. End-to-end Mediastore smoke: same-run floor F1 96.77% -> 98.41% (+1 TP, 0 marginal FP)."
related: [005-upstream-candidate-gap]
tags: [agentic, tool-routing, fixed-floor, s24, traceability]
---

# Spike 006: S24 agentic phase tools

## What This Validates

Given the saved, exact S21 floor for each project, when a controller sees only
runtime component names and candidate inventory counts, then it can select bounded
recovery tools whose independently scored additions improve the fixed floor.

## Hypothesis

The useful unit of autonomy is tool selection, not free-form link prediction.
Two tools are sufficient for the first test:

- `alias_phase4`: exact occurrences of aliases already approved by Phase 1,
  validated by S21's unchanged Phase-4 two-pass gate.
- `anchored_reference`: S24's existing local-anchor resolver and dedicated
  anchored-reference validator.

The controller may call zero, one, or both. It cannot emit candidates or links.

## Pass Gate

- at least one marginal true positive;
- no more than one marginal false positive and marginal precision at least 0.95;
- macro F1 strictly above the same saved S21 floor;
- at least two distinct tool plans across projects;
- no gold data, project identity, or benchmark-derived vocabulary in prompts.

## How to Run

```bash
../.venv/bin/python pilot/s24_agentic_tools_pilot.py
```

## What to Expect

The script writes a JSON trace and one final link CSV per project under
`../results/s24_agentic_tools_pilot_20260724/`. The JSON contains controller
plans, tool inventories, every model call, fixed-floor metrics, marginal
additions, and the pass-gate verdict.

## Investigation Trail

1. S24 already demonstrated 3/3 clean marginal anchored additions, but its
   deterministic eligibility already skips projects with no cases. A controller
   around that single tool would add overhead without meaningful autonomy.
2. Phase-1 caches contain approved, project-specific aliases that S21 injects
   into extraction but does not always recover. Exact residual alias scanning
   supplies a second bounded candidate source while reusing the unchanged
   Phase-4 gate.
3. The fixed-floor replay avoids interpreting stochastic S21 reruns as an
   augmentation delta.
4. Iteration 1 improved macro F1 by 1.41 points and found 8 TPs, but failed the
   precision gate with 4 FPs. The controller's choices were valid; the failures
   were acceptance errors: weak lexical aliases of Phase-1-ambiguous targets,
   and a short anchored target inside a longer approved alias for another
   component.
5. Iteration 2 adds two runtime-grounded constraints before rerunning the same
   gate: weak lexical aliases do not become exact identifiers for ambiguous
   targets, and a longer approved competing alias wins over a short anchored
   target. Neither constraint contains project vocabulary or benchmark scores.

## Results

**VALIDATED.**

Iteration 1 failed the declared precision gate despite improving F1: 8 TP and
4 FP (66.67% marginal precision). After the two runtime-grounding corrections,
iteration 2 passed with 6 TP / 0 FP and macro F1 93.34% → 94.88%.

The promoted production class independently passed the same all-project fixed-
floor protocol with 5 TP / 0 FP, 100% marginal precision, four distinct plans,
and macro F1 93.34% → 94.68% (+1.34pp). A normal runner smoke on Mediastore
completed at F1 98.41%, +1.64pp over its same-run internal S21 floor.

Evidence:

- `results/s24_agentic_tools_pilot_20260724/` — failed iteration preserved;
- `results/s24_agentic_tools_pilot_iter2_20260724/` — passing pilot;
- `results/s24_agentic_promoted_fixed_floor_20260724/` — production replay;
- `results/s24_agentic_codex_e2e_mediastore_20260724/` — live end-to-end smoke.
