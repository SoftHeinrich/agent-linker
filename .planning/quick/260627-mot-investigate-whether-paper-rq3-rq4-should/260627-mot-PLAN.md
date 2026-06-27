---
quick_id: 260627-mot
slug: investigate-whether-paper-rq3-rq4-should
date: 2026-06-27
type: investigation
status: complete
---

# Quick Task: Should paper RQ3/RQ4 report the RQ2 size-aware metrics?

## Description

Decide, grounded on real run results, whether the paper's **RQ3** (validator
contribution) and **RQ4** (module ablation) should report the **RQ2 size-aware
metric suite** (sentence coverage, worst-component F1, harmonic-component F1)
instead of (or alongside) plain macro-F1 and TP/FP counts.

The judgment must be data-grounded, not argued: run the metric pipelines on the
actual ablation link sets and read the deltas.

## Scope

- Source data: the canonical N=3 `s_linker20_union` sweep (v2.6.5),
  `results/v2.6.5_s20union_sonnet` (claude) and `results/v2.6.5_s20union` (gpt),
  via the `phase_cache` pickles per run/project.
- Tooling: `evaluation/mini-rq34/rq34.py` (rebuilds RQ3/RQ4 variant link sets,
  self-validates tp/fp/fn against each run's `ablation_*.json`) and
  `evaluation/mini-rq34/rq34_rq2.py` (composes each variant to doc-to-code and
  scores it with the RQ2 panel).
- For RQ3: NoEntityValid / NoCitation / NoValidator vs Full, per metric.
- For RQ4: EntityOnly / CorefOnly vs Full, per metric.
- Both backends (claude + openai), all 3 runs + run-average; check sign
  robustness across backend × run, not just on the average.

## Out of scope

- No code/prompt changes, no new LLM runs. Existing run artifacts only.
- No paper edits in this task — the deliverable is the judgment + a concrete
  recommendation. Applying it to `working/` is a follow-up.

## Method

Run both `mini-rq34` scripts fresh; confirm `validate=OK` against the ablation
JSONs and that the reports reproduce. Read run-average and per-run deltas from
`reports/rq34_rq2_variants.csv` (RQ3) and `reports/rq34_rq2_linkers.csv` (RQ4).
Decision rule per RQ: a size-aware metric is worth reporting only if its delta is
(a) the same sign as the contribution and (b) robust across both backends and all
three runs. A metric that flips sign across backends, or moves opposite to the
intended conclusion, is rejected.

## Deliverable

`evaluation/mini-rq34/RQ2_LENS_DECISION.md` — the grounded decision with the
delta tables, plus the concrete `results.tex` fix it surfaces.
