---
quick_id: 260627-mot
slug: investigate-whether-paper-rq3-rq4-should
date: 2026-06-27
type: investigation
status: complete
---

# SUMMARY — Should RQ3/RQ4 report the RQ2 size-aware metrics?

**Bottom line (split decision):** **RQ4 — yes**, report worst-component F1 +
harmonic-component F1 (+ coverage): dropping a linker hurts the size-aware
metrics 2.5–3× more than file-F1, consistently across both backends and all 3
runs, which is RQ2's own thesis shown from inside the approach. **RQ3 — no**,
keep TP-killed / FP-removed + macro-ΔF1: validators are a *precision* mechanism,
so the *retained* RQ2 metrics move against them (coverage and harmonic-mean both
**improve** when validators are removed) or **flip sign across backends**
(worst-component F1: −0.077 claude vs +0.011 openai). The one RQ2-family metric
that captures the validator benefit is noise rate, which the paper already
dropped from the suite.

## How it was grounded (real runs)

`mini-rq34/rq34.py` rebuilds the RQ3/RQ4 variant link sets from the canonical
N=3 `s_linker20_union` phase-cache (claude + openai, 5 projects) and
self-validates every Full-variant tp/fp/fn against the run's `ablation_*.json`:
`validate=OK, 15 checked, 0 mismatches` per backend. `rq34_rq2.py` then composes
each variant to doc-to-code and scores it with the RQ2 panel. Both reproduce on
a fresh run; per-run signs are stable within each backend.

Delta convention: `(variant − Full)`. RQ3 variant removes validators → negative
= validators helped. RQ4 variant is one linker → negative = the dropped linker
helped.

## RQ4 — report them (consistent, amplified)

Run-average Δ vs Full (`−` = the linker helped):

| set | Δ file-F1 | Δ worst-comp | Δ harmonic | Δ coverage |
|---|---|---|---|---|
| EntityOnly (claude) | −0.050 | **−0.125** | **−0.141** | −0.095 |
| EntityOnly (openai) | −0.046 | **−0.120** | **−0.143** | −0.107 |
| CorefOnly (claude) | −0.415 | **−0.626** | **−0.705** | −0.483 |
| CorefOnly (openai) | −0.444 | **−0.440** | **−0.583** | −0.495 |

- **Every cell, every metric, all 6 runs negative** — no sign flips.
- Each linker reads as a ~0.05 file-F1 contributor but a 0.12–0.14 contributor
  on the tail; coref-only collapses the worst component to ~0.07 (a whole
  documented component goes untraced).
- Cheap to adopt: the paper already names linker-only macro-F1 in RQ4 prose and
  already reports a coverage delta. These are standalone single-linker sets, so
  the "leave-one-out is contaminated" caveat does not apply.

## RQ3 — do not report the suite (contradicts the narrative)

Run-average Δ from removing both validators (`−` = validators helped):

| metric | claude | openai | robust? |
|---|---|---|---|
| file-F1 | −0.015 | −0.039 | yes — keep this |
| worst-comp F1 | −0.077 | **+0.011** | **no — flips across backends** (same flip every run) |
| harmonic-comp F1 | +0.025 | +0.036 | consistent but **wrong way** (validators hurt it) |
| sentence coverage | +0.028 | +0.052 | consistent but **wrong way** (validators hurt it) |
| noise rate | +0.100 | +0.138 | clean — but **dropped from the suite** as "unstable" |

Validators kill spurious links and lose a few gold ones — a precision move. The
retained RQ2 metrics reward recall/reach, so removing validators improves them.
The only size-aware metric that captures the benefit (noise rate) was explicitly
excluded from the suite in `working/sections/metric.tex`. So no *retained* RQ2
metric supports RQ3; reporting the suite there would put a backend-dependent,
self-contradicting result next to the "validators help" conclusion.

## Concrete paper fix this surfaces

`working/sections/results.tex` (≈L103–105) carries an open TODO to assert
*"worst-component F1 drops when the validators are switched off … cost paid in
the tail."* The data makes that true only on claude (−0.077) and **reversed on
openai** (+0.011). As written it would be a non-robust, backend-specific claim.
**Recommend: cut that sentence, or recast RQ3's tail argument around the
consistent precision/noise effect** rather than worst-component F1.

## Recommended follow-ups (not done here)

1. RQ4: add worst-component + harmonic-component F1 (and coverage) for the
   entity-only / coref-only sets to `working/table/rq4-agents.tex` + RQ4 prose.
2. RQ3: drop the worst-component tail line; keep kill/keep counts + macro-ΔF1.

## Artifacts

- Deliverable: `evaluation/mini-rq34/RQ2_LENS_DECISION.md` (grounded decision).
- Data: `evaluation/mini-rq34/reports/rq34_rq2_variants.csv`,
  `rq34_rq2_linkers.csv` (regenerated this run); base RQ3/RQ4 in
  `rq3_variants.csv`, `rq4_variants.csv`.
- No code/prompt changes; no new LLM calls.
