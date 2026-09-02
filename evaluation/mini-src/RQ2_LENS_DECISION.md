# Should RQ3/RQ4 report the RQ2 size-aware metrics? — grounded decision

**Question.** For the paper, is it better to report RQ3 (validator contribution)
and RQ4 (module ablation) with the RQ2 size-aware suite — sentence coverage,
worst-component F1, harmonic-component F1 — instead of (or on top of) plain
macro-F1 and TP/FP counts?

**How this was decided.** Not by argument — by running the real pipeline.
`rq34.py` rebuilds the RQ3/RQ4 variant link sets from the canonical N=3
`s_linker20_union` phase-cache (claude + openai backends, 5 projects) and
self-validates every Full-variant tp/fp/fn against the run's `ablation_*.json`
(`validate=OK, 15 checked, 0 mismatches`). `rq34_rq2.py` then composes each
variant's SAD-SAM links through recovered SAM-CODE links and scores them with
the RQ2 doc-to-code panel. Both reproduce on a fresh run. All deltas below are
read from `rq34_rq2_variants.csv` and `rq34_rq2_linkers.csv`.

> Frozen record. It was decided on the retired `s_linker20_union` arm and the numbers
> below are that arm's; the *verdict* is what the paper still follows. Those CSVs now
> live at `evaluation/reports/rq34/s21/`, and the current arm's are under
> `evaluation/reports/rq34/s110/`. The engines moved to `evaluation/mini-src/` on
> 2026-09-02.

Reading rule: a delta is `(variant − Full)`. For RQ3 the variant removes the
validators, so a **negative** delta means the validators **helped** that metric.
For RQ4 the variant is a single linker, so a negative delta means the dropped
linker **helped**.

---

## Verdict

| | Report RQ2 metrics? | Why |
|---|---|---|
| **RQ4 — module ablation** | **Yes** | Every size-aware metric moves the same way as file-F1 but 2.5–3× larger, consistently across both backends and all 3 runs. It is the cleanest demonstration of RQ2's own thesis. |
| **RQ3 — validator contribution** | **No** (keep TP-killed / FP-removed + macro-ΔF1) | The retained RQ2 metrics point the *wrong way* or *flip sign across backends*. Only noise rate captures the benefit — and noise was explicitly dropped from the suite as unstable. |

---

## RQ4: yes — the size-aware metrics amplify a consistent story

Effect of dropping one linker (run-average, `−` = the linker helped):

| set | Δ file-F1 | Δ worst-comp F1 | Δ harmonic-comp F1 | Δ coverage |
|---|---|---|---|---|
| EntityOnly (claude) | −0.050 | **−0.125** | **−0.141** | −0.095 |
| EntityOnly (openai) | −0.046 | **−0.120** | **−0.143** | −0.107 |
| CorefOnly (claude) | −0.415 | **−0.626** | **−0.705** | −0.483 |
| CorefOnly (openai) | −0.444 | **−0.440** | **−0.583** | −0.495 |

- **Every cell, every metric, every one of the 6 runs is negative** (no sign
  flips, no exceptions). The contribution is real on the standard ruler and
  larger on the size-aware ones.
- The amplification is the point: the entity linker looks like a +0.05 file-F1
  contributor but a +0.12–0.14 contributor on the tail; coref-only collapses
  the worst component to ~0.07 (a whole documented component goes untraced).
  That is exactly the "size-aware metrics expose what link-F1 hides" claim from
  RQ2, shown from the inside of the approach instead of against baselines.
- Low cost to adopt: the paper already names the linker-only macro-F1 in RQ4
  prose ("the \linkerB-only baseline reaches a macro F1 of 0.xx") and already
  reports a coverage delta in the RQ4 design. Adding worst-component and
  harmonic-component F1 for those same standalone sets needs no new run. These
  are single-linker validated sets, not leave-one-out, so the "LOO is
  contaminated" caveat does not apply.

## RQ3: no — the retained RQ2 metrics contradict the narrative

Effect of removing both validators (run-average, `−` = the validators helped):

| metric | claude | openai | robust? |
|---|---|---|---|
| file-F1 | −0.015 | −0.039 | yes — validators help (this is the metric to keep) |
| worst-comp F1 | **−0.077** | **+0.011** | **no — sign flips across backends** |
| harmonic-comp F1 | +0.025 | +0.036 | consistent but **wrong way** (validators hurt it) |
| sentence coverage | +0.028 | +0.052 | consistent but **wrong way** (validators hurt it) |
| noise rate | +0.100 | +0.138 | yes — validators help — **but dropped from the suite** |

Why this happens: validators are a **precision** mechanism. They kill spurious
links and lose a few gold ones. The retained RQ2 metrics reward **recall /
reach** (coverage, harmonic mean), so removing the validators *improves* them —
on both backends, every run (coverage Δ is +0.022…+0.055 throughout). The one
tail metric that could have helped, worst-component F1, is **not robust**: it
says validators help on claude (−0.077) and that they slightly hurt on openai
(+0.011), with the same flip in every per-run pair. Reporting the suite here
would put a backend-dependent, partly self-contradicting result next to the
"validators help" conclusion.

The only RQ2-family metric that captures the validator benefit cleanly is the
**noise rate** (+0.10–0.14, consistent everywhere) — a precision-side metric.
But `working/sections/metric.tex` explicitly **drops noise rate from the suite**
as "run-to-run unstable and confounded with coverage." So among the *retained*
RQ2 metrics there is none that supports RQ3. Keep RQ3 on its current footing:
the gold-killed / spurious-killed counts and the macro-ΔF1.

## Concrete paper fix this surfaces

`working/sections/results.tex` (≈L103–105) carries an open TODO to fill in:
*"the worst-component F1 column … drops by [delta]pp when the validators are
switched off, so the cost is paid where it matters most: in the tail."* The data
says this is true only on claude (−0.077) and **reverses on openai** (+0.011).
As written it would be a non-robust, backend-specific claim. **Recommend: cut
that sentence, or recast the RQ3 tail argument around the consistent
precision/noise effect** rather than worst-component F1.

---

## One-line answer

Yes for RQ4 (the size-aware metrics consistently amplify each module's
contribution and reinforce RQ2); no for RQ3 (the validators are a precision
story, and the retained size-aware metrics move against them or flip sign — keep
the kill/keep counts and macro-ΔF1, and drop the non-robust worst-component line).
