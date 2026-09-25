# Why teammates' doc-model result does not transfer to doc-code

Date: 2026-09-19 · Arm: s126, GPT-5.6-terra · Reproduce: `python3 analyze.py && python3 analyze_size.py` (output in `REPORT.txt`)

## The observation

On teammates, \approach wins doc-model F1 (.868 vs SWATTR .710) and loses
doc-code F1 (.770 vs TransArC .821). The loss is entirely on recall: doc-code
precision is .795-.800 against TransArC's .753, while doc-code recall is
.742-.746 against .902.

## Measured cause

Teammates is the one project whose doc-code gold is **not** the composition of
its doc-model gold with the model-code map. Composing the gold doc-model links
through the (gold-equal) ARCoTL map yields 6380 file pairs; the doc-code gold
has 8097, so 1717 pairs (21.2%) are gold at the file grain with no gold
(sentence, component) link behind them. They come from 50 sentences that carry
gold doc-code links and no gold doc-model link at all (1664 of the 1717 pairs);
the doc-model gold covers 45 sentences, the doc-code gold 92.

| project | gold doc-model | gold doc-code | closure of gold doc-model | closure ⊆ gold doc-code | gold doc-code beyond the closure |
|---|---|---|---|---|---|
| mediastore | 31 | 59 | 52 | 100.0% | 11.9% (7) |
| teastore | 27 | 707 | 707 | 100.0% | 0.0% |
| **teammates** | **57** | **8097** | **6380** | **100.0%** | **21.2% (1717)** |
| bigbluebutton | 62 | 1529 | 1746 | 84.0% | 4.1% (63) |
| jabref | 18 | 8268 | 8272 | 100.0% | 0.0% |

Splitting doc-code recall on that boundary locates the whole gap:

| system | inside the closure | beyond the closure | doc-code recall |
|---|---|---|---|
| \approach run1 | 93.4% | 4.8% (83/1717) | .746 |
| TransArC | 96.1% | 68.3% (1173/1717) | .902 |

Of the 15.6pp recall gap, **+2.1pp is inside the closure and +13.5pp is beyond
it**. Inside the reachable half the two systems are close; the difference is that
TransArC reaches the unreachable half and \approach does not.

It reaches it through its doc-model errors. SWATTR emits 32 doc-model false
positives; composing them yields 1173 pairs that are gold doc-code links (and
2395 that are not) — sentence 156 → `Common` and sentences 187/188 → `E2E` are
doc-model false positives whose entire file extent (150, 123, 123 files) is gold
doc-code. \approach emits 14 doc-model false positives, and they yield 83 gold
doc-code pairs. The doc-model false positives that pay at the file grain are the
ones on file-heavy components, and SWATTR's land there while \approach's land on
`UI` sentences whose gold file extent is nearly empty (e.g. s50: 2 of 348).

## Is the loss precision, or component size?

Neither, exactly: \approach *leads* doc-code precision on teammates (.798 vs
.753), so no part of the gap is a precision deficit. The gap is recall, and it is
two effects stacked — a real coverage deficit at a size-blind grain, plus a
file-count weighting that amplifies it. `analyze_size.py` separates them.

Per gold component, ordered by how many code files it owns (run 1; per-component
F1 is the `metric.tex` eq:worst slice):

| component | files | gold doc-code | \approach R | TransArC R | ΔR | \approach F1 | TransArC F1 | pairs lost |
|---|---|---|---|---|---|---|---|---|
| UI | 348 | 3622 | 76.9% | 89.9% | -13.0 | .748 | .835 | +471 |
| Common | 150 | 1345 | 55.8% | 94.0% | -38.2 | .716 | .844 | +514 |
| E2E | 123 | 861 | 71.4% | 100.0% | -28.6 | .769 | .824 | +246 |
| Logic | 71 | 1243 | 85.0% | 91.6% | -6.5 | .816 | .832 | +81 |
| Storage | 59 | 746 | 81.0% | 81.0% | +0.0 | .866 | .798 | 0 |
| Client | 40 | 172 | 95.3% | 76.7% | +18.6 | .882 | .537 | -32 |
| Test Driver | 17 | 68 | 100.0% | 75.0% | +25.0 | .889 | .750 | -17 |

The gap is monotone in component size: \approach wins the two smallest
components and loses the four largest, Spearman(size, ΔR) = -0.89. All 1231 lost
pairs beyond `Logic` sit in the three largest components, and both systems link
all seven components at the doc-model grain — so this is not a difference in
*which* components are recovered, it is a difference in cost per missed decision.

Scoring the same two link sets at four grains shows how much of the gap the
file-count weighting carries:

| system | file R | file F1 | (sentence, component) F1 | macro comp. R | worst comp. F1 | harmonic comp. F1 |
|---|---|---|---|---|---|---|
| \approach run1 | .746 | .770 | .636 | .808 | **.716** | **.807** |
| \approach run2 | .744 | .771 | .628 | .805 | **.716** | **.808** |
| \approach run3 | .742 | .770 | .620 | .770 | **.716** | **.789** |
| TransArC | **.902** | **.821** | **.716** | **.869** | .537 | .757 |

Reading down the row: TransArC leads at the file grain (.821 vs .770) and still
leads when every component counts once per sentence (.716 vs .636), so a genuine
coverage deficit is there — those are the beyond-closure sentences. But once
components are weighted equally rather than by file count, the ordering flips:
\approach leads worst-component F1 .716 vs .537 and harmonic-component F1 .807
vs .757, in all three runs. **Teammates is a loss at the file-count-weighted link
grain and a win at the size-aware grain the paper defines in `metric.tex`.**

The mechanism is visible in what each system proposes, per component:

| component | files per link | \approach doc-model links | SWATTR doc-model links | composed file links (\approach / SWATTR) |
|---|---|---|---|---|
| UI | 348 | 11 | 12 | 3828 / 4176 |
| Common | 150 | 5 | 11 | 750 / 1650 |
| E2E | 123 | 6 | 10 | 738 / 1230 |
| Logic | 71 | 19 | 21 | 1349 / 1491 |
| Storage | 59 | 11 | 13 | 649 / 767 |
| Client | 40 | 5 | 8 | 200 / 320 |
| Test Driver | 17 | 5 | 4 | 85 / 68 |

SWATTR emits 81 doc-model links against \approach's 68, but the surplus is not
uniform: it proposes 2.2x as many links on `Common` and 1.7x on `E2E`, the two
components whose *entire* file extent is gold doc-code. Those are exactly the
proposals \approach's judges remove. One doc-model link is worth 112 file links
on average for \approach and 120 for SWATTR.

Across projects the size effect is teammates-specific in sign, not in kind:

| project | gold comps | \approach wins | TransArC wins | median size won | median size lost | Spearman(size, ΔR) |
|---|---|---|---|---|---|---|
| mediastore | 9 | 4 | 0 | 2 | -- | +0.12 |
| teastore | 6 | 4 | 0 | 30 | -- | +0.62 |
| teammates | 7 | 2 | 4 | 40 | 150 | **-0.89** |
| bigbluebutton | 10 | 5 | 1 | 16 | 6 | +0.28 |
| jabref | 6 | 0 | 0 | -- | -- | n/a (ties) |

Pooled over all 38 gold components, Spearman(size, ΔR) = -0.23: teammates is the
only project where \approach's advantage decays with component size.

## Reading

Two independent facts produce the result, and only the second is about our
method:

1. **The two gold standards disagree on teammates.** A system that predicts
   exactly the doc-model gold reaches at most 78.8% doc-code recall there. The
   sentences in the gap are code-level prose — servlet/filter request-flow steps
   (s39-s50), package overviews (s84, s86), configuration files (s75) — that the
   doc-model gold does not annotate but the doc-code gold does. This is a
   property of the benchmark, not of a linker.
2. **The file grain rewards over-proposal where components are large.** One
   (sentence, component) link composes into as many file links as the component
   owns: 348 for `UI`, 150 for `Common`. A doc-model precision error on a large
   component is cheap at the model grain (1 link) and can be worth up to 348 gold
   file links; the same error on a component whose gold file extent is empty
   costs up to 348 false file links. Doc-model precision is therefore not
   monotonically good for transitive doc-code F1 on this project. The
   per-component table above is the direct measurement: the loss is monotone in
   component size (rho = -0.89), it reverses on the two smallest components, and
   it reverses again on the size-aware metrics (worst-component F1 .716 vs .537).

This is consistent with, and a sharper statement of, the size-aware motivation in
`metric.tex`: teammates' link mass is concentrated in a few large components.

## Scope and threats

- N=1 project, one arm, three runs for \approach and a deterministic baseline.
  The closure table shows the disagreement is teammates-specific (mediastore 7
  pairs, bigbluebutton 63, teastore/jabref none), so this explains teammates and
  is not evidence about the approach in general.
- bigbluebutton is the mirror-image case: 16% of the gold doc-model closure is
  *not* in its doc-code gold, so composition there over-produces by construction.
  Not investigated here.
- The analysis attributes recall differences to the closure partition; it does
  not establish what \approach would score if the doc-model gold annotated those
  50 sentences.
- The composition step and the model-code map are identical for both systems
  (ARCoTL's map equals the gold model-code map file-for-file on teammates), so no
  part of the gap is attributable to the mapping stage.
