# explore-tail — tail/coverage metric family vs link-F1 correlation

> **Naming (2026-09-01).** This study says *Silent-Failure Mass (SFM)* and
> *Silent-Failure Count (SFC)*; the shipped metric is the mass form, renamed
> **component miss rate (CMR)** on 2026-08-27. The count form was shipped beside it
> as CMC and dropped on 2026-09-01 — no table or engine read it — so the suite now
> reports the rate only. `sfm_vs_worst.py` calls `metrics.compute_sad_sam` for CMR;
> the other scripts here still re-derive their own candidate metrics, which is the
> point of the exploration (`abandmass_*` matches CMR exactly on `sad-sam`).

Exploration (not a frozen study): sweep a broad family of **tail** and **coverage**
summaries over the per-component F1 distribution and measure each one's Spearman
correlation to the reference **link/file F1**, across 30 (system, project) cells
(6 systems × 5 projects; aalinker = mean of 3 runs, baselines single-shot), for
both `sad-sam` (doc-model) and `sad-code` (doc-code).

Run: `python3 studies/explore-tail/explore.py` → `reports/{cells,corr}_{sad-sam,sad-code}.csv`.
Stdlib only; reuses `evaluation/mini-src/metrics.py` loaders + `metrics.spearman`.
Roots (scorer, phase-state reader, link dump) come from `_roots.py`; override the
dump with `$SOTA_LINKS`.

## Question

The paper's headline tail pair (worst-comp F1, harmonic-comp F1) correlates with
link-F1 at .67/.70 (doc-code) and .79/.83 (doc-model) — too high to claim "adds
info F1 doesn't." Is any other tail/coverage summary **more independent** of
link-F1 while still ranking the approach first?

## Finding — independence lives in COVERAGE-COUNTING, not tail-magnitude

Spearman |rho| vs link-F1, lowest (most independent) first:

| metric | rho doc-model | rho doc-code | family |
|--------|--------------:|-------------:|--------|
| **component coverage** (frac comps w/ ≥1 correct) | **0.51** | **0.41** | coverage |
| **# components missed** (count, silent failures) | **0.53** | **0.43** | coverage |
| frac comps F1≥0.5 | 0.59 | 0.66 | coverage |
| neg coeff-of-variation | 0.82 | 0.70 | spread |
| worst-comp F1 (min) | 0.85 | 0.71 | tail-mag |
| harmonic / geomean / median / CVaR25 / bottom2 | .85–.91 | .73–.81 | tail-mag |
| macro-mean F1 | 0.94 | 0.78 | mean |

- **All tail-MAGNITUDE summaries cluster at .71–.92** — worst, harmonic, geomean,
  CVaR-25%, median, bottom-2, second-worst. They re-rank by F1 magnitude and add
  little beyond link-F1. (Confirms the metric.tex footnote: geomean ≈ harmonic,
  CVaR confirms ranking with smaller margin — none is more independent.)
- **The orthogonal axis is "reach":** component coverage / silent-failure COUNT
  (rho .41–.53 both tasks) — *whether* each component is hit, not *how well*.

## Saturation caveat (matches the metric.tex drop rationale)

`component coverage` per system (macro):

| | S21 GPT | s20u | S21 Claude | Artemis | TransArC |
|--|--:|--:|--:|--:|--:|
| doc-model | 1.000 | .994 | .975 | **.917** | **.940** |
| doc-code  | 1.000 | .993 | .980 | **.947** | **.956** |

- **doc-code: near-saturated** (every system ≥ .947, ~5pp spread) → weak ranker,
  exactly why the paper dropped component coverage. Holds under canonical s21.
- **doc-model: not saturated** — baselines drop to .917/.940 while the approaches
  sit ≥ .975. The count form is cleaner: **S21 GPT misses ~0 components/project,
  both baselines miss ~0.6** (silent component failure).

## Takeaway

If the aim is a metric that adds signal link-F1 lacks, it is **coverage-counting
(silent-failure count / component coverage), not another tail-magnitude variant** —
and it is defensible only on **doc-model**, where it is not saturated. On doc-code
it is saturated (paper correctly dropped it). worst/harmonic remain magnitude
re-rankers (.7–.9 corr) on both tasks.

## Sharpness vs independence frontier (`sharpness.py`, doc-model)

Coverage *fractions* are independent (rho .51) but flat (.92–1.0) — not sharp.
The frontier: on doc-model **no bounded [0,1] metric is both sharp and
independent**. Sharp scores (worst-F1, harmonic, worst-recall, strict-cov; spread
.3–.6) all correlate .85–.90 with link-F1. Independent scores (coverage fractions)
sit at .92–1.0.

The escape is to **stop normalizing — COUNT the silent failures**:

| system | MS | TS | TM | BBB | JR | total abandoned / 40 comps |
|--------|--:|--:|--:|--:|--:|--:|
| S21 GPT (approach) | 0 | 0 | 0 | 0 | 0 | **0** |
| Artemis | 0 | 0 | 1 | 1 | 1 | **3** |
| TransArC | 3 | 0 | 0 | 0 | 0 | **3** |

`0 vs 3` is categorical and independent (rho .53). It works *only* as a count: the
abandoned components are the SMALL ones (link-mass-weighted `abandoned_link_mass`
goes flat, .93–1.0, rho .51), so any fraction divides the rare misses back into the
0.9x band. Harsher bounded bars (recall≥.8 cov, strict full-recovery) sharpen to
spread .27–.37 but rho climbs to .90 — sharpness bought by converging onto link-F1.
