# mini-src — the trace-link metrics

A single self-contained, stdlib-only module (`metrics.py`) that computes **the
paper's metrics** for doc-to-code (`sad-code`) and doc-to-model (`sad-sam`)
trace-link recovery.

This is now the project's **sole** metrics implementation, and the shared core
of the whole `evaluation/` tree: the benchmark layout, the confusion matrix and
F-measures, the gold loaders and the dict-row CSV writer live here once, and
`rq12.py`, `build_dump.py`, `mini-rq34/`, `mini-inequality/` and `../studies/`
all import them. Nothing outside it re-derives an F-measure. The former canonical
stack (`src/lib/metrics_api.py`, `src/bias/component_suite.py`, the RQ2/bias
side-analyses, and the `generate_tables.py` table pipeline) has been retired —
it is preserved on the `master` (legacy) branch (`git show master:<path>`), not
on this `mini` branch. The redundancy analysis that justified the reduction
(legacy `reports/RQ2_METRIC_REDUNDANCY.md`) showed the dropped columns carried
no independent ranking signal (Spearman ρ ≥ 0.85 with a kept metric, ~0
system-pair reversals).

## What it computes

| Task       | Metrics                                                                |
|------------|-----------------------------------------------------------------------|
| `sad-code` | file P/R/F1+F2; per-component F1+F2 (micro); **worst-component F1+F2**; **harmonic-mean component F1+F2** |
| `sad-sam`  | link P/R/F1+F2; **component miss rate/count** (CMR%/CMC) |

**Every F1 ships with its F2.** F2 is recall-weighted (a missed gold link costs
4× a spurious one), which is the asymmetry that matters for recovered links: a
reader can discard a wrong link but cannot see one that was never proposed. The
paper reports the pair in every table, so each F1 key here has an `_f2` twin
computed by the same aggregation — for the tail metrics that means the min and
the harmonic mean *of the per-component F2s*, not the F2 of the worst-F1
component.

The **size-aware suite** (worst-component + harmonic mean) is the paper's
`metric.tex` headline: it weights each architecture component equally rather
than each link pair, so a system that abandons one documented component scores
0 even when file-level F1 stays high. Both flavours zero out on the same
abandoned component. `sad-sam` keeps no per-component view — without enrolment it
collapses onto link F1 (ρ = +1.00).

Dropped as redundant (appendix-only): sentence F1, decision F1, weighted F1,
MCC, MAP, ACF1, NDG, HUS, and the per-component *macro* mean (washes the tail
back out; ρ ≈ 0.85–0.96 with file F1).

### Definitions

- **per-component grouping (D-01):** each `(sentence, file)` link contributes one
  `(sentence, component)` pair per SAM-CODE component that owns the file; files
  mapping to no component are dropped (same rule for gold and result).
- **interface drop (D-12):** `Interface:` model elements are excluded from the
  file→component map. Every interface shares its code extent with a `Component:`
  twin (0 interface-only files) and the doc-to-model gold never links a sentence
  to an interface, so interfaces add no unique code and no documentation signal;
  dropping them makes the component count the distinct architectural units
  (7/10/6/9/6) and removes a per-component distortion on MediaStore/TeaStore. The
  tail metrics are invariant to it (they don't change on any project).
- **per-component F1 (micro):** one P/R/F1 over all `(sentence, component)` pairs;
  `component_f2` is F2 of that same P/R pair.
- **worst-component F1/F2:** minimum per-component F1 (resp. F2) over a project's
  gold components — one abandoned component drives either to 0.
- **harmonic-mean component F1/F2:** harmonic mean of per-component F1 (resp. F2)
  over gold components; also 0 if any component is missed.
- **component miss rate (CMR%) / count (CMC):** the doc-model size-aware metric —
  share of gold (sentence, component) assignments whose component recovers no
  correct link, and the integer count of such components. Reported in **percent**
  (the paper's `\cmr` is the same quantity as a share in [0,1]); named
  *silent-failure mass / SFM* until 2026-08-27.

> A per-sentence **noise rate** (mean FP/(TP+FP) over predicted sentences) was
> reported here until 2026-08-27. It was dropped: the paper never defined it, and
> it is not 1 − precision, so the two figures could not be reconciled by a reader.

## Usage

```bash
# All five projects, bundled TransArc results:
python3 mini-src/metrics.py --task sad-code
python3 mini-src/metrics.py --task sad-sam

# One project, also dump a CSV:
python3 mini-src/metrics.py --task sad-code --project jabref --csv /tmp/panel.csv

# Score arbitrarily-named result CSVs (column dialect auto-detected):
python3 mini-src/metrics.py --task sad-code \
    --results-dir /path/to/run \
    --result-pattern 's_linker21_{project}_links.csv'
```

Benchmark and result roots default to the bundled tree and can be overridden via
`$TRANSARC_BENCHMARK` / `$TRANSARC_RESULTS_DIR` or the `--results-dir` flag.

## Verification

`check.py` is a self-contained regression: it scores the bundled TransArc
results with `metrics.py` and asserts the panel reproduces a frozen golden table
to 1e-4. The goldens were validated at retirement against the then-canonical
`metrics_api` (primary panel) and the interface-dropped `component_suite`
(`worst_component_f1` == `min_comp`).

```bash
python3 mini-src/check.py     # → PASS
```

Bundled TransArc headline averages: sad-code file F1 .80 (F2 .78) / comp F1 .82
(F2 .77) / worst-comp F1 .54 (F2 .51) / harmonic F1 .67 (F2 .65) / cov .75;
sad-sam link F1 .80 (F2 .78) / cov .79 / CMR 7.1%.
