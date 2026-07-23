# mini-src — the trace-link metrics

A single self-contained, stdlib-only module (`metrics.py`, ~450 lines, zero
cross-module imports) that computes **the paper's metrics** for doc-to-code
(`sad-code`) and doc-to-model (`sad-sam`) trace-link recovery.

This is now the project's **sole** metrics implementation. The former canonical
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
| `sad-code` | file P/R/F1; per-component F1 (micro); **worst-component F1**; **harmonic-mean component F1**; sentence coverage; noise rate |
| `sad-sam`  | link P/R/F1; sentence coverage; noise rate                            |

The **size-aware suite** (worst-component F1 + harmonic mean) is the paper's
`metric.tex` headline: it weights each architecture component equally rather
than each link pair, so a system that abandons one documented component scores
0 even when file-level F1 stays high. `sad-sam` keeps no per-component view —
without enrolment it collapses onto link F1 (ρ = +1.00).

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
- **per-component F1 (micro):** one P/R/F1 over all `(sentence, component)` pairs.
- **worst-component F1:** minimum per-component F1 over a project's gold
  components — one abandoned component drives it to 0.
- **harmonic-mean component F1:** harmonic mean of per-component F1 over gold
  components; also 0 if any component is missed.
- **sentence coverage:** fraction of gold sentences with ≥1 *correct* hit.
- **noise rate:** mean over *predicted* sentences of FP/(TP+FP); lower is better.

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

Bundled TransArc headline averages: sad-code file F1 .80 / comp F1 .82 /
worst-comp .54 / harmonic .67 / cov .75 / noise .13; sad-sam link F1 .80 /
cov .79 / noise .14.
