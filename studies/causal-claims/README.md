# Causal-claim error analysis for `paper/sections/results.tex`

`verification/2026-09-17-s126-paper-results-provenance.md` established that every
**number** in `results.tex` regenerates from the committed `s126` CSVs. This
directory checks the other half: the `This is because ...` clauses — the
**mechanism** each number is attributed to.

```bash
python3 studies/causal-claims/audit.py             # full report
python3 studies/causal-claims/audit.py --csv-only  # refresh items.csv only
```

Stdlib only, read-only on `results/`. Every metric comes from
`evaluation/mini-src/metrics.py` and every phase-state reader from
`evaluation/mini-src/rq34.py`, so nothing is re-implemented and no definition can
drift from the paper's own floats. Section 0 of the report is a 16-assertion
reproduction gate against `evaluation/reports/rq34/s126/*.csv`; read nothing below
it unless it passes.

## `items.csv`

One row per `(sentence, component)` pair that is gold, or was predicted by
`\approach`, Artemis or SWATTR in any of the three `terra` runs — 294 rows, 195 of
them gold. Every count in the report is a filter over this table, so each finding
is checkable by hand.

| column | meaning |
|---|---|
| `gold` | in the doc-model gold standard |
| `approach_runs` / `noknow_runs` / `artemis_runs` | in how many of the 3 runs (0–3) |
| `approach_forms` | which proposal form(s) emitted it |
| `swattr` | in the single-shot SWATTR dump |
| `surface` | how the sentence refers to the component (see below) |
| `matched` | the surface string that fired |

`surface` is a property of the **document**, decided by the first rule that fires:

- `canonical` — the sentence writes the catalog name as a contiguous token run
- `alias` — it writes a document alias bound to that component
- `partial_d` — it writes a *distinctive* token of the name
- `partial_g` — it writes only a *generic* head noun of the name (`client`, `server`)
- `none` — no token of the name occurs at all

`alias` is the one category that consults the run's own knowledge table, so it is
downstream of the approach. The `canonical` / `partial` / `none` split is not.

Findings and verdicts: `verification/2026-09-18-results-causal-claims-error-analysis.md`.
