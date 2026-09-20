# Table 2 (tab:rq1) row-wise winner bolding verification

Date: 2026-09-19

## Change

`csv_to_tex.py` gained a second bolding rule beside the existing column-wise
`extrema`: `row_bold`, for tables whose systems are the columns. Nothing is
hardcoded -- the winners are an argmax computed at render time from
`rq1_transposed.csv`, and with `"row_bold": "by_position"` the comparison groups
themselves are derived from the spec's columns (`position_groups()`): every
column prints the same metric tuple in the same order, so the fields sharing a
position are one metric's competitors. Every field holding the group's best
*printed* value in that row is bolded, ties included. For `tab:rq1` (Table 2 in
the built PDF) this yields one group per metric -- precision, recall, F1, F2 --
so each project/task row and the Average row mark their own winner per metric,
and adding or dropping a system column re-derives the groups with no spec edit.

- Comparison is on the rounded value the reader sees, so two systems differing
  only below the third decimal are both bolded rather than one carrying an
  invisible lead.
- The Average row previously bolded every number as row emphasis, which would
  hide the winner marks; the spec now sets `summary_bold_values: False`, so its
  numbers follow the same winner rule and only the row labels stay bold.
- `check_specs()` rejects a `row_bold` group naming a field the table does not
  render (silent no-op otherwise).

## Commands and results

```text
$ python3 -m py_compile evaluation/mini-src/csv_to_tex.py
(ok)

$ python3 evaluation/mini-src/csv_to_tex.py
[csv2tex] 12 tables written under .../evaluation/reports/tex, 1 skipped (no source CSV for arm s126)

$ git status --porcelain evaluation/reports/tex
 M evaluation/reports/tex/rq1-results.tex          # the other 11 tables re-render byte-identical

$ python3 <static assertions over rq1_transposed.csv + rq1-results.tex>
[1] per-metric row winners: 12 rows x 3 systems x 4 metrics -- PASS
[2] mediastore doc-model: SWATTR precision, approach F1, Artemis recall+F2 -- PASS
[3] teastore doc-model 1.000 precision tie: approach and SWATTR both bolded -- PASS
[4] Average row: labels bold, values bold only when they win -- PASS
[5] row_winners: empty cells skipped, min mode, printed-tie, no bolded '--' -- PASS
[6] check_specs rejects a typo'd row_bold field -- PASS
[7] groups derived from the rendered columns; no system/metric list in the spec -- PASS
[8] column-shape guards -- PASS (ragged column set rejected)

$ PAPER_DIR=$PWD/paper python3 evaluation/mini-src/sync_paper.py --only rq paper
synced 24 file(s) into paper (2 absent for this arm)

$ PAPER_DIR=$PWD/paper python3 evaluation/mini-src/sync_paper.py --only rq --check paper
IN SYNC: all 24 paper file(s) match the generated output. (2 absent for this arm)

$ python3 mini-src/check.py
PASS: mini-src/metrics.py reproduces the frozen golden panel (10 cells, sad-code + sad-sam).

$ ./scripts/build-paper.sh
latexmk is required to build the paper (install TeX Live with latexmk).
```

Assertion [1] recomputes the winners from `rq1_transposed.csv` independently of
the renderer and compares them cell by cell against the `\textbf{}` marks in the
rendered `.tex`, for all 144 numbers.

## Measured winners, as now bolded

| Row | Prec. | Rec. | F1 | F2 |
|-----|-------|------|----|----|
| mediastore doc-model | SWATTR | Artemis | approach | Artemis |
| mediastore doc-code | TransArC | Artemis | Artemis | Artemis |
| teastore doc-model | approach, SWATTR (tie, 1.000) | approach | approach | approach |
| teastore doc-code | approach, TransArC (tie, 1.000) | approach | approach | approach |
| teammates doc-model | Artemis | approach | approach | approach |
| teammates doc-code | Artemis | TransArC | TransArC | TransArC |
| bigbluebutton doc-model | SWATTR | approach | approach | approach |
| bigbluebutton doc-code | TransArC | approach | TransArC | approach |
| jabref doc-model | approach | approach, SWATTR (tie, 1.000) | approach | approach |
| jabref doc-code | approach | approach, TransArC (tie, 1.000) | approach | approach |
| **Average doc-model** | approach | approach | approach | approach |
| **Average doc-code** | TransArC | approach | approach | approach |

The SWATTR$\rightarrow$TransArC column is SWATTR on the doc-model rows and
TransArC on the doc-code rows.

## Blocker

Visual PDF validation is blocked by the missing TeX toolchain (no `latexmk`),
not by table generation -- the same blocker recorded in
`rq1-table-redesign-20260913.md`.
