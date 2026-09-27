# RQ1 metric subheader, 2026-09-24

## Configuration

The `s126` RQ1 generator renders six metric columns under the Doc-model and
Doc-code panels. Each column now has the subheader `P/R; \fone/\ftwo` beneath
its system name. The generated score cells remain on one line, with the
existing `\footnotesize` font, 2 pt table-local column padding, and 4 pt
panel gap.

## Commands and text results

Run from the repository root:

```text
$ python3 evaluation/mini-src/csv_to_tex.py
[csv2tex] wrote .../evaluation/reports/tex/rq1-results.tex
[csv2tex] 12 tables written under .../evaluation/reports/tex, 1 skipped (no source CSV for arm s126)

$ cp evaluation/reports/tex/rq1-results.tex paper/table/rq1-results.tex
(no output; exit 0)

$ python3 verification/verify_rq1_side_by_side.py
PASS RQ1: 6 project rows, 144 values copied from evaluation CSV; TeX panels, P/R; F1/F2 subheaders, and bold marks match
PASS gold: 10 Gini cells use .xx; other 55 numeric cells match parsed CSV

$ python3 verification/estimate_rq1_width.py /tmp/ardoco-LinLibertine_R.otf /tmp/ardoco-LinLibertine_RB.otf
ESTIMATE RQ1 width 378.7 pt / 395.8 pt available (17.1 pt spare); font and scores stay at paper size

$ python3 evaluation/mini-src/check.py
PASS: mini-src/metrics.py reproduces the frozen golden panel (10 cells, sad-code + sad-sam).

$ cmp -s evaluation/reports/tex/rq1-results.tex paper/table/rq1-results.tex
(no output; exit 0)

$ git diff --check -- evaluation/mini-src/csv_to_tex.py evaluation/reports/tex/rq1-results.tex
(no output; exit 0)
$ git -C paper diff --check -- table/rq1-results.tex
(no output; exit 0)
```

The width script includes the new subheader text in its font-metric estimate.
No TeX compiler is installed here, so PDF width remains unmeasured.
