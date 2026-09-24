# RQ1 compact task rows, 2026-09-24

## Configuration

The `s126` GPT-5.6-terra RQ1 table is generated from
`evaluation/reports/tex_src/rq1_transposed.csv`. Each of the five projects and
the project average has one printed row for doc-model and one for doc-code.
Precision and recall retain sample SD from three runs, displayed to two
decimals on the score scale. The source CSV values are unchanged.

## Commands and text results

Run from the repository root:

```text
$ python3 evaluation/mini-src/csv_to_tex.py
[csv2tex] wrote .../evaluation/reports/tex/rq1-results.tex
[csv2tex] 13 tables written under .../evaluation/reports/tex, 1 skipped (no source CSV for arm s126)

$ cp evaluation/reports/tex/rq1-results.tex paper/table/rq1-results.tex
(no output; exit 0)

$ python3 verification/verify_rq1_task_rows.py
PASS RQ1: 12 single-line task rows; 36 system cells match source scores and two-decimal SD

$ python3 evaluation/mini-src/check.py
PASS: mini-src/metrics.py reproduces the frozen golden panel (10 cells, sad-code + sad-sam).

$ PAPER_DIR=$PWD/paper python3 evaluation/mini-src/sync_paper.py --check
absent for this arm: rq4-floor.tex
absent for this arm: rq4_floor.csv
IN SYNC: all 28 paper file(s) match the generated output. (2 absent for this arm)

$ git diff --check -- evaluation/mini-src/csv_to_tex.py evaluation/reports/tex/rq1-results.tex verification/verify_rq1_task_rows.py
(no output; exit 0)

$ git -C paper diff --check -- table/rq1-results.tex
(no output; exit 0)

$ bash scripts/build-paper.sh
latexmk is required to build the paper (install TeX Live with latexmk).
(exit 1)
```

The generated TeX and paper copy match byte for byte. A PDF layout check
remains blocked because `latexmk` is absent in this environment.
