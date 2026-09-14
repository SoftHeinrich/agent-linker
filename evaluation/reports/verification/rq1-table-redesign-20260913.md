# RQ1 per-project table redesign verification

Date: 2026-09-13

## Configuration

- Reported backend: GPT-5.6-terra.
- Table 1 rows: five projects plus Average, each with a doc-model and doc-code
  row; systems are columns.
- Visible system subheaders: `Prec./Rec.; F1/F2`.
- Numeric precision: three decimal places throughout.
- Table 2 remains the original four-row macro RQ2 size-aware suite.

## Commands and results

```text
$ python3 evaluation/mini-src/rq_tables.py
[rq_tables] table CSVs written under .../evaluation/reports/tex_src

$ python3 evaluation/mini-src/csv_to_tex.py
[csv2tex] wrote .../evaluation/reports/tex/rq1-results.tex
[csv2tex] wrote .../evaluation/reports/tex/rq2-results.tex

$ python3 evaluation/mini-src/sync_paper.py --only rq paper
synced 24 file(s) into paper (2 absent for this arm)

$ python3 -m py_compile evaluation/mini-src/rq_tables.py evaluation/mini-src/csv_to_tex.py

$ python3 <Table 1 and Table 2 static assertions>
Table verification passed: Table 1 has 12 transposed project/task rows, visible metric subheaders, and three-decimal values; Table 2 has 4 macro rows.

$ ./scripts/build-paper.sh
latexmk is required to build the paper (install TeX Live with latexmk).
```

The evaluation-to-paper generation path and structural checks pass. Visual PDF
validation is blocked by the missing TeX toolchain, not by table generation.
