# RQ1 task panels, 2026-09-23

## Configuration

- Body arm `s126`, GPT-5.6-terra. ArchLinker and ArTEMiS use their three-run means; SWATTR and TransArC use the recorded single pipeline run, as in the prior RQ1 table.
- `rq_tables.py` copies the two task rows for each project from `rq1_transposed.csv` into one `rq1_side_by_side.csv` row. `csv_to_tex.py` renders its 24 score values as six columns: three under Doc-model and three under Doc-code. The table has five project rows and one macro-average row.
- The paper's `acmsmall` template and `\footnotesize` table font are unchanged. Each score cell prints `P/R;F1/F2` on one line. RQ1 alone uses 2 pt `\tabcolsep` and a 4 pt gap between task panels to keep the inline cells within the page width. The project codes were introduced in the generated gold table.

## Commands and text results

Run from the repository root:

```text
$ python3 evaluation/mini-src/rq_tables.py
[rq_tables] wrote .../evaluation/reports/tex_src/rq1_transposed.csv
[rq_tables] wrote .../evaluation/reports/tex_src/rq1_side_by_side.csv
[rq_tables] table CSVs written under .../evaluation/reports/tex_src

$ python3 evaluation/mini-src/csv_to_tex.py
[csv2tex] wrote .../evaluation/reports/tex/rq1-results.tex
[csv2tex] 12 tables written under .../evaluation/reports/tex, 1 skipped (no source CSV for arm s126)

$ cp evaluation/reports/tex/rq1-results.tex paper/table/rq1-results.tex
(no output; exit 0)
$ cp evaluation/reports/tex_src/rq1_side_by_side.csv paper/table/rq1-results.csv
(no output; exit 0)

$ python3 verification/verify_rq1_side_by_side.py
PASS RQ1: 6 project rows, 144 values copied from evaluation CSV; TeX panels and bold marks match
PASS gold: 10 Gini cells use .xx; other 55 numeric cells match parsed CSV

$ python3 evaluation/mini-src/check.py
PASS: mini-src/metrics.py reproduces the frozen golden panel (10 cells, sad-code + sad-sam).

$ python3 evaluation/mini-src/sync_paper.py --check --only gold paper
IN SYNC: all 2 paper file(s) match the generated output.

$ cmp -s evaluation/reports/tex/rq1-results.tex paper/table/rq1-results.tex
(no output; exit 0)
$ cmp -s evaluation/reports/tex_src/rq1_side_by_side.csv paper/table/rq1-results.csv
(no output; exit 0)

$ git diff --check -- evaluation/mini-src/rq_tables.py evaluation/mini-src/csv_to_tex.py evaluation/HOWTO-REGENERATE-RQ.md evaluation/reports/tex/rq1-results.tex
(no output; exit 0)
$ git -C paper diff --check -- table/rq1-results.tex table/rq1-results.csv
(no output; exit 0)

$ curl -fsSL https://mirrors.mit.edu/CTAN/fonts/libertine/opentype/LinLibertine_R.otf -o /tmp/ardoco-LinLibertine_R.otf
(no output; exit 0)
$ curl -fsSL https://mirrors.mit.edu/CTAN/fonts/libertine/opentype/LinLibertine_RB.otf -o /tmp/ardoco-LinLibertine_RB.otf
(no output; exit 0)
$ python3 verification/estimate_rq1_width.py /tmp/ardoco-LinLibertine_R.otf /tmp/ardoco-LinLibertine_RB.otf
ESTIMATE RQ1 width 378.7 pt / 395.8 pt available (17.1 pt spare); font and scores stay at paper size
```

## Width check and limits

The [acmart `acmsmall` geometry](https://mirrors.ctan.org/macros/latex/contrib/acmart/acmart.pdf#page=52) specifies 6.75 in paper width and 46 pt left and right margins, giving approximately 395.8 pt of text width. [ACM's class uses Libertine](https://mirrors.ctan.org/macros/latex/contrib/acmart/acmart.pdf#page=56); the 2017 Libertine regular and bold OTF files were read from the [CTAN font directory](https://mirrors.mit.edu/CTAN/fonts/libertine/opentype/). A font-metric estimate at 8.25 pt, including the displayed bold spans, 2 pt `\tabcolsep`, and the 4 pt panel gap, yielded 378.7 pt for the table body. This leaves about 17.1 pt in the estimate. The first inline rendering with the prior thin spaces around the semicolon and wider local padding exceeded the available width, so the displayed cells use a compact semicolon and the smaller table-local spacing.

There is no TeX compiler (`latexmk`, `pdflatex`, or `tectonic`) installed, so the PDF width was not directly measured. The full `sync_paper.py --check paper` command still exits 1 on an unrelated pre-existing wording difference in `paper/appendix/big-table-perrun.tex` (“component-level measures” in paper versus “component tail” in the generator). A full `git -C paper diff --check` also reports trailing whitespace in a concurrent edit to `sections/results.tex:59`; the requested table files pass the scoped whitespace check above. Neither unrelated file was changed for this task.
