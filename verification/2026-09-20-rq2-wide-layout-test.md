# RQ2 side-by-side layout test

## Configuration

- Source: `evaluation/reports/tex_src/bigtable_rq12_perproject.csv`.
- GPT-5.6-terra body arm: ArchLinker and Artemis mean of three runs; deterministic SWATTR and TransArc are treated as one pipeline, with the stage named under its respective task.
- Five projects plus the five-project Average; precision and recall are omitted.
- Two task groups side by side, three approach columns per group. Each doc-model cell shows link F1/F2 and CMR%; each doc-code cell shows link, worst-component, and harmonic-component F1/F2, in that order.
- The LaTeX fragment uses `tabular*{\linewidth}`, `\scriptsize`, and 2 pt `\tabcolsep`. The visual preview uses a conservative five-inch table width, DejaVu Serif 7 pt body text, and 300 dpi raster output. It is a layout approximation, not a LaTeX render.

## Commands and text results

```text
$ python3 verification/rq2-wide-layout-test.py
PASS: 6 project rows x 6 approach cells; five-inch preview 1500x1035 px; no text extends past page width
Wrote verification/rq2-wide-layout-test.tex
Wrote verification/rq2-wide-layout-test.png
Wrote verification/rq2-wide-layout-test.pdf

$ python3 -m py_compile verification/rq2-wide-layout-test.py
(exit 0)

$ command -v latexmk
(exit 1; no output)

$ command -v pdflatex
(exit 1; no output)
```

The PNG was visually inspected: all project and method labels and values remain inside their columns at the five-inch preview width. The native ACM LaTeX fit has not been verified because this environment has no TeX compiler. `rq2-wide-layout-test-document.tex` is the paper-class wrapper for that check when the toolchain is available.

The test is isolated from `paper/table/rq2-results.tex` and does not change the submitted table.
