# RQ2 side-by-side layout test

Updated: 2026-09-21

## Configuration

- Source: `evaluation/reports/tex_src/bigtable_rq12_perproject.csv`.
- GPT-5.6-terra body arm: ArchLinker and Artemis mean of three runs; deterministic SWATTR and TransArc are treated as one pipeline, with the stage named under its respective task.
- Five projects plus the five-project Average; precision and recall are omitted. F scores use two decimal places; CMR remains a percentage with one decimal place.
- The body has two large task columns. Each contains a metric-by-approach matrix with explicit metric and approach headers. Doc-model places CMR on the link-score row and formats each approach cell as `F1/F2 (CMR%)`; doc-code uses separate link, worst-component, and harmonic-component F1/F2 rows.
- The deterministic pipeline is represented by SWATTR under doc-model and TransArc under doc-code.
- The LaTeX fragment uses `tabular*{\linewidth}`, `\footnotesize`, 2 pt `\tabcolsep`, and nested fixed-width matrices. Doc-model receives 38% and doc-code 43% of `\linewidth`. Its wrapper uses the paper's exact `acmsmall,screen,review,anonymous` class options and top-matter settings and imports `paper/abbrev.tex`.
- The visual preview uses a conservative five-inch table width, DejaVu Serif 6.4 pt body text, and 300 dpi raster output. It is a layout approximation, not a LaTeX render.

## Commands and text results

An initial same-row form retained a separate `Link F1/F2 (CMR%)` metric column beside the three approach values. The ACM build rejected that width:

```text
Overfull \hbox (30.73558pt--38.17558pt too wide) in all six doc-model rows
```

The accepted form moves the common format label into the doc-model header and leaves three approach values in each body row.

```text
$ python3 verification/rq2-wide-layout-test.py
PASS: 6 project rows x 2 task matrices x 3 approaches; all F scores have two decimals; five-inch preview 1500x960 px; no text extends past page width
Wrote verification/rq2-wide-layout-test.tex
Wrote verification/rq2-wide-layout-test.png
Wrote verification/rq2-wide-layout-test.pdf

$ python3 -m py_compile verification/rq2-wide-layout-test.py
(exit 0)

$ /tmp/rq2-tectonic.grZsvV/tectonic -Z shell-escape-cwd=. rq2-wide-layout-test-document.tex
(exit 0; wrote verification/rq2-wide-layout-test-document.pdf)

$ rg 'Overfull|Underfull' /tmp/rq2-tectonic-build.log
(exit 1; no layout warnings)

$ gs -q -dSAFER -dBATCH -dNOPAUSE -sDEVICE=pngalpha -r180 \
    -sOutputFile=rq2-wide-layout-test-template-%d.png \
    rq2-wide-layout-test-document.pdf
(exit 0; rendered both ACM document pages for inspection)
```

The conservative PNG mock-up and the ACM-rendered page were visually inspected. All project, metric, method, and value labels remain inside the table, and the table uses the available text width. The ACM build reports no overfull or underfull boxes. The wrapper uses the same document class, options, top-matter settings, acronym definitions, and system-name macros as `paper/main.tex`; it omits unrelated paper packages and sections.

The test is isolated from `paper/table/rq2-results.tex` and does not change the submitted table.
