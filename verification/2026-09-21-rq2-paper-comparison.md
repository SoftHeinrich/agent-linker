# RQ2 paper comparison insertion

## Change

- Added `paper/table/rq2-wide-comparison.tex` as a second RQ2 float.
- Included it immediately after `table/rq2-results` in `paper/sections/results.tex`.
- Kept the current RQ2 table in place so both layouts can be compared in one build.
- Staged only the new input line from the already-modified results section; the author's other worktree edits remain unstaged.

## Configuration

- Paper template: `acmsmall,screen,review,anonymous` from `paper/main.tex`.
- Alternative table font: `\small`, matching the current RQ2 table.
- F scores: two decimal places; CMR: one decimal place in parentheses beside doc-model link F1/F2.
- Source values: `evaluation/reports/tex_src/bigtable_rq12_perproject.csv`.
- Compiler used for this verification: Tectonic 0.15.0 with its default TeX bundle.

## Commands and results

```text
$ (cd paper && tectonic -Z shell-escape-cwd=. main.tex)
(exit 0; wrote paper/main.pdf, 19 pages)

$ for page_num in $(seq 1 19); do
    gs ... -dFirstPage="$page_num" -dLastPage="$page_num" \
      -sDEVICE=txtwrite -sOutputFile=- paper/main.pdf |
      rg -q 'Alternative RQ2|RQ2 size-aware' && echo "page $page_num"
  done
page 13

$ git -C paper diff --cached --check
(exit 0)
```

Visual inspection of page 13 confirms that the current macro RQ2 table appears directly above the alternative per-project table. The alternative fits within the text width. The full build retains pre-existing overfull boxes in `sections/metric.tex` and `sections/eval.tex`; it reports no overfull or underfull box in `table/rq2-wide-comparison.tex`.

The rendered comparison page is `verification/rq2-paper-comparison-page13.png`.
