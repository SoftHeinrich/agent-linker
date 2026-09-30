# Table 3 (tab:inference-cost) number check and caption edit (2026-09-30)

## Number check

Regenerated the source CSV from raw logs into a scratch directory and diffed it
against the committed copy:

```text
$ cd evaluation/mini-src && python3 inference_cost.py --out <scratch>/ic
PASS: 30 project/run usage records; means of three runs written to <scratch>/ic
$ diff <scratch>/ic/tex_src/inference_cost_by_system.csv ../reports/tex_src/inference_cost_by_system.csv
(no output; identical)
```

Every input/output cell in `paper/table/inference-cost.tex` equals the CSV
value rounded to one decimal (e.g. approach Total 98.562/22.691 -> 98.6/22.7;
Artemis Total 20.034/23.368 -> 20.0/23.4).

Cost column, recomputed at US$2/12 per million input/output tokens:

```text
approach: 98.562*2/1e3 + 22.691*12/1e3 = 0.4694 -> 0.47
Artemis : 20.034*2/1e3 + 23.368*12/1e3 = 0.3205 -> 0.32
input ratio 98.562/20.034 = 4.92 -> "4.9x" in results.tex
```

The table and the prose in results.tex and discussion.tex agree with these numbers.

Drift note: `evaluation/reports/tex/inference-cost.tex` (generator output) has
no Cost column or pricing footnote. The paper copy adds them by hand, so
re-running `sync_paper.py` would drop them.

## Caption edit (paper/table/inference-cost.tex, submodule working tree)

Old: `Recorded inference usage per project, averaged over three runs.`
New: `Mean input/output tokens per project over three runs, with total estimated cost at GPT-5.6-terra pricing.`
(one sentence, 15 words by `wc -w`)

## Blocker

No LaTeX in this session (`latexmk`/`pdflatex` not on PATH, exit 127), so
the PDF was not rebuilt. The edit changes only caption text.
