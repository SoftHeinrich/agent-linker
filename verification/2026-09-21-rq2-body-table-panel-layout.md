# RQ2 body table: paired-panel layout in the table pipeline

Date: 2026-09-21

## Change

The RQ2 body float (`tab:rq2`) is now the paired-panel per-project layout that
was previously rendered beside it as `tab:rq2-wide-comparison`. The layout moved
into the table pipeline; the standalone script and the earlier macro layout were
removed.

- `mini-src/rq_tables.py`: `build_rq2` writes `reports/tex_src/rq2.csv` as nine
  rows -- three project pairs times three systems -- with a `left_*`/`right_*`
  column pair per metric. The projects are the five benchmark projects plus the
  macro `Average`, split in halves, so both panels carry three project blocks.
  Per-project cells come from `RQ12_PERPROJECT.csv`, the `Average` panel from
  `RQ12_BIGTABLE.csv`; no value is computed here.
- `mini-src/csv_to_tex.py`: the RQ2 spec now says `"render": "panels"`, and
  `render_panels()` renders it. `check_specs()` checks a panel spec's bands
  against one panel's width instead of a column list.
- Dropped: `scripts/generate_rq2_split_table.py`,
  `paper/table/rq2-wide-comparison.{tex,csv}`, the second `\input` in
  `paper/sections/results.tex`, and the former macro RQ2 spec (one row per
  system, a `Prec./Rec.; \fone/\ftwo` cell per task).

The rendered table body is byte-identical to the layout verified on 2026-09-21
in `2026-09-21-rq2-split-table-generator.md`:

```bash
diff <(sed -n '/toprule/,/bottomrule/p' paper/table/rq2-wide-comparison.tex) \
     <(sed -n '/toprule/,/bottomrule/p' evaluation/reports/tex/rq2-results.tex)
```

Result: exit 0, no output (run before the old file was removed). The caption,
label (`tab:rq2`) and the generated-by banner are the parts that differ.

## Regeneration and deterministic replay

```bash
cd evaluation
python3 mini-src/rq_tables.py
python3 mini-src/csv_to_tex.py
```

Both outputs were copied aside, both commands rerun, and the copies compared:

```bash
cmp $SCRATCH/rq2.csv reports/tex_src/rq2.csv
cmp $SCRATCH/rq2.tex reports/tex/rq2-results.tex
```

Both exited 0 with no output. Digests:

```text
c11c6ddc7631da81f7904a44cb1ed92f7ab196538763fe2fe18613f680ea15f2  evaluation/reports/tex_src/rq2.csv
b28c7dc540aff45ca2f4d89b4c678fbf8f821c9414e4d1fadd3752b927a408c5  evaluation/reports/tex/rq2-results.tex
```

## Pipeline gates

```bash
python3 mini-src/check.py
```

Result: `PASS: mini-src/metrics.py reproduces the frozen golden panel (10 cells,
sad-code + sad-sam)`, and `OK arm-default every generator reports arm 's126'
(7/7 found)`. The metrics layer is untouched by this change; the gate is
recorded because `check_specs()` runs on import of `csv_to_tex.py`.

```bash
PAPER_DIR=.../paper python3 mini-src/sync_paper.py --only rq
PAPER_DIR=.../paper python3 mini-src/sync_paper.py --only rq --check
```

Results: `synced 24 file(s)` then `IN SYNC: all 24 paper file(s) match the
generated output. (2 absent for this arm)` -- the two absences are
`rq4-floor.{tex,csv}`, which s126 has no floor sweep for.

## ACM-template compilation

Isolated check, from `verification/` (the test file now inputs the body table;
it restates `\cmrname`, which the paper defines in `main.tex`, not `abbrev.tex`):

```bash
/tmp/rq2-tectonic.grZsvV/tectonic rq2-body-table-test.tex
```

Result: exit 0, `rq2-body-table-test.pdf` (49,119 bytes); the log contains no
overfull or underfull box. The rendered page is
`verification/rq2-body-table-test-page2.png`.

Full paper, from `paper/`:

```bash
/tmp/rq2-tectonic.grZsvV/tectonic -Z shell-escape-cwd=. main.tex
```

Result: exit 0, 19 pages. The RQ2 table is Table 3 on page 13, printed once
(`verification/rq2-body-table-paper-page13.png`). Against a build of the paper
submodule at `HEAD` in a detached worktree -- the state with both RQ2 floats --
the warning set does not grow:

| Warning | HEAD (both floats) | this change |
|---|---|---|
| Overfull \hbox `sections/eval:92` | yes | yes |
| Overfull \hbox `sections/metric.tex` lines 119-- | yes | yes |
| Underfull \vbox `sections/motivation:8` | yes | yes |
| Underfull \vbox `sections/results:128` | yes | yes |
| Underfull \vbox `sections/approach:108` | yes | no |
| Underfull \vbox `sections/motivation:28` | yes | no |

No warning points at `table/rq2-results.tex` in either build. The two that
disappear are page-breaking warnings; removing a float from the results section
changes where pages break, so their absence is a side effect of the float count,
not evidence about the table itself.

## Prose

No RQ2 prose depended on the dropped macro layout. `sections/results.tex` quotes
the macro \cmrname{} ($0.6\%$ / $3.3\%$ / $7.1\%$) and the worst- and
harmonic-component \fone{} ($0.77$ / $0.47$, $0.91$ / $0.63$); all of these are
the `Average` panel of the new table. The macro layout's precision and recall
columns are not cited in the RQ2 section; per-system precision and recall remain
in `tab:rq1` and in the appendix per-project table.
