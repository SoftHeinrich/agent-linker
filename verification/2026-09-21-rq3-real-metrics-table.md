# RQ3 table: judge-level kills/keeps + real metrics on both tasks; judges x knowledge grid dropped

Date: 2026-09-21. Arm `s126`, body backend `terra`, mean of three runs.
Scope: evaluation infrastructure and the generated floats only. No paper prose was
edited; `paper/sections/results.tex` is left to the author (see "Open" below). Nothing
was hand-edited in a `.tex`/`.csv` artifact either -- every number below comes out of
the engine -> reshape -> render chain.

## What changed

1. **`tab:rq3-confusion` now prints levels, not deltas.** Rows are judging
   *configurations* -- `Full` (the shipped pipeline), each judge switched off, and
   `No judge` (the whole judging layer off, formerly the `both` row). The five
   `judge off (pp)` delta columns are replaced by the pipeline's own scores in that
   configuration on **both** tasks: doc-model `Prec./Rec.; F1/F2` + CMR, doc-code
   `Prec./Rec.; F1/F2` plus its size-aware pair (worst- and harmonic-component F1/F2,
   the same two bands `tab:rq2` and `tab:rq4` print).
2. **The reject/keep counts keep their old judge-level meaning, and the two rows that
   had none get one.** An off-row still prints the *distinct set* of the judge it names
   -- what that judge kills and keeps while it is on -- unchanged from the delta table
   (`\nameValidator` 100.0 FP / 9.7 TP rejected, 169.0 / 22.0 kept). `Full` prints
   the judges together: the `all_combined` audit row, i.e. the union over the judges,
   not the sum (two judges can reject the same link). `No judge` prints the no-judge
   audit: nothing rejected, and the keeps are every candidate the linkers propose.
   That last row is the one new number, and it comes from a new engine function,
   `rq34.rq3_none_audit`, written as a `none` row of `rq3_validators.csv` (same schema,
   same grain as the per-judge rows). `rq_tables.RQ3_AUDIT_ROW` maps each table row to
   its audit row. Counts are not bolded: the all-off row rejects nothing by
   construction, which is not a win.
   The appendix companion `tab:rq3-runs` (both backends, per run + average) follows the
   same shape.
   **Font size and precision (now global).** Every generated table is `\footnotesize`
   (8pt) and every score cell prints two decimals: the renderer
   grew two knobs, `csv_to_tex.TABLE_SIZE` (the default in both table writers) and
   `csv_to_tex.SCORE_DP` (the `f2`/`f3` kinds alike -- counts, CMR% and pp deltas keep
   their own one-decimal grain, they are not 0..1 scores),
   and all per-spec `"size"` overrides were deleted, so the paper no longer mixes
   `\small` (rq2, rq4, rq4-run*), `\footnotesize` (the three big tables,
   gold_concentration) and `\scriptsize` (rq1). `mini-inequality/motivation.py`, which
   writes `gold_concentration.tex` outside `csv_to_tex`, was moved to the same size by
   hand-in-the-generator. What prompted it, measured in the compiled `main.pdf` of
   2026-09-21 09:57
   (pypdf, text-matrix x cm): the body table's 12 columns overran `\columnwidth`
   (395.8pt) at `\footnotesize`, so `\adjustbox{max width=\columnwidth}` rubber-scaled
   the box by 0.9404 -- 8pt text printed at 7.5pt with the rules thinned to match, and
   the declared size had no effect on the result since the box was width-bound.
   `tab:rq1` (the other wide body table) prints at a true 7.0pt `\scriptsize`;
   `tab:rq2` and `tab:rq4` are narrow enough to print at their declared 9.0pt `\small`.
   Dropping the third decimal is what buys the room for 8pt: it takes ~5 characters out
   of every `Prec./Rec.; \fone/\ftwo` cell. Estimated natural widths at 8pt
   (character model calibrated on the 0.9404 data point above, +-3%): every table now
   fits the 395.8pt measure except `tab:rq3-confusion` (~408pt, `fit` scales ~0.97 ->
   ~7.8pt) and the appendix `tab:rq3-runs` (~486pt, ~0.81 -> ~6.5pt); at three decimals
   `big-table-perproject` (~422pt) and `rq4-bigtable-perproject` (~428pt) were scaling
   too. Next compile confirms: a table that fits has no `cm` scale in its page content
   stream.
   For reference, at `\scriptsize` the RQ3 tabular is ~378pt wide (text part scales with the font, the
   3pt `\tabcolsep` does not), so it fits natively: a true 7pt, no scaling. `fit` stays
   as a guard only. That is the argument for the global size too -- above `\scriptsize`
   the 12-16 column tables overrun the measure and `fit` prints a size nobody declared.
3. **`tab:judges-knowledge` (judging x knowledge, 2x2) is dropped**, with its finding.
   Removed: `rq_tables.build_rq5` + `RQ5_ROWS`/`RQ5_METRICS`, the `rq5.csv` spec in
   `csv_to_tex.SPECS` and `JUDGES_ON_MAP`, the `sync_paper.BODY` entry,
   `reports/tex_src/rq5.csv`, `reports/tex/rq5-knowledge-judges.tex`, and the paper's
   `table/rq5-knowledge-judges.{tex,csv}`. `reports/rq34/s126_noknow{,_luna}/` is
   untouched -- it still feeds the RQ4 `No knowledge` row.
   Counts in HOWTO-REGENERATE-RQ.md drop to 13 `tex_src` CSVs / 12 rendered tables.

`metrics.py` is untouched and every metric cell is still copied from an
already-committed engine CSV. The only new computation is the `none` audit row in item 2,
and it lives in the engine (`rq34.py`), not in the reshape or render layer.

## Verification

```bash
cd /mnt/hostshare/ardoco-home/agent-linker
python3 evaluation/mini-src/rq34.py             # re-score: adds the `none` rows to rq3_validators.csv
python3 evaluation/mini-src/rq_tables.py && python3 evaluation/mini-src/csv_to_tex.py
python3 evaluation/mini-src/check.py            # PASS (frozen golden panel, 10 cells)
python3 evaluation/mini-src/gen_csv_to_temp.py  # all engine CSVs reproduce; repo untouched
PAPER_DIR=$PWD/paper python3 evaluation/mini-src/sync_paper.py --only rq --check
# IN SYNC: all 24 paper file(s) match the generated output. (2 absent for this arm)
python3 verification/2026-09-21-rq3-real-metrics-check.py    # RESULT: PASS
```

`rq34.py` re-scored both backends with `validate=OK` (15 cells checked each) and the
only change under `reports/rq34/s126/` is the eight added `none` rows in
`rq3_validators.csv` (2 backends x [3 runs + average]) -- `rq3_variants.csv`, the
per-project drilldowns and every RQ4 file are byte-identical, which is the evidence that
item 2 added a row and moved no existing number.

`2026-09-21-rq3-real-metrics-check.py` is an independent re-derivation: it reads the
engine CSVs directly (not `rq_tables.py`), re-formats every cell, and compares it
against the rendered `.tex` -- 4 body rows and 32 appendix rows (2 backends x 4 grains x
4 configurations) -- plus the LaTeX structure (band spans + labels == header cells ==
cells in every data row) and the `No judge` row's defining property (the `none` audit
rejects nothing, and its kept true positives equal the full layer's kept TPs plus the
true links the layer costs outright; the false positives deliberately do not add up that
way, since a link one judge rejects the other can keep).

Rendered body table (terra, mean of 3), re-derived from the engine CSVs by
`python3 verification/2026-09-21-rq3-real-metrics-check.py --md` and pasted by that
command's output, not typed:

| Judges | rej FP | rej TP | keep TP | keep FP | doc-model P/R; F1/F2 | CMR% | doc-code P/R; F1/F2 | worst F1/F2 | harm. F1/F2 |
|---|---|---|---|---|---|---|---|---|---|
| Full | 145.7 | 10.7 | 184.0 | 24.0 | .91/.96; .93/.95 | 0.6 | .86/.91; .88/.90 | .77/.78 | .91/.91 |
| name | 100.0 | 9.7 | 169.0 | 22.0 | .75/.99; .84/.91 | 0.6 | .76/.95; .84/.90 | .56/.70 | .82/.90 |
| coref | 53.3 | 2.3 | 34.3 | 2.0 | .77/.97; .85/.92 | 0.0 | .78/.92; .83/.88 | .68/.77 | .86/.90 |
| No judge | 0.0 | 0.0 | 194.7 | 166.3 | .67/1.00; .78/.89 | 0.0 | .71/.96; .80/.88 | .55/.70 | .78/.88 |

The metric columns reproduce the deltas the previous table printed (e.g. No judge
doc-model F1 .935 -> .784 = -15.0pp, F2 .949 -> .891 = -5.8pp), so no metric moved, and
the per-judge counts on the two middle rows are the ones the delta table already carried.
New to this float: the metric *levels* themselves, the doc-code half with its size-aware
pair, the `Full` counts (previously blank, now the judges together) and the `No judge`
counts (previously the union, now 0 rejects with the whole pool kept).

TeX is not installed in this environment, so the floats were not compiled. The
structural check above is what stands in for it; both tables keep the `\adjustbox{max
width=\columnwidth}` wrapper and are now 9 metric/count columns wide (10 header cells in
the body table, against 10 in the delta version it replaces).

## Open (author's call, not done here)

`paper/sections/results.tex` still contains `\input{table/rq5-knowledge-judges}` and the
paragraph that reads `\autoref{tab:judges-knowledge}`, plus the RQ4-answer line about the
judging/knowledge interaction. The document will not build until that `\input` goes. The
RQ3 prose also still describes the removed delta columns ("judge-off block", "-9.9 and
-8.1pp"); its per-judge counts (100.0 / 53.3 rejected FPs) are still exactly what the
table prints on the two middle rows.
