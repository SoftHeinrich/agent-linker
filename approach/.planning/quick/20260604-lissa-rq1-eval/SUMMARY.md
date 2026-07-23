---
slug: lissa-rq1-eval
created: 2026-06-04
completed: 2026-06-04
status: complete
---

# LiSSA RQ1 eval — summary

## What changed

- Cloned `git@github.com:SoftHeinrich/lissa-replication.git` into
  `sota/lissa-replication/` (60 MB; ships pre-computed gpt-4o-mini and
  gpt-5-mini tracelink CSVs for d2m and d2c).
- Added `approach/scripts/v2.6.3/eval_lissa_rq1.py`: a stdlib-only adapter
  that reads LiSSA's `results/tracelinks/d2{m,c}/<project>-gpt-5-mini.csv`,
  converts to `compute_sad_sam_metrics` / `compute_sad_code_metrics` shape,
  and writes apples-to-apples P/R/F1 + the full unified metric suite to
  `evaluation/reports/lissa_*`.
- Filled the LiSSA columns in `writing/gen/table/rq1-doc-to-{model,code}.tex`
  with the computed numbers; em-dashed Teammates / JabRef in d2c (LiSSA
  replication package never ran d2c on them) and the LiSSA d2c macro (would
  otherwise mix a 3-project mean into a 5-project column).
- Updated `.planning/STATE.md` "Quick Tasks Completed".

## Outputs

| File | What |
|------|------|
| `sota/lissa-replication/` | Cloned LiSSA replication package (gpt-4o-mini + gpt-5-mini tracelinks). |
| `approach/scripts/v2.6.3/eval_lissa_rq1.py` | Adapter + driver. |
| `evaluation/reports/lissa_metrics_sad-sam.csv` | Full unified d2m metric suite (link/sentence/component/MCC/MAP/HUS) for all 5 projects. |
| `evaluation/reports/lissa_metrics_sad-code.csv` | Full unified d2c metric suite for 3 projects (em-dash rows for Teammates / JabRef). |
| `evaluation/reports/lissa_rq1_d2m.csv` | Per-project P/R/F1 + TP/FP/FN/N\_pred — link level (the RQ1 d2m table source). |
| `evaluation/reports/lissa_rq1_d2c.csv` | Per-project P/R/F1 + TP/FP/FN/N\_pred — file level (the RQ1 d2c table source). |
| `writing/gen/table/rq1-doc-to-model.tex` | LiSSA P/R/F1 columns populated; macro 0.310 / 0.833 / 0.425. |
| `writing/gen/table/rq1-doc-to-code.tex` | LiSSA cells populated for MS/TS/BBB; Teammates / JabRef and the LiSSA macro em-dashed with caption note. |

## Apples-to-apples sanity check

The per-project TP/FP/FN counts that our re-evaluation produces for d2m exactly
match what LiSSA itself reports in `sota/lissa-replication/results/COMPARISON.md`
(rerun: gpt-5-mini d2m table): mediastore 30/71/1, teastore 26/79/1, teammates
30/282/27, jabref 18/6/0, bigbluebutton 44/237/18. The eval infra reads LiSSA's
raw tracelinks against the same TransArc gold standard, so no metric drift was
introduced.

## Out of scope (deferred)

- SWATTR / TransArc P/R columns (CSV ships F1 only — separate task).
- AALinker d2c row (Phase 43 / Plan 43-02 owns the s_linker19 → sad-code
  composition; the RQ1 d2c AALinker column will fill once that ships).
- gpt-4o-mini LiSSA numbers (paper-aligned model). Easy follow-up — same
  script with `MODEL = "gpt-4o-mini"`.
