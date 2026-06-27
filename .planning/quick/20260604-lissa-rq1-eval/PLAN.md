---
slug: lissa-rq1-eval
created: 2026-06-04
status: complete
---

# Quick task: Fill RQ1 LiSSA cells via transarc-emp eval infra

## Goal

Replace the `LiSSA` `0.000` placeholders in `writing/gen/table/rq1-doc-to-model.tex`
and `writing/gen/table/rq1-doc-to-code.tex` with apples-to-apples P/R/F1
numbers computed by `transarc-emp/src/lib/metrics_api.py` against the same
TransArc gold standards used for SWATTR / TransArc / AALinker. Source data:
the `gpt-5-mini` (flex tier) tracelink CSVs shipped in
`/mnt/hostshare/ardoco-home/sota/lissa-replication/results/tracelinks/`.

## Decisions (user)

- **Model:** `gpt-5-mini` (newer LiSSA run).
- **Method:** re-evaluate raw tracelinks via `compute_sad_sam_metrics` /
  `compute_sad_code_metrics` (apples-to-apples). Do not use LiSSA's
  self-reported numbers.
- **d2c coverage gap:** LiSSA only ran d2c on mediastore, teastore,
  bigbluebutton. teammates and jabref d2c cells become `---` (em-dash),
  documented in a table comment.

## Inputs

- `sota/lissa-replication/results/tracelinks/d2m/<project>-gpt-5-mini.csv`
  (cols: `sentenceId,modelElementId`) — 5 projects.
- `sota/lissa-replication/results/tracelinks/d2c/<project>-gpt-5-mini.csv`
  (cols: `sentenceId,codeFilePath` with `Implementation/` prefix) — 3 projects.
- `transarc-emp/src/lib/metrics_api.py` (`compute_sad_sam_metrics`,
  `compute_sad_code_metrics`).

## Adapter (set construction)

- d2m → `{(modelElementId, sentenceId_str) for row in csv}` → matches
  `load_gs_sad_sam` shape.
- d2c → `{(sentenceId_str, normalize_path(codeFilePath)) for row in csv}` →
  matches `load_result_sad_code` shape (the column conflation is intentional
  in `transarc_error_analysis.py:223`).

## Deliverables

1. `approach/scripts/v2.6.3/eval_lissa_rq1.py` — adapter + driver.
2. `evaluation/reports/lissa_metrics_sad-sam.csv` and
   `evaluation/reports/lissa_metrics_sad-code.csv` — per-project P/R/F1 + the
   full metric suite (same schema as `metrics_sad-sam.csv`).
3. `writing/gen/table/rq1-doc-to-model.tex` and
   `writing/gen/table/rq1-doc-to-code.tex` — LiSSA columns populated; d2c
   teammates/jabref em-dashed; data-source comment updated.

## Out of scope

- Cloning lissa-replication into `sota/` (already done by the orchestrator).
- AALinker / TransArc / SWATTR cells in the same tables (other quick tasks /
  Phase 43).
- RQ2 / RQ3 / RQ4 numbers.
