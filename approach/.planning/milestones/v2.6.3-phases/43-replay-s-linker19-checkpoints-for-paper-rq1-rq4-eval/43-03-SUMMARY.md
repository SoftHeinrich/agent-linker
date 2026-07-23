---
phase: 43-replay-s-linker19-checkpoints-for-paper-rq1-rq4-eval
plan: 03
subsystem: paper/rq1-tables
tags: [rq1, tex, formatter, two-backend, stdlib-only]
requirements: [REQ-V263-02]
dependency_graph:
  requires:
    - "Plan 02 CSVs at results/v2.6.3/{claude,openai}/<project>/sad-{sam,code}.csv"
    - "transarc-emp/src/lib/metrics_api.py (compute_sad_sam_row, compute_sad_code_row, build_avg_row, NA, NA_TEX, _fmt, SCHEMA, LATEX_HEADER, PROJECTS)"
    - "transarc-emp/src/lib/transarc_error_analysis.py (RESULTS, load_result_sad_sam_standalone, load_result_sad_code)"
  provides:
    - "writing/working/tables/metrics_sad-sam.tex (\\label{tab:metrics-sad-sam})"
    - "writing/working/tables/metrics_sad-code.tex (\\label{tab:metrics-sad-code})"
    - "transarc-emp/src/paper/rq1_table.py (build_two_backend_rows, write_two_backend_tex, main)"
  affects:
    - "writing/working/sections/results.tex (Plan 05 will \\input{tables/metrics_sad-sam} and metrics_sad-code)"
tech_stack:
  added: []
  patterns:
    - "Per-backend results-tree materialisation + monkey-patched RESULTS (reuse of metrics_api primitives without copying metric math)"
    - "Two-row LaTeX header with multicolumn backend groups + cmidrule visual separation"
    - "Symlink-lens fallback in CSV root resolution (approach/results vs results)"
key_files:
  created:
    - "transarc-emp/src/paper/rq1_table.py"
    - "writing/working/tables/metrics_sad-sam.tex"
    - "writing/working/tables/metrics_sad-code.tex"
    - "writing/working/tables/ (new directory)"
  modified: []
decisions:
  - "Use a per-backend temp results tree + monkey-patch transarc_error_analysis.RESULTS, restored in finally, so the two-backend wrapper REUSES metrics_api.compute_*_row verbatim (D-02). No metric math is reimplemented in rq1_table.py."
  - "Bridge Plan 02's sad-code.csv schema (sentence, codeID) to the legacy loader schema (modelElementID, codeId) by rewriting the temp copy. Plan 02's sad-sam.csv (modelElementID, sentence, source) is a superset of the legacy schema and is symlink-copied as-is."
  - "Use \\approach{} in captions (the existing abbrev macro from writing/working/abbrev.tex), not a hardcoded 's_linker19' name. Caption text passes through render-time so the macro expands at \\input time."
  - "Backend labels are exactly 'Claude' and 'GPT-5.4' per D-03 — Claude is the LEFT column group."
metrics:
  duration: "~10 minutes"
  completed: "2026-06-05"
  tasks: 2
  files_created: 4
  files_modified: 0
---

# Phase 43 Plan 03: RQ1 Two-Backend Wide-Table Generator Summary

Built the two-backend RQ1 wide-table generator (`transarc-emp/src/paper/rq1_table.py`) and ran it against Plan 02's per-backend replay CSVs to populate the two RQ1 LaTeX tables (`writing/working/tables/metrics_sad-{sam,code}.tex`), each with 6 data rows (5 projects + Macro) under Claude | GPT-5.4 column groups. Stdlib-only; reuses `metrics_api.compute_sad_*_row` primitives verbatim via a per-backend temp results tree.

## What Shipped

**`transarc-emp/src/paper/rq1_table.py`** (new, stdlib-only):

- Exports `main`, `build_two_backend_rows`, `write_two_backend_tex` (plus internal helpers `_resolve_csv_root`, `_materialise_backend_results_tree`, `_run_backend_rows`, `_render_wide_table`, `_project_display`, `_metric_label`).
- CLI shape: `--task {sad-sam,sad-code,both}` (default `both`), `--csv-root <PATH>` (default `/mnt/hostshare/ardoco-home/agent-linker/results/v2.6.3`, with symlink-lens fallbacks), `--tex-out-dir <PATH>` (default `/mnt/hostshare/ardoco-home/agent-linker/writing/working/tables`).
- Reuses `metrics_api.compute_sad_sam_row`, `compute_sad_code_row`, `build_avg_row`, plus `_fmt`, `NA`, `NA_TEX`, `SCHEMA`, `LATEX_HEADER`, `PROJECTS`. Reuses `transarc_error_analysis.RESULTS` via temporary monkey-patch (restored in `finally`).
- Per-backend strategy: build a temp results tree at `<temp>/<project>/sad-sam/sadSamTlr_<project>.csv` (symlink-copy passthrough) and `<temp>/<project>/sad-code/sadCodeTlr_<project>.csv` (column rename `sentence→modelElementID`, `codeID→codeId` to match the legacy loader schema in `transarc_error_analysis.load_result_sad_code`), then point `RESULTS` at it.

**`writing/working/tables/metrics_sad-sam.tex`** (new):

- `\label{tab:metrics-sad-sam}`. 6 rows: MediaStore, TeaStore, TeaMmates, BigBlueButton, JabRef, Macro. 14 numeric columns = 7 metrics × 2 backends (link_f1, sentence_f1, MCC, MAP, ACF1, NDG, HUS — ACF1/NDG inapplicable → `--`). Source-note footer cites `approach/results/v2.6.3/<backend>/<project>/sad-sam.csv`.
- Macro F1 (link, the headline metric): Claude 0.939, GPT-5.4 0.922.

**`writing/working/tables/metrics_sad-code.tex`** (new):

- `\label{tab:metrics-sad-code}`. Same 6 rows. 22 numeric columns = 11 metrics × 2 backends (Link/Sentence/MAP inapplicable → `--`). Source-note footer cites `approach/results/v2.6.3/<backend>/<project>/sad-code.csv`.
- Macro file_f1 (the doc-to-code headline): Claude 0.939, GPT-5.4 0.919.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Duplicate `sad-` prefix in TeX source-note footer**

- **Found during:** Task 2 (initial run produced `sad-sad-sam.csv` / `sad-sad-code.csv` in the footnote).
- **Issue:** `note = "...sad-%s.csv..." % task` was wrong because `task` is already `sad-sam` / `sad-code`.
- **Fix:** Changed format string to `"...%s.csv..." % task` (also matches the actual Plan 02 CSV filenames).
- **Files modified:** `transarc-emp/src/paper/rq1_table.py`
- **Commit:** `transarc-emp@2cbe3ce`
- **Re-verified:** regenerated both TeX files, `grep -c "sad-sad" ...tex` returned 0 lines each.

No other deviations. No architectural changes (Rule 4 not triggered). No auth gates.

## Threat Surface

Plan 03's `<threat_model>` mitigations were honored:

- **T-43-07 (Tampering, RESULTS monkey-patch):** `_run_backend_rows` saves `transarc_error_analysis.RESULTS` before the per-backend block, mutates it inside `try`, and restores it in `finally`. Also cleans up the temp dir via `shutil.rmtree(..., ignore_errors=True)`.
- **T-43-09 (Repudiation, stale TeX):** `Path.write_text` overwrites in full; no append mode.

No new threat surface beyond the plan's register.

## Acceptance Criteria — Verified

### Task 1

- [x] `rq1_table.py` exists and compiles (`python3 -m py_compile` succeeds).
- [x] Imports contain NO `pandas`, `numpy`, `jinja2`, `requests`, `httpx`, `pyyaml`, `tomli`. Actual imports: `__future__`, `argparse`, `csv`, `generate_tables`, `metrics_api`, `os`, `pathlib`, `shutil`, `sys`, `tempfile`, `transarc_error_analysis`.
- [x] `metrics_api` is imported (D-02 reuse contract).
- [x] No `llm_sad_sam*` import.
- [x] `--help` prints usage with `--task`, `--csv-root`, `--tex-out-dir`.
- [x] `grep REQ-V263-02`, `D-02`, `D-03` all present.

### Task 2

- [x] Both TeX files exist with the correct labels.
- [x] Each contains both backend labels (`Claude`, `GPT-5.4`).
- [x] Each contains all five project names (MediaStore, TeaStore, TeaMmates, BigBlueButton, JabRef) AND a Macro row.
- [x] Each contains the data-lineage substring `results/v2.6.3`.
- [x] `git diff --stat src/llm_sad_sam/` → 0 lines (GATE-01 preserved).
- [x] `git diff --stat ../transarc-emp/writing/tables/` → 0 lines (unrelated dir untouched).
- [x] Run executed with `env -i HOME=$HOME PATH=$PATH` (zero LLM env vars present at process start).

## Commits

### agent-linker

| Commit    | Type | Files                                                                  |
| --------- | ---- | ---------------------------------------------------------------------- |
| `234d8bc` | feat | writing/working/tables/metrics_sad-sam.tex, metrics_sad-code.tex       |

### transarc-emp

| Commit    | Type | Files                       |
| --------- | ---- | --------------------------- |
| `497afe9` | feat | src/paper/rq1_table.py      |
| `2cbe3ce` | fix  | src/paper/rq1_table.py      |

## Known Stubs

None. Every numeric cell is populated (`--` denotes an inapplicable metric per `metrics_api.SCHEMA`, not a missing computation). No `\todo{}` placeholders. No hardcoded zeros.

## Self-Check: PASSED

- `[ -f /mnt/hostshare/ardoco-home/transarc-emp/src/paper/rq1_table.py ]` → FOUND
- `[ -f /mnt/hostshare/ardoco-home/agent-linker/writing/working/tables/metrics_sad-sam.tex ]` → FOUND
- `[ -f /mnt/hostshare/ardoco-home/agent-linker/writing/working/tables/metrics_sad-code.tex ]` → FOUND
- agent-linker `234d8bc` → FOUND in git log
- transarc-emp `497afe9` → FOUND in git log
- transarc-emp `2cbe3ce` → FOUND in git log
