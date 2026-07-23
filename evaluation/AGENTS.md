# CLAUDE.md — TransArc-EMP (mini branch)

Project-local guidance for Claude Code. The workspace-level `../CLAUDE.md`
(ARDoCo monorepo) also applies.

## What This Is

The **`mini`** branch: a cleaned-up workspace of small, self-contained,
stdlib-only studies that compute the paper's research-question metrics for
ARDoCo trace-link recovery. Each `mini-*` directory is independently runnable.

- `mini-src/` — RQ1/RQ2 metrics (the sole metrics implementation) + `rq12.py`
  (RQ1/RQ2 big table) + `noenroll.py` (no-enroll inflation baseline). Verify with
  `python3 mini-src/check.py` (frozen golden panel, asserts to 1e-4).
- `mini-inequality/` — RQ2 motivation (gold-link concentration inequality). A
  self-contained GSD sub-project with its **own** `mini-inequality/.planning/`.
- `mini-rq34/` — RQ3 (validator contribution) + RQ4 (per-module ablation).
- `mini-data/` — pruned canonical data: the 15 TransArc result CSVs the studies
  read (`<project>/{sad-code,sad-sam,sam-code}/...Tlr_*.csv`, 5 projects).

> **Legacy lives on `master`.** The full two-pillar history — the retired `src/`
> metrics pipeline, benchmark-bias analyses, all result snapshots, and
> `writing/eval.tex` — is preserved on the `master` (legacy) branch. Nothing was
> deleted; this branch only tracks the mini studies. To consult or revive any of
> it: `git show master:<path>` or `git checkout master -- <path>`.

## Stack

- Python 3, **stdlib only** (`csv`, `json`, `collections`, `dataclasses`, `math`,
  `pathlib`, `re`). No `requirements.txt`, no third-party deps — do not add any.
- Run a study: `python3 mini-<x>/<script>.py`. Outputs land in `reports/` (top
  level) and each study's own `*/reports/`.
- Benchmark data is external:
  `/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark/`
  (`mediastore`, `teastore`, `teammates`, `bigbluebutton`, `jabref`). Override
  roots via `$TRANSARC_BENCHMARK` / `$TRANSARC_RESULTS_DIR`.

## Conventions

- **Self-contained, copy-not-import.** Each `mini-*` study copies shared
  definitions and sanity-checks them for agreement rather than importing across
  directories. Keep it that way — a study must run on its own.
- **Verify after editing metrics:** `python3 mini-src/check.py` must print PASS.
- **No benchmark leakage:** no benchmark-derived word lists in any code
  (workspace rule). Stopwords / generic English only.
- **GSD planning** for `mini-inequality/` lives in its own subdir
  (`mini-inequality/.planning/`); there is no repo-root `.planning/` on this
  branch (the legacy two-pillar planning is on `master`).
