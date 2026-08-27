# CLAUDE.md — TransArc-EMP (mini branch)

Project-local guidance for Claude Code. The workspace-level `../CLAUDE.md`
(ARDoCo monorepo) also applies.

## What This Is

The paper's **table pipeline**: small, self-contained, stdlib-only studies that
compute the research-question metrics for ARDoCo trace-link recovery and render
them into the paper's floats. Each `mini-*` directory is independently runnable.

Scope rule: a directory belongs here only if it feeds a float in
`paper/{table,appendix}/`. Exploratory probes and one-off audits go in
`../studies/` (see `../studies/README.md`) — do not add them back here.

- `mini-src/` — RQ1/RQ2 metrics (the sole metrics implementation) + `rq12.py`
  (RQ1/RQ2 big table) + `rq_tables.py`/`csv_to_tex.py` (the paper's RQ tables) +
  `sync_paper.py` (the paper bridge, with a `--check` drift guard). Verify with
  `python3 mini-src/check.py` (frozen golden panel, asserts to 1e-4).
  **Every F1 is reported with its F2.** When adding an F-measure anywhere, add
  both flavours in the same commit and thread them through
  `metrics.PANELS` → `rq12.COLUMNS` → `rq_tables` → `csv_to_tex.SPECS`; a lone F1
  column is a bug. `csv_to_tex.check_specs()` runs on import and will refuse a
  spec whose `\multicolumn` bands no longer cover its columns.
  The canonical arm is **s_linker92a** (terra = paper body, luna = mirror); the
  s21 / s20union arms were retired from the roster on 2026-08-26.
- `mini-inequality/` — RQ2 motivation (gold-link concentration inequality).
  `motivation.py` writes `paper/table/gold_concentration.tex`, which is why the
  engine lives here; `sync_paper.py` imports it lazily. A self-contained GSD
  sub-project with its **own** `mini-inequality/.planning/`. The retired claim
  audit moved to `../studies/mini-inequality/claim_check.py`.
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
