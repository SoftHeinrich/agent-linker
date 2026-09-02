# CLAUDE.md — TransArc-EMP (mini branch)

Project-local guidance for Claude Code. The workspace-level `../CLAUDE.md`
(ARDoCo monorepo) also applies.

## What This Is

The paper's **table pipeline**: self-contained, stdlib-only code that computes
the research-question metrics for ARDoCo trace-link recovery and renders them
into the paper's floats.

Scope rule: code belongs here only if it feeds a float in
`paper/{table,appendix}/`. Exploratory probes, one-off audits and arm-selection
tools go in `../studies/` (see `../studies/README.md`) — do not add them back here.

- `mini-src/` — **the whole pipeline, and nothing else**: `metrics.py` (the sole
  metrics implementation) → the four engines (`rq12.py` for RQ1/RQ2; `rq34.py`,
  `rq34_rq2.py`, `rq4_floor.py` for RQ3/RQ4) → `rq_tables.py`/`csv_to_tex.py`
  (reshape, then render) → `sync_paper.py` (the paper bridge, with a `--check`
  drift guard). Gates: `python3 mini-src/check.py` (frozen golden panel, asserts
  to 1e-4) and `python3 mini-src/gen_csv_to_temp.py` (regenerates every engine's
  CSVs into a temp dir and diffs them against the committed copies).
  Only the engines compute; reshape and render copy cells, so a number can enter
  the paper only through an engine. See `mini-src/README.md` for the stage table.
  **Every F1 is reported with its F2.** When adding an F-measure anywhere, add
  both flavours in the same commit and thread them through
  `metrics.PANELS` → `rq12.COLUMNS` → `rq_tables` → `csv_to_tex.SPECS`; a lone F1
  column is a bug. `csv_to_tex.check_specs()` runs on import and will refuse a
  spec whose `\multicolumn` bands no longer cover its columns.
  The canonical arm is **s_linker110** (terra = paper body, luna = mirror); the
  s21 / s20union arms were retired from the roster on 2026-08-26. Six modules
  declare `DEFAULT_ARM` and `check.py` fails if any two disagree, so an arm
  cannot be promoted by halves.
- `mini-inequality/` — RQ2 motivation (gold-link concentration inequality).
  `motivation.py` writes `paper/table/gold_concentration.tex`, which is why the
  engine lives here; `sync_paper.py` imports it lazily. A self-contained GSD
  sub-project with its **own** `mini-inequality/.planning/`. The retired claim
  audit and the back-compat table guard moved to `../studies/mini-inequality/`.
- `reports/` — every engine's output. RQ1/RQ2 land directly here; RQ3/RQ4 are
  arm-scoped under `reports/rq34/<arm>/` (plus `<arm>_floor`, `<arm>_noknow`).
  `reports/tex_src/` holds the per-float CSVs and `reports/tex/` the rendered
  tables that `sync_paper.py` copies.
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
- Run a stage: `python3 mini-src/<script>.py`. Outputs land in `reports/`
  (RQ3/RQ4 under `reports/rq34/<arm>/`); `mini-inequality/` keeps its own.
- Benchmark data is external:
  `/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark/`
  (`mediastore`, `teastore`, `teammates`, `bigbluebutton`, `jabref`). Override
  roots via `$TRANSARC_BENCHMARK` / `$TRANSARC_RESULTS_DIR`.

## Conventions

- **One shared core, imported — never re-copied.** `mini-src/metrics.py` holds
  the benchmark layout (roots, project list, gold-standard paths), the confusion
  matrix and F-measures (`prf`, `prf_counts`, `fbeta`), the gold loaders and the
  dict-row CSV writer, and the report roots. Every engine imports it (`import
  metrics as m`; anything outside `mini-src/` adds a `sys.path.insert` on it)
  instead of vendoring a copy: the copies had drifted and had to be hand-checked
  for agreement, and only one of them was pinned by `check.py`. The import is
  in-tree, stdlib-only, no package install, so each script still runs on its own.
  When you need a definition that lives there, import it; when you add one that
  two callers need, put it there. The one deliberate exception is
  `mini-src/_alinker_types.py`, vendored because unpickling the run's phase state
  needs those classes importable under their original module path.
- **Verify after editing metrics:** `python3 mini-src/check.py` must print PASS.
- **No benchmark leakage:** no benchmark-derived word lists in any code
  (workspace rule). Stopwords / generic English only.
- **GSD planning** for `mini-inequality/` lives in its own subdir
  (`mini-inequality/.planning/`); there is no repo-root `.planning/` on this
  branch (the legacy two-pillar planning is on `master`).
