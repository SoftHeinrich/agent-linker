# TransArc-EMP — mini studies

The **table pipeline** for the trace-link-recovery paper: small, **self-contained,
stdlib-only** Python studies that compute the research-question metrics directly
from the ARDoCo benchmark and the recorded TransArc / agent-linker run results, and
render them into the paper's floats. Each study is one directory, runs on a bare
`python3`, and verifies itself against a frozen panel.

Everything here earns its place by feeding a float in `paper/{table,appendix}/`
(`mini-src/sync_paper.py --check` guards that bridge). Side analyses that answered a
design question without producing a reported number live in
[`../studies/`](../studies/README.md).

> **Branch layout.** This is the **`mini`** branch — the active, cleaned-up
> workspace. The full historical two-pillar workspace (the retired `src/`
> metrics pipeline, the benchmark-bias analyses, every result snapshot, and the
> `writing/eval.tex` paper) lives on the **`master`** branch, which is now the
> legacy/full archive. Nothing was deleted — `master` preserves all of it; this
> branch just tracks the mini-studies and their data.

## The studies

| Dir | Question | What it does | Entry points |
|-----|----------|--------------|--------------|
| [`mini-src/`](mini-src/README.md) | **RQ1 / RQ2** — link & component metrics | The project's sole metrics implementation: file/link P/R, per-component, worst-component and harmonic-mean scores — each F1 reported with its recall-weighted F2 — plus the component miss rate (CMR), for `sad-code` (doc-to-code) and `sad-sam` (doc-to-model). Plus the RQ1/RQ2 big table and the CSV→TeX paper-table pipeline. | `metrics.py`, `check.py`, `rq12.py`, `rq_tables.py`, `csv_to_tex.py` |
| [`mini-inequality/`](mini-inequality/README.md) | **RQ2 motivation** — data inequality | Concentration inequality of the gold links (Gini, Lorenz, top-k share, enrollment expansion) — why micro-F1 needs the size-aware suite. Writes `paper/table/gold_concentration.tex` via `sync_paper.py`. Self-contained GSD sub-project (own `.planning/`). | `inequality.py`, `motivation.py` |
| [`mini-rq34/`](mini-rq34/README.md) | **RQ3 / RQ4** — validators & ablation | Validator contribution (cost/benefit, counterfactual macro-F1) and per-module linker ablation (unique TPs, leave-one-out delta, overlap decomposition), at the doc-to-model grain. | `rq34.py` |
| `mini-data/` | — | Pruned canonical data: the 15 TransArc result CSVs the studies actually read (`<project>/{sad-code,sad-sam,sam-code}/...Tlr_*.csv`, 5 projects). | (data) |

`reports/` holds the top-level mini outputs (`RQ12_BIGTABLE.csv`,
`RQ12_PERPROJECT.csv`, and the generated `tex_src/` + `tex/` paper tables); each
study also writes to its own `*/reports/`.

> Frozen leftovers. `RQ2_PANEL.csv`, `RQ2_CELLS.csv`, `RQ2_CORR.csv` and
> `NOENROLL_DOC_CODE.{csv,md}` are the last outputs of scripts retired with the
> s21 arm (`rq2_corr.py`, `noenroll.py`) — nothing regenerates them now. They are
> kept as the provenance for the archived paper floats that cite them; their
> generators are recoverable from git history.

## Prerequisites

- **Python 3, stdlib only** — no `pip install`, no `requirements.txt`, no
  third-party packages. Every script runs with a bare interpreter.
- **External benchmark data** lives outside this repo at
  `/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark/`
  (5 projects: `mediastore`, `teastore`, `teammates`, `bigbluebutton`, `jabref`).
  Override with `$TRANSARC_BENCHMARK`.
- **Bundled run data** is `mini-data/` (TransArc results). Override the results
  root with `$TRANSARC_RESULTS_DIR` or `--results-dir`. `mini-rq34/` and some of
  `mini-src/` additionally read agent-linker run dumps from sibling repos.

## Run & verify

```bash
# RQ1/RQ2 metrics + self-check (must print PASS)
python3 mini-src/metrics.py --task sad-code
python3 mini-src/metrics.py --task sad-sam
python3 mini-src/check.py            # golden-panel regression, asserts to 1e-4
python3 mini-src/rq12.py             # -> reports/RQ12_BIGTABLE.csv, RQ12_PERPROJECT.csv
python3 mini-src/rq_tables.py        # -> reports/tex_src/*.csv (one per paper float)
python3 mini-src/csv_to_tex.py       # -> reports/tex/*.tex

# RQ2 motivation (inequality) — runs its own sanity check vs frozen literals
python3 mini-inequality/inequality.py
python3 mini-inequality/motivation.py

# RQ3/RQ4 (validators + ablation)
python3 mini-rq34/rq34.py
```

See each study's own `README.md` for definitions, provenance, and full options.

## Conventions

- **Stdlib only; one shared core.** `mini-src/metrics.py` is the tree's single
  implementation of the benchmark layout, the F-measures and the gold loaders;
  every study — and the side analyses in `../studies/` — imports it rather than
  keeping a copy, so every reported number is scored by the arithmetic
  `mini-src/check.py` pins. Each study still runs on a bare `python3` with no
  install: the import is in-tree.
- **No benchmark leakage** — no benchmark-derived word lists in any code
  (workspace rule); distributional / structural stats only.
- See [CLAUDE.md](CLAUDE.md) for agent guidance and the workspace
  [../CLAUDE.md](../CLAUDE.md).
