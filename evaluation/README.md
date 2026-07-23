# TransArc-EMP — mini studies

A set of small, **self-contained, stdlib-only** Python studies that compute the
trace-link-recovery paper's research-question metrics directly from the ARDoCo
benchmark and the recorded TransArc / agent-linker run results. Each study is one
directory, runs on a bare `python3`, and verifies itself against a frozen panel.

> **Branch layout.** This is the **`mini`** branch — the active, cleaned-up
> workspace. The full historical two-pillar workspace (the retired `src/`
> metrics pipeline, the benchmark-bias analyses, every result snapshot, and the
> `writing/eval.tex` paper) lives on the **`master`** branch, which is now the
> legacy/full archive. Nothing was deleted — `master` preserves all of it; this
> branch just tracks the mini-studies and their data.

## The studies

| Dir | Question | What it does | Entry points |
|-----|----------|--------------|--------------|
| [`mini-src/`](mini-src/README.md) | **RQ1 / RQ2** — link & component metrics | The project's sole metrics implementation: file/link P/R/F1, per-component F1, worst-component & harmonic-mean F1, sentence coverage, noise rate, for `sad-code` (doc-to-code) and `sad-sam` (doc-to-model). Plus the RQ1/RQ2 big table and the no-enroll inflation baseline. | `metrics.py`, `check.py`, `rq12.py`, `noenroll.py` |
| [`mini-inequality/`](mini-inequality/README.md) | **RQ2 motivation** — data inequality | Concentration inequality of the gold links (Gini, Lorenz, top-k share, enrollment expansion) — why micro-F1 needs the size-aware suite. Self-contained GSD sub-project (own `.planning/`). | `inequality.py`, `motivation.py`, `claim_check.py` |
| [`mini-rq34/`](mini-rq34/README.md) | **RQ3 / RQ4** — validators & ablation | Validator contribution (cost/benefit, counterfactual macro-F1) and per-module linker ablation (unique TPs, leave-one-out delta, overlap decomposition), at the doc-to-model grain. | `rq34.py` |
| `mini-data/` | — | Pruned canonical data: the 15 TransArc result CSVs the studies actually read (`<project>/{sad-code,sad-sam,sam-code}/...Tlr_*.csv`, 5 projects). | (data) |

`reports/` holds the top-level mini outputs (`RQ12_BIGTABLE.csv`, `RQ2_PANEL.csv`,
`NOENROLL_DOC_CODE.{csv,md}`); each study also writes to its own `*/reports/`.

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
python3 mini-src/rq12.py             # -> reports/RQ12_BIGTABLE.csv, RQ2_PANEL.csv
python3 mini-src/noenroll.py         # no-enroll doc-to-code inflation baseline

# RQ2 motivation (inequality) — runs its own sanity check vs frozen literals
python3 mini-inequality/inequality.py
python3 mini-inequality/motivation.py

# RQ3/RQ4 (validators + ablation)
python3 mini-rq34/rq34.py
```

See each study's own `README.md` for definitions, provenance, and full options.

## Conventions

- **Stdlib only; no cross-module imports.** Each `mini-*` study copies shared
  definitions (and sanity-checks them for agreement) rather than importing, so it
  stays runnable in isolation.
- **No benchmark leakage** — no benchmark-derived word lists in any code
  (workspace rule); distributional / structural stats only.
- See [CLAUDE.md](CLAUDE.md) for agent guidance and the workspace
  [../CLAUDE.md](../CLAUDE.md).
