# mini-inequality — trace-link data-inequality study

A self-contained, stdlib-only mini study of the **concentration inequality** of trace
links in the ARDoCo benchmark: how a few large components / files own most of the gold
link mass, quantified with Gini, Lorenz curves, top-k concentration share, and the
enrollment expansion factor — for both `sad-code` (doc-to-code) and `sad-sam`
(doc-to-model), across all five projects.

Its purpose is twofold:

1. **Quantify** the dataset's intrinsic inequality (a property of the gold standard).
2. **Ground the paper** — verify every distributional-inequality claim the
   `alinker-paper` makes, fill its open `XX` placeholders, and demonstrate empirically
   why file/link micro-F1 needs the proposed four-metric suite (per-component F1,
   sentence coverage, noise rate, file-level F1).

## Status

🚧 Defined, not yet built. See `.planning/ROADMAP.md` (Phases 1-3). The engine
(`inequality.py`), the claim-verification report (`CLAIM_CHECK.md`), and the
paper-ready table are produced by those phases.

## Isolation

This directory is a **self-contained GSD sub-project**. Its planning lives in
`mini-inequality/.planning/`. On the `mini` branch there is no repo-root
`.planning/` (the legacy two-pillar planning is preserved on `master`), so this
subdir's planning is the only tracker here.

When running GSD phase commands for this study, target this subdir's planning
(`mini-inequality/.planning/`).

## Conventions (inherited)

- Python 3, **stdlib only** — no `requirements.txt`, no pandas/numpy/matplotlib.
- **No cross-module imports**: definitions are copied from `src/bias/component_suite.py`
  (`_gini`, `gold_gini`) and `mini-src/metrics.py` (enrollment, gold loaders) and
  sanity-checked for agreement, not imported.
- Benchmark/result roots derive from file location; override via `$TRANSARC_BENCHMARK` /
  `$TRANSARC_RESULTS_DIR`.
- No benchmark-derived word lists (workspace leakage rule) — distributional stats only.
