# Roadmap: Data-Inequality Mini-Study

> Isolated sub-project (`mini-inequality/`). Phase numbering is local to this study
> and independent of the repo-root roadmap (which tracks v1.2 Phases 7-9). Branch:
> `gsd/mini-data-inequality`.

## Milestones

- 🚧 **v0.1 Data-Inequality Mini-Study** — Phases 1-3 (active, defined 2026-06-21)

## Phases

### Phase 1: Inequality Engine — INEQ-01, INEQ-02, INEQ-03, OUT-01

**Goal:** A self-contained, stdlib-only `mini-inequality/inequality.py` that computes
the dataset's trace-link concentration inequality (Gini, Lorenz points, top-k share,
expansion factor, cascade) for both tasks across all 5 projects, writes CSV + a
markdown report, and is sanity-checked against canonical numbers.

**Success criteria:**
1. `python3 mini-inequality/inequality.py` emits per-project + average Gini, top-1/top-3 share, min/median/max for sad-code (component & file) and sad-sam.
2. Per-project enrollment expansion factor and the aggregate component-FP → file-FP cascade are computed and reported.
3. Gini agrees with `src/bias/component_suite.py` `gold_gini` (and `eval.tex` `tab:sent_gini` values) within a stated tolerance — printed by a `check`/sanity path.
4. Zero imports from `src/` or `mini-src/`; runs on stdlib only.

### Phase 2: Claim Verification — CLAIM-01, CLAIM-02, CLAIM-03

**Goal:** Turn the Phase-1 numbers into a paper-claim audit: extract every
distributional-inequality claim from the paper, verify it, and fill the open
placeholders.

**Success criteria:**
1. A claims checklist enumerates each inequality claim with its exact source location (alinker-paper `metric.tex`/`eval.tex`/`intro.tex` + local `eval.tex` Ch1).
2. Each claim is labeled MATCH / MISMATCH / STALE with the paper value and the computed value side by side.
3. Every `XX` placeholder in `intro.tex` has a resolved, paste-ready numeric value.
4. A verification report (`mini-inequality/CLAIM_CHECK.md`) records the audit.

### Phase 3: Motivation & Paper Hooks — MOTIV-01, OUT-02

**Goal:** Close the loop from inequality → metric suite: show baselines exploit the
inequality, and emit paper-ready table/figure source.

**Success criteria:**
1. A Top-3 baseline and a random baseline are scored, showing file/link micro-F1 is inflated by the inequality (numbers reported per project + average).
2. The report explicitly connects the inequality evidence to the need for each of the four metrics (per-component F1, sentence coverage, noise rate, file-level F1).
3. A paper-ready Gini/Lorenz/concentration table (TeX or CSV) is generated, with columns matching the paper's inequality tables; optional Lorenz-curve data emitted.

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Inequality Engine | 1/1 | ✓ Complete | 2026-06-21 |
| 2. Claim Verification | 1/1 | ✓ Complete | 2026-06-21 |
| 3. Motivation & Paper Hooks | 1/1 | ✓ Complete | 2026-06-21 |
