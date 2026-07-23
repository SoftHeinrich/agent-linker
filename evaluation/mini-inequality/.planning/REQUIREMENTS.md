# Requirements: Data-Inequality Mini-Study

**Defined:** 2026-06-21
**Core Value:** Faithful, reproducible inequality numbers that ground every paper claim (and fill every `XX` placeholder) and motivate the four-metric suite.

> Isolated sub-project — see `mini-inequality/.planning/PROJECT.md`. Phase numbering
> is local to this study (Phases 1-3) and independent of the repo-root roadmap.

## v0.1 Requirements

### INEQ — Inequality Measurement

- [ ] **INEQ-01**: Compute per-component gold link-count inequality — Gini, Lorenz-curve points, top-k concentration share (top-1/top-3), and min/median/max links-per-component — for both `sad-code` (component level) and `sad-sam`, across all 5 projects, emitted to CSV.
- [ ] **INEQ-02**: Compute per-file (post-enrollment) link concentration and the per-sentence gold links-per-sentence distribution for `sad-code` (mean, max, skew/top-share).
- [ ] **INEQ-03**: Compute the per-project enrollment expansion factor (component decision → file-level pairs) and the aggregate component-FP → file-FP cascade, reproducing the dataset-level inequality drivers.

### CLAIM — Paper Claim Verification

- [ ] **CLAIM-01**: Extract the paper's data-inequality claims into a checklist with exact source locations — `alinker-paper/sections/{metric,eval,intro}.tex` plus local `writing/eval.tex` Ch1 (expansion 1.0×→217.6×, Gini 0.331→0.645, 96.0× / 36→3,457 cascade, long-tail-both-tasks, Top-3/random exploitability).
- [ ] **CLAIM-02**: Verify each extracted claim against the computed numbers, labeling MATCH / MISMATCH / STALE and reporting the empirical value beside the paper's value.
- [ ] **CLAIM-03**: Resolve the `XX` placeholders in `alinker-paper/sections/intro.tex` (e.g. `XX`% gold-mass concentration on three sentences of one project; trivial-baseline file-level F1) with computed values, ready to paste.

### MOTIV — Metric Motivation

- [ ] **MOTIV-01**: Demonstrate empirically that file-level / link-level micro-F1 is dominated by the long tail and a few large components — via a Top-3 baseline and a random baseline that exploit the inequality — establishing why the four-metric suite (per-component F1, sentence coverage, noise rate, file-level F1) is needed.

### OUT — Outputs

- [ ] **OUT-01**: Self-contained study artifacts in `mini-inequality/` — `inequality.py` (stdlib only, no cross-module imports), a `README.md`, and a generated report (markdown + CSV) — including a sanity check that the Gini agrees with `src/bias/component_suite.py` `gold_gini` and the `eval.tex` inequality tables to a stated tolerance.
- [ ] **OUT-02**: Paper-ready output — a Gini/Lorenz/concentration table (TeX or CSV the paper ingests) and optionally Lorenz-curve data (pgfplots-friendly), matching the columns the paper's inequality tables expect.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Writing to repo-root `.planning/` / the v1.2 milestone | Strict isolation per user requirement |
| New trace-link recovery / a linker | This study measures the dataset + scores existing results only |
| Re-deriving the 13-column metric panel | `mini-src/` already covers per-system metrics; this is about the distribution |
| pandas / numpy / matplotlib | Stdlib-only; figures as pgfplots/TeX or plain data |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| INEQ-01 | Phase 1 | Pending |
| INEQ-02 | Phase 1 | Pending |
| INEQ-03 | Phase 1 | Pending |
| OUT-01  | Phase 1 | Pending |
| CLAIM-01 | Phase 2 | Pending |
| CLAIM-02 | Phase 2 | Pending |
| CLAIM-03 | Phase 2 | Pending |
| MOTIV-01 | Phase 3 | Pending |
| OUT-02  | Phase 3 | Pending |

**Coverage:**
- v0.1 requirements: 9 total
- Mapped to phases: 9
- Unmapped: 0 ✓

---
*Requirements defined: 2026-06-21*
*Last updated: 2026-06-21 after initial definition*
