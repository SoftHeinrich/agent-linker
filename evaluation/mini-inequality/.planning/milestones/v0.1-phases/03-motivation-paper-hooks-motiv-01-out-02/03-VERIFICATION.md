---
phase: 03-motivation-paper-hooks-motiv-01-out-02
verified: 2026-06-21T00:00:00Z
status: passed
score: 6/6 must-haves verified
overrides_applied: 0
---

# Phase 3: Motivation & Paper Hooks Verification Report

**Phase Goal:** Show baselines exploit the inequality (file/link micro-F1 inflated), connect the inequality to the four-metric suite, and emit paper-ready table/figure source.
**Verified:** 2026-06-21
**Status:** passed

## Goal Achievement

`motivation.py` was executed directly. It scores Top-3 + random baselines on the GOLD benchmark for both tasks, reuses the Phase-1 engine, writes MOTIVATION.md + baselines.csv and the OUT-02 table/figure source, and fails loud if Top-3 does not beat random. The fail-loud gate did its job during development (it rejected an incorrect weak Top-3 definition).

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `python3 motivation.py` exits 0; writes MOTIVATION.md + baselines.csv + OUT-02 source | ✓ VERIFIED | Ran: exit 0, "MOTIVATION OK"; 5 report files present |
| 2 | Top-3 micro-F1 > random on both tasks; self-checked, fail-loud | ✓ VERIFIED | sad-code 0.353 > 0.149; sad-sam 0.381 > 0.206; `main()` exits 1 otherwise (proven when the weak Top-3 gave 0.090<0.149) |
| 3 | micro-F1 shown beside the 4-metric suite, exposing the content-blind baseline | ✓ VERIFIED | baselines.csv + MOTIVATION.md: top3 micro 0.353/0.381 vs per-comp macro F1 0.186/0.185; gold row = 1.0 ceiling |
| 4 | Driver→metric map + resolved trivial-baseline file-F1; pipeline/approach deferred | ✓ VERIFIED | MOTIVATION.md driver→metric table; "Trivial-baseline file-level F1 = 0.353"; pipeline/approach "deferred" |
| 5 | OUT-02 booktabs .tex + CSV (matching tab columns) + pgfplots Lorenz snippet | ✓ VERIFIED | out02_concentration.tex has toprule/midrule/bottomrule + 0.331; out02_concentration.csv matches engine (mediastore 0.331, teammates 0.645); out02_lorenz.tex has tikzpicture/addplot |
| 6 | Reuses import inequality; copied defs; stdlib + seeded; no system results | ✓ VERIFIED | AST scan: imports {inequality, random, sys, csv, collections}; random.Random(0); no results/ reads |

## Requirements Coverage

- **MOTIV-01** ✓ (Top-3/random baselines scored; micro-F1 inflation shown; suite motivated; per project + average)
- **OUT-02** ✓ (paper-ready Gini/Lorenz table TeX+CSV + Lorenz figure source)

## Notes

- sad-sam Top-3 micro-F1 0.381 independently matches the existing `reports/RQ2_DOC_TO_MODEL_PRESTUDY.md` (0.38) — cross-validates the gold-only reimplementation.
- Strongest-pipeline / \approach file-F1 placeholders remain deferred (need published system scores) — consistent with the gold-only scope.
