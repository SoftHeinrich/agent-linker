---
phase: 01-inequality-engine-ineq-01-ineq-02-ineq-03-out-01
plan: 01-01
subsystem: data-analysis
tags: [gini, lorenz, enrollment, benchmark, inequality, stdlib, ardoco]

requires: []
provides:
  - Self-contained stdlib-only gold-inequality engine (mini-inequality/inequality.py)
  - Per-sentence / per-file / per-component gold concentration (Gini, Lorenz, top-k, min/median/max) for sad-code & sad-sam
  - Enrollment expansion factor (tab:enrollment) and the GOLD structural component→file amplification driver (tab:samcode_skew fan-out)
  - Paper-facing report (reports/INEQUALITY.md) + 5 CSVs (incl. pgfplots Lorenz data)
  - A fail-loud sanity gate that reproduces every eval.tex Ch1 GOLD literal exactly
affects: [Phase 2 Claim Verification, Phase 3 Motivation & Paper Hooks]

tech-stack:
  added: []
  patterns:
    - "Copy-not-import: definitions duplicated from mini-src/metrics.py + component_suite._gini, sanity-checked for agreement (isolation rule)"
    - "Self-validating engine: recompute vs frozen paper literals, fail loud (non-zero exit) on drift"

key-files:
  created:
    - mini-inequality/inequality.py
    - mini-inequality/reports/INEQUALITY.md
    - mini-inequality/reports/inequality_sad_code.csv
    - mini-inequality/reports/inequality_sad_sam.csv
    - mini-inequality/reports/inequality_samcode_skew.csv
    - mini-inequality/reports/inequality_expansion.csv
    - mini-inequality/reports/lorenz_sad_code_sentence.csv
  modified: []

key-decisions:
  - "INEQ-03 cascade RE-PIVOTED to a GOLD structural amplification (files-per-component fan-out); the TransArc actual-error cascade (eval.tex tab:amplification 36→3,457) is system-specific and excluded (user directive 2026-06-21)"
  - "Sanity gate compares Gini within ±0.005 and integer counts exactly; enrollment factor compared at the paper's 1-decimal precision"
  - "Engine reads NO results/ files — purely gold/benchmark"

patterns-established:
  - "Frozen-literal sanity gate: EXPECTED dict mirrors eval.tex tables; run_check() prints a diff table and exits non-zero on any mismatch"

requirements-completed: [INEQ-01, INEQ-02, INEQ-03, OUT-01]

duration: ~40min
completed: 2026-06-21
---

# Phase 1: Inequality Engine — Summary

**A stdlib-only, self-contained engine now measures the ARDoCo benchmark's gold trace-link concentration inequality and reproduces every Chapter-1 inequality number of `writing/eval.tex` exactly, self-checking with a fail-loud gate.**

## Performance

- **Duration:** ~40 min (incl. one scope re-pivot)
- **Completed:** 2026-06-21
- **Tasks:** 4 (scaffold → gold distributions → expansion/amplification → report + sanity gate)
- **Files created:** 7 (engine + 6 generated reports)

## Accomplishments

- **INEQ-01** — per-component gold concentration (Gini, top-k, min/median/max) for sad-code (mapped-only universe, agreeing by construction with `component_suite.gold_gini`) and sad-sam; full Lorenz curve emitted for the per-sentence distribution.
- **INEQ-02** — per-sentence enrolled sad-code distribution reproducing `tab:sent_gini` (Gini **0.331→0.645**, Top-3 %, min/median/max) and a per-file concentration view.
- **INEQ-03** — enrollment expansion reproducing `tab:enrollment` exactly (**525 → 18,660**, 35.5× avg, **217.6×** JabRef) and the GOLD structural component→file amplification = the SAM-CODE files-per-component fan-out (`tab:samcode_skew`, Gini 0.400→0.694, **max 972** = JabRef `logic`, 348 = Teammates `ui`).
- **OUT-01** — self-contained `inequality.py` (stdlib only, zero `src/`/`mini-src/` imports, no `results/` reads) + `reports/` MD/CSV, with a default-on sanity gate (`--check-only` to run just the gate).

## Key Deviation — INEQ-03 re-pivot

Mid-execution, the naive forward cascade (Σ|files(m)| over a system's sad-sam FPs) did **not** reproduce `eval.tex` `tab:amplification` (40→4,697 vs 36→3,457). Investigation showed that table is a **TransArc actual-error attribution** (real sad-code FPs decomposed by transitive cause, from `reports/TRANSARC_EMPIRICAL_STUDY.md`), not a gold property — it belongs to the separate TransArc empirical pillar. Per user directive ("do not use anything TransArc-specific; re-pivot towards benchmark distribution"), INEQ-03's cascade was re-pivoted to the gold-intrinsic fan-out driver, and all system-result inputs were removed.

## Verification

`python3 inequality.py` exits 0 and prints **SANITY CHECK PASSED**; all 39 gate rows PASS (per-sentence Gini ×5, samcode Gini/AEs/enrolled/max ×5 each, enrollment enrolled ×5 + factor ×5 + totals). `--check-only` exits 0; an AST scan confirms stdlib-only imports; the engine contains no `sadSamTlr`/`load_result` paths.
