---
phase: 01-inequality-engine-ineq-01-ineq-02-ineq-03-out-01
verified: 2026-06-21T00:00:00Z
status: passed
score: 8/8 must-haves verified
overrides_applied: 0
---

# Phase 1: Inequality Engine Verification Report

**Phase Goal:** A self-contained, stdlib-only `mini-inequality/inequality.py` computes the benchmark's gold trace-link concentration inequality (Gini, Lorenz, top-k, enrollment expansion, structural component→file amplification) for both tasks across all 5 projects, writes CSV + a markdown report, and is sanity-checked against canonical numbers.
**Verified:** 2026-06-21
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

The engine was executed directly (not trusting SUMMARY claims). `python3 inequality.py` runs end-to-end for all 5 projects, writes 5 CSVs + `reports/INEQUALITY.md`, and its built-in gate reproduces every Chapter-1 GOLD literal of `writing/eval.tex` (per-sentence Gini 0.331→0.645; samcode-skew Gini 0.400→0.694 with AEs/enrolled/max exact; enrollment 525→18,660 / 35.5× / 217.6×). The gate FAILS LOUD (exit 1) on drift — demonstrated during development when an over-tight factor tolerance was caught. The engine is stdlib-only and reads no `results/` files (gold-only, per the INEQ-03 re-pivot).

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `python3 inequality.py` exits 0, emits gold inequality for all 5 projects, writes CSVs + INEQUALITY.md, runs CHECK by default | ✓ VERIFIED | Ran: exit=0, "SANITY CHECK PASSED", 6 files in reports/ |
| 2 | Per-sentence enrolled sad-code Gini reproduces tab:sent_gini within 0.005 (0.331/0.448/0.645/0.472/0.527); Top-3 % | ✓ VERIFIED | Gate rows sent_gini ×5 PASS (max |Δ|=0.0002); CSV sent_top3_pct = 27.1/35.2/20.3/21.3/70.0 |
| 3 | SAM-CODE files-per-AE reproduces tab:samcode_skew: Gini 0.400/0.694/0.452/0.513/0.612; AEs 19/19/14/22/6; enrolled 60/164/1616/730/1956; Max 16/64/348/94/972 | ✓ VERIFIED | Gate rows samcode_gini/aes/enrolled/max ×5 all PASS |
| 4 | Enrollment expansion reproduces tab:enrollment EXACTLY: enrolled 59/707/8097/1529/8268, total 18,660, factors 1.0/10.1/35.5/11.6/217.6, avg 35.5× | ✓ VERIFIED | Gate enroll_enrolled ×5 + total (18660) + raw total (525) + factor ×5 all PASS |
| 5 | Structural component→file amplification reported as a GOLD property (fan-out max 972/348; aggregate = enrollment factor); NO results/ reads, NO TransArc logic | ✓ VERIFIED | INEQUALITY.md "Structural amplification" section (max_fanout 972/348); `grep -i sadSamTlr\|load_result inequality.py` → none |
| 6 | Gini of #sentences-per-component computed for sad-code (mapped-only) & sad-sam via copied `_gini` (gold_gini agreement by construction) | ✓ VERIFIED | inequality_sad_code.csv comp_sent_gini col; inequality_sad_sam.csv comp_sent_gini col; both populated |
| 7 | On mismatch: diff table + non-zero exit; `--check-only` runs just the gate | ✓ VERIFIED | `--check-only` exit=0; earlier intentional factor-tolerance bug produced "SANITY CHECK FAILED" exit=1 (fail-loud proven) |
| 8 | Zero non-stdlib imports; no src/ or mini-src/ imports | ✓ VERIFIED | AST scan: imports = {argparse,collections,csv,json,os,pathlib,sys} only |

## Requirements Coverage

- **INEQ-01** ✓ (per-component + Lorenz + top-k, both tasks, 5 projects, CSV)
- **INEQ-02** ✓ (per-file + per-sentence distribution; the 0.331→0.645 headline)
- **INEQ-03** ✓ (enrollment expansion + GOLD structural amplification; TransArc cascade dropped per user re-pivot)
- **OUT-01** ✓ (self-contained, stdlib-only, no cross-module imports, sanity-checked vs canonical numbers)

## Notes

- Scope deviation logged: INEQ-03's cascade was re-pivoted from the (non-reproducible-from-gold) TransArc `tab:amplification` to the gold-intrinsic fan-out driver, per explicit user directive. CONTEXT.md, PLAN.md, and SUMMARY.md all record this.
- `tab:amplification` (36→3,457) remains available for Phase 2's claim audit as an explicitly system-specific (TransArc) quantity, should the paper claim need cross-referencing there.
