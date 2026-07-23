---
phase: 02-claim-verification-claim-01-claim-02-claim-03
verified: 2026-06-21T00:00:00Z
status: passed
score: 6/6 must-haves verified
overrides_applied: 0
---

# Phase 2: Claim Verification Report

**Phase Goal:** Turn the Phase-1 numbers into a paper-claim audit — extract every distributional-inequality claim, verify it MATCH/MISMATCH/STALE, and fill the open `XX` placeholders.
**Verified:** 2026-06-21
**Status:** passed

## Goal Achievement

`claim_check.py` was executed directly. It reuses the Phase-1 engine (`import inequality`), enumerates all gold distributional-inequality claims with exact source locations, labels each against the engine-computed value, resolves the gold-derivable placeholders, and writes `CLAIM_CHECK.md`. The audit is honest: a claim that did not actually reproduce (C7) was labelled PARTIAL with a documented reason rather than forced to MATCH, and the system-specific cascade was excluded rather than mislabelled.

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `python3 claim_check.py` exits 0, writes CLAIM_CHECK.md | ✓ VERIFIED | Ran: exit 0, "CLAIM CHECK OK (6 MATCH, 0 unexpected MISMATCH)" |
| 2 | Every gold inequality claim enumerated with exact source location (CLAIM-01) | ✓ VERIFIED | CLAIM_CHECK.md Claims table: C1–C8 with `metric.tex:11`, `eval.tex:23/25`, `tab:sent_gini`, `tab:samcode_skew`, `tab:sadcode_conc`, `tab:enrollment`, `tab:amplification` |
| 3 | Each claim labelled with paper vs computed value (CLAIM-02) | ✓ VERIFIED | 6 MATCH (C1–C6), 1 PARTIAL (C7), 1 SYSTEM-SPECIFIC (C8); paper + computed columns populated |
| 4 | Gold-derivable XX placeholders resolved paste-ready (CLAIM-03) | ✓ VERIFIED | Resolved table: 5 projects, 4 metrics, 70% (JabRef); baseline/pipeline F1 → "deferred → Phase 3" |
| 5 | Reuses engine; no src/ or mini-src/ imports | ✓ VERIFIED | AST scan: imports = {inequality, sys, collections}; `import inequality` present |
| 6 | Fail-loud self-check (expected-MATCH not matching → non-zero exit) | ✓ VERIFIED | `main()` collects expect=="MATCH" rows not labelled MATCH and `sys.exit(1)`; C1–C6 all MATCH so exit 0 |

## Requirements Coverage

- **CLAIM-01** ✓ (claims checklist with source locations)
- **CLAIM-02** ✓ (MATCH/MISMATCH/PARTIAL/SYSTEM-SPECIFIC + paper vs computed)
- **CLAIM-03** ✓ (gold-derivable placeholders resolved; rest deferred to Phase 3)

## Notes

- C7 (top-AE 44-48%) is PARTIAL by design — only JabRef reproduces; the divergence (coarse single-AE vs multi-mapped component universe) is documented in CLAIM_CHECK.md. Not a defect; an honest result.
- The cascade claim (36→3,457) is SYSTEM-SPECIFIC, consistent with the Phase-1 re-pivot.
- Baseline/pipeline-F1 placeholders remain for Phase 3 (MOTIV-01).
