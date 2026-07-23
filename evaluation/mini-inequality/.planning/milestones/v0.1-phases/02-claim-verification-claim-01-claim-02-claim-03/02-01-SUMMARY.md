---
phase: 02-claim-verification-claim-01-claim-02-claim-03
plan: 02-01
subsystem: data-analysis
tags: [claim-audit, traceability, paper, placeholders, reuse]

requires:
  - phase: 01-inequality-engine-ineq-01-ineq-02-ineq-03-out-01
    provides: inequality.py engine + reports/* gold numbers (reused via import)
provides:
  - claim_check.py — stdlib audit reusing the engine to verify paper inequality claims
  - CLAIM_CHECK.md — MATCH/MISMATCH/PARTIAL/SYSTEM-SPECIFIC table + resolved XX placeholders
affects: [Phase 3 Motivation & Paper Hooks]

tech-stack:
  added: []
  patterns:
    - "Self-contained reuse: Phase-2 audit imports the study's own engine (import inequality), not src/mini-src"
    - "Fail-loud audit: an expected-MATCH claim that does not match exits non-zero"

key-files:
  created:
    - mini-inequality/claim_check.py
    - mini-inequality/CLAIM_CHECK.md
  modified: []

key-decisions:
  - "Cascade/amplification claim recorded as SYSTEM-SPECIFIC (TransArc actual-error attribution), not audited as MATCH/MISMATCH"
  - "Gold-derivable XX placeholders resolved now (5 projects, 4 metrics, 70% JabRef); baseline/pipeline-F1 placeholders deferred to Phase 3"
  - "C7 (top-AE 44-48%) labelled PARTIAL — only JabRef reproduces; the paper's coarse single-AE grouping differs from the engine's multi-mapped component_suite universe"

patterns-established:
  - "Claim→source→computed→label audit table generated reproducibly from the engine"

requirements-completed: [CLAIM-01, CLAIM-02, CLAIM-03]

duration: ~20min
completed: 2026-06-21
---

# Phase 2: Claim Verification — Summary

**The paper's gold distributional-inequality claims are now audited against the Phase-1 engine — 6 MATCH, 1 PARTIAL, 1 SYSTEM-SPECIFIC — and the gold-derivable `intro.tex` placeholders are resolved (5 projects, 4 metrics, 70% gold mass on 3 JabRef sentences).**

## Accomplishments

- **CLAIM-01** — every gold distributional-inequality claim enumerated with its exact source location across alinker `metric.tex`/`eval.tex`/`intro.tex` + local `writing/eval.tex` Ch1.
- **CLAIM-02** — each claim labelled with paper value vs engine-computed value: C1 expansion 1.0×→217.6×, C2 JabRef fan-out 972, C3 long-tail both tasks, C4 per-sentence Gini 0.331→0.645, C5 70% on 3 sentences, C6 samcode Gini 0.400→0.694 / JabRef 98.6% — all **MATCH**; C7 top-AE 44-48% **PARTIAL** (only JabRef reproduced — granularity difference, documented); C8 cascade 36→3,457 **SYSTEM-SPECIFIC** (excluded TransArc quantity).
- **CLAIM-03** — `intro.tex` placeholders resolved: **5** projects, **4** metrics, **70%** (JabRef, 3 sentences). Baseline/pipeline file-F1 placeholders (L17/54/64) explicitly deferred → Phase 3 (need system scores).

## Honesty notes

- **C7 PARTIAL** was a deliberate down-grade from an over-eager MATCH: the engine's per-project top-AE link shares (e.g. Teammates 22.5%, BBB 14.5%) do NOT reproduce `tab:sadcode_conc` (44.7%, 47.9%) because the paper uses a single coarse top-level component per file while the engine uses the multi-mapped `component_suite` universe. Only JabRef (47.0%) matches. Documented in CLAIM_CHECK.md.

## Verification

`python3 claim_check.py` exits 0, prints "CLAIM CHECK OK (6 MATCH, 0 unexpected MISMATCH)", writes CLAIM_CHECK.md. Reuses `import inequality` (no `src/`/`mini-src/` imports). Fail-loud self-check present.
