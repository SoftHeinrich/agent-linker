---
phase: 09-cross-gpt-5-2-cross-model-validation
plan: 04
subsystem: reporting
tags: [cross-model, gpt-5.4, s_linker13, gate-01, model-provider-property, cross-03, report]

# Dependency graph
requires:
  - phase: 09-cross-gpt-5-2-cross-model-validation/09-01
    provides: CLEAN GATE-06 audit verdict (harness + adapter shim audited; cross-model invocation surface authorised)
  - phase: 09-cross-gpt-5-2-cross-model-validation/09-02
    provides: BBB probe JSON (`ablation_20260531_055235.json`), reused for the bigbluebutton row in the report
  - phase: 09-cross-gpt-5-2-cross-model-validation/09-03
    provides: 4-dataset gpt-5.4 sweep JSON (`ablation_20260531_063446.json`) for MS/TS/TM/JAB rows; informational macro F1 0.9077
provides:
  - CROSS-03 comparison report (`09-CROSS-REPORT.md`) with explicit GATE-01 cross-model verdict, per-dataset Claude vs gpt-5.4 table, variance disclosure, model-provider-property framing, and CROSS-01/02/03 closure mapping
  - Terminal Phase 9 evidence package for the v2.0 generality claim
affects: [v2.0 milestone close-out, phase 9 transition]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Cross-model report = model-provider-property finding (NOT regression investigation); verdict-line template with no follow-on tailoring proposals"
    - "Macro F1 computed inline from per-dataset JSON values (no script artifact) — keeps the report self-contained"
    - "Cross-plan JSON merge by source-file provenance per row (4 sweep + 1 reused probe)"

key-files:
  created:
    - .planning/phases/09-cross-gpt-5-2-cross-model-validation/09-CROSS-REPORT.md
    - .planning/phases/09-cross-gpt-5-2-cross-model-validation/09-04-SUMMARY.md
  modified: []

key-decisions:
  - "GATE-01 cross-model verdict = does NOT hold (macro 0.9077 < 0.93 on gpt-5.4). Framed as model-provider-property finding per D-05, NOT as a defect in s_linker13."
  - "Teammates dominates the macro gap (Δ +0.1531, ~3.06pp of the macro Δ); pattern mirrors v1.0 13d/VAR-04 dotted-path / Java-package convention negative result."
  - "V32 prior cross-model numbers (gpt-5.2 0.906, gpt-5.4 0.877) cited as background only per D-06 — NOT ground truth for s_linker13."
  - "Report is terminal — no follow-on tailoring proposals per D-04 + scope_reduction_prohibition. Phase 9 ships regardless of GATE-01 cross-model outcome per D-05 v2.0 ship rule."
  - "Per-dataset baseline values quoted at rounded precision (MS 0.984, TS 1.000, TM 0.947, BBB 0.821, JAB 1.000) per ABLATION-TABLE.md; macro at 4-decimal precision (0.9506) per D-47a / D-44a JSON-wins precision rule."
  - "BBB row reuses Plan 09-02 probe JSON (`ablation_20260531_055235.json`) per D-03 Step 3 default no-retest policy."

patterns-established:
  - "Cross-model comparison report template: Scope → Per-dataset table → GATE verdict (verbatim template) → Variance disclosure → Model-provider-property framing → Background-context prior evidence → Requirement closure → GATE-06 pointer."
  - "Inline macro F1 verification at report-write time (no script artifact in repo); JSON provenance recorded per row."

requirements-completed: [CROSS-03]  # CROSS-01 + CROSS-02 already marked completed in 09-03-SUMMARY.md; this plan closes the report-deliverable side

# Metrics
duration: ~15min
completed: 2026-05-31
---

# Phase 9 Plan 04: CROSS-03 Comparison Report Summary

**One-liner:** Authored `09-CROSS-REPORT.md` documenting `s_linker13` on gpt-5.4 across all 5 datasets vs the Claude Sonnet v1.0 baseline; macro F1 0.9077 < GATE-01 floor 0.93 (Δ +0.0429 vs Claude 0.9506); verdict framed as model-provider-property finding per D-05, with teammates (Δ +0.1531) as the dominant driver.

## Performance

- **Duration:** ~15 min (plan execution wall-clock; single auto task; no LLM calls, no re-runs)
- **Started:** 2026-05-31 (Plan 09-04 kickoff)
- **Completed:** 2026-05-31
- **Tasks:** 1 (single auto task — author 09-CROSS-REPORT.md)
- **Files created:** 2 (report + this summary)

## Accomplishments

- Created `09-CROSS-REPORT.md` with all 9 required sections (header, scope, per-dataset table, GATE-01 verdict, variance disclosure, model-provider-property framing, background prior evidence, requirement closure, GATE-06 reference).
- Populated per-dataset comparison table from the documented JSON sources (4 rows from Plan 09-03 sweep `ablation_20260531_063446.json` + 1 row reused from Plan 09-02 probe `ablation_20260531_055235.json`); deltas computed inline.
- Verdict line uses template (B) verbatim per the plan's `<interfaces>` block: "GATE-01 does NOT hold cross-model: macro F1 = 0.9077 < 0.93 on gpt-5.4 for s_linker13."
- Variance disclosure cites memory's 5–12 link stdev GPT band + the BBB intra-provider jitter prior (Phase 6 06-SUMMARY.md ~4pp same-session jitter on Claude).
- Background prior evidence section cites V32 gpt-5.2 0.906 and gpt-5.4 0.877 as background only (D-06), not as ground truth for `s_linker13`; notes the informational +3.1pp gain of `s_linker13` vs V32 on the same gpt-5.4 backend.
- Requirement closure section marks CROSS-01 (per-dataset JSON evidence), CROSS-02 (collapsed per D-02), CROSS-03 (this report) all SATISFIED.

## GATE-01 Cross-Model Verdict (verbatim from §3 of 09-CROSS-REPORT.md)

> **GATE-01 does NOT hold cross-model: macro F1 = 0.9077 < 0.93 on gpt-5.4 for s_linker13. Per the standing v2.0 policy (PROJECT.md Key Decisions; CONTEXT D-05), this is a model-provider-property finding, NOT a defect in s_linker13. The harness was audited CLEAN under GATE-06 (09-01-SUMMARY.md), no backend-specific tailoring was introduced, and the result represents the inherent capability difference of gpt-5.4 relative to Claude Sonnet on this task.**

## Macro F1 Reproduction (per §2 of 09-CROSS-REPORT.md)

| Dataset       | Claude Sonnet F1 | gpt-5.4 F1 | Δ (Claude − gpt-5.4) |
|---------------|------------------|------------|----------------------|
| mediastore    | 0.984            | 0.9677     | +0.0163              |
| teastore      | 1.000            | 1.0000     | +0.0000              |
| teammates     | 0.947            | 0.7939     | +0.1531              |
| bigbluebutton | 0.821            | 0.8037     | +0.0173              |
| jabref        | 1.000            | 0.9730     | +0.0270              |
| **Macro**     | **0.9506**       | **0.9077** | **+0.0429**          |

Teammates contributes ~3.06pp of the 4.29pp macro Δ on its own.

## Requirement Closure Mapping

| Requirement | Status     | Evidence in this plan                                                                 |
|-------------|------------|---------------------------------------------------------------------------------------|
| CROSS-01    | SATISFIED  | All 5-dataset JSON evidence cited in `09-CROSS-REPORT.md` §2 (sweep + reused probe). |
| CROSS-02    | SATISFIED  | D-02 collapse documented in `09-CROSS-REPORT.md` §1 + §7 (no `s_linker14.py`; `s_linker13` IS the COMBINE artifact per Phase 8 retro-designation; `08-SUMMARY.md`). |
| CROSS-03    | SATISFIED  | `09-CROSS-REPORT.md` is the report deliverable; this SUMMARY closes the plan side.    |

## Task Commits

1. **Task 1: Author 09-CROSS-REPORT.md** — `db67f72` (docs)

Plan metadata finalised in the next commit (`docs(09-04): finalize summary — CROSS-03 closed`).

## Files Created/Modified

- `.planning/phases/09-cross-gpt-5-2-cross-model-validation/09-CROSS-REPORT.md` — the CROSS-03 deliverable (9 sections, per-dataset table, GATE-01 verdict, model-provider-property framing, requirement closure).
- `.planning/phases/09-cross-gpt-5-2-cross-model-validation/09-04-SUMMARY.md` — this plan close-out.

## Decisions Made

- **Verdict template (B) selected.** Computed macro F1 0.9077 < 0.93 threshold → verbatim template (B) from the plan's `<interfaces>` block applied. Template (A) discarded.
- **Per-dataset baseline values quoted at rounded precision per ABLATION-TABLE.md** (MS 0.984, TS 1.000, TM 0.947, BBB 0.821, JAB 1.000); the macro is reported at 4-decimal precision (0.9506) to match the Claude baseline notation per D-47a / D-44a (JSON wins on any discrepancy; the ABLATION-TABLE row uses 4-decimal macro).
- **Teammates regression pattern attribution.** Report notes the teammates Δ +0.1531 mirrors the v1.0 13d / VAR-04 negative result (dotted-path / Java-package casing convention is regex territory that pure-LLM emission cannot reproduce reliably). This is observation, not a proposal — D-04 / D-05 prohibit closing the gap.
- **Background V32 numbers cited as context only per D-06.** The +3.1pp informational gain of `s_linker13` over V32 on the same gpt-5.4 backend is noted as context, not as a goal of Phase 9.
- **No new linker or prompt files were created** — verified via `git status --short` showing only `.planning/` updates in this plan's diff. Plan 09-01's CLEAN GATE-06 audit carries forward unchanged.

## Deviations from Plan

None — plan executed exactly as written. The plan's automated verification one-liner (single command testing: file existence + GATE-01 line + model-provider-property + macro 0.9506 + variance/jitter token + zero forbidden-token hits) passed on first attempt after one wording tweak that removed an inadvertent "action item" phrase from the framing paragraph (the phrase was meta-descriptive — "this report does NOT contain action items" — but the verify regex was lexical, not semantic, so it was reworded to "this report is deliberately terminal").

## Issues Encountered

None blocking. The single edit recorded above (reword "does NOT contain action items" → "is deliberately terminal") was a lexical-match fix; the meaning of the paragraph was preserved.

## User Setup Required

None — report-authoring plan, no external service configuration introduced, no LLM calls made.

## Hand-Off

- **Phase 9 ready to close.** All 4 plans complete: 09-01 (GATE-06 audit), 09-02 (BBB probe + go signal), 09-03 (4-dataset sweep + reused probe = full 5-dataset evidence), 09-04 (CROSS-03 report).
- **v2.0 milestone unblocked for `/gsd-transition` → milestone-complete.** GATE-01 cross-model verdict is published; Phase 9 ships per D-04 v2.0 ship rule (both verdicts are terminal states).
- **No follow-up plans queued.** Per CONTEXT §deferred + D-04 + scope_reduction_prohibition, no fix-it work is proposed.

## Self-Check: PASSED

- `09-CROSS-REPORT.md` exists at `.planning/phases/09-cross-gpt-5-2-cross-model-validation/09-CROSS-REPORT.md` — verified via `test -f`.
- Plan's automated verification one-liner passes (file exists; GATE-01 line present; "model-provider-property" present; Macro 0.9506 present; "variance" or "jitter" present; zero forbidden-token hits) — verified end-to-end after the lexical-match wording fix recorded under §Deviations.
- Task 1 commit `db67f72` exists in `git log --oneline` — verified.
- No edits to `run_ablation.py`, `llm_client.py`, `s_linker13.py`, or any `prompts*.py` — verified via `git status --short` showing only `.planning/` changes.
- All 5 dataset rows in §2 of the report resolve to a JSON file under `results/ablation_results/` — verified by file presence (`ablation_20260531_063446.json` for MS/TS/TM/JAB; `ablation_20260531_055235.json` for BBB).

---
*Phase: 09-cross-gpt-5-2-cross-model-validation*
*Plan: 04 — CROSS-03 close-out*
*Completed: 2026-05-31*
