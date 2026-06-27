---
phase: 10-scaffolding
plan: 04
subsystem: planning-docs
tags: [gate, cross-model, governance, v2.1]
requires:
  - .planning/REQUIREMENTS.md GATE-01 (loose ≤ 1pp phrasing)
  - .planning/PROJECT.md Key Decisions table
  - .planning/STATE.md Standing Gates section
  - .planning/milestones/v2.0-MILESTONE-AUDIT.md (0.9077 v2.0 CROSS baseline provenance)
provides:
  - Concrete cross-model tolerance T = 1.0pp pinned in two canonical locations
  - Absolute F1 floor 0.8977 derived and recorded
  - Deterministic gate criterion for Phase 12 (PROMPT-02) trim acceptance
  - Deterministic gate criterion for Phase 13 (PROMPT-03) final sweep promotion
affects:
  - Phase 12 PROMPT-02 per-prompt rule-trim ablation acceptance logic
  - Phase 13 PROMPT-03 final-variant promotion sweep
tech-stack:
  added: []
  patterns:
    - additive-edits-only
key-files:
  modified:
    - .planning/PROJECT.md (Key Decisions table: +1 row)
    - .planning/STATE.md (Standing Gates section: 1 line replaced)
  created: []
decisions:
  - GATE-01 cross-model tolerance T = 1.0pp (variant passes iff gpt-5.4 macro F1 ≥ 0.8977 on full 5-dataset sweep)
metrics:
  duration: ~5min
  completed: 2026-05-31
  tasks: 2
  files-touched: 2
commits:
  - 7cae749 docs(10-04): codify GATE-01 cross-model tolerance T=1.0pp in PROJECT.md
  - 74be23c docs(10-04): commit concrete cross-model tolerance T=1.0pp in STATE.md Standing Gates
---

# Phase 10 Plan 10-04: Cross-Model Gate Codify Summary

GATE-01 cross-model tolerance pinned to T = 1.0pp (absolute floor 0.8977) in both PROJECT.md Key Decisions and STATE.md Standing Gates so Phase 12/13 trim acceptance and promotion sweeps can be evaluated deterministically.

## Tasks Completed

| Task | Name                                                                          | Commit  | Files                  |
| ---- | ----------------------------------------------------------------------------- | ------- | ---------------------- |
| 1    | Codify GATE-01 cross-model in PROJECT.md Key Decisions                        | 7cae749 | .planning/PROJECT.md   |
| 2    | Replace loose tolerance in STATE.md Standing Gates with concrete T = 1.0pp    | 74be23c | .planning/STATE.md     |

## Committed Values

- **Baseline:** 0.9077 (gpt-5.4 macro F1 — v2.0 CROSS evidence)
- **Tolerance:** T = 1.0pp
- **Absolute F1 floor:** 0.8977 (= 0.9077 − 0.01)
- **Model:** gpt-5.4
- **Evaluation:** full 5-dataset sweep

## Exact Diff — PROJECT.md

Single additive row appended to the end of the "## Key Decisions" Markdown table (before the "## Evolution" section header):

```diff
+| GATE-01 cross-model tolerance T = 1.0pp (v2.1) | Pins the loose REQUIREMENTS GATE-01 phrasing "≤ 1pp regression" to a concrete numeric tolerance so Phase 12 trim acceptance and Phase 13 promotion sweeps can be evaluated deterministically. Baseline 0.9077 is the v2.0 CROSS evidence on gpt-5.4 (see v2.0-MILESTONE-AUDIT.md "09-CROSS-REPORT.md §GATE-01"). T = 1.0pp means a variant passes iff gpt-5.4 macro F1 ≥ 0.9077 − 0.01 = 0.8977 absolute on the full 5-dataset sweep. | Codified 2026-05-31 (Phase 10, Plan 10-04) |
```

`git diff --stat .planning/PROJECT.md` → 1 file changed, 1 insertion(+), 0 deletions.

No other row in the Key Decisions table modified; "## Evolution", "## Constraints", and all other sections byte-identical.

## Exact Diff — STATE.md

Single-line replacement inside "## Standing Gates (v2.1)" section:

```diff
-- GATE-01 cross-model (v2.1 NEW): gpt-5.4 macro ≥ 0.9077 within ≤ 1pp tolerance (T to be committed in Phase 10)
+- GATE-01 cross-model (v2.1): gpt-5.4 macro ≥ 0.9077 within ≤ 1.0pp tolerance — i.e. variant passes iff gpt-5.4 macro F1 ≥ 0.8977 on full 5-dataset sweep (T = 1.0pp committed Phase 10 Plan 10-04; baseline 0.9077 from v2.0 CROSS evidence; see PROJECT.md Key Decisions row "GATE-01 cross-model tolerance T = 1.0pp (v2.1)")
```

`git diff --stat .planning/STATE.md` → 1 file changed, 1 insertion(+), 1 deletion(-).

GATE-01 Claude row, GATE-02 (v2.1 NEW), GATE-06, and GATE-07 byte-identical. YAML frontmatter unchanged. Current Position, Performance Metrics, Accumulated Context, Deferred Items, and Session Continuity all unchanged.

## Cross-Document Consistency

Both files now share:

| Field          | Value                                            |
| -------------- | ------------------------------------------------ |
| Model          | gpt-5.4                                          |
| Baseline       | 0.9077                                           |
| Tolerance T    | 1.0pp                                            |
| Absolute floor | 0.8977                                           |
| Evaluation     | full 5-dataset sweep                             |
| Provenance     | Phase 10, Plan 10-04 (2026-05-31)                |
| Evidence ref   | v2.0 CROSS evidence / v2.0-MILESTONE-AUDIT.md    |

STATE.md row contains the literal grep-discoverable string
`gpt-5.4 macro ≥ 0.9077 within ≤ 1.0pp tolerance`, allowing the GATE-02
regression test (Plan 10-01) and future audits to find it with a single pattern.

## Verification

All five plan-level checks pass:

1. `grep -F "GATE-01 cross-model tolerance T = 1.0pp" .planning/PROJECT.md` → 1 match
2. `grep -F "gpt-5.4 macro ≥ 0.9077 within ≤ 1.0pp tolerance" .planning/STATE.md` → 1 match
3. `grep -F "0.8977" .planning/PROJECT.md .planning/STATE.md` → 2 matches (one per file)
4. `grep -F "T to be committed in Phase 10" .planning/STATE.md` → 0 matches (loose phrasing removed)
5. Both files reference provenance "Phase 10[, ] Plan 10-04"

## Deviations from Plan

None — plan executed exactly as written. Both tasks were pure additive/single-line replacement edits with no Rule 1/2/3 issues encountered.

## Self-Check: PASSED

- PROJECT.md contains the literal "GATE-01 cross-model tolerance T = 1.0pp" decision row in the Key Decisions table between "## Key Decisions" and "## Evolution".
- STATE.md contains the literal "gpt-5.4 macro ≥ 0.9077 within ≤ 1.0pp tolerance" line in Standing Gates.
- Both files contain the absolute floor "0.8977" exactly once.
- Loose phrasing "T to be committed in Phase 10" removed from STATE.md (0 matches).
- Sibling Standing Gates rows (GATE-01 Claude, GATE-02, GATE-06, GATE-07) confirmed byte-identical.
- "## Evolution" and "## Constraints" sections of PROJECT.md preserved byte-identical.
- Commits 7cae749 (PROJECT.md) and 74be23c (STATE.md) exist on master.
