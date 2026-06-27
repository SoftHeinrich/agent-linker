---
phase: 51-noknow
plan: 03
subsystem: infra
tags: [sweep-scripts, bash, no-knowledge, ablation, cost-logging, resume-markers, gpt, sonnet]

requires:
  - phase: 51-noknow/51-01
    provides: "s_linker20_union_noknow variant registered in run_ablation.py"

provides:
  - "run_s20union_noknow_gpt_n3.sh: resumable GPT No-Knowledge N=3 sweep script into results/v2.6.6_s20union_noknow/gpt"
  - "run_s20union_noknow_sonnet_n3.sh: resumable sonnet No-Knowledge N=3 sweep script into results/v2.6.6_s20union_noknow_sonnet"

affects: [51-04, 51-05]

tech-stack:
  added: []
  patterns:
    - "Per-run PHASE_CACHE_DIR isolation under annotated _noknow BASE (prevents Full cache clobber)"
    - "CUM_CALLS counter + COST progress-log line after each dataset for D-08 cost visibility"

key-files:
  created:
    - run_s20union_noknow_gpt_n3.sh
    - run_s20union_noknow_sonnet_n3.sh
  modified: []

key-decisions:
  - "GPT root: results/v2.6.6_s20union_noknow/gpt (with /gpt/ sub-level, mirrors Full gpt asymmetry)"
  - "Sonnet root: results/v2.6.6_s20union_noknow_sonnet (no /sonnet/ sub-level per Q7 asymmetry)"
  - "Cost logging is call-count only (no dollar field in _calls.json per Landmine 5); actual $ via API dashboard"
  - "Landmine-3 warning comment added to PHASE_CACHE_DIR line in both scripts"
  - "Cost logging block is byte-consistent between gpt and sonnet scripts (same CUM_CALLS logic, same COST line format)"

patterns-established:
  - "No-Knowledge sweep scripts are static-validated only in this plan (bash -n); live execution is plan 51-04"

requirements-completed: [NOKNOW-02]

duration: 15min
completed: 2026-06-21
---

# Phase 51 Plan 03: No-Knowledge N=3 Sweep Scripts Summary

**Two resumable No-Knowledge sweep scripts authored for s_linker20_union_noknow — GPT into results/v2.6.6_s20union_noknow/gpt and sonnet into results/v2.6.6_s20union_noknow_sonnet — with per-dataset cumulative call-count logging and per-run PHASE_CACHE_DIR isolation (GATE-01 safety)**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-06-21T21:15:00Z
- **Completed:** 2026-06-21T21:30:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Authored `run_s20union_noknow_gpt_n3.sh`: verbatim copy of the Full gpt skeleton with VARIANT, BASE, and LOGBASE changed to annotated `_noknow` values; cumulative call-count logging block added; Landmine-3 warning comment on PHASE_CACHE_DIR
- Authored `run_s20union_noknow_sonnet_n3.sh`: verbatim copy of the Full sonnet skeleton with symmetric changes; sonnet root has NO extra `/sonnet/` level per Q7 asymmetry; cost-logging block byte-consistent with gpt script
- Both scripts pass `bash -n` syntax check and are executable (chmod +x); all five acceptance criteria verified per task

## Task Commits

Each task was committed atomically:

1. **Task 1: Author run_s20union_noknow_gpt_n3.sh** - `587bf00` (feat)
2. **Task 2: Author run_s20union_noknow_sonnet_n3.sh** - `6cc345e` (feat)

## Files Created/Modified
- `run_s20union_noknow_gpt_n3.sh` - GPT No-Knowledge N=3 sweep: VARIANT=s_linker20_union_noknow, BASE=results/v2.6.6_s20union_noknow/gpt, D-08 call-count logging, per-run PHASE_CACHE_DIR isolation
- `run_s20union_noknow_sonnet_n3.sh` - sonnet No-Knowledge N=3 sweep: same VARIANT, BASE=results/v2.6.6_s20union_noknow_sonnet (no extra /sonnet/ level), same call-count logging

## Decisions Made
- Cumulative call-count (`CUM_CALLS`) is the cost metric because `_calls.json` records have `token_usage` but no dollar field (Landmine 5); PROGRESS log entry notes "actual $ via API dashboard"
- GPT and sonnet cost-logging blocks are byte-consistent (same python3 one-liner, same COST line format) to ease cross-script comparison
- PHASE_CACHE_DIR per-run isolation retains the Full script's exact pattern (`"$rdir/phase_cache"` under the current run's `BASE/run$i`); the Landmine-3 comment explains why a shared path would clobber Full caches

## Deviations from Plan

None - plan executed exactly as written. Both scripts authored verbatim from their Full skeletons with only the four annotated values changed (VARIANT, BASE, LOGBASE, backend env unchanged) plus the cumulative call-count logging block added.

## Issues Encountered

The plan's `verify` grep pattern `grep -q 'PHASE_CACHE_DIR="$rdir/phase_cache"'` fails in some bash `&&`-chain contexts due to shell interaction with the `"$rdir` substring inside single quotes. Verified via Python subprocess and step-by-step individual greps — the file content is correct ASCII and all acceptance criteria hold.

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes introduced. Both scripts are bash shell scripts that invoke the existing `run_ablation.py` with an existing registered variant. No new threat surface beyond what is already in the threat model for this plan (T-51-07 PHASE_CACHE_DIR isolation — mitigated by per-run isolation and Landmine-3 comment; T-51-08 cost runaway — mitigated by call-count logging and .done resume markers).

## Next Phase Readiness
- Both scripts ready for plan 51-04 (live sweep execution)
- 51-04 will execute `run_s20union_noknow_gpt_n3.sh` and `run_s20union_noknow_sonnet_n3.sh` and produce 30 result cells
- 51-05 (extractor extension) depends on the 30-cell result tree that 51-04 produces

## Self-Check: PASSED

- `run_s20union_noknow_gpt_n3.sh` exists and is executable: FOUND
- `run_s20union_noknow_sonnet_n3.sh` exists and is executable: FOUND
- Commit 587bf00 exists: FOUND
- Commit 6cc345e exists: FOUND

---
*Phase: 51-noknow*
*Completed: 2026-06-21*
