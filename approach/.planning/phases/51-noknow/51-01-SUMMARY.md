---
phase: 51-noknow
plan: 01
subsystem: linker
tags: [no-knowledge, s_linker20_union, gate-01, gate-06, registry, build_linker, ablation]

# Dependency graph
requires: []
provides:
  - "SLinker20Union(no_knowledge=True) constructor flag with strictly-additive Phase-1 guard"
  - "s_linker20_union_noknow registered variant (CANONICAL_VARIANTS + VARIANT_SPECS)"
  - "build_linker kwargs threading (spec.get('kwargs', {}) → cls(..., **extra))"
affects: [51-02, 51-03, 52-score, 53-rq3, 54-rq4]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "VARIANT_SPECS kwargs field: pass non-backend constructor kwargs from registry to build_linker"
    - "Strictly-additive if/else guard on frozen linker: if self.no_knowledge / else: <original code>"

key-files:
  created: []
  modified:
    - src/llm_sad_sam/linkers/experimental/s_linker20_union.py
    - run_ablation.py

key-decisions:
  - "no_knowledge flag uses if/else guard (strictly-additive) — else-branch is byte-identical pre-existing code (GATE-01)"
  - "No _VARIANT_NAME change — run-script PHASE_CACHE_DIR isolation handles cache separation (Landmine 3)"
  - "build_linker kwargs threading uses spec.get('kwargs', {}) default to preserve backward-compatibility for all existing variants (Landmine 1 fix)"
  - "NOKNOW path sets ModelKnowledge() + DocumentKnowledge() directly — pure empty set()/dict(), no hardcoded vocab (GATE-06)"

patterns-established:
  - "VARIANT_SPECS kwargs field: add kwargs=dict(...) to any spec that needs non-backend constructor args; build_linker auto-threads them"

requirements-completed: [NOKNOW-01]

# Metrics
duration: 8min
completed: 2026-06-21
---

# Phase 51 Plan 01: NOKNOW Flag + Registry Summary

**`SLinker20Union` gains a default-off `no_knowledge` constructor flag that skips the 3 layer1 LLM calls and injects empty ModelKnowledge/DocumentKnowledge; `s_linker20_union_noknow` is registered and constructible via `build_linker` through a new VARIANT_SPECS `kwargs` threading mechanism.**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-06-21T00:00:00Z
- **Completed:** 2026-06-21T00:08:00Z
- **Tasks:** 2 of 2
- **Files modified:** 2

## Accomplishments

- Added `no_knowledge: bool = False` param to `SLinker20Union.__init__` (after existing 4 params) with `self.no_knowledge = no_knowledge` in the constructor body
- Wrapped the Phase-1 knowledge acquisition block in a strictly-additive `if self.no_knowledge:` / `else:` guard; the `else:` branch is the pre-existing `_run_parallel(...)` + assignments verbatim (only re-indented); `_save_phase("layer1")` and the two print lines stay outside the guard and fire in both branches
- Registered `s_linker20_union_noknow` in both `CANONICAL_VARIANTS` and `VARIANT_SPECS` (same `SLinker20Union` class, no logic duplicated; `kwargs=dict(no_knowledge=True)`)
- Extended `build_linker` to read `extra = spec.get("kwargs", {})` and pass `**extra` to the constructor — closing the BLOCKING kwargs gap (Landmine 1); all existing specs default to `{}` and are unaffected

## Task Commits

1. **Task 1: Add no_knowledge flag + Phase-1 guard** — `9313d41` (feat)
2. **Task 2: Register s_linker20_union_noknow + thread kwargs through build_linker** — `b2bdf19` (feat)

## Files Created/Modified

- `/mnt/hostshare/ardoco-home/agent-linker/src/llm_sad_sam/linkers/experimental/s_linker20_union.py` — added `no_knowledge` param + assignment; Phase-1 if/else guard
- `/mnt/hostshare/ardoco-home/agent-linker/run_ablation.py` — `CANONICAL_VARIANTS` entry, `VARIANT_SPECS` entry with `kwargs`, `build_linker` kwargs threading

## Decisions Made

- Chose VARIANT_SPECS `kwargs` field approach (Research option a) over subclass to close the `build_linker` gap — least surface change, backward-compatible, no inheritance chain
- `_VARIANT_NAME` unchanged at `"s_linker20_union"` — per Landmine 3 in RESEARCH, run-script per-run `PHASE_CACHE_DIR` isolation is sufficient to keep No-Knowledge and Full caches separate
- NOKNOW print marker is a single `[NOKNOW]` line (not a banner) — matches the verbosity style of the surrounding Phase-1 print statements

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- `s_linker20_union_noknow` is fully constructible: `build_linker('s_linker20_union_noknow').no_knowledge is True` verified
- Full variant backward-compatible: `build_linker('s_linker20_union').no_knowledge is False` verified
- GATE-01 structural condition met: `else:` branch is byte-identical pre-existing code (only re-indented); `_VARIANT_NAME` unchanged
- GATE-06 clean: NOKNOW path uses only `set()` / `{}` empties; no hardcoded alias/vocab literals
- Ready for Phase 51 subsequent plans: run-scripts for the 30-run No-Knowledge sweep, GATE-01 evidence generation, extractor extension

## Self-Check

- `src/llm_sad_sam/linkers/experimental/s_linker20_union.py` — modified (verified by verify command)
- `run_ablation.py` — modified (verified by --list-variants + build_linker test)
- Commit `9313d41` — exists (Task 1)
- Commit `b2bdf19` — exists (Task 2)

## Self-Check: PASSED

---
*Phase: 51-noknow*
*Completed: 2026-06-21*
