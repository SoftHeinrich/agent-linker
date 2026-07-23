---
phase: 01-baseline-and-infrastructure
plan: 03
subsystem: infra
tags: [variant-namespacing, checkpoint, refactor, s_linker12c]

# Dependency graph
requires:
  - phase: 01-baseline-and-infrastructure
    provides: diskcache-based LLM-response cache (01-02) — orthogonal but co-located in checkpoint subsystem
provides:
  - "_VARIANT_NAME class-constant pattern in SLinker12c (template for every 13x variant)"
  - "Namespaced per-phase pickle cache directory derived from self._VARIANT_NAME"
  - "Namespaced per-run log filename derived from self._VARIANT_NAME"
  - "D-07 runtime assertion enforcing variant-name embedding in checkpoint path"
affects: [01-04 (baseline run), 01-05 (s_linker13a copy + flip-the-constant), all future 13x variants]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Per-variant identity via class-level `_VARIANT_NAME` constant"
    - "Fail-fast `assert _VARIANT_NAME in checkpoint_dir` at first cache-dir use"

key-files:
  created: []
  modified:
    - src/llm_sad_sam/linkers/experimental/s_linker12c.py

key-decisions:
  - "D-07 placement inside `_checkpoint_dir()` body (per-call site), not `__init__` — confirmed by user (RESEARCH.md A2/Q1 RESOLVED). Method body is the per-call site for the per-variant pickle cache; LLMBackend.CHECKPOINT path is namespaced separately by LLMClient and is NOT the D-07 target."

patterns-established:
  - "Pattern 1 (_VARIANT_NAME constant): every linker class declares `_VARIANT_NAME = \"s_linkerXX\"` immediately after the class line; `_checkpoint_dir` and `_save_log` derive their identity from it; D-07 assertion guards the path."

requirements-completed: [INFRA-05]

# Metrics
duration: 6 min
completed: 2026-05-13
---

# Phase 1 Plan 3: SLinker12c Variant Namespacing Summary

**Class-level `_VARIANT_NAME = "s_linker12c"` constant in SLinker12c; `_checkpoint_dir` and `_save_log` derive their identity from it; D-07 fail-fast assertion fires inside `_checkpoint_dir` body when the computed path omits the variant name.**

## Performance

- **Duration:** ~6 min
- **Started:** 2026-05-13T17:00:00Z (approx.)
- **Completed:** 2026-05-13T17:06:00Z (approx.)
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Declared `_VARIANT_NAME = "s_linker12c"` as the first member of the `SLinker12c` class body (line 72) so the constant is unmistakable.
- Refactored `_checkpoint_dir` (~L1118) to build the per-phase pickle cache path from `self._VARIANT_NAME` instead of the hardcoded `"s_linker12c"` literal, and added the D-07 assertion immediately after the path-construction line.
- Refactored `_save_log` (~L1141) so the log filename prefix is derived from `self._VARIANT_NAME` (`f"{self._VARIANT_NAME}_{ds}_..."`), removing the second hardcoded literal.
- Verified at runtime: instantiating `SLinker12c(backend=LLMBackend.CLAUDE)` and calling `_checkpoint_dir('/tmp/fake_dataset.txt')` returns `./results/phase_cache/s_linker12c/fake_dataset` (unchanged layout) and the assertion passes.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add `_VARIANT_NAME` constant + namespaced `_checkpoint_dir`/`_save_log` + D-07 assertion** — `519de18` (refactor)

## Files Created/Modified
- `src/llm_sad_sam/linkers/experimental/s_linker12c.py` — Added `_VARIANT_NAME` class constant (line 72); replaced hardcoded `"s_linker12c"` literal in `_checkpoint_dir` path (line 1121) with `self._VARIANT_NAME` and added D-07 assertion (lines 1122–1126); replaced hardcoded `"s_linker12c"` literal in `_save_log` filename (line 1145) with `self._VARIANT_NAME`. (+13 / −2 lines.)

## Decisions Made
None beyond the locked CONTEXT.md decisions (D-03, D-04, D-07). The plan-level "D-07 placement note (2026-05-08)" confirmed the assertion lives inside `_checkpoint_dir()` body — that placement was followed exactly.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

The plan referenced approximate line numbers (L1181–1186 / L1204–1211) but the actual file is 1149 lines and the methods sit at L1118 and L1141. The plan's `<interfaces>` block intentionally documented this as approximate context and the unique-string edits matched anyway, so this is not a deviation.

## Acceptance Criteria Results

All eight acceptance criteria PASS:

1. `grep -q "_VARIANT_NAME = \"s_linker12c\""` exits 0 — PASS (line 72)
2. `grep -c "self\._VARIANT_NAME"` returns 4 (≥3 required) — PASS
3. `grep -q "assert self\._VARIANT_NAME in d"` exits 0 — PASS (line 1126)
4. `grep -q 'os.path.join(cache_dir, "s_linker12c"'` exits non-zero — PASS (literal removed)
5. `grep -q 'f"s_linker12c_{ds}'` exits non-zero — PASS (literal removed)
6. Smoke test prints `OK ./results/phase_cache/s_linker12c/fake_dataset` and exits 0 — PASS
7. Module import succeeds, no syntax errors — PASS
8. All four methods (`_checkpoint_dir`, `_save_log`, `_save_phase`, `_log`) still present — PASS

## Verification (plan `<verification>`)

- `grep -c "_VARIANT_NAME"` returns 5 (≥4 required: 1 declaration + 4 usages) — PASS
- The hand-test of the negative-assertion case was deliberately skipped per the plan ("skip in normal verification — the in-method assertion is exercised on every cache-dir use").

## Self-Check: PASSED

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Pattern is now established for 01-05 (`s_linker13a`): copy `s_linker12c.py` and flip the single `_VARIANT_NAME = "s_linker12c"` line to `_VARIANT_NAME = "s_linker13a"`. No other path/filename edits required for namespacing.
- 01-04 (12c baseline run) can proceed: on-disk pickle layout under `./results/phase_cache/s_linker12c/<ds>/` is unchanged, so any pre-existing checkpoints remain valid.

---
*Phase: 01-baseline-and-infrastructure*
*Completed: 2026-05-13*
