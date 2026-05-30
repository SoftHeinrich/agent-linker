---
phase: 05-promote-and-ablation-artifact
plan: 01
subsystem: infra
tags: [promotion, registration, keep-decision]

requires:
  - phase: 04-alias-scope-and-coref-fold
    provides: s_linker13f.py canonical chain-winner
provides:
  - s_linker13.py (canonical promotion of 13f)
  - run_ablation.py registration (CANONICAL_VARIANTS + VARIANT_SPECS)
  - PROJECT.md KEEP-decision row for _has_standalone_mention
affects: [05-02, 05-03]

tech-stack:
  added: []
  patterns: [byte-equivalent promotion via cp + class/constant rename]

key-files:
  created:
    - src/llm_sad_sam/linkers/experimental/s_linker13.py
  modified:
    - run_ablation.py
    - .planning/PROJECT.md

key-decisions:
  - "Canonical flag: description string AND explicit canonical=True dict key (D-44d planner discretion)"
  - "No build_linker smoke test — REGISTRATION OK + importlib check are sufficient SC-1"
  - "Cache namespace results/phase_cache/s_linker13/ pre-check: clean state, no stale dir"

patterns-established:
  - "Promotion provenance dual-marker: docstring KEEP field + top-of-file KEEP comment block"

requirements-completed: [PROMO-01, PROMO-02]

duration: 5min
completed: 2026-05-29
---

# Phase 05-01: Promote s_linker13f to s_linker13 — Summary

**s_linker13f promoted as canonical s_linker13 (PROMO-01); KEEP-decision for `_has_standalone_mention` logged in PROJECT.md (PROMO-02).**

## Performance

- **Duration:** ~5 min
- **Completed:** 2026-05-29
- **Tasks:** 3 of 3 (file copy + edits; run_ablation registration; smoke tests)
- **Files modified:** 3 (1 created, 2 edited)

## Accomplishments

1. Copied `s_linker13f.py` → `s_linker13.py`; applied D-44 edits:
   - class `SLinker13f` → `SLinker13`
   - `_VARIANT_NAME = "s_linker13"`
   - rewrote module docstring with `REMOVED_FROM:` chaining 12c via 13a→13b→13c→13e→13f, `RULES_REMOVED:` listing all 6 cumulative removals, `KEEP:` naming `_has_standalone_mention`
   - inserted KEEP-decision comment block citing Spike 002 RISKY, EXT-01/EXT-02 deferrals, and forward-pointer to METHODOLOGY.md
   - banner `print("SLinker13f ...")` → `print("SLinker13 (canonical promotion of s_linker13f, Phase 5)")`
2. Registered `s_linker13` in `run_ablation.py`: appended to `CANONICAL_VARIANTS` after `s_linker13f`; added `VARIANT_SPECS` entry with `description` containing "canonical promotion" AND explicit `canonical=True` flag.
3. Appended KEEP-decision row to `PROJECT.md` Key Decisions table (contains all six LOCKED tokens: `_has_standalone_mention`, `Spike 002`, `O(N×M)`, `EXT-01`, `EXT-02`, `KEPT (Phase 5`).

## Verification

- `python3 -c "import ast; ast.parse(open('src/.../s_linker13.py').read())"` → AST OK
- Import smoke test: `from run_ablation import CANONICAL_VARIANTS, VARIANT_SPECS; assert 's_linker13' in ...` → REGISTRATION OK
- `importlib.import_module('llm_sad_sam.linkers.experimental.s_linker13').SLinker13` → class name = `SLinker13`, `_VARIANT_NAME` = `s_linker13`
- BENCHMARK_TABOO audit: CLEAN (file is byte-equivalent to 13f modulo docstring/class/constant/banner; no new prompt text introduced)
- 13a-13f files remain in tree (D-43b satisfied)
- Cache namespace `results/phase_cache/s_linker13/` clean (no stale dir to remove)

## Deviations

- (a) Cache pre-check returned "no preexisting cache" — no deletion needed.
- (b) `build_linker` smoke test skipped: SC-1 satisfied by REGISTRATION OK + importlib check, planner discretion per D-44d.
- (c) Canonical flag implemented as BOTH description-string token AND explicit `canonical=True` key (belt-and-suspenders per D-44d).

## SC mapping

- D-54 SC-1 (`s_linker13.py` exists, importable, registered) → PASS
- D-54 SC-2 (PROJECT.md KEEP row exists) → PASS
