---
phase: 47-ship
plan: 01
subsystem: linker
tags: [python, prompt-minimization, standalone-module, runner-registration]

# Dependency graph
requires:
  - phase: 46-minimize
    provides: "Phase 46 frozen minimized prompt constants in tests/scratch/prompts_v5.py and tests/scratch/s_linker19.py"
provides:
  - "src/llm_sad_sam/linkers/experimental/s_linker20.py — standalone minimized-prompt linker variant (no inheritance, all constants inlined)"
  - "run_ablation.py gains --variants s_linker20 support (CANONICAL_VARIANTS + VARIANT_SPECS)"
affects: [48-sweep, 49-close]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Standalone flat-copy linker: duplicate-over-inherit pattern, all prompt constants inlined at module level"
    - "Phase 46 kept-cut inventory documented in inline comment block"

key-files:
  created:
    - src/llm_sad_sam/linkers/experimental/s_linker20.py
  modified:
    - run_ablation.py

key-decisions:
  - "Zero occurrences of s_linker19/prompts_v5 token in s_linker20.py — documentary mentions removed from docstrings to satisfy EDIT H grep test"
  - "VARIANT_SPECS description text avoids s_linker19 token to comply with zero-occurrence acceptance criterion"
  - "COR-05 tombstone placed on single line to satisfy grep test for exact phrase"

patterns-established:
  - "GATE-01 SHA-256 verification confirmed: s_linker19.py (05c413d0), prompts_v5.py (2f8b9968), s_linker13_min.py (083d92ae) — all byte-equal at plan close"

requirements-completed: [REQ-V264-08]

# Metrics
duration: 8min
completed: 2026-06-09
---

# Phase 47 Plan 01: SHIP — Create s_linker20.py and Register in Runner Summary

**Standalone minimized-prompt SLinker20 created with all 13 Phase 46 inlined constants and registered in run_ablation.py as --variants s_linker20.**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-06-09T10:08:58Z
- **Completed:** 2026-06-09T10:16:12Z
- **Tasks:** 2 of 2
- **Files modified:** 2 (1 created, 1 modified)

## Accomplishments

- Created `src/llm_sad_sam/linkers/experimental/s_linker20.py` as a 1086-line standalone module — no inheritance from any prior class, no external prompt import
- Inlined all 13 Phase 46 minimized prompt constants (AMBIGUITY_FEW_SHOT="", DOC_KNOWLEDGE_JUDGE_EXAMPLES="", plus 11 unchanged or minimized constants)
- Applied all 5 builder-method text changes (CUT-AMB-02, CUT-EXT-01, CUT-VAL-02, CUT-COR-03, CUT-COR-04) with COR-05 tombstone preserved verbatim
- Registered s_linker20 in run_ablation.py with experimental=True, canonical=False; s_linker19 remains absent from runner
- GATE-01 verified: all three frozen files byte-equal at plan close (SHA-256 confirmed)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create standalone s_linker20.py** - `de3b48e` (feat)
2. **Task 2: Register s_linker20 in run_ablation.py** - `a267a96` (feat)

**Plan metadata:** (docs commit below)

## Files Created/Modified

- `src/llm_sad_sam/linkers/experimental/s_linker20.py` — New standalone linker variant; SLinker20 class, _VARIANT_NAME="s_linker20", 13 minimized constants inlined, identical logic to SLinker19
- `run_ablation.py` — Added s_linker20 to CANONICAL_VARIANTS (line 118) and VARIANT_SPECS (lines 751–767)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Tombstone phrase split across lines**
- **Found during:** Task 1 verification
- **Issue:** The `_prompt_coref` method body had `Be\nconservative —` split across two lines, failing the grep acceptance test for the exact phrase `Be conservative — only include resolutions you are CERTAIN about.`
- **Fix:** Joined the phrase onto a single line while preserving the exact text
- **Files modified:** src/llm_sad_sam/linkers/experimental/s_linker20.py
- **Commit:** de3b48e

**2. [Rule 1 - Bug] Docstring mentions of s_linker19/prompts_v5 violated EDIT H zero-occurrence requirement**
- **Found during:** Task 1 verification
- **Issue:** EDIT H requires zero occurrences of the token "s_linker19" and "prompts_v5" in the file; initial docstrings mentioned them
- **Fix:** Replaced descriptive references with neutral language ("no superclass dependency", "external prompt module import block")
- **Files modified:** src/llm_sad_sam/linkers/experimental/s_linker20.py
- **Commit:** de3b48e

**3. [Rule 1 - Bug] VARIANT_SPECS description contained "s_linker19" token**
- **Found during:** Task 2 verification
- **Issue:** Acceptance criteria require zero occurrences of s_linker19 in run_ablation.py; the planned description text said "Same logic as s_linker19"
- **Fix:** Changed to "Same logic as the preceding paper variant"
- **Files modified:** run_ablation.py
- **Commit:** a267a96

## Known Stubs

None — s_linker20.py is a complete functional implementation.

## Threat Flags

None — pure file-construction phase; no new network endpoints, auth paths, or trust boundary changes.

## Self-Check: PASSED

- FOUND: src/llm_sad_sam/linkers/experimental/s_linker20.py (1086 lines, >= 900 min)
- FOUND: commit de3b48e (Task 1)
- FOUND: commit a267a96 (Task 2)
- GATE-01: s_linker19.py SHA-256 05c413d0 MATCH; prompts_v5.py 2f8b9968 MATCH; s_linker13_min.py 083d92ae MATCH
- All 13 constants at module level: confirmed (grep returns 13 lines)
- class SLinker20 + _VARIANT_NAME="s_linker20": confirmed
- AMBIGUITY_FEW_SHOT="" + DOC_KNOWLEDGE_JUDGE_EXAMPLES="": confirmed
- COR-05 tombstone "Be conservative — only include resolutions you are CERTAIN about": confirmed
- Builder openers (AMB-02, EXT-01, VAL-02, COR-03/04): confirmed
- Zero s_linker19/prompts_v5 tokens in s_linker20.py: confirmed
- Zero s_linker19 tokens in run_ablation.py: confirmed
- run_ablation.py: s_linker20 in CANONICAL_VARIANTS and VARIANT_SPECS with experimental=True, canonical=False: confirmed
