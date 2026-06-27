---
phase: 46-minimize
plan: 02
subsystem: minimize-amb-cuts
tags:
  - AMB
  - CUT-AMB-01
  - CUT-AMB-02
  - drop-block
  - pleonasm
  - cross-section-batch
  - tests/scratch
  - SAD_SAM_LINKER_SOURCE
dependency_graph:
  requires:
    - tests/scratch/s_linker19.py (post-46-01 baseline)
    - tests/scratch/prompts_v5.py (post-46-01 baseline)
    - tests/harness/ SAD_SAM_LINKER_SOURCE toggle + ACCEPTED_PREFIXES (from 46-01)
    - .planning/phases/45-audit/s_linker20-PROMPT-AUDIT.md (CUT-AMB-01, CUT-AMB-02 rows)
    - BENCHMARK_TABOO.md (GATE-06 re-isolation)
  provides:
    - .planning/phases/46-minimize/s_linker20-MINIMIZE-LOG.md AMB section populated (2 rows + closing blockquote)
    - tests/scratch/s_linker19.py:274 — opener pleonasm cut applied (kept)
    - tests/scratch/prompts_v5.py:30 — AMBIGUITY_FEW_SHOT drop-by-empty applied (kept)
    - smallest-passing identifier for AMBIGUITY_FEW_SHOT = `drop` (no Family A/B emitted per D-06)
  affects:
    - Wave-2 plan 46-05 (CUT-EXT-01 uses same `components` batch vocab as CUT-AMB-02)
    - Wave-2 plan 46-06 (CUT-VAL-02 uses same `components` batch vocab as CUT-AMB-02)
    - Wave-3 plan 46-08 (Pareto Summary reads AMB row count + LOC totals + smallest-passing-id)
tech-stack:
  added: []
  patterns:
    - per-cut atomic commit (D-04) with SHA-backfill follow-up docs commit (deviation: see Deviations)
    - drop-by-empty (preserve constant binding so scratch import resolves)
    - GATE-06 v2.1 isolation: bare `component` in per-dataset sections is generic SE noun (clean)
key-files:
  created:
    - .planning/phases/46-minimize/46-02-SUMMARY.md
  modified:
    - tests/scratch/s_linker19.py (line 274 opener)
    - tests/scratch/prompts_v5.py (line 30 AMBIGUITY_FEW_SHOT body)
    - .planning/phases/46-minimize/s_linker20-MINIMIZE-LOG.md (AMB section: 2 rows + closing blockquote)
decisions:
  - "CUT-AMB-02 trialled first per D-02 risk-ascending (low risk before high); kept"
  - "CUT-AMB-01 trialled second; drop-block passed harness on first trial — smallest-passing = drop; LOC saved = 7"
  - "No Family A / Family B trials needed (audit assigned `clean` verdict to AMB; D-06 only emits rewording families for benchmark-leak verdicts)"
  - "SHA-backfill of self-referential commit_sha is incompatible with --amend (rewrites SHA); resolved with follow-up docs commit per cut. Pattern recommendation for 46-03..07: always commit cut as feat/chore with placeholder SHA, then immediately commit a small docs(46-NN): backfill — ... commit with the actual SHA."
metrics:
  duration_minutes: 8
  tasks_completed: 2
  commits_produced: 3
  files_modified: 3
  completed_date: 2026-06-08
---

# Phase 46 Plan 02: AMB Section Cuts Summary

JWT-style atomic-per-cut trial of the 2 AMB candidates from the Phase 45 audit:
CUT-AMB-02 (pleonasm, low-risk) and CUT-AMB-01 (drop-block on AMBIGUITY_FEW_SHOT,
high-risk) — both **kept** under `SAD_SAM_LINKER_SOURCE=scratch` with the
`tests/test_s_linker20_prompt_ambiguity.py @ phase_1_model` gate (5 snapshots).

## Per-cut Outcomes

| cut_id | verdict | snapshot_delta | gate06_isolation | loc_saved | commit_sha |
|---|---|---|---|---|---|
| CUT-AMB-02 | kept | 0/5 | clean | 0 | `0710510` |
| CUT-AMB-01 | kept | 0/5 | clean (no after-text) | 7 | `dfad56a` |

**Total LOC saved in AMB section:** 7
**Smallest-passing identifier for AMBIGUITY_FEW_SHOT:** `drop` (drop-by-empty per 46-RESEARCH §9 Q9 (a))

## CUT-AMB-02 (pleonasm) — kept

- **File:** `tests/scratch/s_linker19.py:274` (`_prompt_ambiguity` opener)
- **Before:** `Classify these software architecture component names.`
- **After:** `Classify these component names.`
- **Replacement vocabulary:** `components` bare — pre-decided cross-section
  batch vocab from 46-01 (closes 1 of 3 sites; CUT-EXT-01 and CUT-VAL-02 close
  the remaining 2 in 46-05 / 46-06).
- **Harness gate:** `SAD_SAM_LINKER_SOURCE=scratch pytest
  tests/test_s_linker20_prompt_ambiguity.py -x` → 5/5 snapshots passed.
  `reconstruct_ambiguity_inputs` anchors on `^NAMES:`, so opener change is
  harness-safe by construction (46-RESEARCH §6.2 prediction confirmed).
- **GATE-06 evidence:** Grep `classify|these|component|names` against
  `BENCHMARK_TABOO.md` returned only bare-`component` hits in per-dataset
  sections as anaphoric uses (e.g. "Teammates component", "JabRef component",
  "BBB — Presentation Conversion component"). Cleared under the v2.1 isolation
  precedent (`component` is a generic SE noun in per-dataset anaphor position);
  `classify`, `these`, `names` produced zero per-dataset hits. **Clean.**
- **Commit:** `0710510 feat(46-02): keep CUT-AMB-02 — software architecture component names -> component names`

## CUT-AMB-01 (drop AMBIGUITY_FEW_SHOT) — kept

- **File:** `tests/scratch/prompts_v5.py:30` (`AMBIGUITY_FEW_SHOT` body)
- **Before:** Multi-line triple-quoted body (Examples 1+2 using the `"Scheduler"`
  worked few-shot, ARCHITECTURAL/AMBIGUOUS classification rationales).
- **After:** `""` (drop-by-empty; constant binding preserved).
- **Drop strategy:** Per 46-RESEARCH §9 Q9 (a) the constant must remain defined
  so the `from tests.scratch.prompts_v5 import (..., AMBIGUITY_FEW_SHOT, ...)`
  line in scratch `s_linker19.py` continues to resolve. Verified via
  `python -c "from tests.scratch.prompts_v5 import AMBIGUITY_FEW_SHOT; assert
  AMBIGUITY_FEW_SHOT == ''"` → OK.
- **Harness gate:** 5/5 snapshots passed. `reconstruct_ambiguity_inputs`
  anchors on `^NAMES:` and `NOW CLASSIFY THE NAMES ABOVE.` — both still
  present in the post-cut prompt — so the f-string interpolation of an empty
  `AMBIGUITY_FEW_SHOT` is harness-safe (46-RESEARCH §4.4 MED-HIGH prediction
  confirmed).
- **GATE-06 evidence:** After-text is `""`; trivially clean (no tokens to grep).
- **D-03 protocol:** drop-block passed on first trial ⇒ smallest-passing = `drop`.
  No Family A / Family B trials needed (none emitted at audit time per D-06 —
  AMB verdict is `clean`, not `benchmark-leak`).
- **Behavioral caveat (46-RESEARCH §4.4):** drop verdict reflects harness
  compatibility only; behavioral effect on judge calibration (zero-shot vs
  few-shot) is NOT observable in this phase. Phase 48 sweep validates behavior
  on gpt-5.4.
- **Commit:** `dfad56a feat(46-02): keep CUT-AMB-01 — drop AMBIGUITY_FEW_SHOT (drop-by-empty)`

## Cross-section pleonasm batch outcome

CUT-AMB-02 closes the AMB site of the recurring `software architecture …`
opener pleonasm. The other two sites (CUT-EXT-01 in `_prompt_extraction`
opener and CUT-VAL-02 in `_prompt_validation` opener) are trialled in plans
46-05 and 46-06 using the same `components` bare replacement vocabulary
pre-decided by 46-01's MINIMIZE-LOG batch header. With CUT-AMB-02 kept, the
batch is 1/3 closed; the Pareto Summary at 46-08 will cross-reference the
three rows as one conceptual cut.

## GATE-01 verification

`git diff --stat src/llm_sad_sam/linkers/experimental/s_linker19.py
src/llm_sad_sam/linkers/experimental/prompts_v5.py
src/llm_sad_sam/linkers/experimental/s_linker13_min.py` is **empty** at phase
close. Verified after each per-cut commit (`0710510`, `dfad56a`) and the
follow-up SHA-backfill commit (`1d3d9f0`). The three GATE-01-protected sources
in `src/llm_sad_sam/linkers/experimental/` are byte-equal vs HEAD throughout.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 — Blocking issue] SHA-backfill incompatible with `git commit --amend`**

- **Found during:** Task 2 commit step (Step 5a, "Backfill the row's commit_sha after commit")
- **Issue:** The per-cut atomic-commit protocol asks for the `commit_sha` cell
  to be backfilled inside the same commit that introduces the cut, by amending.
  But `git commit --amend` after editing the LOG rewrites the SHA being
  backfilled — a self-referential bootstrap problem. An amend cycle produced
  an obsolete intermediate SHA (`20cbde3`) and a final SHA (`dfad56a`) that
  did not match the value embedded in the LOG.
- **Fix:** Resolved the inconsistency by adding a small follow-up docs commit
  (`1d3d9f0 docs(46-02): backfill CUT-AMB-01 commit_sha to current parent`)
  that points the row at the stable parent SHA `dfad56a`. The feat commit
  remains atomic to the cut; the backfill commit is metadata-only and does
  not touch `tests/scratch/*` or any GATE-01-protected source.
- **Files modified:** `.planning/phases/46-minimize/s_linker20-MINIMIZE-LOG.md`
  only.
- **Commit:** `1d3d9f0`
- **Recommendation for 46-03..07:** Apply the same pattern — commit the cut
  as `feat`/`chore` with a placeholder SHA, then immediately commit a small
  `docs(46-NN): backfill — ...` with the actual SHA. Do NOT use `--amend`
  for SHA backfill.

### Other Deviations

None. CUT-AMB-02 and CUT-AMB-01 were both kept on first trial with no
unexpected harness behavior. No revert, no unsafe verdict, no GATE-06 hit
escalation.

## Commits Produced

| # | Hash | Type | Subject |
|---|------|------|---------|
| 1 | `0710510` | feat | keep CUT-AMB-02 — software architecture component names -> component names |
| 2 | `dfad56a` | feat | keep CUT-AMB-01 — drop AMBIGUITY_FEW_SHOT (drop-by-empty) |
| 3 | `1d3d9f0` | docs | backfill CUT-AMB-01 commit_sha to current parent |

Two atomic per-cut commits (feat) + one metadata follow-up (docs).

## Verification

- [x] Exactly 2 cut rows exist under `<!-- SECTION:AMB:START --> ... <!-- SECTION:AMB:END -->`.
- [x] Row order matches D-02 risk-ascending: CUT-AMB-02 first, CUT-AMB-01 second.
- [x] CUT-AMB-02 verdict ∈ {kept, reverted, unsafe} → `kept`.
- [x] CUT-AMB-01 verdict ∈ {kept, kept-original} → `kept`.
- [x] CUT-AMB-02 gate06_isolation = `clean`; CUT-AMB-01 = `clean (no after-text)`.
- [x] LOC saved values are non-negative integers (0 + 7 = 7 total).
- [x] CUT-AMB-02 commit is `feat(46-02): …`; CUT-AMB-01 commit is `feat(46-02): …`.
- [x] GATE-01: `git diff --stat` on s_linker19.py + prompts_v5.py + s_linker13_min.py is empty.
- [x] No edits outside the AMB anchor in the MINIMIZE-LOG; no edits under `src/llm_sad_sam/` or `tests/harness/`.
- [x] AMB closing blockquote summarises smallest-passing = `drop`.

## Self-Check

| Item | Result |
|------|--------|
| `.planning/phases/46-minimize/46-02-SUMMARY.md` | FOUND |
| `.planning/phases/46-minimize/s_linker20-MINIMIZE-LOG.md` AMB section: 2 rows + closing blockquote | FOUND |
| `tests/scratch/s_linker19.py:274` opener = `Classify these component names.` | FOUND |
| `tests/scratch/prompts_v5.py:30` AMBIGUITY_FEW_SHOT = `""` | FOUND |
| Commit `0710510` (CUT-AMB-02 keep) | FOUND in `git log --oneline --all` |
| Commit `dfad56a` (CUT-AMB-01 keep) | FOUND in `git log --oneline --all` |
| Commit `1d3d9f0` (SHA backfill) | FOUND in `git log --oneline --all` |
| GATE-01 post-phase `git diff --stat` on s_linker19.py + prompts_v5.py + s_linker13_min.py | empty (exit 0) |

## Self-Check: PASSED
