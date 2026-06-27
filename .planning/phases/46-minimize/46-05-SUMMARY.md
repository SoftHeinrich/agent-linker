---
phase: 46-minimize
plan: 05
subsystem: prompt-minimization
tags: [minimize, ext, pleonasm, batch-member-2, cross-section]
requires:
  - 46-01 (scratch baseline + ACCEPTED_PREFIXES + MINIMIZE-LOG scaffold + batch vocab `components`)
  - 46-02 (CUT-AMB-02 — cross-section pleonasm batch member 1 of 3, sha 0710510)
  - phase 44 harness + phase 45 audit row CUT-EXT-01
provides:
  - EXT section of `s_linker20-MINIMIZE-LOG.md` populated with single CUT-EXT-01 row
  - tests/scratch/s_linker19.py:331 mutated to `Extract ALL references to components from this document.`
  - 2 of 3 cross-section pleonasm-batch sites closed (AMB-02 + EXT-01 kept; VAL-02 pending in 46-06)
affects:
  - 46-06 (downstream cross-section pleonasm batch closer — CUT-VAL-02)
  - 46-08 (Pareto Summary — rolls AMB-02 + EXT-01 + VAL-02 as one conceptual batch)
  - phase 47 (SHIP) — kept-cut after-text is the new s_linker20 `_prompt_extraction` opener
tech-stack:
  added: []
  patterns:
    - "per-cut atomic commit with backfilled SHA via separate docs commit (46-02 / 46-04 pattern)"
    - "GATE-06 v2.1 isolation precedent: bare `components` as generic SE noun anaphor cleared in per-dataset schema-header context"
    - "harness-safe opener rewrite: reconstruct_extraction_inputs anchors on ^COMPONENTS: and \\nDOCUMENT:\\n"
key-files:
  created:
    - .planning/phases/46-minimize/46-05-SUMMARY.md
  modified:
    - tests/scratch/s_linker19.py
    - .planning/phases/46-minimize/s_linker20-MINIMIZE-LOG.md
decisions:
  - "CUT-EXT-01 verdict = kept (RC=0 + GATE-06 clean; 18/18 snapshots pass under SAD_SAM_LINKER_SOURCE=scratch)"
  - "Atomic commit per D-04: feat commit + separate docs SHA-backfill commit (mirrors 46-02 / 46-04 pattern)"
  - "EXT closing note placed inside <!-- SECTION:EXT:END --> per 46-RESEARCH section symmetry"
metrics:
  duration_minutes: 6
  tasks_completed: 1
  files_touched: 3
  loc_saved: 0
  snapshots_passed: 18
  snapshots_total: 18
  cuts_kept: 1
  cuts_reverted: 0
  cuts_unsafe: 0
completed: 2026-06-08
---

# Phase 46 Plan 05: EXT Section Minimization Summary

**One-liner:** Trialled and kept CUT-EXT-01 — pleonasm batch member 2 of 3 (`Extract ALL references to software architecture components` -> `Extract ALL references to components`) at `tests/scratch/s_linker19.py:331`, 18/18 snapshots pass, GATE-06 clean, GATE-01 byte-equal preserved.

## Outcome

CUT-EXT-01: **kept**. Single trial cut, single kept verdict, single atomic feat commit + one docs SHA-backfill commit.

| cut_id | verdict | snapshot_delta | gate06_isolation | loc_saved | commit_sha |
|---|---|---|---|---|---|
| CUT-EXT-01 | kept | 0/18 | clean | 0 | `fbfbcb9` |

### What was cut

- **Site:** `tests/scratch/s_linker19.py:331` — `_prompt_extraction` opener (drifted from audit-time line 323 because upstream Wave-2 plan 46-02 mutated nearby AMB lines).
- **Before:** `Extract ALL references to software architecture components from this document.`
- **After:** `Extract ALL references to components from this document.`
- **Replacement vocabulary:** `components` (bare), per the cross-section pleonasm batch vocab pre-decided in the 46-01 MINIMIZE-LOG header (`components` collapses to the noun the `COMPONENTS:` slot already names downstream).

### Trial evidence

- **Gated tests:** `SAD_SAM_LINKER_SOURCE=scratch pytest tests/test_s_linker20_prompt_extraction.py -x --tb=short`
- **Result:** 18 collected, 18 passed in 0.08s (snapshot_delta = `0/18`). This is the largest single-cut gating in the entire audit (`phase_2_framing_c_pass1` + `phase_2_framing_c_pass2` parametrized across 5 projects + extra axes).
- **Harness safety mechanism:** `reconstruct_extraction_inputs` (`tests/harness/inputs.py`) anchors on `^COMPONENTS:` and `\nDOCUMENT:\n` — the opener line itself carries no anchor responsibility, so a strict-prefix rewording is harness-invariant. Confirms 46-RESEARCH §6.2's HIGH-confidence "kept" expectation.

### GATE-06 re-isolation

Grep on after-text tokens:

```
grep -niwE 'extract|references|components|document' BENCHMARK_TABOO.md | grep -v '^[0-9]*:#'
```

- `extract`: **0 hits**.
- `references`: **0 hits**.
- `components`: **5 hits**, all in per-dataset `Components:` schema-section column headers (BENCHMARK_TABOO.md lines 7, 12, 17, 22, 27). The bare word is the schema column header, not a benchmark vocabulary item. Cleared as a generic SE noun anaphor per Phase 45 v2.1 isolation precedent — same clearance basis as CUT-AMB-02's reasoning cell (MINIMIZE-LOG line 59) and Phase 45 CUT-VAL-03 / CUT-COR-01.
- `document`: **1 hit** at BENCHMARK_TABOO.md line 100, which is methodology prose (`document the inspection`), not dataset vocabulary.

Verdict: **clean**.

## Cross-section pleonasm-batch progress

This plan closes site 2 of 3 in the `software architecture …` opener pleonasm batch:

| batch_member | plan | sha     | verdict |
|---|---|---|---|
| CUT-AMB-02 (Phase 1 Ambiguity opener) | 46-02 | `0710510` | kept |
| **CUT-EXT-01 (Phase 2 Extraction opener)** | **46-05** | **`fbfbcb9`** | **kept** |
| CUT-VAL-02 (Phase 4 Validation opener)| 46-06 | (pending) | — |

All three share the pre-decided `components` (bare) replacement vocabulary committed to the 46-01 MINIMIZE-LOG header. The Pareto Summary (46-08, Wave 3) will cross-reference them as one conceptual batch with a combined LOC-saved row.

## Gates

| Gate | Result |
|---|---|
| GATE-01 (byte-equal: s_linker19.py + prompts_v5.py + s_linker13_min.py) | `git diff --stat` empty post-commit and post-backfill — **OK** |
| GATE-06 (cross-dataset isolation on after-text) | clean — bare `components` cleared per v2.1 precedent, `extract`/`references` zero hits, `document` only in methodology prose |
| Snapshot gating (D-04 / 46-RESEARCH §3.2) | 18/18 pass under `SAD_SAM_LINKER_SOURCE=scratch` |
| D-04 atomic commit | one feat commit (`fbfbcb9`) + one docs SHA-backfill commit (`94a405e`) — mirrors 46-02 / 46-04 cadence |
| Post-commit deletion check (`git diff --diff-filter=D HEAD~1 HEAD`) | empty (zero deletions) |

## Commits

| sha | type | message |
|---|---|---|
| `fbfbcb9` | feat | `feat(46-05): keep CUT-EXT-01 — software architecture components -> components (batch with AMB-02 + VAL-02)` |
| `94a405e` | docs | `docs(46-05): backfill CUT-EXT-01 commit_sha + EXT closing note` |

## Deviations from Plan

None. Plan executed exactly as written:

- Cut applied to the exact target line (line drifted from 323 -> 331 because of 46-02's upstream AMB edits — anticipated by the plan's "robust to drift if Task 1 of 46-02 mutated nearby lines" guard).
- Verdict matches 46-RESEARCH §6.2's HIGH-confidence prediction (`kept`).
- Two-commit pattern (feat + docs SHA-backfill) is identical to 46-02 / 46-04 — the plan's commit message template allowed a `(sha)` placeholder, which a separate docs commit then resolves.

## Threat-model retrospective

| Threat ID | Mitigation outcome |
|---|---|
| T-46-EXT-01 (Tampering — frozen sources) | `git diff --quiet` exit 0 pre- and post-commit; GATE-01 byte-equal preserved on all three frozen sources. |
| T-46-EXT-02 (Information disclosure — benchmark vocab in after-text) | GATE-06 re-grep returned clean (5 `components` hits in schema headers cleared per v2.1 precedent; methodology-prose `document` hit not dataset vocab; `extract`/`references` zero hits). |
| T-46-EXT-03 (Coverage gap — 18-snapshot gating not exercised) | Bare `pytest tests/test_s_linker20_prompt_extraction.py -x` with no `-k`/`-m` filters collected and passed all 18 parametrized tests; exit code 0. |

## Self-Check: PASSED

| Item | Result |
|------|--------|
| `.planning/phases/46-minimize/46-05-SUMMARY.md` | FOUND |
| `.planning/phases/46-minimize/s_linker20-MINIMIZE-LOG.md` (EXT row + closing note) | FOUND (regex check passed) |
| `tests/scratch/s_linker19.py` contains `Extract ALL references to components from this document.` | FOUND (line 331) |
| `tests/scratch/s_linker19.py` contains 0 hits for `Extract ALL references to software architecture components` | confirmed |
| Commit `fbfbcb9` (feat — CUT-EXT-01 keep) | FOUND in `git log` |
| Commit `94a405e` (docs — SHA backfill + closing note) | FOUND in `git log` |
| GATE-01 post-commit `git diff --stat` on s_linker19.py + prompts_v5.py + s_linker13_min.py | empty (exit 0) |
| Post-commit deletion check (`git diff --diff-filter=D HEAD~1 HEAD`) | empty (zero deletions) |
