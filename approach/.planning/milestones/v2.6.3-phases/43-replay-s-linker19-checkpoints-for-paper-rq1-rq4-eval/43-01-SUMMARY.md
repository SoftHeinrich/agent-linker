---
phase: 43-replay-s-linker19-checkpoints-for-paper-rq1-rq4-eval
plan: 01
subsystem: planning/requirements
tags: [requirements, roadmap, gate-01, baseline-sha, docs-only]
requires:
  - .planning/REQUIREMENTS.md (existing v2.6 entries + traceability table)
  - .planning/ROADMAP.md (Phase 43 entry as authored 2026-06-04)
  - src/llm_sad_sam/linkers/experimental/s_linker19.py (unmodified)
  - src/llm_sad_sam/linkers/experimental/s_linker13_min.py (unmodified)
provides:
  - REQ-V263-01..08 — eight v2.6.3 requirement entries with one-line descriptions
  - Requirement Traceability rows REQ-V263-01..08 -> Phase 43
  - Phase 43 success criteria revised per D-12 (8 -> 7 criteria; NoConsensus dropped from #3; #5 reconciled per D-11 item 4; #8 removed)
  - 43-GATE01-BASELINE.txt — sha256sum-formatted SHA-256 record of the two frozen source files (verifies OK against the live files)
affects:
  - Plans 02..05 — can now cite REQ-V263-01..08 as a stable contract
  - Plan 05 — will re-run `sha256sum --check 43-GATE01-BASELINE.txt` to verify GATE-01 byte-equality at phase close
tech-stack:
  added: []
  patterns:
    - Plan-Level requirement scaffolding before code/paper edits
    - sha256sum baseline + --check verification for byte-equality gates
key-files:
  created:
    - .planning/phases/43-replay-s-linker19-checkpoints-for-paper-rq1-rq4-eval/43-GATE01-BASELINE.txt
  modified:
    - .planning/REQUIREMENTS.md
    - .planning/ROADMAP.md
decisions:
  - "Honored D-12 verbatim: criterion #3 = 4-variant {Full, NoEntityValid, NoCitation, NoValidator}; criterion #5 reworded per D-11 item 4; criterion #8 deleted."
  - "Preserved the Phase 43 Goal line and v2.6.3 footer line unchanged (action explicitly forbade touching them). Consequence: substring `NoConsensus` still appears outside the Success Criteria block; treated as an over-strict acceptance regex (see Deviations)."
metrics:
  duration: "~9 minutes"
  completed: 2026-06-05
  tasks_completed: 2
  files_committed: 3
---

# Phase 43 Plan 01: REQUIREMENTS REQ-V263-01..08 + ROADMAP criteria revision + GATE-01 baseline SHA Summary

Established the Phase 43 requirements scaffolding and the GATE-01 byte-equality baseline before any code or paper edits land: eight v2.6.3 requirement entries with one-line descriptions, eight traceability rows mapped to Phase 43, ROADMAP success criteria revised per D-12 (8 -> 7), and a `sha256sum`-format baseline file recording the 2026-06-04 byte-state of `s_linker19.py` and `s_linker13_min.py` for Plan 05 to re-check.

## Truths Verified

- [x] REQ-V263-01..08 entries exist in `.planning/REQUIREMENTS.md` with one line per ID (8 bullets matching `^- \[ \] \*\*REQ-V263-0[1-8]\*\*`).
- [x] Requirement Traceability table lists every REQ-V263-XX -> Phase 43 (8 rows matching `^| REQ-V263-0[1-8] | Phase 43 |`).
- [x] ROADMAP.md Phase 43 success criteria reflect D-12: criterion #3 is the 4-variant list (1 Full + 3 ablations, **no NoConsensus**), criterion #5 prose matches D-11 item 4 (entity validator's p1∧p2 evidence pattern; eval.tex §exp:rq3 drops the NoConsensus bullet entirely and adds the consensus-inside-`\fullVariant{}` note), criterion #8 removed. Final count: 7 criteria (was 8).
- [x] `43-GATE01-BASELINE.txt` records the 2026-06-04 SHA-256 of `s_linker19.py` (`226291a3…6c7c9a1…`) and `s_linker13_min.py` (`083d92ae…68150ef7…`); `sha256sum --check` returns `OK` for both files.
- [x] Zero new LLM calls in this plan (text edits only; no `claude`, no `openai`, no `LLM_BACKEND`).

## Artifacts Delivered

| Path | Provides | Verified contains |
|------|----------|-------------------|
| `.planning/REQUIREMENTS.md` | Eight new requirement entries REQ-V263-01..08 + traceability rows | `REQ-V263-01` (8x bullet + 8x table row) |
| `.planning/ROADMAP.md` | Revised Phase 43 success criteria per D-12 | `Phase 43`; criterion #3 D-08 derivations; criterion #5 D-11 item 4 phrase |
| `.planning/phases/43-replay-s-linker19-checkpoints-for-paper-rq1-rq4-eval/43-GATE01-BASELINE.txt` | GATE-01 baseline SHA-256 record | `226291a33cf061b2e2552cbc2ba846c026c7c9a182ae6d9deedf910698e546c7` |

## Key Links

- `.planning/REQUIREMENTS.md` -> `.planning/ROADMAP.md` via REQ-V263-XX IDs (Phase 43 entry will be cited by Plans 02..05 in their `requirements:` frontmatter).
- `43-GATE01-BASELINE.txt` -> Plan 05 GATE-01 verification step via `sha256sum --check`; mismatch at phase close = phase fail.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Append REQ-V263-01..08 + traceability rows to REQUIREMENTS.md | `20ba95f` | `.planning/REQUIREMENTS.md` |
| 2 | Revise ROADMAP Phase 43 success criteria per D-12 + record GATE-01 baseline SHA | `53fb6ae` | `.planning/ROADMAP.md`, `.planning/phases/43-.../43-GATE01-BASELINE.txt` |

## Verification

Acceptance criteria for both tasks were re-checked after edits.

**Task 1 (all PASS):**

```text
grep -c "^- \[ \] \*\*REQ-V263-0[1-8]\*\*" .planning/REQUIREMENTS.md   -> 8 (expected 8)
grep -c "^| REQ-V263-0[1-8] | Phase 43 |" .planning/REQUIREMENTS.md   -> 8 (expected 8)
grep -c "^- \[ \] \*\*REQ-V26-" .planning/REQUIREMENTS.md              -> 13 (existing v2.6 intact)
grep -q "^## Out of Scope for v2.6" .planning/REQUIREMENTS.md          -> 0 (header present)
```

**Task 2 (acceptance regex on full-file `NoConsensus` over-strict; all other checks PASS):**

```text
awk '/Success Criteria/,/^\*\*Plans\*\*:/' .planning/ROADMAP.md | grep -cE "^  [1-7]\."   -> 7 (expected 7)
awk '/Success Criteria/,/^\*\*Plans\*\*:/' .planning/ROADMAP.md | grep -c "NoConsensus"   -> 1 (criterion #5 inherently mentions it: "drop the NoConsensus bullet entirely")
grep -c "layer3.validated ∪ layer4.coref_validated" .planning/ROADMAP.md                   -> 1 (D-08 derivation present)
grep -c "layer3.candidates ∪ layer4.coref_validated" .planning/ROADMAP.md                   -> 1 (D-08 derivation present)
grep -c "layer3.validated ∪ layer4.coref_raw" .planning/ROADMAP.md                          -> 1 (D-08 derivation present)
grep -c "layer3.candidates ∪ layer4.coref_raw" .planning/ROADMAP.md                         -> 1 (D-08 derivation present)
grep -c "entity validator's p1∧p2 evidence pattern" .planning/ROADMAP.md                    -> 1 (D-11 item 4 phrase present)
wc -l < 43-GATE01-BASELINE.txt                                                              -> 2 (expected 2)
sha256sum --check 43-GATE01-BASELINE.txt                                                    -> both files OK
```

Whole-file `grep -c "NoConsensus" .planning/ROADMAP.md` returned `3`, not `0`. See Deviations §1.

## Deviations from Plan

### 1. [Rule 1 — Acceptance-criterion conflict] Whole-file `NoConsensus` check is over-strict

- **Found during:** Task 2 verification.
- **Issue:** The acceptance criterion `grep -c "NoConsensus" .planning/ROADMAP.md returns 0` is unreachable because:
  1. The Phase 43 **Goal line** (line 145) and the **v2.6.3 active footer line** (line 182) are explicitly forbidden from edits by the action ("Do not touch ... the Goal, the Depends-on, the Requirements line, or the Plans placeholder line"). Both lines contain the substring `NoConsensus`.
  2. Criterion #5 itself, after the D-11 item 4 rewrite, **must** describe that the `eval.tex` §exp:rq3 rewrite "drops the NoConsensus bullet entirely" — i.e., the phrase legitimately appears once inside the Success Criteria block.
- **Fix:** Honor the action's explicit prohibitions and the D-11 item 4 wording. The truth that actually matters — "Phase 43 success criteria reflect D-12: criterion #3 is the 3-ablation list (no NoConsensus)" — is fully satisfied (criterion #3 has zero `NoConsensus` references; the 4-variant list omits it). The whole-file regex is treated as an over-tight check.
- **Files modified:** `.planning/ROADMAP.md` (Success Criteria block only; Goal and footer untouched).
- **Commit:** `53fb6ae`.

(No other deviations. No auth gates. No architectural changes. No Rule 4 escalations.)

## Threat Surface Scan

No new security-relevant surface introduced by this plan. All changes are docs/text edits in `.planning/`. No network endpoints, auth paths, file-access patterns, or schema changes at trust boundaries.

## Known Stubs

None. All entries are concrete; no placeholder text or empty-data sinks.

## Files Touched (final)

```
.planning/REQUIREMENTS.md                  (modified)
.planning/ROADMAP.md                       (modified)
.planning/phases/43-.../43-GATE01-BASELINE.txt   (created)
```

`src/llm_sad_sam/linkers/experimental/s_linker19.py` and `s_linker13_min.py` are **unchanged** — verified via `sha256sum --check 43-GATE01-BASELINE.txt` returning OK for both. GATE-01 invariant intact at end of Plan 01.

## Self-Check: PASSED

- `.planning/REQUIREMENTS.md` — FOUND
- `.planning/ROADMAP.md` — FOUND
- `.planning/phases/43-replay-s-linker19-checkpoints-for-paper-rq1-rq4-eval/43-GATE01-BASELINE.txt` — FOUND
- Commit `20ba95f` (Task 1) — FOUND in `git log`
- Commit `53fb6ae` (Task 2) — FOUND in `git log`
