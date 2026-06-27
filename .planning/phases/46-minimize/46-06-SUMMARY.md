---
phase: 46-minimize
plan: "06"
subsystem: prompt-minimization
section_anchor: VAL
tags: [minimize, val, scratch-trial, gate-01, gate-06, cross-section-batch, lexicon-handoff]
dependency_graph:
  requires:
    - 46-01 (scratch bootstrap + ACCEPTED_PREFIXES wiring + tombstone pre-fill)
    - 46-02 (CUT-AMB-02 — pleonasm batch site 1/3, sha 0710510)
    - 46-05 (CUT-EXT-01 — pleonasm batch site 2/3, sha fbfbcb9)
  provides:
    - VAL section of MINIMIZE-LOG populated (3 trialled rows + tombstone backfill)
    - Cross-section pleonasm batch CLOSED 3/3 (CUT-VAL-02 = site 3/3)
    - VAL-03 → COR-01 lexicon handoff string `noun phrase that refers back` (CUT-VAL-03 kept)
    - CUT-VAL-04 tombstone protected (sha eec7fb8)
  affects:
    - 46-07 (consumes VAL-03 lexicon handoff for CUT-COR-01)
    - 46-08 (Pareto Summary rolls VAL section + cross-section batch cross-references)
    - 47 (SHIP) — kept-cut after-text inlined from tests/scratch/{s_linker19.py, prompts_v5.py}
tech_stack:
  added: []
  patterns:
    - "Risk-ascending trial within section (D-02): CUT-VAL-02 (low) → CUT-VAL-01 (med) → CUT-VAL-03 (med-high) → CUT-VAL-04 (protected)"
    - "Universal-noun replacement (`counterparts` → `matching entities`; `role-referential phrase` → `noun phrase that refers back`)"
    - "Cross-section batch participation (CUT-VAL-02 + CUT-AMB-02 + CUT-EXT-01 share `components` bare vocab)"
    - "Allow-empty docs commit for tombstone protection + separate bookkeeping commit for SHA backfill (no --amend)"
key_files:
  created:
    - .planning/phases/46-minimize/46-06-SUMMARY.md
  modified:
    - tests/scratch/s_linker19.py (line 347 opener mutation — CUT-VAL-02)
    - tests/scratch/prompts_v5.py (line 82 VALIDATION_RULES — CUT-VAL-01; lines 94-100 COREF_VALIDATION_FOCUS — CUT-VAL-03)
    - .planning/phases/46-minimize/s_linker20-MINIMIZE-LOG.md (VAL section table + closing blockquote + tombstone SHA backfill)
decisions:
  - "CUT-VAL-02 kept: harness compatibility validated through 46-01 ACCEPTED_PREFIXES wiring; 24/24 snapshots pass; pleonasm batch site 3/3 closed"
  - "CUT-VAL-01 kept: VALIDATION_RULES body content (not opener) — reconstructor unaffected; 24/24 pass; `counterparts` replaced with universal noun `matching entities`"
  - "CUT-VAL-03 kept: COREF_VALIDATION_FOCUS body content (not opener) — reconstructor unaffected; 24/24 pass; replacement vocab `noun phrase that refers back` written to LOG for 46-07 to reuse on CUT-COR-01"
  - "CUT-VAL-04 protected: P1_FOCUS qualified-name X.Y.Z clause is empirically validated load-bearing per prompts_v5.py:5-22 docstring + experiment_dotted_path_rename.py; allow-empty docs commit + separate bookkeeping backfill commit"
metrics:
  duration: "~12 min wall-clock"
  completed_date: "2026-06-08"
  trials_attempted: 3
  trials_kept: 3
  trials_reverted: 0
  trials_unsafe: 0
  tombstones_protected: 1
  snapshots_passed: "24/24 per cut (×3 = 72 total)"
  loc_saved_section: 0
  commits_emitted: 6
---

# Phase 46 Plan 06: VAL Section Trial Cuts — Summary

VAL section of `s_linker20-MINIMIZE-LOG.md` populated with 4 rows in D-02 risk-ascending order (CUT-VAL-02 → CUT-VAL-01 → CUT-VAL-03 trialled; CUT-VAL-04 tombstone protected). All three trialled cuts **kept** with 24/24 snapshots passing under `SAD_SAM_LINKER_SOURCE=scratch` and clean GATE-06 re-grep against `BENCHMARK_TABOO.md`. CUT-VAL-02 closes site 3/3 of the cross-section `software architecture …` opener pleonasm batch. CUT-VAL-03 kept established the integration contract for plan 46-07: replacement vocabulary `noun phrase that refers back` recorded for shared lexicon with CUT-COR-01.

## Per-cut verdicts

| cut_id | verdict | snapshot_delta | gate06 | loc_saved | commit_sha | wave-2 dependency note |
|---|---|---|---|---|---|---|
| CUT-VAL-02 | kept | 0/24 | clean | 0 | `d82e5a9` | Pleonasm batch site 3/3 (with AMB-02 `0710510` + EXT-01 `fbfbcb9`) |
| CUT-VAL-01 | kept | 0/24 | clean | 0 | `5118c32` | VALIDATION_RULES body swap — reconstructor unaffected |
| CUT-VAL-03 | kept | 0/24 | clean | 0 | `8c195bc` | COREF_VALIDATION_FOCUS body swap; sets VAL-03 → COR-01 lexicon |
| CUT-VAL-04 | protected | n/a | n/a | 0 | `eec7fb8` | P1_FOCUS X.Y.Z tombstone — behaviorally load-bearing, NOT trialled |

## Commits emitted (6 total)

1. `d82e5a9` — `feat(46-06): keep CUT-VAL-02 — Validate component references … -> Validate components in a document. (batch with AMB-02 + EXT-01)`
2. `5118c32` — `feat(46-06): keep CUT-VAL-01 — counterparts -> matching entities`
3. `8c195bc` — `feat(46-06): keep CUT-VAL-03 — role-referential phrase -> noun phrase that refers back`
4. `eec7fb8` — `docs(46-06): protect CUT-VAL-04 — P1_FOCUS qualified-name X.Y.Z clause behaviorally protected` (allow-empty)
5. `10ad787` — `docs(46-06): backfill CUT-VAL-04 tombstone commit_sha`
6. `80784b6` — `docs(46-06): VAL section closing note + COR-01 lexicon handoff`

## Cross-section pleonasm batch state (CLOSED 3/3)

| batch member | section | site | sha | plan |
|---|---|---|---|---|
| CUT-AMB-02 | AMB | 1/3 | `0710510` | 46-02 |
| CUT-EXT-01 | EXT | 2/3 | `fbfbcb9` | 46-05 |
| CUT-VAL-02 | VAL | 3/3 | `d82e5a9` | **46-06 (this plan)** |

All three share the `components` bare replacement vocabulary pre-decided in the 46-01 MINIMIZE-LOG header. The Pareto Summary (46-08, Wave 3) will roll the three commits together as one conceptual batch with combined LOC-saved (= 0 — all substring rewordings) and cross-reference footer.

## VAL-03 → COR-01 lexicon handoff (integration contract for plan 46-07)

CUT-VAL-03 was **kept**. The chosen replacement vocabulary is:

> **`noun phrase that refers back`**

Plan 46-07's Task 2 (CUT-COR-01) **must** read this string from:
1. The CUT-VAL-03 row's `reasoning` cell in `.planning/phases/46-minimize/s_linker20-MINIMIZE-LOG.md`, AND
2. The VAL section closing blockquote (also in the same LOG file).

Then apply the SAME wording to the `role-referential noun phrase` span in `COREF_RULES` (`tests/scratch/prompts_v5.py:102`) so the two cuts stay lexically aligned across `COREF_VALIDATION_FOCUS` (VAL-03) and `COREF_RULES` (COR-01).

## Per-cut isolation evidence

### CUT-VAL-02 — pleonasm batch site 3/3
- **Before:** `Validate component references in a software architecture document. {focus}` (tests/scratch/s_linker19.py:347)
- **After:** `Validate components in a document. {focus}`
- **Harness compatibility:** `reconstruct_validation_inputs` (`tests/harness/inputs.py:279-291`) consumes both pre/post openers via `ACCEPTED_PREFIXES` tuple pre-wired by 46-01 (entry 2)
- **24/24 snapshots passed** across `phase_4_twopass_p1` + `phase_4_twopass_p2` + `phase_5_coref_validation`
- **GATE-06 re-grep:** `validate` 0 hits; `components` 5 hits (per-dataset `Components:` schema column headers — generic SE noun anaphor, cleared per Phase 45 v2.1 isolation precedent); `document` 1 hit (line 100 methodology prose — not dataset vocab)

### CUT-VAL-01 — counterparts → matching entities
- **Before:** `…including counterparts.` in VALIDATION_RULES (tests/scratch/prompts_v5.py:82)
- **After:** `…including matching entities.`
- **Harness compatibility:** VALIDATION_RULES is body content (not opener); reconstructor unaffected
- **24/24 snapshots passed**
- **GATE-06 re-grep:** `approve`/`sentence`/`treats`/`architectural`/`participant`/`including`/`matching`/`entities` all 0 hits; `component` hits = generic SE noun anaphor (cleared per Phase 45 v2.1 isolation precedent)

### CUT-VAL-03 — role-referential phrase → noun phrase that refers back
- **Before:** `…or similar role-referential phrase in this sentence actually refer to…` in COREF_VALIDATION_FOCUS (tests/scratch/prompts_v5.py:94-100)
- **After:** `…or similar noun phrase that refers back in this sentence actually refer to…`
- **Syntactic agreement preserved** (plural `refer` agrees with the disjunction subject `"the pronoun, 'it', 'they', 'the service', or similar noun phrase…"`)
- **Asymmetric design untouched:** the empirically load-bearing narrower focus (~4 FP reduction on bigbluebutton coref per prompts_v5.py:90-93 docstring) is preserved
- **Harness compatibility:** COREF_VALIDATION_FOCUS is body content (not opener); reconstructor unaffected; the 5 `phase_5_coref_validation` snapshots exercise this constant
- **24/24 snapshots passed**
- **GATE-06 re-grep:** `noun`/`phrase`/`refers`/`back`/`coref`/`resolution`/`pronoun`/`sentence`/`named` all 0 hits; `component` hits = generic SE noun anaphor (cleared)

### CUT-VAL-04 — P1_FOCUS X.Y.Z tombstone (protected, NOT trialled)
- **Protected clause:** `…and not just as a qualified-name identifier (e.g. a package- or member-access path X.Y.Z)?` in P1_FOCUS (prompts_v5.py:68-74)
- **Empirical evidence (prompts_v5.py docstring lines 5-22 + experiment_dotted_path_rename.py):** catches 2/3 code-path FPs on gpt-5.4 + 1/3 on Claude Sonnet with 0 collateral damage on the 4-TP control set; strict joint improvement over prior `dotted-path identifier` wording (which catches 0/3 on Sonnet)
- **Threat:** Phase 45 T-45-VAL-02 — Phase 46 MUST NOT cut
- **No scratch edits attempted.** Allow-empty docs commit (`eec7fb8`) emitted with body quoting before-text + empirical evidence + threat reference. Separate bookkeeping commit (`10ad787`) backfills the `(assigned by 46-06)` placeholder in the Protected Tombstones section row.

## GATE-01 verification

`git diff --stat src/llm_sad_sam/linkers/experimental/s_linker19.py src/llm_sad_sam/linkers/experimental/prompts_v5.py src/llm_sad_sam/linkers/experimental/s_linker13_min.py` returns empty after every commit (continuous GATE-01 hold across all 6 commits in this plan). The frozen sources are byte-equal to HEAD throughout — all cut work lives in `tests/scratch/` mirrors.

## Deviations from plan

None. Plan executed exactly as written:
- D-02 risk-ascending order honored (CUT-VAL-02 → CUT-VAL-01 → CUT-VAL-03 → tombstone).
- 6 commits emitted (matches plan upper-bound estimate of 5–6: 3 trial + 1 protect + 1 bookkeeping + 1 closing-note).
- All three trialled cuts kept on first attempt; no reverts, no unsafe verdicts.
- CUT-VAL-04 tombstone never edited in scratch (per CONTEXT in-scope §).

## Self-Check: PASSED

| Item | Result |
|------|--------|
| `tests/scratch/s_linker19.py` line 347 opener mutated to `Validate components in a document. {focus}` | FOUND |
| `tests/scratch/prompts_v5.py` line 82 VALIDATION_RULES contains `including matching entities.` | FOUND |
| `tests/scratch/prompts_v5.py` COREF_VALIDATION_FOCUS contains `noun phrase that refers back` | FOUND |
| `.planning/phases/46-minimize/s_linker20-MINIMIZE-LOG.md` VAL section table has 3 trialled rows | FOUND |
| `.planning/phases/46-minimize/s_linker20-MINIMIZE-LOG.md` VAL closing blockquote present | FOUND |
| `.planning/phases/46-minimize/s_linker20-MINIMIZE-LOG.md` CUT-VAL-04 tombstone row commit_sha = `eec7fb8` (placeholder removed) | FOUND |
| Commit `d82e5a9` (CUT-VAL-02 kept) in `git log` | FOUND |
| Commit `5118c32` (CUT-VAL-01 kept) in `git log` | FOUND |
| Commit `8c195bc` (CUT-VAL-03 kept) in `git log` | FOUND |
| Commit `eec7fb8` (CUT-VAL-04 protect, allow-empty docs) in `git log` | FOUND |
| Commit `10ad787` (CUT-VAL-04 SHA backfill bookkeeping) in `git log` | FOUND |
| Commit `80784b6` (VAL closing note) in `git log` | FOUND |
| GATE-01 final: `git diff --stat` on three frozen sources empty | PASS (zero output) |
| Post-commit deletion check (`git diff --diff-filter=D HEAD~6 HEAD`) | empty (zero deletions) |
