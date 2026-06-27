---
quick_id: 260601-bfe
phase: quick
plan: 01
date: 2026-06-01
mode: quick-close
one_liner: "Close v2.2 milestone: ship s_linker13_min unchanged, register Probe D as gpt-5.4-only opt-in carve-out, create v2.3 kickoff seed with two proven prereqs"
tags: [v2.2, milestone-close, opt-in-carve-out, probe-d, v2.3-seed, gate-07]
key-files:
  modified:
    - src/llm_sad_sam/linkers/experimental/s_linker14_probe_d_upstream_clean.py
    - run_ablation.py
    - .planning/ROADMAP.md
    - .planning/MILESTONES.md
    - .planning/PROJECT.md
    - .planning/STATE.md
  created:
    - .planning/milestones/v2.2-ROADMAP.md
    - .planning/milestones/v2.2-REQUIREMENTS.md
    - .planning/milestones/v2.2-MILESTONE-AUDIT.md
    - .planning/v2.2-prep/v2.2-MILESTONE-CLOSE-SUMMARY.md
    - .planning/v2.3-prep/v2.3-KICKOFF-SEED.md
decisions:
  - "v2.2 ships s_linker13_min unchanged (no Pareto-positive cross-backend mechanism found)"
  - "Probe D registered canonical=False gpt-5.4-only opt-in (BBB Claude FAIL was cache-confounded)"
  - "Voyager v4 deferred to v2.3 with per-backend cache infra + Probe A' vocab-aligned R3 as proven prereqs"
---

# Quick Task 260601-bfe: v2.2 Milestone Close SUMMARY

## Outcome

v2.2 milestone SHIPPED 2026-06-01. Three commits, three tasks, zero LLM calls, zero frozen artifact changes.

## Tasks Executed

### Task 1 — Probe D opt-in carve-out marker (commit e479ad7)

**Files:** `s_linker14_probe_d_upstream_clean.py`, `run_ablation.py`

Prepended v2.2 OPT-IN CARVE-OUT banner to module docstring (38 lines: status, mediastore/BBB STRONG_PASS evidence, BBB Claude FAIL confound explanation, per-backend cache infra description, v2.3 handoff pointer). Updated VARIANT_SPECS description with carve-out status + NOT canonical + gating policy. Updated CANONICAL_VARIANTS inline comment.

Verified: `s_linker13_min canonical=True` preserved; `s_linker14_probe_d_upstream_clean canonical=False` preserved; both modules import cleanly.

### Task 2 — v2.2 milestone artifacts + close SUMMARY (commit 1727bec)

**Files created:** `milestones/v2.2-ROADMAP.md`, `milestones/v2.2-REQUIREMENTS.md`, `milestones/v2.2-MILESTONE-AUDIT.md`, `v2.2-prep/v2.2-MILESTONE-CLOSE-SUMMARY.md`

Modeled on v2.1 counterparts. v2.2-ROADMAP has phases 14–18 outcome table, declined tracks with negative-finding pointers, standing gates (GATE-01 through GATE-08), key numbers, carry-forward to v2.3. v2.2-REQUIREMENTS has 3/3 requirements closed (SHIP-V22-MIN-UNCHANGED, SHIP-V22-PROBE-D-OPT-IN, DEFER-V4-TO-V23) plus deferred-to-v2.3 carry-forward table (10 rows). v2.2-MILESTONE-AUDIT verdict: PASSED, 6 methodological contributions, 6 publishable findings. Close SUMMARY: ~200 lines covering trimmed scope, surviving artifact, opt-in carve-out evidence table, declined tracks (Probe A/A'/B/C), v2.3 handoff, cost ($3 total), gates compliance.

### Task 3 — Top-level updates + v2.3 kickoff seed (commit 5831478)

**Files modified:** `ROADMAP.md`, `MILESTONES.md`, `PROJECT.md`, `STATE.md`
**File created:** `v2.3-prep/v2.3-KICKOFF-SEED.md`

ROADMAP: v2.2 milestone row + v2.2 details block + Next Milestone section replaced with v2.3 anchor including both proven prereqs and link to kickoff seed.

MILESTONES: v2.2 entry inserted at top (above v2.0) with stats table, per-dataset carve-out table, 4 key v2.2 lessons, full file link block.

PROJECT: Current State updated to v2.2 SHIPPED; opt-in carve-out line added; v2.2 Past Milestones bullet; Next Milestone Candidates heading renamed to v2.3+; v2.2 close Key Decision row; footer date updated.

STATE: frontmatter updated (milestone_name, stopped_at, last_activity, last_updated); GATE-08 added to Standing Gates; new Opt-in Carve-Out section; Deferred Items updated (Voyager v4 DEFERRED, Self-Refine DECLINED as primary, Upstream SHIPPED opt-in, 3 new rows for cache infra/Probe A' vocab/Claude re-test); Session Continuity updated.

v2.3-KICKOFF-SEED: anchor (Voyager v4 multi-role), Prereq 1 (per-backend cache infra with impl pointer), Prereq 2 (Probe A' vocab-aligned R3 with impl pointer), open sub-questions, costs-already-spent, "What v2.3 Should NOT Do" (4 items), "What v2.3 SHOULD Do First" (4 steps).

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check

All artifact files exist. All registry invariants preserved. Only `s_linker14_probe_d_upstream_clean.py` modified in `src/` (docstring only). Frozen artifacts byte-equal.
