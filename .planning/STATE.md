---
gsd_state_version: 1.0
milestone: v2.6.4
milestone_name: — IN PROGRESS)
status: executing
stopped_at: Phase 48 plan 02 complete (verdict formalization — REQ-V264-09 FAIL 88.9%, GATE-06 clean, GATE-08 PASS $7.71)
last_updated: "2026-06-09T13:30:00Z"
last_activity: 2026-06-09 -- Phase 48 plan 02 complete (verdict + gate records)
progress:
  total_phases: 6
  completed_phases: 5
  total_plans: 22
  completed_plans: 22
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-05 for v2.6.4 kickoff)

**Core value:** Every rule removed and every prompt-rule trimmed must hold macro F1 ≥ 0.93 on Claude Sonnet AND gpt-5.4 macro within tolerance of the v2.0 baseline (0.9077) — or be rejected. Generality first (GATE-06).
**Current focus:** Phase 48 — sweep

## v2.6.3 — SHIPPED 2026-06-05

Paper RQ1–RQ4 cells populated via s_linker19 checkpoint replay. Phase 43 closed. Zero new LLM calls. GATE-01 byte-equal. See `.planning/milestones/v2.6.3-MILESTONE-AUDIT.md`.

## v2.6.2 — SHIPPED 2026-06-03 + post-ship 17d/17e extension

Multi-framing exploration complete. ICSE decision (FINAL): use s_linker17a naming.
Post-ship: 17d (wrong hypothesis) + **17e breakthrough** (92.3% GPT, +3.2pp vs s15, FP 31→14).
Unified validation concept validated: ALL links (framing + coref) pass the same quality gate.
See `.planning/milestones/v2.6.2-MILESTONE-AUDIT.md`.

### v2.6.2 results (reference) — GPT-5.4

| Dataset | s15 | 17a | 17b (k=2) | 17c (union) | 17d | **17e** |
|---------|-----|-----|-----------|-------------|-----|---------|
| MediaStore | 91.8% | 90.3% | 92.1% | 92.1% | 90.3% | **94.9%** |
| TeaStore | 96.4% | 94.7% | 94.5% | 96.4% | 98.2% | 96.3% |
| TeaMmates | 82.5% | 82.3% | 78.3% | 80.0% | 80.3% | **89.8%** |
| BigBlueButton | 77.6% | 74.1% | 70.2% | 77.0% | 75.0% | **80.4%** |
| JabRef | 97.3% | 90.0% | 90.0% | 90.0% | 90.0% | **100.0%** |
| **Macro** | **89.1%** | 86.3% | 85.0% | 87.1% | 86.8% | **92.3%** |
| FP | 31 | 37 | 37 | 43 | 43 | **14** |

Logs: `logs/v2.6.1_s17ab_claude.log` (17a/17b, GPT-5.4, misleading name);
`logs/v2.6.2_s17c_gpt.log` (17c); `logs/v2.6.2_s17d_gpt.log` (17d);
`logs/v2.6.2_s17e_gpt.log` (17e).

## Frozen / pending

- **v2.7 (Phases 38–42)** — FROZEN. Resume after v2.6.4. ⚠ Phases 40–41 predicated on keeping training — re-evaluate when v2.7 resumes.
- **v2.6 close (Phase 37)** — GATE-06 'Persistence' taboo fix + v2.6 audit, deferred.

## Current Position

Phase: 48 (sweep) — ALL PLANS COMPLETE; Phase 49 next
Plan: 2 of 2 (complete)
Status: Phase 48 complete — verdict formalized; REQ-V264-09 FAIL, GATE-06 clean, GATE-08 PASS
Last activity: 2026-06-09 -- Phase 48 plan 02 complete (verdict + gate records; Phase 49 ready)

```
Progress: v2.6.4 [█████████████████████████████░] 5/6 phases
          Phase 44 HARNESS  [x] complete (2026-06-07)
          Phase 45 AUDIT    [x] complete (2026-06-08) — 19 cut candidates, 1 benchmark-leak
          Phase 46 MINIMIZE [x] complete (2026-06-08) — 12 kept, 14 LOC saved, leak eliminated
          Phase 47 SHIP     [x] complete (2026-06-09) — s_linker20 shipped, GATE-01/06 pass
          Phase 48 SWEEP    [x] complete (2026-06-09) — MARGINAL FAIL (macro 88.9%; GATE-06 clean; GATE-08 $7.71 PASS)
          Phase 49 CLOSE    [ ] not started
```

## v2.6.4 Roadmap Summary

| Phase | Goal | Key REQs | Cost |
|-------|------|----------|------|
| 44 — HARNESS | Golden-replay fixture harness for all 6 s19 prompt sites | REQ-V264-01, REQ-V264-02 | $0 |
| 45 — AUDIT | Per-constant + per-builder audit doc with generality verdicts + cut candidates | REQ-V264-03, REQ-V264-04 | $0 |
| 46 — MINIMIZE | Pareto cut loop driven by golden tests + candidate list | REQ-V264-05, REQ-V264-06, REQ-V264-07 | $0 |
| 47 — SHIP | `s_linker20.py` standalone + runner wired + GATE-01 verify | REQ-V264-08, GATE-01 | $0 |
| 48 — SWEEP | 5-dataset gpt-5.4 macro F1 sweep on s_linker20; floor ≥ 91.3% | REQ-V264-09, GATE-06, GATE-08 | ≤ $20 |
| 49 — CLOSE | Final gate audit, MILESTONES.md, archive | GATE-01/06/08 (final) | $0 |

**Floor:** s_linker20 gpt-5.4 macro ≥ 91.3% (= s17e 92.3% − T 1.0pp)
**Budget cap:** ≤ $20 total (all LLM calls; zero for harness + audit + minimize + ship)
**s17e per-dataset reference:** MS 94.9%, TS 96.3%, TM 89.8%, BBB 80.4%, JAB 100.0%

## Standing Gates (into v2.6.4)

- **GATE-01**: PASS (carried from v2.6.3). s_linker13_min.py + s_linker19.py byte-equal. ✅
- **GATE-06**: PASS (carried). Zero benchmark-derived vocabulary in prompt constants. ✅
- **GATE-08**: Active — budget cap ≤ $20 for Phase 48 macro F1 sweep. 🔄

## Canonical Artifact (current)

- **`src/llm_sad_sam/linkers/experimental/s_linker13_min.py`** (v2.1 PROMOTED, `canonical=True`, unchanged)
- Claude Sonnet macro F1: 0.9506 | gpt-5.4 macro F1: 0.9069

## Experimental Artifact (current — paper variant)

- **`src/llm_sad_sam/linkers/experimental/s_linker19.py`** (`experimental=True`, `canonical=False`)
- Paper RQ1–RQ4 reference variant. BYTE-EQUAL FROZEN. Do not modify.
- gpt-5.4 macro: see v2.6.3 RQ1 tables. Claude macro: 93.9% (doc-to-SAM).

## v2.6.4 Target Artifact

- **`src/llm_sad_sam/linkers/experimental/s_linker20.py`** (CREATED Phase 47 plan 01, commit de3b48e)
- Standalone (no inheritance from s19); minimized inlined constants; experimental=True, canonical=False.
- Registered in run_ablation.py (commit a267a96). Target: gpt-5.4 macro ≥ 91.3%.

## Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260602-d1w | investigate latency implications of switching gpt backend to flex tier | 2026-06-02 | — | [260602-d1w-investigate-latency-implications-of-swit](./quick/260602-d1w-investigate-latency-implications-of-swit/) |
| 20260604-lissa-rq1-eval | Clone lissa-replication into sota/; re-evaluate gpt-5-mini d2m/d2c tracelinks via metrics_api.py; fill RQ1 LiSSA cells | 2026-06-04 | — | [20260604-lissa-rq1-eval](./quick/20260604-lissa-rq1-eval/) |

## s_linker20 Sweep Results (Phase 48 Plan 01)

| Dataset | s20 F1 | s17e F1 | Delta | Status |
|---------|--------|---------|-------|--------|
| MediaStore | 96.7% | 94.9% | +1.8pp | PASS |
| TeaStore | 98.1% | 96.3% | +1.8pp | PASS |
| TeaMmates | 83.3% | 89.8% | -6.5pp | FAIL |
| BigBlueButton | 75.0% | 80.4% | -5.4pp | FAIL |
| JabRef | 91.4% | 100.0% | -8.6pp | FAIL |
| **Macro** | **88.9%** | **92.3%** | **-3.4pp** | **FAIL** |

REQ-V264-09: FAIL (macro 88.9% < floor 91.3%). GATE-08: PASS ($7.71 < $20).
Log: `logs/v2.6.4_s_linker20_gpt.log`

## Session Continuity

Last session: 2026-06-09T13:30:00Z
Stopped at: Phase 48 plan 02 complete (verdict formalization — REQ-V264-09 FAIL, GATE-06 clean, GATE-08 PASS $7.71)
Resume file: None
Next action: Phase 49 CLOSE (milestone audit + MILESTONES.md + GATE-01/06/08 final)

## Performance Metrics

| Phase | Plan | Duration | Notes |
|-------|------|----------|-------|
| Phase 46 P08 | 25 | - tasks | - files |
| Phase 47 | P01 | 8 min | 2 tasks, 2 files (1 created, 1 modified) |
| Phase 47 | P02 | 2 min | 2 tasks, 2 files (1 created, 1 modified) |
| Phase 48 | P01 | 24 min | 2 tasks, 1 file committed (log); sweep 99 calls, macro 88.9% MARGINAL FAIL |
| Phase 48 | P02 | 10 min | 2 tasks, 1 file created (48-02-SUMMARY.md); verdict + GATE-06/08 records |
