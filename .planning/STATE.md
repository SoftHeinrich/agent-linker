---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: blocked
stopped_at: Plan 01-05 GATE-05 hard reject (s_linker13a BBB F1 = 0.796 vs 12c 0.844; delta -0.048pp)
last_updated: "2026-05-15T18:55:00Z"
last_activity: 2026-05-15 -- Plan 01-05 GATE-05 hard reject (s_linker13a; ablation_20260515_184127.json)
progress:
  total_phases: 5
  completed_phases: 0
  total_plans: 5
  completed_plans: 4
  percent: 80
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-21)

**Core value:** Every rule removed from `s_linker12c` and replaced by an LLM primitive must hold macro F1 ≥ 93% (no dataset >2pp below 12c baseline) — or be rejected. Deliverable: defensible claim that traceability linking works without hand-crafted structural rules.
**Current focus:** Phase 01 — baseline-and-infrastructure

## Current Position

Phase: 01 (baseline-and-infrastructure) — BLOCKED at Plan 05 GATE-05
Plan: 5 of 5 (executed Tasks 1–2; Task 3 checkpoint hard-rejected; Task 4 not executed)
Status: Blocked — GATE-05 hard reject on s_linker13a BBB delta = -0.048pp (< -0.02 threshold)
Last activity: 2026-05-15 -- Plan 01-05 SUMMARY written; phase 1 cannot close without re-route

Progress: [████████░░] 80% (Plan 05 not counted as complete — gate rejected)

### Plan 05 Blocker Detail

- s_linker13a hard-tier results: teammates F1=0.931 (delta -0.007, marginal-ok), bigbluebutton F1=0.796 (delta -0.048, hard-reject)
- Spike 001 LLM trailing-word enrichment produced ZERO new aliases on either hard-tier dataset; regression is from Tier-2 run-to-run drift after adding a new LLM call
- Caller routes (see 01-05-SUMMARY §Phase 1 Status):
  1. Re-run hard-tier for variance (cheapest, ~24 min)
  2. Per-document gate the trailing-word step
  3. Strengthen Spike 001 prompt rule 2
  4. Re-scope Phase 1: drop VAR-01 or pull a different first-removal variant

## Performance Metrics

**Velocity:**

- Total plans completed: 0
- Average duration: -
- Total execution time: 0.0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**

- Last 5 plans: (none)
- Trend: -

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Pre-roadmap: Base = `s_linker12c` (not 12e); 12d/12e are enrichment side-experiments
- Pre-roadmap: Ablation unit = linker variant (not individual rule); full pipeline F1 is the signal
- Pre-roadmap: Keep `_has_standalone_mention` tentatively (RISKY per Spike 002 O(N×M)); formalize in Phase 5
- Pre-roadmap: VAR-02 + VAR-03 grouped into Phase 2 (tightly coupled: 13c depends on 13b)
- Pre-roadmap: VAR-05 run twice on hard tier before full sweep (widest blast radius in chain)

### Pending Todos

None yet.

### Blockers/Concerns

- SUMMARY.md does not exist in `.planning/research/` — research phase may not have run a SUMMARY step; ARCHITECTURE.md and PITFALLS.md were used directly.

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| v2 | EXT-01: Spike on replacing `_has_standalone_mention` | Deferred | Roadmap creation |
| v2 | EXT-02: Drop dotted-path guard in `_has_standalone_mention` | Deferred | Roadmap creation |
| v2 | EXT-03: GPT-5.2 cross-model re-evaluation of s_linker13 | Deferred | Roadmap creation |

## Session Continuity

Last session: --stopped-at
Stopped at: Phase 1 context gathered
Resume file: --resume-file

**Planned Phase:** 1 (Baseline and Infrastructure) — 5 plans — 2026-05-08T16:02:52.874Z
