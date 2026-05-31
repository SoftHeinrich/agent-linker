---
gsd_state_version: 1.0
milestone: v2.1
milestone_name: Cleanup + Prompt Simplification
status: planning
stopped_at: "Phase 11 (Research) complete — survey + supplement shipped, PROMPT-05 closed. Next action: Phase 12 — Trim Ablation."
last_updated: "2026-05-31T10:15:00.000Z"
last_activity: 2026-05-31
progress:
  total_phases: 4
  completed_phases: 2
  total_plans: 5
  completed_plans: 5
  percent: 50
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-31 for v2.1 kickoff)

**Core value:** Every rule removed from `s_linker13`/its prompts must hold macro F1 ≥ 0.93 on Claude Sonnet AND gpt-5.4 macro within ≤ 1pp of 0.9077 — or be rejected. Every retained prompt + helper must read as project-agnostic to a reviewer (GATE-06). Nothing currently runnable breaks.
**Current focus:** Phase 12 — Trim Ablation — Phases 10+11 COMPLETE

## Current Position

Phase: 12 of 4 — Phases 10 (Scaffolding) + 11 (Research) COMPLETE
Plan: — of TBD (Phase 12 not yet planned)
Status: Ready to plan
Last activity: 2026-05-31 — Phase 11 verification passed; survey + supplement shipped (PROMPT-05 closed)

Progress: [█████░░░░░] 50%

## Performance Metrics

**Velocity:**

- Total plans completed: 2
- Average duration: ~15min
- Total execution time: ~27min

**By Phase:**

| Phase | Plan | Duration | Tasks | Files | Commit |
|-------|------|----------|-------|-------|--------|
| 10    | 01   | ~15min   | 2     | 2     | 98cdca2 |
| 10    | 02   | ~12min   | 1     | 1     | eae3028 |

*Updated after each plan completion*
| Phase 10 P04 | ~5min | 2 tasks | 2 files |

## Standing Gates (v2.1)

- GATE-01: macro F1 ≥ 0.93; BBB tolerance 6pp; other-dataset 2pp (Claude Sonnet)
- GATE-01 cross-model (v2.1): gpt-5.4 macro ≥ 0.9077 within ≤ 1.0pp tolerance — i.e. variant passes iff gpt-5.4 macro F1 ≥ 0.8977 on full 5-dataset sweep (T = 1.0pp committed Phase 10 Plan 10-04; baseline 0.9077 from v2.0 CROSS evidence; see PROJECT.md Key Decisions row "GATE-01 cross-model tolerance T = 1.0pp (v2.1)")
- GATE-02 (v2.1 NEW): frozen-compat regression test; all CANONICAL_VARIANTS produce F1 matching v2.0 baseline JSON
- GATE-06: generality audit — every new prompt + helper passes BENCHMARK_TABOO scan AND reviewer-defensibility check
- GATE-07: every promoted variant registered in CANONICAL_VARIANTS + VARIANT_SPECS; standalone file; structured docstring

## Accumulated Context

### Decisions

- v2.1 kickoff: `s_linker13.py`, `prompts_v2.py`, and existing helper modules are frozen; cleanup lands in `_clean` / `v3+` / `_min` siblings
- v2.1 kickoff: cross-model gate uses gpt-5.4 (v2.0 CROSS baseline 0.9077) with tolerance ≤ 1pp
- v2.1 kickoff: coarse granularity → 4 phases (10–13); sequential dependency chain
- Plan 10-02 (CLEAN-02): helper_v3.py is a single file (not per-concern split); `build_component_profile` lifts `self.model_knowledge` / `self.doc_knowledge` to explicit parameters; `MENTION_TYPES` is duplicated rather than re-imported from frozen `SLinker13d` to keep helper_v3 free of variant-class coupling.
- [Phase 10]: Plan 10-04 (GATE-01): cross-model tolerance pinned to T = 1.0pp; absolute F1 floor 0.8977 = 0.9077 − 0.01; recorded in PROJECT.md Key Decisions and STATE.md Standing Gates.

### Pending Todos

Next action: execute Plan 10-03 (s_linker13_clean — depends on 10-02) and Plan 10-04 (cross-model gate codify).

### Blockers/Concerns

None.

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| v2.2+ | EXT-04: Emit-biased boundary prompting on alias-discovery (BBB variance ~3pp → ~1pp) | Deferred | v2.0 kickoff |
| v2.2+ | EXT-upstream: Upstream-tier rule removal (extraction/coref tier) | Deferred | v2.0 close |
| v2.2+ | ADAPTER-01: Multi-model backend-adaptive harness layer | Deferred | v2.0 close |

## Session Continuity

Last session: 2026-05-31T06:12:47.539Z
Stopped at: Completed Plan 10-02 (helper_v3 extraction). Next action: execute Plan 10-03 (s_linker13_clean) — depends on 10-02.
Resume file: None
