---
gsd_state_version: 1.0
milestone: v2.0
milestone_name: Complete Rule Removal + Cross-Model
status: active
last_updated: "2026-05-30T00:00:00Z"
last_activity: 2026-05-30 -- v2.0 milestone started; defining requirements
progress:
  total_phases: 0
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-30 for v2.0 kickoff)

**Core value:** Every rule removed from `s_linker12c`/`s_linker13` and replaced by an LLM primitive must hold macro F1 ≥ 93% AND read as project-agnostic to a reviewer — or be rejected.
**Current focus:** v2.0 — defining requirements

## Current Position

Milestone: v2.0 (Complete Rule Removal + Cross-Model — Generality First)
Phase: Not started (defining requirements)
Plan: —
Status: Defining requirements
Last activity: 2026-05-30 — Milestone v2.0 started after v1.0 close

Progress: [          ] 0% (phase count TBD by roadmap)

## v2.0 Scope (active)

- **EXT-01** — Project-agnostic LLM replacement of `_has_standalone_mention` (relaxed budget)
- **EXT-02** — Drop dotted-path guard in `_has_standalone_mention` (gated on EXT-01)
- **LLM-COMBINE** — `s_linker14`: stack-or-unify combined LLM primitives (decision after EXT-01)
- **EXT-03** — GPT-5.2 cross-model re-evaluation of `s_linker13` and `s_linker14`

## Standing Gates (carried from v1.0)

- GATE-01: macro F1 ≥ 0.93; BBB tolerance 6pp; other-dataset 2pp
- GATE-05: hard-tier auto-approve thresholds (TM ≥ -0.01, BBB ≥ -0.06)
- **GATE-06 (NEW for v2.0):** generality audit — every new prompt + helper passes BENCHMARK_TABOO scan AND reviewer-defensibility check; zero hardcoded project-tailored values in prompts or logic

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- v2.0 kickoff: hard generality constraint applies to every phase (GATE-06)
- v2.0 kickoff: LLM-COMBINE stack-vs-unify decision deferred until EXT-01 signal lands
- v2.0 kickoff: EXT-04 (variance band) explicitly out of scope — defer to robustness milestone
- Carry from v1.0: Claude Sonnet only; standalone linker files; no benchmark-derived words in prompts

### Pending Todos

None yet.

### Blockers/Concerns

None.

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| v2.1+    | EXT-04: Emit-biased boundary prompting on alias-discovery (BBB variance ~3pp → ~1pp) | Deferred | v2.0 kickoff (2026-05-30) |

## Session Continuity

Last session: v1.0 close
Stopped at: v2.0 requirements not yet defined
Resume file: —
