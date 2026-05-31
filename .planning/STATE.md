---
gsd_state_version: 1.0
milestone: v2.0
milestone_name: — Complete Rule Removal + Cross-Model — Generality First
status: executing
stopped_at: Phase 6 closed empty (negative); Phase 7 auto-skipped; advancing to Phase 8
last_updated: "2026-05-31T03:30:00.000Z"
last_activity: 2026-05-31 -- Phase 06 closed empty (negative result), Phase 07 auto-skipped per gating
progress:
  total_phases: 4
  completed_phases: 2
  total_plans: 9
  completed_plans: 9
  percent: 50
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-30 for v2.0 kickoff)

**Core value:** Every rule removed from `s_linker12c`/`s_linker13` and replaced by an LLM primitive must hold macro F1 ≥ 93% AND read as project-agnostic to a reviewer — or be rejected.
**Current focus:** Phase 08 — COMBINE `s_linker14` Stack-or-Unify (Phase 06 closed empty; Phase 07 auto-skipped per gating)

## Current Position

Milestone: v2.0 (Complete Rule Removal + Cross-Model — Generality First)
Phase: 08 (COMBINE — s_linker14 Stack-or-Unify Combined LLM Primitives) — STARTING
Plan: 0 of TBD
Status: Phase 6 negative-closed, Phase 7 skipped, advancing to Phase 8
Last activity: 2026-05-31 -- Phase 06 closed empty (EXT-01 not viable, recall gap upstream); Phase 07 auto-skipped

Progress: [█████     ] 50% (2/4 phases complete — Phase 6 negative-close, Phase 7 auto-skip)

## v2.0 Scope (active)

- **Phase 6 — EXT-01** — Project-agnostic LLM replacement of `_has_standalone_mention` (relaxed budget)
- **Phase 7 — EXT-02** — Drop dotted-path guard in `_has_standalone_mention` (gated on Phase 6 pass)
- **Phase 8 — COMBINE-01/02/03** — `s_linker14`: stack-or-unify combined LLM primitives (decision at phase start using EXT-01 signal)
- **Phase 9 — CROSS-01/02/03** — GPT-5.2 cross-model evaluation harness + report for `s_linker13` and `s_linker14`

## Standing Gates (carried from v1.0 + new for v2.0)

- GATE-01: macro F1 ≥ 0.93; BBB tolerance 6pp; other-dataset 2pp
- GATE-05: hard-tier auto-approve thresholds (TM ≥ -0.01, BBB ≥ -0.06)
- **GATE-06 (NEW for v2.0):** generality audit — every new prompt + helper passes BENCHMARK_TABOO scan AND reviewer-defensibility check; zero hardcoded project-tailored values in prompts or logic. **Recorded per phase in SUMMARY.md.**
- GATE-07: every promoted variant registered in `CANONICAL_VARIANTS` + `VARIANT_SPECS`; standalone file; structured docstring with `REMOVED_FROM` / `RULES_REMOVED`.

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- v2.0 kickoff: hard generality constraint applies to every phase (GATE-06)
- v2.0 kickoff: LLM-COMBINE stack-vs-unify decision deferred to Phase 8 start, using EXT-01 cost/quality signal from Phase 6
- v2.0 kickoff: EXT-04 (variance band) explicitly out of scope — defer to robustness milestone
- v2.0 roadmap: Phase 7 (EXT-02) is gated — only attempted if Phase 6 passes dual floor + GATE-06; otherwise skipped with documented negative result
- Carry from v1.0: Claude Sonnet only; standalone linker files; no benchmark-derived words in prompts

### Pending Todos

None yet. Next action: `/gsd-plan-phase 6`.

### Blockers/Concerns

None.

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| v2.1+    | EXT-04: Emit-biased boundary prompting on alias-discovery (BBB variance ~3pp → ~1pp) | Deferred | v2.0 kickoff (2026-05-30) |

## Session Continuity

Last session: --stopped-at
Stopped at: Phase 6 context gathered
Resume file: --resume-file
