---
gsd_state_version: 1.0
milestone: null
milestone_name: between milestones — v2.1 archived; v2.2 prep delivered
status: idle
stopped_at: "v2.1 SHIPPED + archived 2026-06-01. s_linker13_min PROMOTED as canonical. v2.2-prep Voyager-TLR pilot v2 (3 fresh-start gpt-5.4 splits) delivered 2026-06-01 — verdict: SPLIT-FRAGILE, mean held-out lift -0.05 pp, do not promote. Awaiting v2.2 kickoff with revised anchor."
last_updated: "2026-06-01T06:00:00.000Z"
last_activity: 2026-06-01 — voyager-v2 pilot ran 3 splits (gpt-5.4, fresh-start, full intermediate logging); rollup + 3 per-split SUMMARYs committed to .planning/v2.2-prep/
progress:
  total_phases: 0
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-01 for v2.1 archive)

**Core value:** Every rule removed and every prompt-rule trimmed must hold macro F1 ≥ 0.93 on Claude Sonnet AND gpt-5.4 macro within tolerance of the v2.0 baseline (0.9077) — or be rejected. Generality first (GATE-06).
**Current focus:** Between milestones. v2.1 archived. Awaiting v2.2 kickoff anchored by Voyager-TLR pilot result or other v2.1-deferred candidate.

## Current Position

Phase: none active
Plan: none active
Status: idle — v2.1 archived; no live milestone

Progress: [          ] 0%

## Standing Gates (carried forward to next milestone)

- **GATE-01** (carried from v2.1): macro F1 ≥ 0.93 on Claude Sonnet AND gpt-5.4 macro F1 ≥ 0.8977 absolute (T = 1.0pp off v2.0 0.9077 anchor). v2.1 Scenario-E relaxation framework (cross-model floor 0.89, per-dataset drop ≤ −4pp for runtime-mechanism variants) available if v2.2 needs aggressive trim mechanism exploration.
- **GATE-02** (carried from v2.1): frozen-compat regression test — all CANONICAL_VARIANTS produce F1 matching v2.0 baseline JSON. `tests/test_v20_baseline_regression.py` covers `s_linker13`, `s_linker13_clean`, `s_linker13_min`, plus 7 EXTENSION variants.
- **GATE-06**: generality audit — every new prompt + helper passes BENCHMARK_TABOO scan AND reviewer-defensibility check. v2.1 cross-dataset isolation methodology (`12-05-SUMMARY-REVISIT.md`) is the operational test for runtime LLM discovery.
- **GATE-07**: every promoted variant registered in CANONICAL_VARIANTS + VARIANT_SPECS; standalone file; structured docstring.

## Canonical Artifact (current)

- **`src/llm_sad_sam/linkers/experimental/s_linker13_min.py`** (v2.1 PROMOTED, `canonical=True`)
- Claude Sonnet macro F1: 0.9506
- gpt-5.4 macro F1: 0.9069
- Composition: trim1 (distilled Tier-1 judge rubric) + trim9 (runtime Tier-2 seed disambiguation rubric)
- Dependencies: `prompts_v3` + `helper_v3` + `s_linker13_clean_v3`

## Accumulated Context

### Decisions

Carried forward from v1.0 + v2.0 + v2.1 (see PROJECT.md Key Decisions for the consolidated table). Per-milestone decision log archived at the respective `milestones/v{X}-ROADMAP.md`.

### Pending Todos

No active milestone todos. v2.2 anchor candidates listed in PROJECT.md "Next Milestone Candidates" section.

### Blockers/Concerns

None.

## Deferred Items (v2.2 candidates)

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| v2.2+ | Voyager-TLR train-test methodology (gpt-5.4 pilot result) | **DECLINED — v2.2-prep pilot v2 (3 splits, fresh-start gpt-5.4) confirmed SPLIT-FRAGILE: mean +0/-0 pp, range [-1.92, +1.09], best below trim1 by 2.82 pp; see `.planning/v2.2-prep/voyager-v2-rollup.md`** | v2.2-prep close (2026-06-01) |
| v2.2+ | ADAPTER-01: Multi-model backend-adaptive prompts (re-opened by v2.1 trim4/5/6/7 single-FP rejections) | Deferred | v2.0 close + v2.1 re-flagged |
| v2.2+ | Self-Refine layered on accepted variants (proposer-side recovery for TS regression) | Deferred | v2.1 close |
| v2.2+ | Extended Thinking on judge stages (Tier-1 ambiguity calibration recovery on gpt-5.4) | Deferred | v2.1 close |
| v2.2+ | Upstream-tier rule removal (extraction/coref tier — v2.0 EXT-01 evidence) | Deferred | v2.0 close |
| v2.2+ | Link provenance data structure | Deferred | v2.1 Phase 12 |
| v2.2+ | EXT-04: Emit-biased boundary prompting on alias-discovery (BBB variance band 3pp → 1pp) | Deferred | v2.0 kickoff |
| v2.2+ | Problem-statement system message preamble (Erdős-comparison discussion) | Deferred | v2.1 close |
| v2.2+ | Verifier-driven iteration (Erdős-comparison Pilot A) | Deferred | v2.1 close |
| v2.2+ | Cached problem-grounded rubric (Erdős-comparison Pilot B) | Deferred | v2.1 close |

## Session Continuity

Last session: 2026-06-01T06:00:00.000Z
Stopped at: v2.2-prep Voyager-TLR pilot v2 SHIPPED — 3 fresh-start gpt-5.4 splits (replication, BBB-in-train acid test, rotated hold-out), full intermediate state saved (json+pkl) per iter. Verdict: SPLIT-FRAGILE, do not promote. Voyager pulled from v2.2 candidate queue; v2.2 anchor candidates remain ADAPTER-01, Self-Refine, Extended Thinking on judge, Upstream-tier rule removal, Problem-statement preamble, Verifier-driven iteration. Awaiting v2.2 kickoff decision on revised anchor.
Resume file: None
