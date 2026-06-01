---
gsd_state_version: 1.0
milestone: null
milestone_name: between milestones — v2.1 archived; v2.2 PROBE WAVE delivered (1 STRONG survivor)
status: idle
stopped_at: "v2.2 RANGE A' BBB + Probe D cache-fix wave complete 2026-06-01 (gpt-5.4 only). (1) Range A' BBB gpt-5.4 = WEAK_PASS (R5 0/8 reject, iter0→iter1 -0.24pp; v4 architecture is mediastore-viable, BBB-inactive). (2) Probe D cache-key fix verified: per-backend (text_stem,comp_hash,backend,model) key, gpt-5.4 BBB sanity re-run STRONG_PASS (+1.12pp vs anchor; -2.17pp vs original 0.7965 = gpt-5.4 variance band on identical rubric). Claude re-test methodologically ready, deferred to next turn per user directive. Recommend hybrid path: ship s_linker13_min as v2.2 canonical, add Probe D opt-in gpt-5.4-only flag, defer v4 architecture to v2.3."
last_updated: "2026-06-01T08:05:00.000Z"
last_activity: 2026-06-01 — gpt-5.4-only wave: Range A' BBB (WEAK_PASS) + Probe D cache-fix sanity (SANITY_PASS). Per-task + combined SUMMARYs committed under .planning/v2.2-prep/. Cost ~$0.55 of $10 envelope. Awaiting user decision on v2.2 milestone scope (3 options in rollup: ship s_linker13_min unchanged + Probe D conditional flag; defer v4; re-test Claude with cache fix).
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
| v2.2+ | Voyager v4 multi-role architecture (R1-R5) | **DECLINED — v2.2 PROBE WAVE Probe A confirmed R5 100% reject rate (falsification criterion hit) on mediastore gpt-5.4; see `.planning/v2.2-prep/probe-A-voyager-v4-SUMMARY.md`** | v2.2-prep PROBE WAVE close (2026-06-01) |
| v2.2+ | ADAPTER-01: Multi-model backend-adaptive prompts (re-opened by v2.1 trim4/5/6/7 single-FP rejections) | Deferred | v2.0 close + v2.1 re-flagged |
| v2.2+ | Self-Refine layered on accepted variants (proposer-side recovery for TS regression) | **PROBE WEAK_PASS — Probe C matched gpt-5.4 mediastore anchor exactly (+0.00004pp); see `.planning/v2.2-prep/probe-C-selfrefine-SUMMARY.md`. Awaiting user decision on Range test.** | v2.2-prep PROBE WAVE (2026-06-01) |
| v2.2+ | Extended Thinking on judge stages (Tier-1 ambiguity calibration recovery on gpt-5.4) | Deferred | v2.1 close |
| v2.2+ | Upstream-tier rule removal (extraction/coref tier — v2.0 EXT-01 evidence) | **PROBE STRONG_PASS — Probe D runtime coref rubric lifted mediastore gpt-5.4 +1.59pp (0.9677 → 0.9836, matches Claude baseline); see `.planning/v2.2-prep/probe-D-upstream-SUMMARY.md`. Awaiting user approval to Range-test on BBB.** | v2.2-prep PROBE WAVE (2026-06-01) |
| v2.2+ | Link provenance data structure | Deferred | v2.1 Phase 12 |
| v2.2+ | EXT-04: Emit-biased boundary prompting on alias-discovery (BBB variance band 3pp → 1pp) | Deferred | v2.0 kickoff |
| v2.2+ | Problem-statement system message preamble (Erdős-comparison discussion) | **DECLINED-COMPOSED — Probe B (preamble + cached rubric) regressed mediastore gpt-5.4 -5.24pp; preamble alone not isolated. Re-test of preamble-only optional but de-prioritized.** | v2.2-prep PROBE WAVE (2026-06-01) |
| v2.2+ | Verifier-driven iteration (Erdős-comparison Pilot A) | Deferred | v2.1 close |
| v2.2+ | Cached problem-grounded rubric (Erdős-comparison Pilot B) | **DECLINED — Probe B composition included this mechanism; -5.24pp regression on mediastore gpt-5.4.** | v2.2-prep PROBE WAVE (2026-06-01) |

## Session Continuity

Last session: 2026-06-01T07:32:00.000Z
Stopped at: v2.2 PROBE WAVE COMPLETE — 4 parallel probes ran on mediastore gpt-5.4 from `s_linker13_clean_v3`. Scoreboard: **Probe D STRONG_PASS** (runtime coref rubric, +1.59pp, matches Claude baseline); Probe C WEAK_PASS (Self-Refine, +0.00004pp); Probe B FAIL (preamble+rubric, -5.24pp); Probe A PROBE_FAIL (Voyager v4, R5 100% reject — falsification criterion hit). Budget spent: ~$2-3 of $40 envelope. v2.2 anchor candidate is Probe D's upstream coref-rubric mechanism. Next decision required: approve Range-test of Probe D on bigbluebutton gpt-5.4 (~$3-5).
Resume file: .planning/v2.2-prep/v2.2-PROBE-WAVE-SUMMARY.md
