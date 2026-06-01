---
gsd_state_version: 1.0
milestone: null
milestone_name: "between milestones — v2.2 archived 2026-06-01; v2.3 anchor = Voyager v4 multi-role with proven per-backend cache infra + Probe A' vocab fix"
status: idle
stopped_at: "v2.2 SHIPPED 2026-06-01 (Probe-Wave Trimmed Close). Canonical s_linker13_min unchanged (Claude 0.9506, gpt-5.4 0.9069). Probe D opt-in carve-out registered (gpt-5.4 only). Voyager v4 multi-role + per-backend cache infrastructure + Probe A' vocab-aligned R3 carried to v2.3 as proven prereqs. See .planning/milestones/v2.2-MILESTONE-AUDIT.md."
last_updated: "2026-06-01T09:30:00.000Z"
last_activity: "2026-06-01 — v2.2 milestone close (quick-mode). Created milestones/v2.2-ROADMAP|REQUIREMENTS|MILESTONE-AUDIT, v2.2-prep/v2.2-MILESTONE-CLOSE-SUMMARY, v2.3-prep/v2.3-KICKOFF-SEED. Probe D variant docstring + registry description tagged 'v2.2 OPT-IN CARVE-OUT (gpt-5.4 only)'. ROADMAP/MILESTONES/PROJECT updated."
progress:
  total_phases: 0
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-01 for v2.2 close)

**Core value:** Every rule removed and every prompt-rule trimmed must hold macro F1 ≥ 0.93 on Claude Sonnet AND gpt-5.4 macro within tolerance of the v2.0 baseline (0.9077) — or be rejected. Generality first (GATE-06).
**Current focus:** Between milestones. v2.2 archived 2026-06-01. v2.3 anchor: Voyager v4 multi-role with proven per-backend cache infrastructure + Probe A' vocab-aligned R3.

## Current Position

Phase: none active
Plan: none active
Status: idle — v2.2 archived 2026-06-01; v2.3 anchor seeded

Progress: [          ] 0%

## Standing Gates (carried forward to next milestone)

- **GATE-01** (carried from v2.1): macro F1 ≥ 0.93 on Claude Sonnet AND gpt-5.4 macro F1 ≥ 0.8977 absolute (T = 1.0pp off v2.0 0.9077 anchor). v2.1 Scenario-E relaxation framework (cross-model floor 0.89, per-dataset drop ≤ −4pp for runtime-mechanism variants) available if v2.3 needs aggressive trim mechanism exploration.
- **GATE-02** (carried from v2.1): frozen-compat regression test — all CANONICAL_VARIANTS produce F1 matching v2.0 baseline JSON. `tests/test_v20_baseline_regression.py` covers `s_linker13`, `s_linker13_clean`, `s_linker13_min`, plus 7 EXTENSION variants.
- **GATE-06**: generality audit — every new prompt + helper passes BENCHMARK_TABOO scan AND reviewer-defensibility check. v2.1 cross-dataset isolation methodology (`12-05-SUMMARY-REVISIT.md`) is the operational test for runtime LLM discovery.
- **GATE-07**: every promoted variant registered in CANONICAL_VARIANTS + VARIANT_SPECS; standalone file; structured docstring.
- **GATE-08** (new in v2.2): cost-per-improvement audit — any mechanism with marginal gain (e.g. WEAK_PASS < +0.001pp) must justify doubled LLM cost or be DECLINED as primary.

## Canonical Artifact (current)

- **`src/llm_sad_sam/linkers/experimental/s_linker13_min.py`** (v2.1 PROMOTED, `canonical=True`, unchanged in v2.2)
- Claude Sonnet macro F1: 0.9506
- gpt-5.4 macro F1: 0.9069
- Composition: trim1 (distilled Tier-1 judge rubric) + trim9 (runtime Tier-2 seed disambiguation rubric)
- Dependencies: `prompts_v3` + `helper_v3` + `s_linker13_clean_v3`

## Opt-in Carve-Out (v2.2 shipped 2026-06-01)

- **`src/llm_sad_sam/linkers/experimental/s_linker14_probe_d_upstream_clean.py`** (`canonical=False`, gpt-5.4 only)
- Mediastore gpt-5.4: F1 0.9836 (+1.59pp vs anchor 0.9677, matches Claude 0.9836 baseline)
- BBB gpt-5.4 (mean of 2 observations): F1 0.7857 (+2.2pp vs anchor 0.7636)
- BBB Claude: FAIL (-4.23pp) — CONFOUNDED by cross-backend cache reuse; per-backend cache fix landed, re-test methodologically ready but not run this milestone
- Mechanism: runtime per-dataset LLM-built coref rubric replaces static `COREF_RULES`
- Gating policy: enable ONLY when `LLM_BACKEND == openai`

## Accumulated Context

### Decisions

Carried forward from v1.0 + v2.0 + v2.1 + v2.2 (see PROJECT.md Key Decisions for the consolidated table). Per-milestone decision log archived at the respective `milestones/v{X}-ROADMAP.md`.

### Pending Todos

No active milestone todos. v2.3 anchor and prereqs listed in `.planning/v2.3-prep/v2.3-KICKOFF-SEED.md`.

### Blockers/Concerns

None.

## Deferred Items (v2.3 candidates)

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| v2.3+ | Voyager-TLR train-test methodology (gpt-5.4 pilot result) | **DECLINED — v2.2-prep pilot v2 (3 splits, fresh-start gpt-5.4) confirmed SPLIT-FRAGILE: mean +0/-0 pp, range [-1.92, +1.09], best below trim1 by 2.82 pp; see `.planning/v2.2-prep/voyager-v2-rollup.md`** | v2.2-prep close (2026-06-01) |
| v2.3+ | Voyager v4 multi-role architecture (R1-R5) | **DEFERRED to v2.3** — Probe A' fix narrowed R3/R5 deadlock; mediastore STRONG_PASS but BBB WEAK_PASS (R5 0/8). v4 is mediastore-viable, BBB-inactive on gpt-5.4. Carried to v2.3 with proven vocab-aligned R3 + per-backend cache infra as named prereqs. See `.planning/v2.3-prep/v2.3-KICKOFF-SEED.md`. | v2.2 close (2026-06-01) |
| v2.3+ | ADAPTER-01: Multi-model backend-adaptive prompts (re-opened by v2.1 trim4/5/6/7 single-FP rejections) | Deferred | v2.0 close + v2.1 re-flagged |
| v2.3+ | Self-Refine layered on accepted variants (proposer-side recovery for TS regression) | **DECLINED as primary** — Probe C WEAK_PASS +0.00004pp on mediastore; iter-1 doubled judge cost without changing approved set (GATE-08 flag). Contingent only if v2.3 mainline fails. | v2.2 close (2026-06-01) |
| v2.3+ | Extended Thinking on judge stages (Tier-1 ambiguity calibration recovery on gpt-5.4) | Deferred | v2.1 close |
| v2.3+ | Upstream-tier rule removal (extraction/coref tier — v2.0 EXT-01 evidence) | **SHIPPED as opt-in carve-out** — Probe D variant `s_linker14_probe_d_upstream_clean` registered `canonical=False` gpt-5.4-only. See `.planning/milestones/v2.2-ROADMAP.md`. | v2.2 close (2026-06-01) |
| v2.3+ | Link provenance data structure | Deferred | v2.1 Phase 12 |
| v2.3+ | EXT-04: Emit-biased boundary prompting on alias-discovery (BBB variance band 3pp → 1pp) | Deferred | v2.0 kickoff |
| v2.3+ | Problem-statement system message preamble (Erdős-comparison discussion) | **DECLINED-COMPOSED — Probe B (preamble + cached rubric) regressed mediastore gpt-5.4 -5.24pp; preamble alone not isolated. Re-test of preamble-only optional but de-prioritized.** | v2.2-prep PROBE WAVE (2026-06-01) |
| v2.3+ | Verifier-driven iteration (Erdős-comparison Pilot A) | Deferred | v2.1 close |
| v2.3+ | Cached problem-grounded rubric (Erdős-comparison Pilot B) | **DECLINED — Probe B composition included this mechanism; -5.24pp regression on mediastore gpt-5.4.** | v2.2-prep PROBE WAVE (2026-06-01) |
| v2.3+ | Per-backend cache infrastructure for runtime LLM rubrics | **PROVEN in v2.2** — per-(text_stem, comp_hash, backend, model) cache key landed in `s_linker14_probe_d_upstream_clean.py`; SANITY_PASS verified. Carry to v2.3 as proven prereq. | v2.2 close (2026-06-01) |
| v2.3+ | Probe A' vocab-aligned R3 (discourse/syntactic terms) | **PROVEN mediastore, NOT BBB** — discourse-vocab R3 narrows R3/R5 deadlock; mediastore STRONG_PASS (+1.69pp); BBB WEAK_PASS (R5 0/8, F1 -0.24pp). Carry to v2.3 as starting point. | v2.2 close (2026-06-01) |
| v2.3+ | Claude Probe D re-test with per-backend cache fix | **METHODOLOGICALLY READY** — cache-key isolation landed; fresh Claude rubric will be built on re-run. Cost ~$1.5. Decides whether Probe D extends to cross-backend or stays gpt-5.4-only. | v2.2 close (2026-06-01) |

## Session Continuity

Last session: 2026-06-01T09:30:00.000Z
Stopped at: v2.2 milestone close (quick-mode 260601-bfe). Created milestone artifacts, registered Probe D opt-in carve-out, seeded v2.3 anchor. STATE clean for v2.3 kickoff.
Resume file: .planning/v2.3-prep/v2.3-KICKOFF-SEED.md
