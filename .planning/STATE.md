---
gsd_state_version: 1.0
milestone: v2.3
milestone_name: Trained Multi-Role Prompt Replacement (β architecture)
status: roadmap-defined
last_updated: "2026-06-01T00:00:00.000Z"
last_activity: 2026-06-01
progress:
  total_phases: 6
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-01 for v2.2 close)

**Core value:** Every rule removed and every prompt-rule trimmed must hold macro F1 ≥ 0.93 on Claude Sonnet AND gpt-5.4 macro within tolerance of the v2.0 baseline (0.9077) — or be rejected. Generality first (GATE-06).
**Current focus:** v2.3 roadmap defined. Next action: `/gsd-plan-phase 14` to produce Phase 14 plans (β Training Harness Infrastructure — all code, no LLM budget).

## Current Position

Phase: 14 (defined, not started)
Plan: —
Status: Roadmap defined — ready for Phase 14 planning
Last activity: 2026-06-01 — Roadmap v2.3 written (Phases 14–19)

```
[Phase 14]──▶[Phase 15]──▶[Phase 16]──▶[Phase 17 (cond.)]──▶[Phase 19]
                │               │
                └──KILL──▶[Phase 18 (Compact-B)]──────────────────────▶[Phase 19]
```

## Standing Gates (carried forward to next milestone)

- **GATE-01** (carried from v2.1): macro F1 ≥ 0.93 on Claude Sonnet AND gpt-5.4 macro F1 ≥ 0.8977 absolute (T = 1.0pp off v2.0 0.9077 anchor). Applies to canonical artifacts only. `s_linker14_voyager` (experimental=True) is NOT bound to GATE-01 — it is bound only to the 0.87 floor.
- **GATE-02** (carried from v2.1): frozen-compat regression test — all CANONICAL_VARIANTS produce F1 matching v2.0 baseline JSON. `tests/test_v20_baseline_regression.py` covers `s_linker13`, `s_linker13_clean`, `s_linker13_min`, plus EXTENSION variants. Extended with `s_linker14_voyager` snapshot on registration.
- **GATE-06**: generality audit — every new prompt + helper passes BENCHMARK_TABOO scan AND reviewer-defensibility check. Bank-entry boundary is the active gate for v2.3 (D's pattern proposals grep'd + critic'd before insertion).
- **GATE-07**: every promoted variant registered in CANONICAL_VARIANTS + VARIANT_SPECS; standalone file; structured docstring.
- **GATE-08** (new in v2.2): cost-per-improvement audit — v4 must justify ~$60–100 gpt-5.4 training cost via STRONG promotion OR documented failure mode publishable as negative finding.

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

## v2.3 Roadmap Summary

**Architecture:** β (L + O + D-with-CoT-A + P). All roles defined in `.planning/v2.3-prep/v2.3-ARCHITECTURE.md`.
**Phases defined:** 6 (14 infra → 15 probe → 16 range → 17 confirmation [cond.] → 18 compact-B [cond.] → 19 close)
**Requirements:** 15 active (REQ-V23-01 through REQ-V23-15) + 5 standing gates (GATE-01/02/06/07/08)
**Promotion bar:** STRONG ≥ 0.9173 / WEAK [0.87, 0.9173) / FAIL < 0.87 on gpt-5.4 macro F1
**Budget:** ~$100 gpt-5.4 total (Probe $5–10 → Range $15–25 → Confirmation $40–60)
**Fallback:** Compact-B (R345 single CoT) auto-triggers on v4 FAIL

| Phase | Goal | Budget |
|-------|------|--------|
| 14 | All β code components built and tested | $0 (no LLM) |
| 15 | Probe tier: go/no-go verdict in 1–2 outer passes | $5–10 gpt-5.4 |
| 16 | Range tier: converge + 5-dataset 3-tier verdict | $15–25 gpt-5.4 |
| 17 (cond.) | Confirmation tier: 3-split sweep + registration + artifacts | $40–60 gpt-5.4 |
| 18 (cond.) | Compact-B fallback if v4 FAIL | $10–20 gpt-5.4 |
| 19 | Milestone close: audit + archive | $0 |

## Accumulated Context

### Decisions

Carried forward from v1.0 + v2.0 + v2.1 + v2.2 (see PROJECT.md Key Decisions for consolidated table). v2.3 new decisions:
- β architecture chosen (L + O + D-with-CoT-A + P) over γ (separate R5) and α (fully merged)
- Endpoint (A) Voyager-bank canonical; (B) runtime-rubric demoted to contingency
- Dual-artifact policy: s_linker13_min canonical=True retained; s_linker14_voyager experimental=True
- Backend = gpt-5.4 only; Claude only if super necessary
- Oracle = text-aware (mode i); Distillator = text-blind; leak defense at bank-entry boundary

### Pending Todos

None. Phase 14 planning is the next action.

### Blockers/Concerns

None.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260601-bfe | Ship v2.2 as s_linker13_min + opt-in Probe D runtime coref rubric (gpt-5.4 only) carve-out; defer v4 to v2.3 with proven per-backend cache infrastructure + vocab fix | 2026-06-01 | 5831478 | [260601-bfe-ship-v2-2-as-s-linker13-min-opt-in-probe](./quick/260601-bfe-ship-v2-2-as-s-linker13-min-opt-in-probe/) |

## Deferred Items (v2.4+ candidates)

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| v2.4+ | Claude cross-model verification of v4 | Pending — only run if reviewers require | v2.3 kickoff |
| v2.4+ | (B-new) runtime bank-builder | Logged — rejected for v2.3; future if (A) shows per-doc adaptation gap | v2.3 kickoff |
| v2.4+ | ADAPTER-01: Multi-model backend-adaptive prompts | Deferred | v2.0 close + v2.1 re-flagged |
| v2.4+ | Self-Refine layered on accepted variants | DECLINED as primary — Probe C WEAK_PASS +0.00004pp; GATE-08 flag | v2.2 close |
| v2.4+ | Extended Thinking on judge stages | Deferred | v2.1 close |
| v2.4+ | Link provenance data structure | Deferred | v2.1 Phase 12 |
| v2.4+ | EXT-04: Emit-biased boundary prompting on alias-discovery | Deferred | v2.0 kickoff |
| v2.4+ | Claude Probe D re-test with per-backend cache fix | METHODOLOGICALLY READY — cost ~$1.5; out-of-scope for v2.3 per backend policy | v2.2 close |

## Session Continuity

Last session: 2026-06-01T00:00:00.000Z
Stopped at: v2.3 roadmap written (Phases 14–19). ROADMAP.md + v2.3-ROADMAP.md + REQUIREMENTS.md traceability updated.
Resume file: .planning/milestones/v2.3-ROADMAP.md
Next action: `/gsd-plan-phase 14`
