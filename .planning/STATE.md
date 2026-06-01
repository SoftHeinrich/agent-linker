---
gsd_state_version: 1.0
milestone: v2.3
milestone_name: Trained Multi-Role Prompt Replacement (β architecture)
status: in-progress
last_updated: "2026-06-01T08:36:26.000Z"
last_activity: 2026-06-01
progress:
  total_phases: 6
  completed_phases: 1
  total_plans: 6
  completed_plans: 6
  percent: 17
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-01 for v2.2 close)

**Core value:** Every rule removed and every prompt-rule trimmed must hold macro F1 ≥ 0.93 on Claude Sonnet AND gpt-5.4 macro within tolerance of the v2.0 baseline (0.9077) — or be rejected. Generality first (GATE-06).
**Current focus:** Phase 14 COMPLETE. Next action: Phase 15 (Probe Tier — real gpt-5.4 training runs, $5–10 budget).

## Current Position

Phase: 15 (Phase 14 complete, Phase 15 not started)
Plan: —
Status: Phase 14 shipped — ready for Phase 15 planning
Last activity: 2026-06-01 — Phase 14 β harness infrastructure complete

```
[Phase 14 ✅]──▶[Phase 15]──▶[Phase 16]──▶[Phase 17 (cond.)]──▶[Phase 19]
                    │               │
                    └──KILL──▶[Phase 18 (Compact-B)]──────────────────────▶[Phase 19]
```

## Phase 14 Deliverables (SHIPPED 2026-06-01)

| Deliverable | File | Status |
|-------------|------|--------|
| Standalone linker consumer | `src/llm_sad_sam/linkers/experimental/s_linker14_voyager.py` | ✅ |
| β Training harness (L+O+D+P) | `scripts/voyager_train_tlr_v4_beta.py` | ✅ |
| Registration (CANONICAL_VARIANTS + VARIANT_SPECS) | `run_ablation.py` | ✅ experimental=True, canonical=False |
| GATE-06 helpers (gate06_ok + reviewer_critic_stub) | `scripts/voyager_train_tlr_v4_beta.py` | ✅ callable + tested |
| Cache adapter (VOYAGER4B_CACHE_ROOT) | `scripts/voyager_train_tlr_v4_beta.py` | ✅ |
| Unit tests (32 passing) | `tests/test_s_linker14_voyager_registration.py` | ✅ |
| GATE-02 regression | test_gate02_frozen_artifacts_unchanged | ✅ PASS |

**Phase 14 success criteria verified:**
1. ✅ Dry-run mode runs end-to-end (no LLM calls) — `test_dry_run_probe_single_project` passes
2. ✅ `s_linker14_voyager` instantiates in axiom-only / empty bank mode
3. ✅ Registered with `experimental=True`, `canonical=False`, GATE-07 docstring
4. ✅ `gate06_ok` + `reviewer_critic_stub` callable + unit-tested
5. ✅ Cache roundtrip + VOYAGER4B_CACHE_ROOT override works
6. ✅ GATE-02 frozen-compat regression test passes

## Standing Gates (carried forward)

- **GATE-01**: macro F1 ≥ 0.93 Claude AND gpt-5.4 ≥ 0.8977. Applies to canonical only. `s_linker14_voyager` (experimental=True) bound to 0.87 floor only.
- **GATE-02**: frozen-compat regression — all CANONICAL_VARIANTS unaffected. `s_linker14_voyager` added to CANONICAL_VARIANTS with experimental=True.
- **GATE-06**: BENCHMARK_TABOO grep + reviewer_critic at bank-entry boundary. Helpers built in Phase 14.
- **GATE-07**: `s_linker14_voyager` registered in CANONICAL_VARIANTS + VARIANT_SPECS; standalone file; structured docstring.
- **GATE-08**: cost audit — v4 must justify ~$60–100 training cost via STRONG promotion or published negative finding.

## Canonical Artifact (current)

- **`src/llm_sad_sam/linkers/experimental/s_linker13_min.py`** (v2.1 PROMOTED, `canonical=True`, unchanged)
- Claude Sonnet macro F1: 0.9506 | gpt-5.4 macro F1: 0.9069

## Phase 15 Plan

**Goal:** Run β training on mainline split (gpt-5.4, $5–10). Cheap-kill gate: macro < 0.87 after pass 2 → KILL v4.
**Action:** `/gsd-plan-phase 15` then `/gsd-execute-phase 15`
**Budget:** $5–10 gpt-5.4

## Accumulated Context

### Decisions (carried from v2.2 + Phase 14)
- β architecture (L + O + D-with-CoT-A + P) — locked
- `s_linker14_voyager` standalone (no inheritance from s_linker13_clean variants)
- Bank format: slot-uniform 9 slots, per-project during training, aggregated final_bank.json
- Cache: per-(text_stem, comp_hash, backend, model), VOYAGER4B_CACHE_ROOT override
- GATE-06 at bank-entry (Phase 14 built helpers; Phase 15+ applies them to real LLM outputs)
- reviewer_critic_stub advisory-only in Phase 14; real LLM critic activates Phase 15+

### Blockers/Concerns
None.

## Deferred Items (v2.4+ candidates)

(unchanged from prior state — see PROJECT.md for full table)

## Session Continuity

Last session: 2026-06-01T08:36:26.000Z
Stopped at: Phase 14 complete. All 32 tests pass. 
Resume file: .planning/milestones/v2.3-ROADMAP.md
Next action: `/gsd-plan-phase 15` (Probe Tier — real LLM training run, $5–10 gpt-5.4)
