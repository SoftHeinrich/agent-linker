---
gsd_state_version: 1.0
milestone: v2.6
milestone_name: Phase Summary
status: Phase 34 KILL verdict. Phases 35+36 SKIPPED. Phase 37 is next (unconditional).
stopped_at: context exhaustion at 75% (2026-06-02)
last_updated: "2026-06-02T12:13:13.353Z"
last_activity: 2026-06-02 — Phase 34 probe KILL ([TEST] macro 84.86% < 87%). v4 axiom Gap 2 over-aggressive (TM 81.97%). Assessor confirmed active (9 decisions).
progress:
  total_phases: 14
  completed_phases: 2
  total_plans: 10
  completed_plans: 5
  percent: 14
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-02 for v2.5 close)

**Core value:** Every rule removed and every prompt-rule trimmed must hold macro F1 ≥ 0.93 on Claude Sonnet AND gpt-5.4 macro within tolerance of the v2.0 baseline (0.9077) — or be rejected. Generality first (GATE-06).
**Current focus:** v2.6 — ILinker4 + LLM-Driven Training + Axiom Re-run

## Current Position

Phase: Phase 37 (Milestone Close) — not started
Plan: —
Status: Phase 34 KILL verdict. Phases 35+36 SKIPPED. Phase 37 is next (unconditional).
Last activity: 2026-06-02 — Phase 34 probe KILL ([TEST] macro 84.86% < 87%). v4 axiom Gap 2 over-aggressive (TM 81.97%). Assessor confirmed active (9 decisions).
Next action: Begin Phase 37 — milestone audit, GATE-01/07/08 verification, v2.6-MILESTONE-AUDIT.md, PROJECT.md update, STATE.md close.

```
Progress: [========                      ] 29% (2/7 phases)
```

## v2.5 Outcome

**Verdict: WEAK** — cross-split macro F1 = 89.1% (gpt-5.4, 5-dataset, 12-pattern bank).
**Oracle cache fix validated**: split-2 committed 12 patterns in Pass 1 (vs v2.4: 0/5).
**15-slot expansion**: 5/6 new slots populated with committed patterns.
**GATE-01**: s_linker13_min unchanged (gpt-5.4 0.9069, Claude 0.9506). ✅

See `.planning/milestones/v2.5-MILESTONE-AUDIT.md` for full audit.

## v2.6 Roadmap

**Phases 31–37.** See `.planning/milestones/v2.6-ROADMAP.md` for full detail.

| Phase | Goal | Key REQs |
|-------|------|----------|
| 31 | ILinker4 (Voyager-native) + prompt hygiene audit | REQ-V26-01, REQ-V26-02 |
| 32 | v5 training loop: OD merge + LLM Assessor + cross-split redesign + log structure | REQ-V26-03 through REQ-V26-06 |
| 33 | Axiom gap fixes: SCN (14 FNs), gerund FPs (7), coref alias (line 1004) | REQ-V26-08, REQ-V26-09, REQ-V26-10 |
| 34 | Probe tier (2-pass mainline, [TRAIN]/[TEST] separate, budget ≤$10) | REQ-V26-11 |
| 35 | Range tier (conditional on Probe CONTINUE, budget ≤$25) | REQ-V26-12 |
| 36 | Confirmation tier (conditional on Range ≥0.87, 3-split, budget ≤$60) | REQ-V26-13 |
| 37 | Milestone close (unconditional, GATE-01/07/08 audit) | GATE-01, GATE-07, GATE-08 |

**REQ-V26-07 (GATE-01 regression) applies throughout all phases.**
**Total training budget cap: ≤$80 (Phases 34–36).**

## Standing Gates (post-v2.5, into v2.6)

- **GATE-01**: PASS. s_linker13_min unchanged. gpt-5.4 0.9069, Claude 0.9506. ✅
- **GATE-06**: PASS. All bank patterns and axiom diffs clean (0 benchmark vocabulary). ✅
- **GATE-07**: PASS. DEFAULT_BANK_PATH = v2.5 cross_split_final_bank.json. Docstring updated. ✅
- **GATE-08**: PASS. Total ~$62 under $80 cap (v2.5). ✅

## Canonical Artifact (current)

- **`src/llm_sad_sam/linkers/experimental/s_linker13_min.py`** (v2.1 PROMOTED, `canonical=True`, unchanged)
- Claude Sonnet macro F1: 0.9506 | gpt-5.4 macro F1: 0.9069

## Experimental Artifact (current)

- **`src/llm_sad_sam/linkers/experimental/s_linker14_voyager.py`** (`experimental=True`, `canonical=False`)
- v2.5 WEAK: gpt-5.4 cross-split macro F1 0.8911 | Bank: 12 patterns (8 slots populated of 15)
- DEFAULT_BANK_PATH: `results/voyager_v4b_v25/confirmation/cross_split_final_bank.json`

## v2.6 Known Debts (from v2.5)

| Priority | ID | Description |
|----------|----|-------------|
| HIGH | C-1 | BBB split strategy — SEED_EXTRACTION_RULES + SEED_ACTOR_RULES empty (ILinker4 needed) |
| HIGH | C-2 | TM FP reduction — FM-1/FM-2 patterns documented; LLM Assessor should commit them |
| HIGH | C-3 | LLM Assessor gate — replace F1-delta Gate A+B with error-set reasoning |
| HIGH | C-4 | Axiom gaps: SCN (14 FNs), gerund FPs (7), coref alias (line 1004 code path) |
| MEDIUM | C-5 | Cross-split redesign — independent per-split train/eval vs axiom-only baseline |
| LOW | C-6 | Flex tier cost optimization — VIABLE. Enable `OPENAI_SERVICE_TIER=flex` in .env. Trial on Phase 34. See quick task 260602-d1w. |

## Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260602-d1w | investigate latency implications of switching gpt backend to flex tier | 2026-06-02 | — | [260602-d1w-investigate-latency-implications-of-swit](./quick/260602-d1w-investigate-latency-implications-of-swit/) |

## Session Continuity

Last session: 2026-06-02T12:13:13.350Z
Stopped at: context exhaustion at 75% (2026-06-02)
Resume file: None
Next action: Monitor `logs/voyager_v5/probe_p36_rollback.log`; if TM train ≥82% proceed to Tier C coref-extension design (task 9)
Last activity: 2026-06-02 — launched Tier-A rollback probe (PID 1505398, detached nohup)
