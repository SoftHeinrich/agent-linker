---
gsd_state_version: 1.0
milestone: v2.7
milestone_name: BBB Recall Closure (Tier C + Partial-Injection Port + Recall-Oracle)
status: v2.7 roadmap defined; v2.6 close pending at Phase 37 (folds in GATE-06 'Persistence' taboo fix); Phases 38–42 plan stubs written.
stopped_at: 2026-06-02 plan landed
last_updated: "2026-06-02T15:00:00Z"
last_activity: 2026-06-02 — v2.7 roadmap synthesized from HANDOFF + pending todos + voyager-improvement notes. F1 gate dropped. v2.8 partial-injection + recall-oracle pulled into v2.7. Phases 38–42 plan stubs written.
progress:
  total_phases: 5
  completed_phases: 0
  total_plans: 17
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-02 for v2.5 close)

**Core value:** Every rule removed and every prompt-rule trimmed must hold macro F1 ≥ 0.93 on Claude Sonnet AND gpt-5.4 macro within tolerance of the v2.0 baseline (0.9077) — or be rejected. Generality first (GATE-06).
**Current focus:** v2.7 — BBB Recall Closure (Tier C axiom + s10 partial-injection port + recall-oracle training redesign). No F1 milestone gate.

## Current Position

Phase: Phase 37 (v2.6 Milestone Close) — not started. After close, Phase 38 + Phase 39 start in parallel (v2.7).
Plan: —
Status: v2.6 close pending. v2.7 phase plans (38–42) drafted. F1 gate intentionally dropped for v2.7.
Last activity: 2026-06-02 — v2.7 roadmap landed. Phases 38–42 plan stubs in `.planning/phases/`.
Next action: Phase 37 task 9 — GATE-06 'Persistence' taboo regex fix in `scripts/voyager_train_tlr_v5.py:104-113`. Then v2.6-MILESTONE-AUDIT.md. Then milestone flip to v2.7 and start Phase 38 + 39 parallel.

```
Progress: v2.6 [############################  ] 6/7 — close pending
          v2.7 [                              ] 0/5 — not started
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

Last session: 2026-06-02T13:58:22.620Z
Stopped at: context exhaustion at 76% (2026-06-02)
Resume file: None
Next action: Monitor `logs/voyager_v5/probe_p36_rollback.log`; if TM train ≥82% proceed to Tier C coref-extension design (task 9)
Last activity: 2026-06-02 — launched Tier-A rollback probe (PID 1505398, detached nohup)
