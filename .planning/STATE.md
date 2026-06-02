---
gsd_state_version: 1.0
milestone: v2.6
milestone_name: ILinker4 + LLM-Driven Training + Axiom Re-run
status: in_progress
last_updated: "2026-06-02T07:00:00.000Z"
last_activity: 2026-06-02
progress:
  total_phases: 7
  completed_phases: 1
  total_plans: 1
  completed_plans: 1
  percent: 14
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-02 for v2.5 close)

**Core value:** Every rule removed and every prompt-rule trimmed must hold macro F1 ≥ 0.93 on Claude Sonnet AND gpt-5.4 macro within tolerance of the v2.0 baseline (0.9077) — or be rejected. Generality first (GATE-06).
**Current focus:** v2.6 — ILinker4 + LLM-Driven Training + Axiom Re-run

## Current Position

Phase: Phase 32 (LLM-Driven Training Loop v5) — not started
Plan: —
Status: Phase 31 complete — ILinker4 wired, audit clean
Last activity: 2026-06-02 — Phase 31 shipped (ilinker4.py + s_linker14_voyager.py update + AUDIT-REPORT.md)
Next action: Begin Phase 32 — build voyager_train_tlr_v5.py: OD merge, LLM Assessor, cross-split redesign, [TRAIN]/[TEST] log separation.

```
Progress: [====                          ] 14% (1/7 phases)
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
| LOW | C-6 | Flex tier cost optimization (deferred to v2.7+) |

## Session Continuity

Last session: 2026-06-02T06:00:00.000Z
Stopped at: v2.6 roadmap created (milestones/v2.6-ROADMAP.md written)
Resume file: None
Next action: Begin Phase 31 (`/gsd-plan-phase 31`)
