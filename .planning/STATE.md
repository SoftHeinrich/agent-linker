---
gsd_state_version: 1.0
milestone: v2.5
milestone_name: Oracle Cache Fix + 15-Slot Expansion + Re-run
status: complete
last_updated: "2026-06-02T06:00:00.000Z"
last_activity: 2026-06-02
progress:
  total_phases: 6
  completed_phases: 6
  total_plans: 6
  completed_plans: 6
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-02 for v2.5 close)

**Core value:** Every rule removed and every prompt-rule trimmed must hold macro F1 ≥ 0.93 on Claude Sonnet AND gpt-5.4 macro within tolerance of the v2.0 baseline (0.9077) — or be rejected. Generality first (GATE-06).
**Current focus:** v2.5 COMPLETE — v2.6 kickoff is next.

## Current Position

Phase: 30 — Milestone Close
Plan: 30-P1
Status: complete
Last activity: 2026-06-02 — Phase 30 complete (v2.5 closed, WEAK verdict, 89.1% cross-split macro F1)
Next action: v2.6 kickoff — BBB split strategy revision + TM FP reduction (see v2.5-MILESTONE-AUDIT.md Debts C-1/C-2)

```
Progress: [##############################] 100% (6/6 phases)
```

## v2.5 Outcome

**Verdict: WEAK** — cross-split macro F1 = 89.1% (gpt-5.4, 5-dataset, 12-pattern bank).
**Oracle cache fix validated**: split-2 committed 12 patterns in Pass 1 (vs v2.4: 0/5).
**15-slot expansion**: 5/6 new slots populated with committed patterns.
**GATE-01**: s_linker13_min unchanged (gpt-5.4 0.9069, Claude 0.9506). ✅

See `.planning/milestones/v2.5-MILESTONE-AUDIT.md` for full audit.

## v2.5 Design Decisions (Locked)

All locked in v2.5 requirements and milestone audit (2026-06-01 to 2026-06-02).

### A-1 Fix: Oracle Cache Contamination — RESOLVED

Oracle cache key in `voyager_train_tlr_v4_beta.py` lines 461–463 now includes `bch = _bank_content_hash(bank)`.
Split-2 committed 12 patterns in Pass 1 vs 0/5 in v2.4. Fix validated.

### A-3 Fix: Probation Variance — RESOLVED

`delta >= 0.005` threshold correctly blocks O+D on passes 2–5 (LLM run-to-run variance).
Side effect: ≥3/5 passes criterion (REQ-V25-11 SC-2) structurally impossible — expected correct behavior.

### A-4 Fix: D Slot Steering — RESOLVED

Underfilled-slot list in D prompt steered Distillator toward 5/6 new slots in Phase 27 Pass 1.

### B-1: 15-Slot Expansion — RESOLVED

6 new slot constants defined; ILinker3Injected subclass active; 4 inline prompts wired.
5/6 new slots have committed patterns (SEED_EXTRACTION_RULES + SEED_ACTOR_RULES not populated).

## Standing Gates (post-v2.5)

- **GATE-01**: PASS. s_linker13_min unchanged. gpt-5.4 0.9069, Claude 0.9506. ✅
- **GATE-06**: PASS. All bank patterns and axiom diffs clean (0 benchmark vocabulary). ✅
- **GATE-07**: PASS. DEFAULT_BANK_PATH = v2.5 cross_split_final_bank.json. Docstring updated. ✅
- **GATE-08**: PASS. Total ~$62 under $80 cap. ✅

## Canonical Artifact (current)

- **`src/llm_sad_sam/linkers/experimental/s_linker13_min.py`** (v2.1 PROMOTED, `canonical=True`, unchanged)
- Claude Sonnet macro F1: 0.9506 | gpt-5.4 macro F1: 0.9069

## Experimental Artifact (current)

- **`src/llm_sad_sam/linkers/experimental/s_linker14_voyager.py`** (`experimental=True`, `canonical=False`)
- v2.5 WEAK: gpt-5.4 cross-split macro F1 0.8911 | Bank: 12 patterns (8 slots)
- DEFAULT_BANK_PATH: `results/voyager_v4b_v25/confirmation/cross_split_final_bank.json`

## v2.6 Kickoff Candidates

See `v2.5-MILESTONE-AUDIT.md` Debts section for full list. Priority order:

| Priority | ID | Description |
|----------|----|-------------|
| HIGH | C-1 | BBB split strategy — add BBB-in-train partition to all splits (or dedicated BBB split) |
| HIGH | C-2 | TM FP reduction — Oracle FM-1/FM-2 patterns now documented; need pass to commit them |
| MEDIUM | C-3 | SEED_EXTRACTION_RULES + SEED_ACTOR_RULES via BBB-in-train rerun |
| MEDIUM | C-4 | Section-context axiom redesign (`2026-06-01-design-better-axioms-section-context-responsibility.md`) |
| LOW | C-5 | Flex tier cost optimization (`260601-flex-tier-integration.md`) |

## Pending Todos (post-v2.5)

- `260601-flex-tier-integration.md` — deferred to v2.6+
- `2026-06-01-design-better-axioms-section-context-responsibility.md` — v2.6 candidate
- `2026-06-01-implement-refined-v3-axiom-diffs-feasibility-study.md` — v2.6 candidate
- `2026-06-02-redesign-voyager-training-gate-and-cross-split-logic.md` — v2.6 candidate (empirical gate + cross-split fix)

## Session Continuity

Last session: 2026-06-02T06:00:00.000Z
Stopped at: v2.5 milestone close complete
Resume file: None
Next action: v2.6 kickoff (`/gsd-new-milestone`)
