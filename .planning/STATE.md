---
gsd_state_version: 1.0
milestone: v2.6.1
milestone_name: Axiom FP Root-Cause Fixes (PATCH)
status: ACTIVE — v2.6.1 patch milestone initialized 2026-06-02. v2.7 (Phases 38–42) and pending v2.6 close (Phase 37) FROZEN. Sole active focus = three isolated axiom FP fixes.
stopped_at: 2026-06-02 v2.6.1 workspace initialized
last_updated: "2026-06-02T16:30:00Z"
last_activity: 2026-06-02 — v2.6.1 patch milestone created from FP-fix todo. v2.7 frozen. Workspace set up — v2.6.1-ROADMAP + phase plan written.
progress:
  total_phases: 1
  completed_phases: 0
  total_plans: 1
  completed_plans: 0
  percent: 0
frozen:
  - "v2.7 (Phases 38–42) — BBB Recall Closure. Stubs intact at .planning/phases/38–42 + milestones/v2.7-ROADMAP.md."
  - "v2.6 close (Phase 37) — GATE-06 'Persistence' taboo fix + v2.6 audit. Deferred until v2.6.1 closes."
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-02 for v2.5 close)

**Core value:** Every rule removed and every prompt-rule trimmed must hold macro F1 ≥ 0.93 on Claude Sonnet AND gpt-5.4 macro within tolerance of the v2.0 baseline (0.9077) — or be rejected. Generality first (GATE-06).
**Current focus:** v2.6.1 — Axiom FP Root-Cause Fixes (PATCH on shipped v2.6 axiom B-variant `61e038`). Three isolated slot patches in `prompts_v4_axiom.py`. Success: TM macro > 82.26%, no MS/TS/JAB regression.

## FROZEN (do not touch until v2.6.1 closes)

- **v2.7 (Phases 38–42)** — BBB Recall Closure. Roadmap + phase stubs intact. See `milestones/v2.7-ROADMAP.md` (freeze banner).
- **v2.6 close (Phase 37)** — GATE-06 'Persistence' taboo fix + v2.6 audit. Deferred behind v2.6.1.

## Current Position

Milestone: **v2.6.1 (ACTIVE)** — PATCH. See `.planning/milestones/v2.6.1-ROADMAP.md`.
Phase: Phase v2.6.1-01 (Axiom FP Root-Cause Fixes) — not started.
Plan: `.planning/phases/v2.6.1-axiom-fp-fixes/PLAN.md`
Source: `.planning/todos/pending/2026-06-02-improve-prompts-v4-axiom-three-root-cause-fp-fixes.md`
Next action: review s_linker15 validation run (below). Decide whether to repeat for GPT-variance averaging / run isolated FP-fix attribution, then close v2.6.1.

### s_linker15 validation (2026-06-02, gpt-5.4, axiom-only + 3 FP fixes)

| Dataset | F1 | P | R | FP | FN |
|---------|----|----|----|----|----|
| MediaStore | 91.8% | 93.3 | 90.3 | 2 | 3 |
| TeaStore | 96.4% | 93.1 | 100.0 | 2 | 0 |
| TeaMMates | 82.5% | 75.4 | 91.2 | 17 | 5 |
| BigBlueButton | 77.6% | 83.3 | 72.6 | 9 | 17 |
| JabRef | 97.3% | 94.7 | 100.0 | 1 | 0 |
| **Macro** | **89.1%** | | | **31** | |

Log: `logs/v2.6.1_s_linker15.log`; CSV/JSON: `results/v2.6.1/`.

### s_linker15 dual-backend (2026-06-02/03)

| Dataset | GPT-5.4 | Claude Sonnet |
|---------|---------|---------------|
| MediaStore | 91.8 | 95.1 |
| TeaStore | 96.4 | 96.4 |
| TeaMMates | 82.5 | 91.4 |
| BigBlueButton | 77.6 | 83.5 |
| JabRef | 97.3 | 97.3 |
| **Macro** | **89.1** | **92.7** |

Claude run: `logs/v2.6.1_s15_claude_tm_bbb_jab.log` (TM/BBB/JAB) + cached MS/TS from
the killed 4-variant compare run; JSON `results/v2.6.1_claude/`.

**Cross-backend + historical findings:**
- No-training thesis holds: GPT 89.1 ties trained s14 (89.11); Claude 92.7.
- Claude beats GPT by +3.6pp macro, concentrated in TM (+8.9) and BBB (+5.9); FP 12 vs 31.
- BBB Claude 83.5 ≈ canonical s13_min (~85) and > s13 v1.0 (82.1). Dropping training costs ~0 on BBB.
- Pre-existing gaps (NOT from v2.6.1): (a) trim tax — early rich s_linkers (s2–s12) hit BBB
  0.90–0.96 on Claude vs s15 83.5; (b) canonical gap — s15 Claude 92.7 vs s13_min 95.06 (−2.3pp).
- BBB FN-dominated (14 FN) both backends → recall ceiling = frozen-v2.7 Tier-C/partial-injection target.
- Cached BBB history (all Claude unless noted) in `results/ablation_results/`:
  s2 .92, s3 .90, s5 .91, s6 .92, s8 .91, s9 .92, s10 .94, s11 .88–.96, s12b .93;
  GPT-5.4 BBB only exists s13+: s13 .80, s13_min ~.76, s14 trained .74, s15 .78.

**Headline finding:** axiom-only (NO training) = **89.1% macro** ≈ v2.5 trained bank (89.1%)
and ≈ B-variant floor (88.99%). Dropping training loses nothing — supports the v2.6.1 thesis.
TM 82.5% ≥ 82.26% target; MS/TS/BBB/JAB all ≥ their bcae0e baselines (no regression).
**Caveat:** single gpt-5.4 run; run-to-run variance ±5–12 links. TM FP still 17 (= baseline) —
the 3 FP fixes' specific TM-FP reduction is NOT established by one combined run; needs a repeat
and/or per-error attribution to confirm.

```
Progress: v2.6.1 [#######################       ] code done; validation landed
          v2.6 close [############################  ] FROZEN — Phase 37 pending
          v2.7 [                              ] FROZEN — 0/5
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
