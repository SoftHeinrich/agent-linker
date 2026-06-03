---
gsd_state_version: 1.0
milestone: v2.6.2
milestone_name: Multi-Framing Extraction Design
status: v2.6.2 SHIPPED 2026-06-03. v2.7 FROZEN. v2.6 close (Phase 37) still deferred.
stopped_at: milestone complete — ICSE decision final; awaiting git tag
last_updated: "2026-06-03T09:32:00.000Z"
last_activity: 2026-06-03 — 17c ablation complete (GPT-5.4 87.1%); ICSE decision finalized; docs updated
progress:
  total_phases: 4
  completed_phases: 4
  total_plans: 4
  completed_plans: 4
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-02 for v2.5 close)

**Core value:** Every rule removed and every prompt-rule trimmed must hold macro F1 ≥ 0.93 on Claude Sonnet AND gpt-5.4 macro within tolerance of the v2.0 baseline (0.9077) — or be rejected. Generality first (GATE-06).
**Current focus:** v2.6.2 — Multi-Framing Extraction Design (s_linker17a/17b ICSE architecture exploration). v2.7 FROZEN.

## v2.6.1 — SHIPPED 2026-06-03 (tag v2.6.1)

s_linker15 no-training axiom linker shipped. macro 89.1% gpt-5.4 / 92.7% Claude. Training proven
non-additive (s15 = trained s14 on gpt). FP fixes fire on Claude (TM FP 17→6), inert on gpt.
See `.planning/milestones/v2.6.1-MILESTONE-AUDIT.md`. Results below retained for reference.

## v2.6.2 — SHIPPED 2026-06-03 (final with 17c, 2026-06-03)

Multi-framing extraction design exploration complete. All three unified variants tested on GPT-5.4.
17a (rename-only) ≈ s15 within variance. 17b (k=2) regresses −4.1pp. 17c (union) regresses −2.0pp
but +12 FPs. Union > k=2 by +2.1pp but s15 integrated approach still wins both.
**ICSE decision (FINAL): use s_linker17a naming (Framing A/B/C) for the paper architecture.**
See `.planning/milestones/v2.6.2-MILESTONE-AUDIT.md`.

### v2.6.2 results (reference) — GPT-5.4

| Dataset | s_linker15 | s_linker17a | s_linker17b (k=2) | s_linker17c (union) |
|---------|-----------|------------|------------|---------------------|
| MediaStore | 91.8% | 90.3% | 92.1% | 92.1% |
| TeaStore | 96.4% | 94.7% | 94.5% | 96.4% |
| TeaMmates | 82.5% | 82.3% | 78.3% | 80.0% |
| BigBlueButton | 77.6% | 74.1% | 70.2% | 77.0% |
| JabRef | 97.3% | 90.0% | 90.0% | 90.0% |
| **Macro** | **89.1%** | **86.3%** | **85.0%** | **87.1%** |

Logs: `logs/v2.6.1_s17ab_claude.log` (17a/17b, GPT-5.4, misleading name);
`logs/v2.6.2_s17c_gpt.log` (17c, GPT-5.4).

## Frozen / pending

- **v2.7 (Phases 38–42)** — FROZEN. Resume after v2.6.2. ⚠ Phases 40–41 predicated on keeping training — re-evaluate when v2.7 resumes.
- **v2.6 close (Phase 37)** — GATE-06 'Persistence' taboo fix + v2.6 audit, deferred.

## Current Position

Milestone: **v2.6.2 (shipped)**. v2.7 frozen.
Next action: git tag v2.6.2 → optionally run 17a/17b on Claude backend → decide whether to tune 17b k-threshold or proceed to v2.7.

### v2.6.1 results (reference) — s_linker15 axiom-only + 3 FP fixes

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

Last session: 2026-06-03T07:06:50.621Z
Stopped at: context exhaustion at 82% (2026-06-03)
Resume file: None
Next action: Monitor `logs/voyager_v5/probe_p36_rollback.log`; if TM train ≥82% proceed to Tier C coref-extension design (task 9)
Last activity: 2026-06-02 — launched Tier-A rollback probe (PID 1505398, detached nohup)
