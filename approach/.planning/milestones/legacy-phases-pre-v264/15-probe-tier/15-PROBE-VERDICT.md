---
phase: 15-probe-tier
tier: probe
backend: openai
model: gpt-5.4
split: mainline
train_projects: [mediastore, teastore, teammates]
date: 2026-06-01
verdict: CONTINUE
cheap_kill_threshold: 0.87
final_train_macro_f1: 0.9152
passes_run: 1
requirements_closed: [REQ-V23-07, REQ-V23-13, REQ-V23-14]
next_action: Phase 16 Range Tier
---

# Phase 15: Probe Tier Verdict

## Summary

CONTINUE: training-project macro F1 0.9152 after pass 1 >= 0.87 cheap-kill threshold; harness converged at pass 1 (committed macro 0.9152 > CONVERGENCE_THRESHOLD 0.90); Phase 16 Range Tier proceeds.

## Per-Pass Results

| Pass | MS F1  | TS F1  | TM F1  | Train Macro (after L) | Committed Macro | Committed | Notes |
|------|--------|--------|--------|-----------------------|-----------------|-----------|-------|
| 1    | 0.9508 | 0.9474 | 0.8264 | 0.9082                | 0.9152          | true      | Converged (0.9152 > 0.90); pass 2 skipped by harness. probation_delta=+0.0070. |

(Pass 2 not executed — harness detected convergence after pass 1; probe_summary.json passes_run=1 per log evidence.)

## Verdict Evidence

- **Cheap-kill threshold**: 0.87 (locked in harness, REQ-V23-05 mapped via REQ-V23-07).
- **Final training-project macro F1**: 0.9152
- **Comparison vs threshold**: 0.9152 >= 0.87 — CONTINUE
- **Pass-1 → pass-2 delta**: N/A (passes_run == 1; convergence triggered at pass 1)
- **Rollbacks observed**: None. Pass 1 committed=true (committed_macro_f1=0.9152, probation_delta=+0.0070 >= 0).

## Bank Saturation (per-project)

| Project | Patterns across 9 slots | Source file |
|---------|-------------------------|-------------|
| mediastore | 6 | results/voyager_v4_beta/mainline/mediastore_bank.json |
| teastore   | 3 | results/voyager_v4_beta/mainline/teastore_bank.json |
| teammates  | 3 | results/voyager_v4_beta/mainline/teammates_bank.json |

Note: mediastore shows 6 patterns because a subsequent dry-run appended 3 dry-run placeholder patterns (p_004, p_005, p_006 in AMBIGUITY_RULES slot) after the original run. The 3 real LLM-generated patterns (p_001 DOC_KNOWLEDGE_EXTRACTION_RULES, p_002 DOC_KNOWLEDGE_JUDGE_RULES, p_003 VALIDATION_RULES) are the probe outputs; the 3 mock placeholders are dry-run artifacts. Teastore and teammates bank files contain only the 3 real patterns (cross-project via Distillator); their AMBIGUITY_RULES slots are empty.

## Cost

- **Total prompt tokens**: N/A (harness does not call LLMClient.get_session_usage() in run_probe)
- **Total completion tokens**: N/A
- **Total tokens**: N/A
- **Dollar estimate**: ~$5-7 based on assumed ~$0.50-0.70/call on gpt-5.4 rate × 10 LLM calls (1 pass: 3 L + 3 O + 1 D + 3 P = 10 calls). Per probe.log: "Token counts not available; estimate from probe_summary.json pass count: 1 pass, 3 L runs + 3 O runs + 1 D run + 3 P runs = 10 LLM calls."
- **Budget cap (REQ-V23-14)**: $10 — status: under (estimated ~$5-7)
- **Cache hits** (if logged): N/A (no cache_hit lines in probe.log)

## GATE-06 Status

- Taboo-grep rejects logged: 0 blockers (4 advisory warnings observed — lines 235, 239, 243, 247 of probe.log)
- Advisory critic rejects logged: 0 (advisory mode, non-blocking per RESEARCH.md GATE-06 section)
- GATE-06 verdict: PASS (probe.log line 251: "[GATE-06] accepted=3 rejected=0"; zero taboo blockers triggered)

## Next Action

CONTINUE verdict — Phase 16 Range Tier proceeds.

- **Next command**: `/gsd-plan-phase 16` (Range Tier — train to convergence, 5-dataset eval, $15-25 budget).
- **What Phase 16 does**: Run β training to convergence (macro >= 0.90 on train projects, or max 5 outer passes). Evaluate aggregated bank on all 5 datasets. Compute 3-tier verdict (STRONG / WEAK / FAIL) against locked bar (STRONG >= 0.9173, WEAK [0.87, 0.9173), FAIL < 0.87).
- **Warm start**: Per-project bank files from Phase 15 probe are available at results/voyager_v4_beta/mainline/{mediastore,teastore,teammates}_bank.json if Phase 16 supports warm-start.

## Anomalies / Notes

1. **probe_summary.json overwritten by dry-run**: After the real probe run completed (verdict=CONTINUE, final_macro=0.9152, runs_ran=1, 3 projects), a subsequent dry-run execution overwrote probe_summary.json with dry-run values (verdict=KILL, final_train_macro_f1=0.5, only mediastore, dry_run=true). The authoritative record is probe.log line 483: "[PROBE] verdict=CONTINUE final_macro=0.9152". This verdict document uses probe.log + 15-01-SUMMARY.md as the source of truth; probe_summary.json is a corrupted artifact. **All numbers in this document are sourced from probe.log (the actual run output), not from the overwritten probe_summary.json.**
2. **Taboo token warnings (advisory)**: Oracle prompts for all 3 projects triggered GATE-06 advisory taboo warnings for project names (mediastore, teastore, teammates, datastore) at probe.log lines 235, 239, 243, 247. These are advisory, non-blocking — harness logged and continued. All 3 D proposals passed GATE-06 filter (accepted=3, rejected=0).
3. **Early convergence at pass 1**: Harness hit CONVERGENCE_THRESHOLD=0.90 after pass 1 (committed macro 0.9152 > 0.90). Pass 2 was skipped automatically. CHEAP_KILL_THRESHOLD=0.87 was not stress-tested.
4. **TM (teammates) F1 lower**: TM base F1=0.8264 vs MS=0.9508 and TS=0.9474. Teammates is the hardest dataset (8 components, 198 sentences, generic aliases). Macro still exceeded convergence threshold.
5. **Token tracking unavailable**: LLMClient.get_session_usage() is not called by run_probe(). Token counts not in probe.log. Cost estimated from pass count.

## Artifacts

- `logs/voyager_v4_beta/probe.log` (primary verdict source — see line 483)
- `results/voyager_v4_beta/mainline/probe_summary.json` (CORRUPTED — overwritten by dry-run; see Anomalies)
- `results/voyager_v4_beta/mainline/pass1_summary.json` (dry-run values — see Anomalies)
- `results/voyager_v4_beta/mainline/mediastore_bank.json` (6 patterns: 3 real + 3 dry-run placeholders)
- `results/voyager_v4_beta/mainline/teastore_bank.json` (3 patterns: real LLM-generated)
- `results/voyager_v4_beta/mainline/teammates_bank.json` (3 patterns: real LLM-generated)
- `.planning/phases/15-probe-tier/15-01-SUMMARY.md` (Plan 1 execution record confirming CONTINUE verdict)

## Requirements Closed

| REQ | Evidence |
|-----|----------|
| REQ-V23-07 | Mainline split MS+TS+TM probe completed (1 pass, converged); verdict=CONTINUE published in this document |
| REQ-V23-13 | Per-pass macro F1 logged (pass 1: committed_macro=0.9152); probe tier capped at 1 pass (convergence; <= 5 max) |
| REQ-V23-14 | gpt-5.4 token counts not directly available from harness; estimated ~$5-7 (vs $10 cap); under budget |
