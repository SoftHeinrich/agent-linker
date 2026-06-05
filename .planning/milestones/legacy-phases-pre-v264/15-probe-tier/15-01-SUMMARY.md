---
phase: 15-probe-tier
plan: 1
subsystem: training-run
tags: [voyager, gpt-5.4, openai, probe-tier, tlr, training]

# Dependency graph
requires:
  - phase: 14
    provides: "voyager_train_tlr_v4_beta.py harness + SLinker14Voyager linker consumer (dry-run verified, 32 tests passing)"
provides:
  - "probe_summary.json verdict=CONTINUE with final_train_macro_f1=0.9152"
  - "pass1_summary.json with per-project F1s (MS=0.9508, TS=0.9474, TM=0.8264) and committed=true"
  - "Per-project trained banks: mediastore_bank.json, teastore_bank.json, teammates_bank.json (3 patterns each, 3/9 slots)"
  - "probe.log full stdout capture of L+O+D+GATE-06+P loop"
affects:
  - 15-P2 (reads probe_summary.json and probe.log for PROBE-VERDICT.md)
  - 16-range-tier (proceeds only because verdict=CONTINUE)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "results/ gitignored files committed with git add -f (consistent with prior results/ablation_results/ pattern)"
    - "Voyager probe tier: converged in 1 pass (macro 0.9152 > CONVERGENCE_THRESHOLD 0.90)"

key-files:
  created:
    - logs/voyager_v4_beta/probe.log
    - results/voyager_v4_beta/mainline/probe_summary.json
    - results/voyager_v4_beta/mainline/pass1_summary.json
    - results/voyager_v4_beta/mainline/mediastore_bank.json
    - results/voyager_v4_beta/mainline/teastore_bank.json
    - results/voyager_v4_beta/mainline/teammates_bank.json
    - results/voyager_v4_beta/mainline/pass1_distillator.json
    - results/voyager_v4_beta/mainline/pass1_mediastore_oracle.json
    - results/voyager_v4_beta/mainline/pass1_teastore_oracle.json
    - results/voyager_v4_beta/mainline/pass1_teammates_oracle.json
    - logs/voyager_v4_beta/.gitkeep
  modified: []

key-decisions:
  - "reviewer_critic_stub kept advisory (advisory=True) for probe tier — no upgrade to blocking LLM call; taboo-grep gate sufficient for REQ-V23-09 in probe mode"
  - "Single invocation ran both passes as a loop; harness converged at pass 1 (macro 0.9152 > 0.90 threshold), skipping pass 2 automatically"
  - "results/ gitignored files committed with git add -f to persist probe evidence, consistent with prior ablation_results/ tracking"
  - "No token count in dollars available (LLMClient.get_session_usage() not called by run_probe); navigation hint appended to probe.log"

patterns-established:
  - "Probe tier verdict: CONTINUE path activates Phase 16 Range Tier"
  - "Bank slot distribution: DOC_KNOWLEDGE_EXTRACTION_RULES (1), DOC_KNOWLEDGE_JUDGE_RULES (1), VALIDATION_RULES (1) — other 6 slots empty after pass 1"

requirements-completed: [REQ-V23-07, REQ-V23-13, REQ-V23-14]

# Metrics
duration: 7min
completed: 2026-06-01
---

# Phase 15 Plan 1: Probe Tier Run Summary

**gpt-5.4 β probe on MS+TS+TM mainline split converged at pass 1 with macro F1=0.9152, verdict=CONTINUE (threshold 0.87 exceeded), 3 bank patterns committed across 3 projects**

## Performance

- **Duration:** 7 min (wall clock; ~5 min of actual LLM inference)
- **Started:** 2026-06-01T09:05:01Z
- **Completed:** 2026-06-01T09:12:21Z
- **Tasks:** 2/2
- **Files modified:** 11 created (logs + results)

## Accomplishments
- Ran voyager_train_tlr_v4_beta.py probe on 3 mainline train projects (MS, TS, TM) with gpt-5.4
- Pass 1 macro F1 = 0.9152 (above CONVERGENCE_THRESHOLD=0.90), harness converged without pass 2
- 3 patterns proposed, 3 accepted by GATE-06, probation committed (delta +0.0070 >= 0)
- verdict=CONTINUE in probe_summary.json — Phase 16 Range Tier proceeds

## Pass Results

| Pass | MS F1  | TS F1  | TM F1  | Train Macro (L) | After Probation | Committed | Converged |
|------|--------|--------|--------|-----------------|-----------------|-----------|-----------|
| 1    | 0.9508 | 0.9474 | 0.8264 | 0.9082          | 0.9152          | Yes       | Yes       |

**Verdict:** CONTINUE (final_train_macro_f1=0.9152 >= cheap_kill_threshold=0.87)

## Probe Command

```bash
python scripts/voyager_train_tlr_v4_beta.py probe \
    --projects mediastore,teastore,teammates \
    --backend openai \
    --model gpt-5.4 \
    2>&1 | tee logs/voyager_v4_beta/probe.log
```

## Task Commits

Each task was committed atomically:

1. **Task 1: Pre-flight checks and log directory creation** - `9b502ca` (chore)
2. **Task 2: Run β probe harness on mainline train split (gpt-5.4)** - `5900232` (feat)

**Plan metadata:** See SUMMARY commit below.

## Files Created

- `logs/voyager_v4_beta/probe.log` — full stdout capture of L+O+D+GATE-06+P loop including taboo warnings and verdict line
- `results/voyager_v4_beta/mainline/probe_summary.json` — verdict=CONTINUE, tier=probe, split=mainline, final_train_macro_f1=0.9152
- `results/voyager_v4_beta/mainline/pass1_summary.json` — per-project F1s, delta, committed=true, converged=true
- `results/voyager_v4_beta/mainline/mediastore_bank.json` — 3 patterns in slots: DOC_KNOWLEDGE_EXTRACTION_RULES, DOC_KNOWLEDGE_JUDGE_RULES, VALIDATION_RULES
- `results/voyager_v4_beta/mainline/teastore_bank.json` — same 3 patterns (cross-project via Distillator)
- `results/voyager_v4_beta/mainline/teammates_bank.json` — same 3 patterns
- `results/voyager_v4_beta/mainline/pass1_distillator.json` — 3 raw proposals from D role
- `results/voyager_v4_beta/mainline/pass1_mediastore_oracle.json` — 3 failure modes (Oracle for MS)
- `results/voyager_v4_beta/mainline/pass1_teastore_oracle.json` — 3 failure modes (Oracle for TS)
- `results/voyager_v4_beta/mainline/pass1_teammates_oracle.json` — 5 failure modes (Oracle for TM)
- `logs/voyager_v4_beta/.gitkeep` — directory tracking marker

## Decisions Made

1. **reviewer_critic_stub kept advisory for probe tier**: CONTEXT.md noted "activates real LLM in Phase 15" but RESEARCH.md confirmed the stub already enforces taboo-grep (blocking) and advisory critique is sufficient for REQ-V23-09 at probe tier. Upgrade deferred to Phase 16 Range if over-acceptance becomes a problem.
2. **Single invocation for both passes**: Harness loops internally; no manual pass-2 trigger needed. Harness detected convergence (macro 0.9152 > 0.90) and stopped after pass 1 naturally.
3. **git add -f for results/**: The `.gitignore` excludes `results/` but prior result files are tracked; used `git add -f` consistently with existing repo practice.
4. **No dollar cost available**: `LLMClient.get_session_usage()` is not called by `run_probe()`. Token counts not in probe.log. Navigation hint appended for Plan 2. Estimated ~$5-7 for 10 LLM-heavy calls at gpt-5.4 rates.

## Deviations from Plan

None — plan executed exactly as written.

The reviewer_critic_stub advisory mode was confirmed as intentional (RESEARCH.md Assumption A1 holds). The results/ gitignore workaround was expected (prior project pattern).

## Token Usage

Token counts not available in probe.log (harness does not call `LLMClient.get_session_usage()` after `run_probe()`). Per RESEARCH.md Pitfall 4, this was expected. Navigation hint appended to probe.log:

```
[TOKENS] No per-role token tracking in harness (LLMClient.get_session_usage() not called by run_probe). Token counts not available; estimate from probe_summary.json pass count: 1 pass, 3 L runs + 3 O runs + 1 D run + 3 P runs = 10 LLM calls.
```

Estimated cost: 1 pass × 10 LLM-heavy calls × ~$0.50-0.70/call on gpt-5.4 ≈ $5-7 (within $10 probe budget).

## Issues Encountered

1. **Taboo token warnings**: The Oracle prompts for all 3 projects triggered GATE-06 taboo warnings for project names (mediastore, teastore, teammates, datastore). These are advisory warnings, not blocking failures — the harness logs them and continues. All 3 proposals passed GATE-06 filter. This is expected behavior per RESEARCH.md anti-patterns section.
2. **pass2_summary.json not created**: Harness converged at pass 1 (macro 0.9152 > CONVERGENCE_THRESHOLD 0.90), so pass 2 was skipped. This is correct behavior per plan acceptance criteria: "If passes_run == 2: file results/voyager_v4_beta/mainline/pass2_summary.json exists" — passes_run=1 so this file is not required.

## Anomalies

- **Early convergence at pass 1**: The harness hit CONVERGENCE_THRESHOLD=0.90 after pass 1 (committed macro 0.9152). This is a positive outcome — no cheap-kill triggered, verdict=CONTINUE with a higher-than-threshold F1. The CHEAP_KILL_THRESHOLD=0.87 was not tested because pass 2 was skipped.
- **TM (teammates) F1 significantly lower**: TM base F1=0.8264 vs MS=0.9508 and TS=0.9474. This reflects teammates being a harder dataset (8 components, 198 sentences, many generic aliases like "GAE Datastore"). The macro still exceeded convergence threshold.
- **No pass2_summary.json**: Expected when convergence happens at pass 1.

## Next Phase Readiness

- Plan 2 (Wave 2) can proceed: reads `probe_summary.json` (verdict=CONTINUE, final_train_macro_f1=0.9152) and creates `15-PROBE-VERDICT.md` + STATE.md update
- Phase 16 Range Tier is unblocked (CONTINUE verdict confirmed)
- Bank files are persisted and ready for Phase 16 warm-start if needed

---
*Phase: 15-probe-tier*
*Completed: 2026-06-01*
