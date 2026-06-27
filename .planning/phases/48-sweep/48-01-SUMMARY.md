---
phase: 48-sweep
plan: 01
subsystem: benchmarking
tags: [gpt-5.4, ablation, macro-f1, s_linker20, sweep, llm-evaluation]

requires:
  - phase: 47-ship
    provides: s_linker20.py standalone variant registered in run_ablation.py with GATE-01/06 clean

provides:
  - Full 5-dataset gpt-5.4 sweep result for s_linker20 (logs/v2.6.4_s_linker20_gpt.log)
  - Per-call token logs for GATE-08 cost reconstruction (results/llm_logs/s_linker20_openai_*_calls.json)
  - Verdict: MARGINAL FAIL — macro F1 88.9% < floor 91.3%; BBB and TM regressions vs s17e

affects: [49-close]

tech-stack:
  added: []
  patterns: []

key-files:
  created:
    - logs/v2.6.4_s_linker20_gpt.log
    - results/llm_logs/s_linker20_openai_mediastore_20260609_124400_calls.json
    - results/llm_logs/s_linker20_openai_teastore_20260609_124525_calls.json
    - results/llm_logs/s_linker20_openai_teammates_20260609_124932_calls.json
    - results/llm_logs/s_linker20_openai_bigbluebutton_20260609_130037_calls.json
    - results/llm_logs/s_linker20_openai_jabref_20260609_130348_calls.json
  modified: []

key-decisions:
  - "VERDICT-PHASE SEMANTICS honored: no source edits despite regression"
  - "Macro F1 88.9% recorded as-is — MARGINAL FAIL vs 91.3% floor"
  - "GATE-08 satisfied: upper-bound cost $7.71 < $20 cap"

patterns-established: []

requirements-completed: [REQ-V264-09, GATE-08]

duration: 24min
completed: 2026-06-09
---

# Phase 48 Plan 01: s_linker20 gpt-5.4 Sweep Summary

**5-dataset gpt-5.4 sweep on s_linker20 completed with MARGINAL FAIL verdict: macro F1 88.9% vs floor 91.3%; BBB regressed -5.4pp and TM -6.5pp vs s17e baseline**

## Performance

- **Duration:** 24 min (clock time including API latency)
- **Started:** 2026-06-09T12:44:00Z (mediastore first call)
- **Completed:** 2026-06-09T13:08:38Z (jabref final)
- **Tasks:** 2 of 2
- **Files modified:** 1 (log created; results/llm_logs/* gitignored)

## Accomplishments

- Full 5-dataset gpt-5.4 sweep on s_linker20 executed and captured in log
- Backend and model confirmed correct: `Backend: openai (gpt-5.4)` in log header — model-name trap bypassed
- All 5 datasets completed (99 total LLM calls across mediastore/teastore/teammates/bigbluebutton/jabref)
- GATE-08 pre-flight and post-run cost evidence recorded: total 237,760 tokens, upper-bound $7.71 < $20
- Verdict-phase invariant held: zero source file edits during measurement

## Task Commits

1. **Task 1: Pre-flight verification** — No separate commit (verification only, no artifact produced)
2. **Task 2: 5-dataset sweep execution** — `fd93cd0` (feat)

**Plan metadata:** (pending docs commit)

## Pre-Flight Cost Estimate (GATE-08 Pre-Flight Evidence)

Recorded before any spend. Source: RESEARCH §Q2, based on empirical s_linker19 token counts (identical architecture).

| Pricing Scenario | Total Cost Estimate | Headroom vs $20 cap |
|-----------------|--------------------|--------------------|
| gpt-4o-like flex (realistic low) | ~$0.36–$0.72 | $19.28–$19.64 |
| gpt-4o-like standard (conservative) | ~$0.72 | $19.28 |
| gpt-4-turbo-like (pessimistic) | ~$2.70 | $17.30 |
| Codebase GPT-4 formula (upper bound) | ~$7.53 | $12.47 |

**Budget verdict pre-flight:** VERY LOW RISK. All scenarios < $20. No per-dataset staged abort required.

**Mediastore sanity-stop condition:** If mediastore alone cost > $5 (anomalous pricing tier), stop and investigate. Mediastore completed at 14 calls / ~24k prompt tokens — well within normal range.

## Sweep Execution

**Exact command run:**
```bash
LLM_BACKEND=openai OPENAI_MODEL_NAME=gpt-5.4 python run_ablation.py --variants s_linker20 2>&1 | tee logs/v2.6.4_s_linker20_gpt.log
```

**Log header (confirmed correct):**
```
Backend: openai (gpt-5.4)
Datasets: mediastore, teastore, teammates, bigbluebutton, jabref
Variants: s_linker20
```

**OpenAI 500 errors during run:** Two retry events on BBB Phase 4 validation and one on Phase 5 coreference. All recovered successfully via the built-in 3-attempt retry. No impact on result correctness.

## Results

### Per-Dataset Results (from log `s_linker20:` lines)

| Dataset | P | R | F1 | TP | FP | FN | Time | Calls |
|---------|---|---|----|----|----|----|------|-------|
| mediastore | 100.0% | 93.5% | **96.7%** | 29 | 0 | 2 | 51s | 14 |
| teastore | 100.0% | 96.3% | **98.1%** | 26 | 0 | 1 | 85s | 13 |
| teammates | 79.4% | 87.7% | **83.3%** | 50 | 13 | 7 | 247s | 40 |
| bigbluebutton | 92.9% | 62.9% | **75.0%** | 39 | 3 | 23 | 665s | 22 |
| jabref | 94.1% | 88.9% | **91.4%** | 16 | 1 | 2 | 191s | 10 |
| **Macro avg** | — | — | **88.9%** | — | **17** | — | 1239s | **99** |

### Comparison vs s17e Reference (2pp Tolerance Fence)

| Dataset | s17e F1 | s20 F1 | Delta | Floor (−2pp) | Status |
|---------|---------|--------|-------|--------------|--------|
| MediaStore | 94.9% | 96.7% | **+1.8pp** | ≥92.9% | PASS (+above) |
| TeaStore | 96.3% | 98.1% | **+1.8pp** | ≥94.3% | PASS (+above) |
| TeaMmates | 89.8% | 83.3% | **−6.5pp** | ≥87.8% | **FAIL** |
| BigBlueButton | 80.4% | 75.0% | **−5.4pp** | ≥78.4% | **FAIL** |
| JabRef | 100.0% | 91.4% | **−8.6pp** | ≥98.0% | **FAIL** |
| **Macro** | **92.3%** | **88.9%** | **−3.4pp** | **≥91.3%** | **FAIL** |

### Verdict: MARGINAL FAIL (Outcome B)

**REQ-V264-09: FAIL** — macro F1 88.9% < required floor 91.3%.

- Regressions in 3 of 5 datasets (TM, BBB, JAB)
- MediaStore and TeaStore show improvement (+1.8pp each) — extractino and validation working well
- TeaMmates regression: FP 13 from coreference (coreference validation over-permissive for TM's generic-English sentences)
- BigBlueButton regression: FN 23, recall 62.9% (extraction validation filtering too aggressively; 6 of 42 extracted candidates were coref-sourced, suggesting entity extraction is the primary deficit)
- JabRef regression: F1 91.4% vs 100.0% — 2 FN, 1 FP (small dataset, high variance)

Per VERDICT-PHASE SEMANTICS: this result is recorded as-is. No source edits were made.

## GATE-08 Cost Evidence (Post-Run)

**Token counts from results/llm_logs/s_linker20_openai_*_calls.json:**

| Dataset | LLM Calls | Prompt Tokens | Completion Tokens | Total |
|---------|-----------|---------------|-------------------|-------|
| mediastore | 14 | 24,487 | 2,410 | 26,897 |
| teastore | 13 | 24,968 | 2,444 | 27,412 |
| teammates | 40 | 100,614 | 7,260 | 107,874 |
| bigbluebutton | 22 | 51,909 | 5,644 | 57,553 |
| jabref | 10 | 16,413 | 1,611 | 18,024 |
| **TOTAL** | **99** | **218,391** | **19,369** | **237,760** |

**Cost estimate (codebase GPT-4 formula, known upper bound):**
- Prompt: 218,391 × $0.00003 = $6.55
- Completion: 19,369 × $0.00006 = $1.16
- **Total upper bound: $7.71**

**GATE-08: SATISFIED** — $7.71 < $20 cap. Actual gpt-5.4 cost almost certainly under $3.

## Files Created/Modified

- `logs/v2.6.4_s_linker20_gpt.log` — Full sweep stdout capture (321 lines; 5 per-dataset results + Macro avg)
- `results/llm_logs/s_linker20_openai_mediastore_20260609_124400_calls.json` — 14-call token log (gitignored)
- `results/llm_logs/s_linker20_openai_teastore_20260609_124525_calls.json` — 13-call token log (gitignored)
- `results/llm_logs/s_linker20_openai_teammates_20260609_124932_calls.json` — 40-call token log (gitignored)
- `results/llm_logs/s_linker20_openai_bigbluebutton_20260609_130037_calls.json` — 22-call token log (gitignored)
- `results/llm_logs/s_linker20_openai_jabref_20260609_130348_calls.json` — 10-call token log (gitignored)
- `results/ablation_results/ablation_20260609_130348.json` — Runner output JSON (gitignored)
- `results/phase_cache/s_linker20/openai/*/` — Phase pickles layer1–final per dataset (gitignored)

## Decisions Made

1. Macro F1 88.9% recorded as-is — no source edits per VERDICT-PHASE SEMANTICS (phase measures only)
2. GATE-08 satisfied: upper-bound cost $7.71 < $20 cap
3. OpenAI 500 retries on BBB treated as transient infrastructure noise (recovered automatically, no effect on validity)
4. Phase 49 CLOSE will receive this MARGINAL FAIL result and record the full verdict in MILESTONES.md

## Deviations from Plan

None — plan executed exactly as written. The regression result (macro 88.9% < 91.3% floor) is the honest measurement, not a deviation. VERDICT-PHASE SEMANTICS explicitly anticipates and mandates recording regressions without fix.

## Issues Encountered

**OpenAI API 500 errors on BBB:** Three transient 500 errors during BBB Phase 4 validation and Phase 5 coreference. The built-in 3-attempt retry with 2s backoff recovered each time. Run completed successfully. No re-billing impact. Documented as transient infrastructure noise.

## Known Stubs

None — this plan produces a measurement artifact (log), not UI or data-sourced components.

## Threat Flags

None — measurement-only phase. No new network endpoints, auth paths, or schema changes introduced.

## Next Phase Readiness

Phase 49 (CLOSE) receives this result:
- `logs/v2.6.4_s_linker20_gpt.log` exists and is committed
- Verdict: REQ-V264-09 FAIL, GATE-08 PASS
- Per-dataset breakdown documented for milestone audit
- Regressions concentrated in TM (coref FP) and BBB (recall deficit)

## Self-Check: PASSED

- logs/v2.6.4_s_linker20_gpt.log: FOUND
- .planning/phases/48-sweep/48-01-SUMMARY.md: FOUND
- Commit fd93cd0: FOUND
- Backend header `Backend: openai (gpt-5.4)`: PASS
- 5 `s_linker20:` dataset lines: PASS
- Macro avg line: PASS (88.9%)
- Token log files (5 datasets): FOUND (gitignored)

---
*Phase: 48-sweep*
*Completed: 2026-06-09*
