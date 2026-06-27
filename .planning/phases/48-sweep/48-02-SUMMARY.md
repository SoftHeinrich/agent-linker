---
phase: 48-sweep
plan: 02
subsystem: benchmarking
tags: [verdict, gate-06, gate-08, req-v264-09, macro-f1, s_linker20, measurement]

requires:
  - phase: 48-sweep
    plan: 01
    provides: logs/v2.6.4_s_linker20_gpt.log + per-call token logs

provides:
  - REQ-V264-09 verdict (FAIL — macro 88.9% < floor 91.3%)
  - GATE-06 clean confirmation (zero benchmark-derived vocabulary collisions)
  - GATE-08 cost assertion (upper-bound $7.71 < $20 cap)
  - Consolidated verdict table for Phase 49 (MILESTONE CLOSE) consumption

affects: [49-close]

tech-stack:
  added: []
  patterns: []

key-files:
  created:
    - .planning/phases/48-sweep/48-02-SUMMARY.md
  modified: []

key-decisions:
  - "REQ-V264-09 recorded as FAIL (macro 88.9% < 91.3% floor) — verdict-phase invariant honored, no source edits"
  - "GATE-06 re-verified clean: zero taboo-vocabulary collisions in BENCHMARK_TABOO.md"
  - "GATE-08 confirmed PASS: upper-bound cost $7.71 reconstructed from token logs < $20 cap"
  - "3 of 5 datasets breach −2pp fence (TM −6.5pp, BBB −5.4pp, JAB −8.6pp); MS/TS improve"

patterns-established: []

requirements-completed: [REQ-V264-09, GATE-06, GATE-08]

duration: 10min
completed: 2026-06-09
---

# Phase 48 Plan 02: Verdict Formalization + Gate Records Summary

**Per-dataset F1 and macro extracted from logs/v2.6.4_s_linker20_gpt.log; REQ-V264-09 verdict FAIL (macro 88.9% < floor 91.3%); GATE-06 clean; GATE-08 PASS ($7.71 upper bound)**

## Performance

- **Duration:** 10 min
- **Completed:** 2026-06-09
- **Tasks:** 2 of 2
- **Files modified:** 1 (48-02-SUMMARY.md created)

## Accomplishments

- Per-dataset F1 + macro re-extracted from the actual log (confirmed 5 lines + Macro avg line)
- Comparison table built vs s17e reference with −2pp fence per dataset
- REQ-V264-09 verdict rendered: FAIL with precise per-dataset breach analysis
- GATE-06 re-grep executed verbatim; result: `GATE-06 clean`
- GATE-08 cost reconstructed from results/llm_logs/s_linker20_openai_*_calls.json token sums; $7.71 < $20
- Verdict-phase invariant maintained: zero source file edits
- Consolidated verdict ready for Phase 49 CLOSE consumption

## Task Commits

1. **Task 1: Extract per-dataset F1 + macro + render verdict** — committed below
2. **Task 2: Re-verify GATE-06 + reconstruct GATE-08 cost** — committed below

---

## VERDICT TABLE (Phase 49 Entry Point)

### Raw Provenance Lines (from logs/v2.6.4_s_linker20_gpt.log)

**Per-dataset `s_linker20:` lines (grep 's_linker20:'):**
```
  s_linker20: P=100.0% R=93.5% F1=96.7% TP=29 FP=0 FN=2 (51s)
  s_linker20: P=100.0% R=96.3% F1=98.1% TP=26 FP=0 FN=1 (85s)
  s_linker20: P=79.4% R=87.7% F1=83.3% TP=50 FP=13 FN=7 (247s)
  s_linker20: P=92.9% R=62.9% F1=75.0% TP=39 FP=3 FN=23 (665s)
  s_linker20: P=94.1% R=88.9% F1=91.4% TP=16 FP=1 FN=2 (191s)
```

**Macro avg line (grep 'Macro avg'):**
```
Macro avg        | F1 88.9% FP  17
```

*(Dataset order in log: mediastore, teastore, teammates, bigbluebutton, jabref — confirmed from log header and per-dataset timing pattern)*

### Per-Dataset Results (s_linker20 observed)

| Dataset | P | R | F1 | TP | FP | FN | Time | LLM Calls |
|---------|---|---|----|----|----|----|------|-----------|
| MediaStore | 100.0% | 93.5% | **96.7%** | 29 | 0 | 2 | 51s | 14 |
| TeaStore | 100.0% | 96.3% | **98.1%** | 26 | 0 | 1 | 85s | 13 |
| TeaMmates | 79.4% | 87.7% | **83.3%** | 50 | 13 | 7 | 247s | 40 |
| BigBlueButton | 92.9% | 62.9% | **75.0%** | 39 | 3 | 23 | 665s | 22 |
| JabRef | 94.1% | 88.9% | **91.4%** | 16 | 1 | 2 | 191s | 10 |
| **Macro avg** | — | — | **88.9%** | — | **17** | — | 1239s | **99** |

### Comparison vs s17e Reference (−2pp Tolerance Fence)

| Dataset | s17e F1 | Floor (−2pp) | s20 F1 (observed) | Delta vs s17e | Within fence? |
|---------|---------|--------------|-------------------|---------------|---------------|
| MediaStore | 94.9% | ≥ 92.9% | 96.7% | **+1.8pp** | YES — above s17e |
| TeaStore | 96.3% | ≥ 94.3% | 98.1% | **+1.8pp** | YES — above s17e |
| TeaMmates | 89.8% | ≥ 87.8% | 83.3% | **−6.5pp** | **NO — breaches by 4.5pp** |
| BigBlueButton | 80.4% | ≥ 78.4% | 75.0% | **−5.4pp** | **NO — breaches by 3.4pp** |
| JabRef | 100.0% | ≥ 98.0% | 91.4% | **−8.6pp** | **NO — breaches by 6.6pp** |
| **Macro** | **92.3%** | **≥ 91.3%** | **88.9%** | **−3.4pp** | **NO — below floor** |

**Total FP:** s20 = 17 (vs s17e reference FP = 14). Net change: +3 FP.

---

## REQ-V264-09 VERDICT

**STATUS: FAIL**

**Reason:** Macro F1 88.9% is below the required floor of 91.3% (−3.4pp gap). Additionally, 3 of 5 datasets breach the −2pp per-dataset fence:
- TeaMmates: 83.3% vs floor ≥ 87.8% (breach = 4.5pp)
- BigBlueButton: 75.0% vs floor ≥ 78.4% (breach = 3.4pp)
- JabRef: 91.4% vs floor ≥ 98.0% (breach = 6.6pp)

Two datasets improved: MediaStore (+1.8pp to 96.7%) and TeaStore (+1.8pp to 98.1%), confirming extraction/validation works well on smaller, cleaner documents.

**Regression analysis (recorded verbatim — no fix attempted):**
- **TeaMmates (−6.5pp):** FP = 13 (vs s17e FP 14 total reference). Coreference validation over-permissive for TM's generic English sentences, injecting false positives.
- **BigBlueButton (−5.4pp):** FN = 23, Recall 62.9% — aggressive extraction filtering. BBB has 62 ground-truth links; 23 missed suggests entity extraction is the primary deficit rather than coref.
- **JabRef (−8.6pp):** Small dataset (18 GT links); 2 FN + 1 FP accounts for the full drop. High variance dataset.

**Per VERDICT-PHASE SEMANTICS (RESEARCH §Q7):** This result is recorded as-is. No source edits were made to s_linker20.py or any prompt constant. The FAIL verdict is the honest measurement result that Phase 49 CLOSE documents in MILESTONES.md.

---

## GATE-06 RESULT

**Command run (verbatim):**
```bash
test -z "$(grep -niwE 'grouping|encompasses|matching|noun|phrase|refers|back|topic|surrounding|section' BENCHMARK_TABOO.md)" && echo "GATE-06 clean" || echo "GATE-06 FAIL"
```

**Output:** `GATE-06 clean`

**STATUS: PASS** — Zero benchmark-derived vocabulary tokens appear in BENCHMARK_TABOO.md at the neutral-vocabulary intersection. The neutralized vocabulary in s_linker20's inlined constants (Phase 46 lexical cuts) does not collide with benchmark-specific terminology.

---

## GATE-08 COST EVIDENCE

**Token reconstruction from results/llm_logs/s_linker20_openai_*_calls.json:**

```
  s_linker20_openai_bigbluebutton_20260609_130037_calls.json: 22 calls, prompt=51909, compl=5644
  s_linker20_openai_jabref_20260609_130348_calls.json: 10 calls, prompt=16413, compl=1611
  s_linker20_openai_mediastore_20260609_124400_calls.json: 14 calls, prompt=24487, compl=2410
  s_linker20_openai_teammates_20260609_124932_calls.json: 40 calls, prompt=100614, compl=7260
  s_linker20_openai_teastore_20260609_124525_calls.json: 13 calls, prompt=24968, compl=2444
TOTAL: prompt=218391, completion=19369, total=237760
Estimated cost (codebase GPT-4 formula, UPPER BOUND): $7.7139
GATE-08: PASS (<= $20)
```

| Dataset | LLM Calls | Prompt Tokens | Completion Tokens | Total Tokens |
|---------|-----------|---------------|-------------------|--------------|
| bigbluebutton | 22 | 51,909 | 5,644 | 57,553 |
| jabref | 10 | 16,413 | 1,611 | 18,024 |
| mediastore | 14 | 24,487 | 2,410 | 26,897 |
| teammates | 40 | 100,614 | 7,260 | 107,874 |
| teastore | 13 | 24,968 | 2,444 | 27,412 |
| **TOTAL** | **99** | **218,391** | **19,369** | **237,760** |

**Cost calculation (codebase GPT-4 formula — known upper bound vs actual gpt-5.4 pricing):**
- Prompt: 218,391 × $0.00003 = $6.5517
- Completion: 19,369 × $0.00006 = $1.1621
- **Total upper-bound: $7.7139**

**STATUS: PASS** — $7.71 < $20 cap. Actual gpt-5.4 spend is almost certainly under $3 (codebase formula overestimates using legacy GPT-4 rates). Budget cap fully satisfied.

---

## CONSOLIDATED REQUIREMENT OUTCOMES

| Requirement | Status | Evidence |
|-------------|--------|----------|
| REQ-V264-09 | **FAIL** | Macro F1 88.9% < floor 91.3%; 3/5 datasets breach −2pp fence |
| GATE-06 | **PASS (clean)** | grep output: `GATE-06 clean` — zero taboo-vocabulary collisions |
| GATE-08 | **PASS** | Upper-bound cost $7.71 < $20 cap (reconstructed from token logs) |

**Phase 49 CLOSE receives:** REQ-V264-09 FAIL result with full per-dataset breakdown. Phase 49 must document this as a non-passing v2.6.4 milestone result in MILESTONES.md.

---

## Decisions Made

1. REQ-V264-09 recorded as FAIL (macro 88.9% < floor 91.3%) — verdict-phase invariant maintained; no source edits
2. GATE-06 re-grep confirms clean (carried from Phase 47; no change to s_linker20 prompt constants)
3. GATE-08 confirmed PASS: $7.71 upper-bound from reconstructed token totals (237,760 tokens)
4. Phase 49 CLOSE to document the MARGINAL FAIL in MILESTONES.md and freeze v2.6.4

## Deviations from Plan

None — plan executed exactly as written. All data extracted from the actual log file; results confirmed consistent with 48-01-SUMMARY. Verdict-phase invariant honored (zero source edits).

## Known Stubs

None — this plan produces a measurement verdict artifact (SUMMARY), not UI or data-sourced components.

## Threat Flags

None — verdict-only phase. No new network endpoints, auth paths, file access patterns, or schema changes introduced. The threat mitigations documented in the plan (T-48-01 extraction assertion, T-48-02 verdict-phase invariant) were both satisfied:
- T-48-01: Exactly 5 `s_linker20:` lines confirmed + Macro avg line confirmed; raw provenance recorded.
- T-48-02: No source files modified on FAIL (verified via `git diff --name-only -- 'src/**'` = empty).

## Self-Check: PASSED

- logs/v2.6.4_s_linker20_gpt.log: FOUND (5 s_linker20: lines + Macro avg)
- `grep -c 's_linker20:' logs/v2.6.4_s_linker20_gpt.log` = 5: PASS
- `grep 'Macro avg' logs/v2.6.4_s_linker20_gpt.log` = "Macro avg | F1 88.9% FP 17": PASS
- Macro F1 88.9% < floor 91.3%: MACRO FAIL (correct)
- GATE-06 grep output: "GATE-06 clean": PASS
- GATE-08 upper-bound $7.71 < $20: PASS
- results/llm_logs/s_linker20_openai_*_calls.json (5 files): FOUND (token totals confirmed)
- git diff --name-only -- 'src/**': empty (no source edits): PASS
- .planning/phases/48-sweep/48-02-SUMMARY.md: CREATED

---

*Phase: 48-sweep*
*Completed: 2026-06-09*
