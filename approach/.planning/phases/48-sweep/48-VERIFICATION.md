---
phase: 48-sweep
verified: 2026-06-09T00:00:00Z
status: human_needed
score: 3/5 success criteria verified (2 empirical FAIL — see verdict framing)
overrides_applied: 0
human_verification:
  - test: "Milestone-level decision required: REQ-V264-09 is an empirical FAIL (macro 88.9% < 91.3% floor). The Phase 48 deliverable is complete and gate-clean. Before milestone v2.6.4 can be closed, a human/milestone owner must explicitly accept the negative result and record the disposition in MILESTONES.md (e.g., freeze as negative result, spawn v2.6.5 with prompt repair, or re-scope the milestone floor)."
    expected: "An explicit written decision in MILESTONES.md: either 'v2.6.4 closes as negative result — s_linker20 Pareto-minimized prompts regress the s17e breakthrough floor' or 'v2.6.4 spawns remediation phase (v2.6.5)'"
    why_human: "The floor miss is an empirically measured outcome, not a code defect. Automated verification cannot decide whether a negative experimental result closes a milestone as a failed hypothesis or triggers a new development phase. That decision belongs to the milestone owner."
---

# Phase 48: s_linker20 gpt-5.4 Sweep — Verification Report

**Phase Goal:** s_linker20 is validated at gpt-5.4 macro F1 >= 91.3% across all 5 datasets within the $20 budget cap, confirming that Pareto-minimized prompts do not regress the 17e-line breakthrough floor.

**Verified:** 2026-06-09

**Status:** human_needed

**Re-verification:** No — initial verification

**Verdict-Phase Framing:** This is a measurement/verdict phase. Its deliverable is a valid, gate-clean gpt-5.4 sweep with an honestly recorded PASS/FAIL verdict. That deliverable is COMPLETE. The macro-F1 floor miss is the *content* of the verdict — an empirical negative result — not a closeable code gap. Criteria #2 and #3 are recorded as EMPIRICAL FAIL, not as defects to fix. The phase cannot and must not be re-planned to "fix" the regression (re-sweeping identical code reproduces the same result; changing prompts is a new milestone, out of scope for Phase 48).

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | log `logs/v2.6.4_s_linker20_gpt.log` exists, is committed, records a COMPLETED 5-dataset gpt-5.4 sweep on s_linker20 | VERIFIED | `git show fd93cd0 --name-only` confirms log added; log header: `Backend: openai (gpt-5.4)`, `Variants: s_linker20`; 5 `s_linker20:` result lines confirmed (`grep -c 's_linker20:'` = 5) |
| 2 | Macro F1 >= 91.3% | EMPIRICAL FAIL | Log: `Macro avg \| F1 88.9% FP 17`. Arithmetic independently verified: (96.7+98.1+83.3+75.0+91.4)/5 = 88.90%. Gap: −3.4pp below floor. |
| 3 | No individual dataset drops more than 2pp vs s17e (MS 94.9, TS 96.3, TM 89.8, BBB 80.4, JAB 100.0) | EMPIRICAL FAIL | TM: 83.3% vs floor 87.8% (−6.5pp, breach 4.5pp); BBB: 75.0% vs floor 78.4% (−5.4pp, breach 3.4pp); JAB: 91.4% vs floor 98.0% (−8.6pp, breach 6.6pp). MS and TS improve (+1.8pp each). |
| 4 | GATE-06 re-verified clean | VERIFIED | Verifier ran gate command directly: `test -z "$(grep -niwE 'grouping|encompasses|matching|noun|phrase|refers|back|topic|surrounding|section' BENCHMARK_TABOO.md)" && echo "GATE-06 clean"` — output: `GATE-06 clean` |
| 5 | Total API cost <= $20 (GATE-08) | VERIFIED | 237,760 total tokens (218,391 prompt + 19,369 completion); upper-bound cost $7.71 using codebase GPT-4 formula (known overestimate); $7.71 < $20. Token sums consistent across 48-01-SUMMARY and 48-02-SUMMARY. |

**Score:** 3/5 success criteria verified. Criteria #2 and #3 are empirical FAILs (the floor miss IS the phase verdict — not a defect to close).

**Verdict-phase invariant check:** Zero src file modifications during Phase 48. Confirmed: `git diff --stat HEAD -- 'src/llm_sad_sam/linkers/experimental/s_linker19.py' 's_linker20.py' 's_linker13_min.py'` = empty. Commit fd93cd0 added only `logs/v2.6.4_s_linker20_gpt.log`. Subsequent commits (bbca9bc, b6d1e05) touch only `.planning/` docs.

---

## Log Provenance Verification

Raw `s_linker20:` lines from `logs/v2.6.4_s_linker20_gpt.log` (verified against file directly):

```
s_linker20: P=100.0% R=93.5% F1=96.7% TP=29 FP=0 FN=2 (51s)    [mediastore]
s_linker20: P=100.0% R=96.3% F1=98.1% TP=26 FP=0 FN=1 (85s)    [teastore]
s_linker20: P=79.4%  R=87.7% F1=83.3% TP=50 FP=13 FN=7 (247s)  [teammates]
s_linker20: P=92.9%  R=62.9% F1=75.0% TP=39 FP=3 FN=23 (665s)  [bigbluebutton]
s_linker20: P=94.1%  R=88.9% F1=91.4% TP=16 FP=1 FN=2 (191s)   [jabref]
```

Macro avg line: `Macro avg | F1 88.9% FP 17`

Log line count: 320 lines (SUMMARY claims 321 — 1-line discrepancy, immaterial; all data lines present).

Numbers in 48-01-SUMMARY.md and 48-02-SUMMARY.md match the log exactly.

---

## Per-Dataset Comparison vs s17e Reference

| Dataset | s17e F1 | Floor (−2pp) | s20 F1 (log) | Delta | Within fence? |
|---------|---------|--------------|-------------|-------|---------------|
| MediaStore | 94.9% | 92.9% | 96.7% | +1.8pp | PASS — above s17e |
| TeaStore | 96.3% | 94.3% | 98.1% | +1.8pp | PASS — above s17e |
| TeaMmates | 89.8% | 87.8% | 83.3% | −6.5pp | EMPIRICAL FAIL |
| BigBlueButton | 80.4% | 78.4% | 75.0% | −5.4pp | EMPIRICAL FAIL |
| JabRef | 100.0% | 98.0% | 91.4% | −8.6pp | EMPIRICAL FAIL |
| **Macro** | **92.3%** | **91.3%** | **88.9%** | **−3.4pp** | **EMPIRICAL FAIL** |

Arithmetic independently verified by verifier: macro = (96.7+98.1+83.3+75.0+91.4)/5 = 88.90%.

---

## Requirements Coverage

| Requirement | Status | Evidence |
|-------------|--------|----------|
| REQ-V264-09 | EMPIRICAL FAIL | Macro 88.9% < required floor 91.3%; 3/5 datasets breach −2pp fence. Phase 48 measured and honestly recorded this result. The requirement is not met, and cannot be met by re-running Phase 48 (identical code, identical result). Disposition requires milestone-level human decision. |
| GATE-06 | PASS | Verifier re-ran gate command directly; output: `GATE-06 clean`. REQUIREMENTS.md marker status: `[ ]` (unchecked — GATE-06 final check is assigned to Phase 49, not Phase 48). Phase 48 re-verification task is COMPLETE per 48-02-SUMMARY. |
| GATE-08 | PASS | Upper-bound cost $7.71 < $20 cap. REQUIREMENTS.md marker: `[x]` (checked and annotated with Phase 48 evidence). |

**Note on GATE-06 REQUIREMENTS.md marker:** The `[ ]` (unchecked) status in REQUIREMENTS.md for GATE-06 reflects that its *final* check is assigned to Phase 49 CLOSE. The Phase 48 re-verification task itself is documented as COMPLETE in 48-02-SUMMARY.md with the verbatim grep evidence. This is consistent with the roadmap assignment table which shows `GATE-06 | Phase 48 (re-verify); Phase 49 (final)`.

---

## Gate Verification

### GATE-06 (Benchmark Vocabulary Isolation)

Command run by verifier (verbatim, from project root):
```
test -z "$(grep -niwE 'grouping|encompasses|matching|noun|phrase|refers|back|topic|surrounding|section' BENCHMARK_TABOO.md)" && echo "GATE-06 clean" || echo "GATE-06 FAIL"
```
Result: `GATE-06 clean`

Status: **PASS**

### GATE-08 (Budget Cap)

Token totals from phase artifacts: 218,391 prompt + 19,369 completion = 237,760 total tokens.
Upper-bound cost (codebase GPT-4 formula): 218,391 × $0.00003 + 19,369 × $0.00006 = $6.55 + $1.16 = **$7.71**.
$7.71 < $20 cap.

Status: **PASS**

### GATE-01 (Source Integrity — Sanity Check)

`git diff --stat HEAD -- src/llm_sad_sam/linkers/experimental/s_linker19.py src/llm_sad_sam/linkers/experimental/s_linker20.py src/llm_sad_sam/linkers/experimental/s_linker13_min.py` = empty (no output).

Phase 48 commits (fd93cd0, bbca9bc, b6d1e05, 457d7eb) touch only `logs/`, `.planning/` — zero src modifications.

Status: **UNAFFECTED** (verdict-phase invariant held)

---

## Anti-Patterns Found

None. This is a measurement-only phase. No source files were modified. The log file and SUMMARY docs contain no stub, placeholder, or TBD/FIXME markers relevant to the deliverable.

---

## Behavioral Spot-Checks

Step 7b: SKIPPED — verdict phase produces a log artifact and documentation only. No new runnable entry points or API endpoints introduced.

---

## Human Verification Required

### 1. Milestone Disposition Decision for REQ-V264-09 FAIL

**Test:** Review the empirical negative result (macro F1 88.9% vs floor 91.3%) and decide: (a) close v2.6.4 as a negative result with the hypothesis "Pareto-minimized prompts do not regress s17e" falsified, or (b) spawn a remediation phase (v2.6.5) targeting the TM/BBB/JAB regressions.

**Expected:** An explicit written decision recorded in `.planning/MILESTONES.md` — either marking v2.6.4 as a closed negative result or creating a v2.6.5 milestone entry with scope definition.

**Why human:** The floor miss is a genuine empirical outcome, not a defect in Phase 48's execution. The phase ran correctly on the correct model with gate-clean prompts. Automated verification can confirm the numbers and the gate status, but cannot decide whether a negative experimental result closes a research milestone as a failed hypothesis or triggers a new development iteration. This requires the milestone owner's judgment.

---

## Gaps Summary

No closeable gaps exist in Phase 48's execution. The phase deliverable — a valid, gate-clean 5-dataset gpt-5.4 sweep with an honestly recorded PASS/FAIL verdict — is COMPLETE.

The 2 criteria marked EMPIRICAL FAIL (macro F1 and per-dataset fence) are the *content* of the verdict, not failures of the phase to execute. Re-planning or re-executing Phase 48 would reproduce the same result because s_linker20's prompt constants are unchanged.

The single `human_needed` item above (milestone disposition) is required before v2.6.4 can close. It is a milestone-level governance decision, not a Phase 48 gap.

---

_Verified: 2026-06-09_
_Verifier: Claude (gsd-verifier)_
