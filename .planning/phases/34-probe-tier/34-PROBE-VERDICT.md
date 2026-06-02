---
phase: 34-probe-tier
tier: probe
backend: openai
model: gpt-5.4
split: mainline
train: [mediastore, teastore, teammates]
test: [bigbluebutton, jabref]
date: 2026-06-02
verdict: KILL
cheap_kill_threshold: 0.87
final_train_macro_f1: 0.9058
final_test_macro_f1: 0.8486
---

# Phase 34: Probe Tier Verdict

## Verdict: KILL

**Reason**: [TEST] macro F1 = 84.86% < 87.0% cheap-kill threshold after pass 2.

Per conditionality rules: Phase 35 (Range Tier) and Phase 36 (Confirmation Tier) are
SKIPPED. Proceed directly to Phase 37 (Milestone Close).

## Per-Pass Results

| Pass | [TRAIN] macro | [TEST] macro | Committed | Assessor decisions |
|------|--------------|-------------|-----------|-------------------|
| 1    | 90.58%       | 84.86%      | ✅ True   | 9 (7 accept, 2 reject) |
| 2    | 90.58%       | 84.86%      | ❌ False  | 0 (variance filter) |

Pass 2: delta=-0.0156 < 0.005 → OD+Assessor skipped by variance filter. [TEST] cache hit.

## Per-Dataset [TEST] F1

| Dataset | F1 |
|---------|-----|
| bigbluebutton | 72.41% |
| jabref        | 97.30% |
| **Macro**     | **84.86%** |

## [TRAIN] F1 (Pass 1)

| Dataset | F1 |
|---------|-----|
| mediastore | 93.33% |
| teastore   | 96.43% |
| teammates  | 81.97% |

## LLM Assessor Validation (REQ-V26-11 SC2)

Assessor was active in Pass 1 with 9 decisions:
- 7 accepted (rationale cited specific FP/FN sentences) ✅
- 2 rejected
- Multiple `revise` cycles observed (pattern revised once, then accepted/rejected)
- Gate A + Gate B F1-delta path is fully replaced ✅

## GATE-06 Warnings

`WARNING: prompt taboo tokens ['datastore']` appeared multiple times in Pass 1
Assessor output. The taboo filter triggered `revise` cycles. Final accepted patterns
were cleaned of the taboo token. GATE-06 is operational; patterns with benchmark
vocabulary were prevented from entering the bank.

## Root Cause Analysis

**Primary cause: v4 axiom Gap 2 gerund rejection is aggressive.**
The `SEED_DISAMBIGUATION_RULES` v4 change (self-referential capability = OTHER) is
rejecting more seeds than expected, especially for teammates (81.97% train F1 vs
expected ~88-90%). Pass 1 log shows many gerund rejections:
- "Describes only what the Storage component itself does" → OTHER
- "Refers to the component's own usage/capabilities" → OTHER
These rejections are FNs when the sentence IS a valid trace link. The axiom
change for Gap 2 over-applies to sentences that are valid links.

**Secondary cause: BBB test generalization remains weak.**
BBB test F1 = 72.41% was the same in v2.5 (73.7%). Training on MS+TS+TM does
not produce patterns that transfer to BBB's document style.

## Comparison to v2.5

| Metric | v2.5 | v2.6 Probe |
|--------|------|-----------|
| Axiom-only floor (BBB+JAB) | ~85-87% | 84.86% (estimated — no axiom-only baseline run) |
| Pass 1 [TEST] macro | — | 84.86% |
| Verdict | CONTINUE (89.1% @ Range) | KILL |

## Findings for Phase 37 Documentation

1. **LLM Assessor is operational**: 9 decisions with evidence-based rationale. Gate A/B
   replacement confirmed (REQ-V26-11 SC2). ✅
2. **OD merge is operational**: Single-role OD received FP/FN sentences and produced
   failure modes + proposals in one call. ✅
3. **[TRAIN]/[TEST] log separation works**: Both metrics logged per-pass. ✅
4. **v4 axiom Gap 2 may be over-aggressive**: The gerund rejection rule in
   `SEED_DISAMBIGUATION_RULES` appears to cause recall regressions on teammates.
   The "own capabilities without external participant" criterion is being applied
   too broadly. **Candidate fix for v2.7**.
5. **BBB generalization ceiling unchanged**: BBB test F1 ~72-73% appears to be an
   inherent ceiling for MS+TS+TM training. ILinker4 SEED slots still empty (no
   committed patterns) — may be needed for BBB improvement.
6. **GATE-06 operational**: Taboo filter caught 'datastore' in proposed patterns and
   triggered revision cycles as designed. ✅

## Next Action

Phase 37 (Milestone Close) — unconditional.
Log: `logs/voyager_v5/probe_p34.log` (tee failed on missing dir; use
`results/voyager_v5/mainline/probe_summary.json` instead).
