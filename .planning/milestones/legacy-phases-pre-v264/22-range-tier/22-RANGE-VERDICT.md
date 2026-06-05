# Phase 22 — Range Tier Verdict

**Date:** 2026-06-01
**Split:** mainline (MS+TS+TM train, BBB+JAB test)
**Backend:** gpt-5.4

## Run Summary

| Pass | Train macro | Committed | Patterns accepted | Probation |
|------|-------------|-----------|-------------------|-----------|
| 1    | 0.8966      | False      | 3 (ROLLBACK −0.0066) | gate fired |
| 2    | 0.8913      | True       | 3                 | passed    |
| 3    | 0.9095      | True       | 2                 | passed    |
| 4    | 0.9043      | True       | 3                 | passed    |
| 5    | 0.9208      | False      | 3 (ROLLBACK −0.0036) | gate fired |

- Passes run: 5 (cap hit; not converged)
- Convergence threshold: 0.90 — not reached consistently
- Patterns committed: 8 (passes 2+3+4)
- Final committed_macro: 0.9208

## Bank State

Final bank: 14 patterns across 7/9 slots (6 from probe + 8 from range).

| Slot | Patterns |
|------|----------|
| AMBIGUITY_RULES | 5 |
| COREF_RULES | 1 |
| DOC_KNOWLEDGE_EXTRACTION_RULES | 4 |
| DOC_KNOWLEDGE_JUDGE_RULES | 1 |
| SEED_DISAMBIGUATION_RULES | 1 |
| VALIDATION_RULES | 2 |
| AMBIGUITY_FEW_SHOT / DOC_KNOWLEDGE_JUDGE_EXAMPLES / ENTITY_EXTRACTION_RULES | 0 each |

Bank persisted: `results/voyager_v4_beta/mainline/final_bank.json`

## 5-Dataset Evaluation (gpt-5.4, trained bank)

| Dataset      | F1    | FP |
|--------------|-------|----|
| mediastore   | 96.7% |  0 |
| teastore     | 90.9% |  3 |
| teammates    | 83.9% | 15 |
| bigbluebutton| 77.6% |  9 |
| jabref       | 100.0%|  0 |
| **Macro avg**| **89.8%** | 27 |

## Comparison Table

| Condition             | Macro F1 |
|-----------------------|----------|
| Phase 20 axiom-only   | 87.6%    |
| Phase 22 trained bank | 89.8%    |
| Trained lift          | +2.2pp   |
| v2.3 Range result     | 89.8%    |
| v2.4 vs v2.3          | 0.0pp    |

**Observation:** v2.4 Range delivers identical 5-dataset macro to v2.3 Range (89.8%). Axiom improvements (Phase 20-P2) show no net uplift at this evaluation tier. TM remains at 83.9% (0 improvement from D-2/D-3 axiom fixes). Lift over axiom-only is same +2.2pp as v2.3.

## Verdict

**WEAK** — 89.8% ∈ [0.87, 0.9173)

Phase 23 Confirmation proceeds (89.8% ≥ 0.87 threshold satisfied).

Phase 23 goal: validate that fixed traceability gate resolves split-2 empty-bank failure mode (REQ-V24-01). Cross-split macro may differ from mainline if split diversity produces different pattern consensus.

## Gate Behavior

Traceability gate (Gate A + Gate B, fixed in Phase 20) fired correctly:
- Pass 1 ROLLBACK: probation delta −0.0066 < 0 → patterns discarded correctly
- Pass 5 ROLLBACK: probation delta −0.0036 < 0 → patterns discarded correctly
- No false passes (unlike v2.3 broken gate which committed harmful patterns)

Gate fix validated as operational. REQ-V24-01 (implementation) satisfied.

## Logs

- Range run: `logs/voyager_v4_beta/range.log`
- Eval: `logs/voyager_v4_beta/eval_range.log`
- Range summary: `results/voyager_v4_beta/mainline/range_summary.json`
