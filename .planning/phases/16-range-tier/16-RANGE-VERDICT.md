---
phase: 16-range-tier
tier: range
backend: openai
model: gpt-5.4
split: mainline
train_projects: [mediastore, teastore, teammates]
test_projects: [bigbluebutton, jabref]
date: 2026-06-01
verdict: WEAK
strong_threshold: 0.9173
weak_floor: 0.87
final_train_macro_f1: 0.9208
final_5dataset_macro_f1: 0.898
passes_run: 5
converged: false
requirements_closed: [REQ-V23-05, REQ-V23-07, REQ-V23-13, REQ-V23-14, REQ-V23-15]
next_action: Phase 17 Confirmation Tier
---

# Phase 16: Range Tier Verdict

## Summary

WEAK verdict — 5-dataset macro F1 = 89.8% (gpt-5.4, trained bank), above the 0.87 floor but below STRONG threshold (0.9173); Phase 17 Confirmation Tier proceeds. Trained bank adds +2.2pp over axiom-only floor (87.6%), with TM as the biggest beneficiary (+6.1pp) and TS showing slight regression (-2.0pp); full convergence not reached after 5-pass cap.

## Training Results

| Pass | MS F1 | TS F1 | TM F1 | Macro-L | Probation Δ | Committed | Notes |
|------|-------|-------|-------|---------|-------------|-----------|-------|
| 1    | 0.9333 | 0.9310 | 0.8254 | 0.8966 | -0.0066 | false | ROLLBACK — 3 patterns discarded |
| 2    | 0.9333 | 0.9310 | 0.8095 | 0.8913 | +0.0281 | true  | COMMIT — committed_macro=0.9194 |
| 3    | 0.9677 | 0.9455 | 0.8154 | 0.9095 | +0.0052 | true  | COMMIT — committed_macro=0.9147 |
| 4    | 0.9333 | 0.9434 | 0.8361 | 0.9043 | +0.0067 | true  | COMMIT — committed_macro=0.9110 |
| 5    | 0.9508 | 0.9615 | 0.8500 | 0.9208 | -0.0036 | false | ROLLBACK — 3 patterns discarded; final bank = pass 4 state |

Final committed bank: pass 4 state (14 patterns, 6 non-empty slots).
LLM variance visible: MS oscillates 0.933–0.968, TM 0.810–0.850 across passes with same bank.

## 5-Dataset Evaluation (s_linker14_voyager, gpt-5.4, trained bank)

| Dataset | Precision | Recall | F1 | FP | FN |
|---------|-----------|--------|----|-----|-----|
| mediastore    | 100.0% | 93.5% | 96.7% |  0 |  — |
| teastore      |  90.6% | 91.2% | 90.9% |  3 |  — |
| teammates     |  77.6% | 91.2% | 83.9% | 15 |  5 |
| bigbluebutton |  83.3% | 72.6% | 77.6% |  9 | 17 |
| jabref        | 100.0% | 100.0% | 100.0% |  0 |  0 |
| **Macro**     | — | — | **89.8%** | **27** | — |

## Axiom-Only Comparison (REQ-V23-15)

| Source | MS | TS | TM | BBB | JAB | 5-Dataset Macro F1 | Notes |
|--------|----|----|----|----|-----|--------------------|-------|
| s_linker14_voyager (trained bank) | 96.7% | 90.9% | 83.9% | 77.6% | 100.0% | **89.8%** | primary result |
| s_linker14_voyager (axiom-only, empty bank) | 95.1% | 92.9% | 77.8% | 74.8% | 97.3% | **87.6%** | prompts_v3_axiom floor |
| s_linker13_min (hand-authored prompts_v3) | — | — | — | — | — | **90.69%** | canonical reference (gpt-5.4, Phase 14 baseline) |

Lift from trained bank over axiom-only floor: **+2.2pp**

Per-dataset lift:

| Dataset | Axiom-only | Trained | Lift | Notes |
|---------|-----------|---------|------|-------|
| mediastore | 95.1% | 96.7% | +1.6pp | |
| teastore | 92.9% | 90.9% | **-2.0pp** | slight regression — patterns may overfit MS/TM failure modes |
| teammates | 77.8% | 83.9% | **+6.1pp** | largest beneficiary |
| bigbluebutton | 74.8% | 77.6% | +2.8pp | |
| jabref | 97.3% | 100.0% | +2.7pp | |

Lift from trained bank over s_linker13_min canonical: **-0.89pp** (trained s_linker14_voyager trails hand-authored canonical by 0.89pp at Range tier — Confirmation Tier aims to close this gap).

## Bank Saturation

| Project | Patterns (9 slots) | Non-empty slots | Source |
|---------|--------------------|-----------------|--------|
| mediastore | 14 total (3 dry-run + 11 real) | 6 | results/voyager_v4_beta/mainline/mediastore_bank.json |
| teastore   | 11 real | 6 | results/voyager_v4_beta/mainline/teastore_bank.json |
| teammates  | 11 real | 6 | results/voyager_v4_beta/mainline/teammates_bank.json |
| **final_bank (aggregated)** | **14 real** | **6/9** | results/voyager_v4_beta/mainline/final_bank.json |

Non-empty slots: AMBIGUITY_RULES (5), DOC_KNOWLEDGE_EXTRACTION_RULES (4), DOC_KNOWLEDGE_JUDGE_RULES (1), VALIDATION_RULES (2), COREF_RULES (1), SEED_DISAMBIGUATION_RULES (1).

Empty slots (axiom-only): AMBIGUITY_FEW_SHOT, DOC_KNOWLEDGE_JUDGE_EXAMPLES, ENTITY_EXTRACTION_RULES.

**Saturation signal:** D proposed 2–3 patterns every pass, but probation rolled back passes 1 and 5; committed deltas shrank from +0.028 → +0.005 → +0.007. Classic diminishing-return saturation curve after pass 2 big commit.

## Verdict Evidence

- **3-tier bar**: STRONG ≥ 0.9173 / WEAK [0.87, 0.9173) / FAIL < 0.87
- **Final 5-dataset macro F1 (gpt-5.4)**: 0.8980
- **Verdict**: **WEAK**
- **Rationale**: 89.8% clears the 0.87 floor by +2.8pp but misses STRONG by 1.93pp. TM (83.9%) and BBB (77.6%) are the drag — both datasets have high FP counts on GPT that the trained bank only partially addresses.

## Cost (REQ-V23-14)

- Range training (5 passes × ~10 LLM calls/pass = ~50 calls @ ~$0.50–0.70/call): ~$25–35
- 5-dataset evaluation (trained bank): ~$5–8
- 5-dataset evaluation (axiom-only): ~$5–8
- **Total Phase 16**: ~$35–51 — **over $25 cap** (range ran to pass 5 cap; expected 1–2 passes in plan estimate)
- Note: range ran 5 passes with no convergence; cost overage driven by saturation without convergence signal

## GATE-06 Status

- Taboo-grep advisory warnings: **27** (project names mediastore/teastore/teammates appearing in O prompts — expected, non-blocking)
- D proposal blockers (hard rejects): **0** across all 5 passes
- GATE-06 verdict: **PASS** (0 hard rejects; advisory warnings logged but non-blocking per Phase 15 precedent)

## Next Action

WEAK verdict — Phase 17 Confirmation Tier proceeds.
- **Next command**: `/gsd-plan-phase 17` (3-split sweep, $40–60 budget)
- Phase 17 runs Voyager v2 splits 1+2+3, cross-split aggregation, dual-artifact registration.

## Anomalies / Notes

- **Two rollbacks (passes 1 and 5)**: Final bank is pass 4 state, not pass 5. Pass 5 L macro (0.9208) is the highest observed training F1 but those 3 patterns failed probation.
- **TS regression (-2.0pp)**: Axiom-only TS=92.9% > trained TS=90.9%. Patterns learned from MS/TM failure modes may introduce noise on TS. Investigate in Phase 17 per-split analysis.
- **TM/BBB GPT gap**: TM 83.9% (GPT) vs 91.4% (Claude) = -7.5pp; BBB 77.6% (GPT) vs 82.1% (Claude) = -4.5pp. These datasets have been flagged for targeted Claude re-runs when per-dataset evidence is needed.
- **Cost overage**: ~$35–51 vs $25 cap. Overrun driven by 5-pass saturation run; budget estimate assumed 1–2 passes to convergence. GATE-08 justification: WEAK positive finding + axiom floor quantified = publishable artifact.

## Requirements Closed

| REQ | Evidence |
|-----|----------|
| REQ-V23-05 | 3-tier verdict computed: WEAK (final macro 0.8980 vs thresholds 0.9173/0.87) |
| REQ-V23-07 | Range tier complete on mainline split MS+TS+TM; verdict documented |
| REQ-V23-13 | 5 passes run; did not converge (accepted > 0 each committed pass); pass-by-pass macros in Training Results table |
| REQ-V23-14 | Total Phase 16 cost ~$35–51 vs $25 cap — over cap; cost overrun documented |
| REQ-V23-15 | Axiom-only comparison: trained 89.8% vs axiom-only 87.6% (+2.2pp lift); vs s_linker13_min 90.69% (-0.89pp gap) |
