# Phase 41: Probe + Range Training Re-runs

**Status**: not started
**Depends on**: Phases 38 + 39 + 40
**Budget**: ≤ $35 gpt-5.4 (probe ≤ $10 + range ≤ $25)

## Goal

The v2.7 stack (Tier C axiom + partial-injection port + coref dampening + dual-objective recall-oracle loop + per-split cross-validation) is exercised by a 2-pass probe followed by a convergence-range run on the mainline split (MS+TS+TM+BBB train, JAB hold-out). Per-dataset and macro F1 reported as evidence. **No F1 gate; no confirmation tier.** Verdict is qualitative + diagnostic.

## Background

Per v2.7 roadmap: numeric F1 is evidence, not gate. v2.7 ships if architecture extensions land cleanly and recall-oracle path produces at least one BBB FN-recovery commit. Phase 41 is the empirical validation of Phases 38+39+40; failure-to-recover is a documentable finding, not a blocker.

## Subtasks

1. **17a — Probe (2-pass mainline)**
   - Run `voyager_train_tlr_v5.py probe` with new stack.
   - 2 outer passes max; per-pass [TRAIN]/[TEST] F1 logged.
   - Verdict question: does the recall-oracle loop produce at least one BBB FN-recovery accept by Assessor?

2. **17b — Range (convergence, max 5 passes)**
   - Run `voyager_train_tlr_v5.py range` on mainline split.
   - Convergence at [TRAIN] macro ≥ 0.90 or 5-pass cap.
   - Per-dataset F1 on all 5 datasets at final pass.
   - Lift over axiom-only baseline quantified per project.

3. **17c — Cross-split per-split validation (no consensus)**
   - 3 splits × independent training × per-held-out axiom-only baseline.
   - Variance table: did all splits converge similarly? Is mainline bank a representative pick?
   - **No 4th confirmation tier.** Cross-split table is the final F1 evidence in v2.7.

## Success Criteria

1. Probe 2 passes complete on mainline split without crash; [TRAIN]/[TEST] F1 logged separately per pass.
2. Recall-oracle path produces ≥ 1 accept with `delta_FN < 0` citing BBB-specific FN sentence in either probe or range run.
3. Range converges or hits cap; per-project final F1 reported for all 5 projects.
4. Cross-split table (3 splits, per-split axiom-only vs trained, no consensus) produced.
5. GATE-08: total Phase 41 spend ≤ $35 gpt-5.4. Per-tier spend recorded.
6. GATE-01: `s_linker13_min` regression check still PASS post-run (run as sanity verification).
7. Verdict document `.planning/phases/41-training-rerun/41-VERDICT.md` produced with: macro F1 table, BBB FN-recovery count, accept/reject breakdown, comparison to v2.6 axiom-only floor.

## Verdict Categories (no F1 gate, qualitative)

| Verdict | Meaning |
|---|---|
| **SHIP** | Recall-oracle produced ≥ 1 BBB FN-recovery accept; per-split cross-val variance ≤ ±2pp; GATE-01 holds. Architecture lands. |
| **PARTIAL** | Recall-oracle path executes but produces no BBB FN-recovery accepts; per-split variance high; documents the structural gap for v2.8. |
| **REGRESSION** | Trained bank performs worse than axiom-only on mainline held-out; document loop instability finding. |

All three verdicts are acceptable v2.7 outcomes. SHIP and PARTIAL both proceed to Phase 42 close.

## Risk

- **Recall-oracle produces zero BBB accepts** — dual-objective guard too tight, or BBB FN patterns too hard for Assessor to articulate. Outcome: PARTIAL verdict. Documents as v2.8 candidate.
- **Cost overrun** if range hits 5-pass cap with many proposals. Mitigation: hard kill at $35 spend; partial-range result still produces a verdict.
- **Per-split variance high** — cross-split semantics fix surfaces instability that was hidden by Jaccard consensus. Mitigation: this is the intended signal; report it.

## Out of Scope

- Confirmation tier (intentionally removed).
- F1 thresholds.
- New training-loop changes mid-phase (lock changes at Phase 40 exit).

## Plans

- TBD (17a probe)
- TBD (17b range)
- TBD (17c cross-split)
