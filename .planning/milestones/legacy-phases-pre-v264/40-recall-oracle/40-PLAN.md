# Phase 40: Cross-Split Validation Redesign + Recall-Oracle (FULL)

**Status**: not started
**Depends on**: Phase 38 + Phase 39 (axiom + injection paths in place to evaluate against)
**Budget**: $0 (no LLM training; infrastructure only)

## Goal

`scripts/voyager_train_tlr_v5.py` is restructured so (a) cross-split runs as per-split validation against axiom-only baseline (no Jaccard consensus dedup), (b) BBB joins the training set, (c) the Assessor evaluates patterns with dual objectives (delta_FP AND delta_FN tracked separately, not collapsed into net F1), and (d) TM-driven precision rules are preserved via objective weighting that prevents recall-improving rules from regressing precision below a guard. v2.7 closes the structural deficit identified in `.planning/notes/2026-06-02-voyager-improvement-ideas.md` #6: voyager could only learn precision rules from TM, never recall rules, because BBB was held out.

## Background

Per todo `2026-06-02-redesign-voyager-training-gate-and-cross-split-logic.md`: current cross-split = Jaccard dedup across 3 splits sharing MS+TS via L-cache → trivial consensus, not validation. Real purpose: each split validates its trained bank on its held-out test. v2.7 fixes this.

Per voyager-improvement-ideas #6: TM (train) = FP-dominant. BBB (test) = FN-dominant. Voyager has no FN-recovery learning path because all training data lacks recall failures. Putting BBB in training opens this path BUT risks losing TM's FP rules. Dual-objective gate is the architectural answer.

User chose FULL scope: BBB into training + dual-objective gate (not bolt-on RecallOracle role).

## Subtasks

1. **13 — Cross-split as validation**
   - Remove Jaccard consensus / pattern dedup across splits in `voyager_train_tlr_v5.py`.
   - Each split trains independently on its train projects; bank evaluated against axiom-only baseline on its held-out test set.
   - Verdict per split: "does training improve F1 on held-out vs axiom-only?" — variance reported, not aggregated.
   - Final deployed bank = mainline (MS+TS+TM+BBB train, JAB hold-out) — not a consensus.

2. **16a — BBB into training set**
   - Mainline split: MS + TS + TM + BBB train; JAB held-out.
   - Cross-split rotations: each split rotates a different held-out from {MS, TS, TM, BBB, JAB}.
   - L-runs include BBB; OD prompt sees BBB FN/FP sentences.

3. **16b — Dual-objective Assessor gate**
   - Assessor input grows: per-pattern proposal + bank state + (FP set, FN set) before/after on each train project.
   - Assessor verdict: `accept` requires `delta_FN < 0 OR (delta_FN <= 0 AND delta_FP < 0)`. Reject if `delta_FP > guard` (guard tunable, e.g. +1 link per project) regardless of FN improvement.
   - Rationale must cite specific FN sentence(s) recovered OR FP sentence(s) blocked.
   - Removal_targets carry forward (per-pattern revert from v2.6 stays).

4. **16c — Precision-rule preservation**
   - Track TM precision-rule patterns committed under v2.6 (FM-1, FM-2 etc).
   - Dual-objective gate weight: if a new pattern increases TM delta_FP above guard, reject even if BBB delta_FN improves. Prevents recall-rule cascading into TM precision loss.

## Success Criteria

1. `voyager_train_tlr_v5.py` no longer contains Jaccard dedup or cross-split pattern intersection logic.
2. Mainline split runs MS+TS+TM+BBB train, JAB held-out (config + execution path verified end-to-end).
3. Assessor verdict log shows at least one `accept` with `delta_FN < 0` rationale citing a BBB-specific FN sentence — confirming recall-rule learning path is active.
4. Assessor verdict log shows at least one `reject` with `delta_FP > guard` rationale — confirming precision guard is active.
5. Per-split validation table format: split | train_set | test_set | axiom-only F1 | trained F1 | delta. No consensus column.
6. GATE-01 unchanged (Phase 40 is infrastructure-only; `s_linker13_min` untouched).
7. GATE-06: new Assessor prompt text + cross-split verdict template pass benchmark-vocabulary check.

## Risk

- **BBB in training → BBB-specific patterns generalize poorly to MS/TS** (overfitting to BBB FN patterns). Mitigation: per-split validation surfaces this immediately — if mainline-trained bank under-performs on its held-out, evidence of overfit.
- **Dual-objective gate too strict → no patterns accepted** (paralysis). Mitigation: guard tunable; default soft (+1 FP per project). Worst case, soften further.
- **Assessor prompt growth** — dual-objective rationale adds tokens. Mitigation: per-project FP/FN truncated to top-K sentences in prompt.

## Out of Scope

- New benchmark datasets (5-dataset only).
- Per-pattern A/B testing with persistent state across passes.
- Removing Assessor entirely (kept; dual-objective is a verdict expansion, not replacement).

## Plans

- TBD (subtask 13)
- TBD (subtask 16a)
- TBD (subtask 16b)
- TBD (subtask 16c)
