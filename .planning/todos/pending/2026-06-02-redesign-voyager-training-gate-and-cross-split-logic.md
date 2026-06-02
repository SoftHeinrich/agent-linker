---
created: 2026-06-02T05:40:02.000Z
title: Redesign Voyager training gate (empirical) and fix cross-split purpose
area: tooling
resolves_phase: 32
files:
  - scripts/voyager_train_tlr_v4_beta.py
  - scripts/voyager_train_tlr_v4_beta.py:789
  - scripts/voyager_train_tlr_v4_beta.py:821
  - scripts/voyager_train_tlr_v4_beta.py:1113
---

## Problem

v2.4/v2.5 training analysis revealed two structural design flaws in the Voyager training loop
that prevent it from closing the gap to canonical (s_linker13_min at 90.69%).

### Flaw 1 — Gate B is precision-biased and blocks all recall improvements

Current Gate B accept condition: `fixes_cited_fm=True AND causes_new_error=False`.

This creates a one-sided ratchet:
- FP-reducing rules (block links) → Gate B asks "does blocking cause FNs?" → rarely yes → ACCEPT
- FN-reducing rules (add links) → Gate B asks "does approving cause FPs?" → almost always yes → REJECT

Evidence: all 3 Gate B rejections in Range Pass 1 were recall-improving rules (coref propagation,
alias induction, scope expansion). All 12 committed rules are precision filters only.
Result: bank can only ever reduce FPs, never recover FNs. BBB's 20 FNs are structurally unreachable.

Root issue: Gate B judges a rule's abstract semantics ("would this cause new errors?") rather than
its EMPIRICAL effect on actual FP+FN counts. An abstract semantic prediction is inherently biased
toward rejecting expansive rules, because any rule that adds links COULD add wrong links in theory.
What matters is whether net F1 improves and neither FP nor FN count gets significantly worse.

### Flaw 2 — Cross-split is used as aggregation, should be cross-validation

Current behavior: 3 splits train independently → their banks are deduped (Jaccard ≥ 0.6) →
patterns surviving ≥2 splits are kept as "consensus." This was designed to measure generalization
but doesn't: all splits share MS+TS via L-cache → same results → trivial 3/3 "consensus."

Actual purpose of cross-split: validate that a trained bank generalizes to held-out test sets.
Split 1 tests on BBB+JAB; Split 2 tests on TM+JAB; Split 3 tests on MS+BBB. The right question
is "does Split 1's bank perform well on its test set?" — not "do all 3 splits propose the same patterns?"

No cross-split dedup should happen. Each split produces an independent bank validated against
its own held-out test. The cross-split result table shows variance/consistency of training, not
a consensus bank to deploy.

---

## Solution

### A. Replace Gate B with an Empirical Evaluator role

New training gate question: **"Does this new skill reduce FP AND FN, or at minimum reduce total
errors net-positive without worsening either FP or FN by more than N links?"**

Implementation approach — add a new LLM role "Assessor" (or retask Gate B):

1. After Distillator proposes a pattern, apply it to the bank as a provisional addition.
2. Run a mini L-run on a held-out slice of the training set (or all training projects at low cost).
3. Compare FP count before vs after; compare FN count before vs after.
4. Gate condition: `(delta_FP <= 0 AND delta_FN <= 0) OR (net_delta_errors < -threshold AND max(delta_FP, delta_FN) <= 1)`.
5. LLM Assessor role sees actual error counts + changed sentences and judges whether the change is acceptable.

This replaces abstract semantic prediction with empirical outcome measurement. Recall-improving
rules that also reduce FNs will pass. Rules that purely add FPs will fail. Tradeoffs (reduce FN by 2,
add FP by 1) can be explicitly approved with human-interpretable rationale.

**Cost consideration:** Mini L-run per proposal is expensive. Options:
- Run L on training set only (not held-out) — cheaper but risks overfitting signal
- Run L on a 10-sentence held-out slice per project — cheap, still empirical
- Batch all proposals for one pass → single L run comparing all variants — best cost

### B. Fix cross-split: validation mode, no aggregation

Remove cross-split dedup and consensus logic entirely. Replace with:

1. Each split trains independently → produces `split_N_final_bank.json`.
2. Each split evaluates its bank on its held-out test set → reports per-split F1.
3. Cross-split verdict table shows variance: if all 3 splits produce F1 within ±2pp of target, training is stable.
4. Final deployed bank = mainline bank (MS+TS+TM train), not a consensus artifact.

Cross-split result is a STABILITY CHECK, not an aggregation input.

### C. Consider merging Oracle + Assessor (architecture simplification)

Current pipeline: L → O (oracle) → D (distillator) → Gate A → Gate B → commit.

Proposed: L → O+D (oracle proposes patterns directly, text-aware) → Assessor (empirical L mini-run
+ LLM judgment on actual delta) → commit.

Rationale: Oracle already sees FP/FN sentences. If Oracle also proposes patterns (merging D role),
it has full context: error description + text context + proposed fix. Then Assessor validates
empirically. This removes the "text-blind distillator" limitation (CoT-A) that prevents D from
writing precise rules for specific document patterns without text access.

Alternatively: keep O+D separate but make Gate B = Assessor with empirical L-run backing.
Simpler change, lower refactor risk.

### D. Variance problem stays

MIN_COMMIT_DELTA=0.005 correctly filters gpt-5.4 noise but means only Pass 1 ever commits.
With the empirical Assessor, this may improve: if the assessor uses a held-out slice, variance
in that slice is lower than full 5-dataset variance. Worth testing whether delta threshold can
be loosened or eliminated when using per-proposal empirical testing.
