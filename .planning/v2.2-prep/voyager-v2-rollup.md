---
phase: v2.2-prep
plan: voyager-v2-rollup
subsystem: prompt-curriculum
tags: [voyager, train-test, gpt-5.4, fresh-start, rollup, cross-split, transfer-analysis]
key-files:
  created:
    - results/voyager_pilot_v2/crossplit_comparison.{json,pkl}
    - .planning/v2.2-prep/voyager-v2-rollup.md
  modified: []
decisions:
  - "Voyager-style prompt-curriculum transfer is SPLIT-FRAGILE on gpt-5.4 SAD-SAM. Two of three splits show a positive held-out lift (+0.66 pp, +1.09 pp); one shows a clear regression (-1.92 pp). Mean lift across 3 splits is -0.05 pp — statistically indistinguishable from zero. Range [-1.92, +1.09] spans 3 pp."
  - "Skills learned in each split are split-specific in surface vocabulary (split 1: service/facade/controller; split 2: broker/gateway/connector; split 3: generic principle statements) but converge on the same 2 axes — judge conservatism (do not link without explicit responsibility) and coref liberalism (propagate links through anaphora). The mechanism is real; the calibration is split-dependent."
  - "GATE-06 + reviewer-defensibility are NECESSARY but NOT SUFFICIENT for transfer. All 17 distilled rules across 3 splits passed both audits; split 3 still regressed. Defensibility ≠ generalisation."
  - "Voyager v2 does NOT beat trim1-distilled baseline (best macro all-5 across splits = 0.8891 vs trim1's 0.9173 from Phase 12 pilot). Mechanism does not earn a v2.2 shipping slot on its own; deferred per Phase 12 verdict, confirmed here across 3 splits."
metrics:
  total_duration_min: 46.1
  total_llm_calls: 72
  total_wallclock_s: 2770
  splits_run: 3
  splits_positive: 2
  splits_negative: 1
  best_held_out_lift_pp: 1.09
  worst_held_out_lift_pp: -1.92
  mean_held_out_lift_pp: -0.05
---

# Voyager-TLR v2 — Cross-Split Rollup

**One-liner:** Three fresh-start gpt-5.4 splits answer the v2.2-prep central question: Voyager-style prompt-curriculum transfer for SAD-SAM TLR is **SPLIT-FRAGILE**. Mean lift = −0.05 pp (held-out macro), range [−1.92, +1.09]. Two splits pass, one regresses. The mechanism produces defensible skill banks but the calibration is split-dependent — generality survives lexical audits (GATE-06 + reviewer critic) but does not survive distributional shift between training and held-out projects.

---

## 1. Scoreboard

| Split | Train | Test | Bank | Distilled | Axiom F1 (HO) | Distilled F1 (HO) | Δ pp | Verdict |
|---|---|---|---|---|---|---|---|---|
| 1 (replication) | MS+TS+TM | BBB+JAB | 9 | 6 | 0.8658 | **0.8725** | **+0.67** | pass |
| 2 (BBB-in-train) | MS+TS+BBB | TM+JAB | 8 | 6 | 0.8818 | **0.8927** | **+1.09** | **pass (best)** |
| 3 (rotated hold-out) | TS+TM+JAB | MS+BBB | 6 | 5 | 0.8718 | **0.8526** | **−1.92** | **regression** |
| **Mean** | — | — | 7.67 | 5.67 | 0.8731 | 0.8726 | **−0.05** | inconclusive |

### Per-project drill-down (distilled F1 only)

|  | MS | TS | TM | BBB | JAB |
|---|---|---|---|---|---|
| s_linker13_clean baseline | 0.9836 | 1.0000 | 0.9381 | 0.8036 | 0.9730 |
| Split 1 distilled | 0.9153 (T-sanity) | 0.9818 (T-sanity) | 0.8033 (T-sanity) | **0.7719** (HO) | **0.9730** (HO) |
| Split 2 distilled | 0.9508 (T-sanity) | 0.9286 (T-sanity) | **0.8125** (HO) | 0.7788 (T-sanity) | **0.9730** (HO) |
| Split 3 distilled | **0.9333** (HO) | 0.9455 (T-sanity) | 0.7752 (T-sanity) | **0.7719** (HO) | 0.9730 (T-sanity) |

JAB is invariant (0.9730 → 1 FP that survives all skill banks). The other projects show a clear pattern: distilled F1 ≤ axiom F1 *on every project except teammates in split 2*. The mechanism *only ever wins on a project that is held-out AND of similar architectural style to a training project*.

## 2. Did Voyager-style transfer hold across splits? — **No, it is split-fragile.**

The directive's central question was: *does the +0.16 pp transfer from the prior Phase 12 pilot replicate, and does it hold under different train/test cuts?*

Three layered answers:

### 2.1 Replication on the original split — YES, with larger effect (+0.66 pp vs +0.16 pp)

Split 1 (fresh-start gpt-5.4, no Claude warm-start) reproduces the direction of the prior pilot's lift. The magnitude is larger because the fresh-start axiom floor is lower (0.8658 vs 0.8830). Distilled bank is 6 rules; prior pilot's was also 6 rules; rule content is broadly similar (service-facade-controller approval + COREF resolution + processor/utility blocking). Replication is solid.

### 2.2 Acid test (BBB-in-train) — YES, BBB-derived skills transfer to teammates (+2.18 pp on TM)

Split 2 is the strongest single-project lift of the entire pilot family. Training on BBB (the hardest domain — 87 sentences, 12 components, error pool of 17-19 FNs) produces connector/broker/gateway-flavoured rules that transfer cleanly to teammates' similar UI/Logic/Storage structure. The hypothesis "harder training → more transfer signal" survives.

### 2.3 Stability test (rotated hold-out) — NO, the mechanism regresses on MS+BBB (-1.92 pp)

Split 3 holds out MS and BBB and trains on TS+TM+JAB. JAB and TS converge immediately (no feedback), so the bank is structurally teammates-derived. The 5 distilled rules are *more generic* than split 1's or split 2's — but the calibration is wrong for MS and BBB. Both held-out projects regress.

**Bottom line:** transfer direction depends on whether the training set vocabulary covers the held-out set's architectural surface. When it does (splits 1, 2), modest positive lift. When it does not (split 3), regression.

## 3. Skill-Bank Differences Across Splits

| Aspect | Split 1 (MS+TS+TM) | Split 2 (MS+TS+BBB) | Split 3 (TS+TM+JAB) |
|---|---|---|---|
| Patterns learned | 9 | 8 | 6 |
| Outer passes to converge | 2 | 1 | 1 |
| LLM calls (training) | 18 | 10 | 8 |
| Vocabulary flavour | service / facade / controller | connector / broker / gateway / SFU | generic principles |
| All in DOC_KNOWLEDGE_JUDGE_RULES? | 7/9 (2 COREF) | 8/8 | 4/6 (2 COREF) |
| GATE-06 rejection rate | 57% (12/21) | 20% (2/10) | 33% (3/9) |
| Distilled count | 6 | 6 | 5 |
| Reviewer-defensibility | 6/6 kept | 6/6 kept | 5/5 kept |

**Same kinds of patterns?** *Partially.* All three splits converge on 2 universal axes:
- **Judge conservatism:** "Block link unless sentence assigns explicit responsibility/interface to component."
- **Coref liberalism:** "Propagate component reference through pronouns/anaphora when discourse focus is preserved."

But the *surface vocabulary* of the rules differs by split, and that surface vocabulary determines which held-out documents the rules over- vs under-trigger on. So the rules differ in a *substantively-meaningful-for-F1* way even when they agree at the abstract level.

## 4. Does any split beat trim1?

| System | Macro all-5 (best of 3 splits if applicable) |
|---|---|
| s_linker13_clean (Phase 10 canonical, gpt-5.4) | 0.9077 |
| trim1 distilled judge rules (Phase 12-03, gpt-5.4) | 0.9173 |
| Voyager v2 split 1 distilled (gpt-5.4) | 0.8891 |
| Voyager v2 split 2 distilled (gpt-5.4) | 0.8887 |
| Voyager v2 split 3 distilled (gpt-5.4) | 0.8798 |
| **Best Voyager v2** | **0.8891 (split 1)** |
| **Δ vs trim1** | **−2.82 pp** |

**No.** Best Voyager v2 macro is 2.82 pp below trim1. Hand-distilled (trim1) judge rules from Phase 12 outperform LLM-distilled rules learned from FP/FN feedback at every split tested. This confirms the Phase 12 pilot verdict: *Voyager-style curriculum learning and v2.1's hand-distilled prompts are competing for the same effect, and the hand-distilled version wins.*

## 5. Mechanism Diagnosis (why split-fragile?)

Three contributing factors emerged from the per-iter intermediate dumps:

### 5.1 Feedback signal is dominated by the noisiest training project

| Split | Project that contributed most feedback iters | Bank size at split end |
|---|---|---|
| 1 | teammates (5/9 iters had feedback) | 9 |
| 2 | bigbluebutton (3/4 iters had feedback) | 8 |
| 3 | teammates (3/4 iters had feedback) | 6 |

When the noisiest project is *teammates*, the bank inherits teammates' surface vocabulary (service/facade/logic/storage). When it's BBB, the bank inherits BBB's vocabulary (connector/SFU/pub-sub). The other training projects often hit early-stop (F1 ≥ 0.95) without producing feedback, so they don't anchor the rules to a third domain.

### 5.2 GATE-06 + reviewer-defensibility filter LEXICAL leakage, not DISTRIBUTIONAL bias

Every distilled rule across all 3 splits passes both gates. None of them mention benchmark terms; all read as universal architecture principles. Yet split 3 still regresses by 1.92 pp. **The gates catch overfitting at the word level but not at the calibration level.** A rule like "approve link to facade/controller when sentence identifies it as locus of behaviour" is universal vocabulary but *empirically miscalibrated* for documents where the controller language is incidental rather than load-bearing (e.g., MS's MediaManagement description in the split 3 hold-out).

### 5.3 Convergence threshold is reached BEFORE saturating feedback signal

Two of three splits converged in 1 outer pass (4-5 inner iters total). Split 3's bank has only 6 patterns; the 0.90 macro threshold was reached too early to accumulate broader coverage. A higher threshold (e.g., 0.95) would force more outer passes, but past Phase 12 evidence shows teammates F1 plateaus around 0.83 — running more outer passes won't lift it. There's no easy fix in the loop's stopping rule.

## 6. Defensibility Audit (paper-readiness)

All 3 splits have:
- Frozen `distilled_skills.json` + matching pkl.
- Full per-iter intermediate state (json + pkl) saved.
- Full feedback prompts + raw LLM responses saved verbatim.
- Full distill + reviewer prompts + raw responses saved verbatim.
- GATE-06 audit log (rejected_patterns.json) with per-pattern reasons.
- Reviewer-defensibility log (reviewer_call.json) with per-pattern verdicts.

The pilot is fully auditable and reproducible. A reviewer could, from the on-disk artefacts:
- Replay any single iteration's LLM call by reading `iter_<N>_feedback_call.json:feedback_prompt`.
- Verify GATE-06 by re-running the taboo regex on each pattern.
- Inspect every accepted/rejected/dropped pattern with its reason code.
- Compare per-split bank vocabulary by diffing `skill_bank.json` between splits.

## 7. Recommendation for v2.2

**Do NOT promote Voyager-style curriculum as a v2.2 shipping plan.** Justifications:

1. **Best result still underperforms trim1** by 2.82 pp (Phase 12 pilot's distilled judge baseline already shipped).
2. **Mean held-out lift is zero** across 3 splits (range [-1.92, +1.09]). The +0.66 pp from the original split is split-dependent.
3. **Split-fragility is not paper-defensible** — promoting a mechanism that regresses on 1/3 random splits would require justifying which splits "count" — methodologically thin.

**Possible v2.2 directions that ARE supported by this pilot's data:**

- **Verifier-driven iteration** (already in deferred queue): the per-iter feedback loop is a generic primitive; could be applied to a per-decision verifier rather than a pattern-bank curriculum.
- **Backend-adaptive prompts** (ADAPTER-01, deferred): the gpt-5.4 vs Claude vocabulary asymmetry seen in Phase 12 + 13 also appears here in the GATE-06 rejection-rate per-split swing (20% – 57%).
- **Self-Refine on accepted variants** (deferred): the per-iter regression we saw on teammates (F1 0.84 → 0.81 → 0.75 across patterns) shows that pattern injection can *increase* error before stabilising — a per-link self-refinement loop could catch that transient.

## 8. Deviations from Plan

### Auto-fixed Issues

None across the 3 splits. All runs completed within budget cap (72/200 calls, 46/360 min).

### Honest non-deviations

- Each split started with EMPTY bank (verified by inspecting each split's `split_config.json:fresh_start=true` plus first iter row's `skills_before=0`).
- prompts_v3_axiom.py untouched.
- No shipping artefacts modified.

## 9. Budget Accounting

| Split | Calls used | Calls budget | Elapsed (s) | Wall budget (s) |
|---|---|---|---|---|
| 1 | 30 | 200 | 1179 | 21600 |
| 2 | 22 | 200 | 792 | 21600 |
| 3 | 20 | 200 | 799 | 21600 |
| **Total** | **72** | 600 | **2770** | 64800 |

72 LLM calls is well under the directive's $80 cap (assuming ~$0.20/gpt-5.4 call → ~$14.40; even at $1/call we'd be at $72). 46 min wallclock vs the 6h cap. Budget was not a binding constraint at any point.

## 10. Threat Flags

None. No code surface changes outside the new sibling script + fresh output tree. The v1 pilot script and all v2.1 production artefacts are untouched.

## 11. Self-Check: PASSED

- `results/voyager_pilot_v2/crossplit_comparison.{json,pkl}` — exists, 3 split rows.
- All 3 splits have complete per-split artefact set (verified per-split SUMMARY self-checks).
- `scripts/voyager_train_tlr_v2.py` — exists, supports `--split-id`, `--fresh-start`, `--save-intermediate`, `rollup` phase.
- All 3 per-split SUMMARYs exist at `.planning/v2.2-prep/voyager-v2-split{1,2,3}-summary.md`.
- Cross-split rollup table reproduces from on-disk `test_results.json` of each split.
- Mean held-out delta computed from on-disk values: (+0.67 + 1.09 − 1.92) / 3 = −0.053 pp.

All claims verified against on-disk artefacts.
