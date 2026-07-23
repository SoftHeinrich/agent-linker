---
phase: v2.2-prep
plan: voyager-v2-split3
subsystem: prompt-curriculum
tags: [voyager, train-test, gpt-5.4, fresh-start, split3, rotated-holdout, regression]
key-files:
  created:
    - results/voyager_pilot_v2/split3_rotated_holdout/skill_bank.{json,pkl}
    - results/voyager_pilot_v2/split3_rotated_holdout/distilled_skills.{json,pkl}
    - results/voyager_pilot_v2/split3_rotated_holdout/rejected_patterns.{json,pkl}
    - results/voyager_pilot_v2/split3_rotated_holdout/distill_call.{json,pkl}
    - results/voyager_pilot_v2/split3_rotated_holdout/reviewer_call.{json,pkl}
    - results/voyager_pilot_v2/split3_rotated_holdout/test_results.{json,pkl}
    - results/voyager_pilot_v2/split3_rotated_holdout/train_trajectory.{json,pkl}
    - results/voyager_pilot_v2/split3_rotated_holdout/split_config.{json,pkl}
    - results/voyager_pilot_v2/split3_rotated_holdout/iter_states/iter_*_{state,feedback_call,predictions_vs_gold}.{json,pkl}
  modified: []
decisions:
  - "Split 3 (train TS+TM+JAB, test MS+BBB) is a REGRESSION: distilled macro 0.8526 vs axiom 0.8718 = -1.92 pp on held-out. Both held-out projects degrade (MS -1.75 pp, BBB -2.09 pp adding 3 FPs). This is the split-fragility result."
  - "Smaller bank (6 patterns vs split 1's 9 / split 2's 8) — TM was the only feedback-rich training project; JAB and TS converge in 1 iter and contribute almost no patterns. The bank ends up TM-skewed and TM-skewed-skills over-trigger on MS and BBB."
  - "Voyager-style transfer is SPLIT-FRAGILE. Two of three splits show a positive held-out lift (+0.66 pp, +1.09 pp); one shows a clear regression (-1.92 pp). Mean lift across all 3 splits is -0.06 pp — statistically indistinguishable from zero."
metrics:
  duration_min: 13.3
  total_llm_calls: 20
  total_wallclock_s: 799
  patterns_in_bank: 6
  patterns_in_distilled: 5
  patterns_gate06_rejected: 3
  outer_passes_completed: 1
---

# Voyager-TLR v2 — Split 3 (rotated hold-out) Summary

**One-liner:** Rotating the hold-out to MS+BBB and training on TS+TM+JAB produces a clean REGRESSION: held-out macro drops from axiom 0.8718 to distilled 0.8526 = **−1.92 pp**. Both held-out projects degrade. The 6-pattern bank is dominated by teammates-flavoured feedback (TM was the only training project that produced sustained errors); these rules over-trigger on the unseen MS and BBB documents. Conclusion: Voyager-style transfer is SPLIT-FRAGILE.

---

## 1. Run Context

| Aspect | Value |
| --- | --- |
| Split | 3 — rotated hold-out |
| Train projects | teastore, teammates, jabref |
| Test projects | mediastore, bigbluebutton |
| Backend | OpenAI gpt-5.4 (all 20 calls) |
| Starting bank | EMPTY |
| Wallclock | 13.3 min (799 s) |
| LLM calls | 20 (training: 5 linker + 3 feedback = 8; distill: 1; reviewer: 1; test: 10) |
| Convergence | 1 outer pass at macro 0.918 (TM 0.8167 + JAB 0.9730 + TS 0.9643) |
| Budget | 200 / 6 h cap — used 20 / 799 s |

## 2. Training Trajectory

| Outer | Inner | Project | F1 | FP | FN | Bank size before |
|---|---|---|---|---|---|---|
| 0 | 0 | teammates | 0.8197 | 15 | 7 | 0 |
| 0 | 1 | teammates | 0.7805 | 18 | 9 | 2 |
| 0 | 2 | teammates | 0.8167 | 14 | 8 | 4 |
| 0 | 0 | jabref | **0.9730** | 1 | 0 | 6 |
| 0 | 0 | teastore | 0.9643 | 2 | 0 | 6 |

**Convergence at outer 0** (macro 0.918 ≥ 0.90).

JAB and TS each ran only 1 iter — both above the 0.95 early-stop threshold after just one linker run. Almost all the feedback (and therefore all the patterns) came from teammates. **Skill bank is structurally biased toward the teammates fault profile.**

## 3. Skill Bank (6 patterns, all GATE-06 clean)

| # | Scope | Gist |
|---|---|---|
| 1 | DOC_KNOWLEDGE_JUDGE_RULES | Approve link only when sentence states explicit responsibility/interface/impl artifact/structural role |
| 2 | DOC_KNOWLEDGE_JUDGE_RULES | Block link for sentences that mention only generic tech/platform/conceptual package without explicit responsibility |
| 3 | DOC_KNOWLEDGE_JUDGE_RULES | Approve when reference to controller/facade/service/API/domain object identifies locus of behavior or structure |
| 4 | COREF_RULES | Resolve pronoun to nearest previously introduced component while discourse focus is preserved |
| 5 | COREF_RULES | Propagate component link through anaphoric chains while subject continuity is preserved |
| 6 | (unspecified — dedupe-dropped) — see distilled output for the surviving 5 |

### 3.1 GATE-06 audit

| Source | Patterns | Taboo hits |
|---|---|---|
| Final skill bank | 6 | **0** |
| Final distilled bank | 5 | **0** |
| Patterns rejected during training | 3 | terms: `persistence` (2), `datastore` (1) |

GATE-06 rejection rate 33% (3/9), in between split 1 (57%) and split 2 (20%). Teammates' storage vocabulary still drags some patterns into taboo territory.

## 4. Distillation (5 universal rules)

Distillation kept all but one of the patterns (one dropped as a near-clone of pattern 3). The 5 surviving rules are visibly more *generic* than splits 1 or 2:

1. Link only when sentence states explicit responsibility/interface/impl artifact/structural role.
2. Don't link for generic tech/platform/conceptual package descriptions.
3. Link to controller/facade/service/API/domain object when locus of behavior/structure.
4. Resolve pronouns to nearest discourse-focus component.
5. Propagate component link through anaphoric chains under subject continuity.

Reviewer-defensibility critic: all 5 rated `defensible`. The criteria themselves are reasonable as universal rules — *the problem is they are too generic to produce a positive lift on documents whose surface vocabulary differs from teammates'*. The rule "link to facade/service/controller/API" is exactly what mediastore's MediaManagement and BBB's Recording Service get over-mapped onto.

## 5. Held-out Evaluation

| Project | Split | s_linker13_clean | Axiom-only | Distilled | Δ vs axiom |
|---|---|---|---|---|---|
| **mediastore** | **test** | 0.9836 | 0.9508 | **0.9333** | **−1.75 pp** |
| **bigbluebutton** | **test** | 0.8036 | 0.7928 | **0.7719** | **−2.09 pp** |
| teastore | train_sanity | 1.0000 | 0.9818 | 0.9455 | −3.64 |
| teammates | train_sanity | 0.9381 | 0.8197 | 0.7752 | −4.45 |
| jabref | train_sanity | 0.9730 | 0.9730 | 0.9730 | 0.00 |
| **Macro all-5** | — | 0.9396 | 0.9036 | 0.8798 | **−2.39 pp** |
| **Macro held-out** | test | 0.8936 | 0.8718 | **0.8526** | **−1.92 pp** |

**Regression on every single project except jabref.** The distilled rules add 3 FPs on BBB and 1 FN on mediastore (Apr stricter coref propagation rule blocks a previously-correct link).

## 6. Central Question (split 3 isolated)

**Did the Voyager mechanism transfer in this split?** *No — it actively hurt.* Both held-out projects regress, both train-sanity projects (TS, TM) also regress.

**Why?** Two compounding factors:
- **Bank is teammates-skewed.** Only TM produced sustained feedback (3 iters). The 6 patterns reflect TM-specific FP/FN profiles (15 raw TM FPs from outer-0 iter 0 dominate the feedback signal).
- **Patterns over-generalised.** Rules 1, 2, 3 are abstract enough to *apply* to MS and BBB documents, but their calibration is wrong for those domains. Rule 3 ("link to controller/facade/service/API") fires on MS's MediaManagement and BBB's Recording Service surfaces that gold doesn't trace.

The 5 distilled rules pass GATE-06 and reviewer-defensibility, yet still produce a regression. **GATE-06 and reviewer-defensibility are necessary but not sufficient — they catch leakage but not over-generalisation.**

## 7. Deviations from Plan

### Auto-fixed Issues

None.

### Honest non-deviations

- Empty starting bank (verified).
- prompts_v3_axiom.py untouched.

## 8. Threat Flags

None.

## 9. Self-Check: PASSED

- Skill bank: 6 patterns, 0 taboo hits, pkl present.
- Distilled bank: 5 patterns, 0 taboo hits, pkl present.
- Rejected log: 3 entries with reasons.
- Distill / reviewer call captures: prompt + raw response saved.
- Test results: 5 projects, macros computed.
- Train trajectory: 5 rows.
- Per-iter intermediate dumps: 5 state + 3 feedback + 5 predictions JSON+PKL pairs.
- Split config: `fresh_start=true`, `save_intermediate=true`.
