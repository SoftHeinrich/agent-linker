# Voyager-TLR Train-Test Pilot — RESUMED AND COMPLETED on gpt-5.4 (2026-06-01)

## RESUMED AND COMPLETED — 2026-06-01

**See:** [`12-VOYAGER-PILOT-GPT-SUMMARY.md`](./12-VOYAGER-PILOT-GPT-SUMMARY.md) for full results, methodology, central-question answer, and held-out F1 vs all baselines.

**Headline numbers (gpt-5.4 only, no Claude in resumption):**

| Held-out (BBB + JAB) | Macro F1 |
|---|---|
| Voyager distilled (6 rules) | **0.8846** |
| Voyager axiom-only floor | 0.8830 |
| s_linker13 gpt-5.4 baseline | 0.8884 |
| trim1 gpt-5.4 baseline | 0.8883 |
| trim9 gpt-5.4 baseline | 0.8774 |

**Verdict:** Train-test methodology transfers operationally, but the +0.16 pp lift over the axiom-only floor is within gpt-5.4 run-to-run variance and below the shipped Phase 12 distilled baselines. Skills survive frozen test-time injection without overfit collapse; per-dataset variance dominates (JAB +2.7 pp, BBB −2.4 pp).

**Total cost of resumption:** 18 LLM calls, ~12.5 min wallclock, 0 taboo hits across both skill bank (10) and distilled bank (6).

**Commits:** `2c21298` (--resume support). Resumption artefacts in `results/voyager_pilot/` (gitignored, on-disk only).

---

## Original deferral note (preserved for history)

**Stopped:** 2026-05-31
**Reason:** Claude API usage budget — prioritize gpt-5.4-driven exploration first per user directive.

## What Was Built

Phase 0 (axiom prompts + variant) + Phase 1 (training script) shipped before halt:

| Artifact | Path |
|---|---|
| Axiom-only prompts | `src/llm_sad_sam/linkers/experimental/prompts_v3_axiom.py` |
| Skill-learned variant | `src/llm_sad_sam/linkers/experimental/s_linker13_skill_learned_clean.py` |
| Training script | `scripts/voyager_train_tlr.py` |
| Skill bank seed | `results/voyager_pilot/skill_bank.json` (empty) |
| Run log dir | `results/voyager_pilot/run_log/` |

Commits: `d7b9fba` (Phase 0), `4965c87` (Phase 1 script).

## What Was Not Run

- Phase 1 training loop never executed end-to-end (halted mid-mediastore Round 1).
- Phase 2 distillation step not run.
- Phase 3 test-on-held-out evaluation not run.
- No skill bank entries learned.
- No verdict on whether the train-test methodology transfers to TLR.

## Resumption Path

When budget allows:

```bash
python scripts/voyager_train_tlr.py \
  --train mediastore teastore teammates \
  --test bigbluebutton jabref \
  --max-inner-iters 3 \
  --max-outer-passes 3 \
  --convergence-threshold 0.93 \
  --skill-bank-path results/voyager_pilot/skill_bank.json
```

Hard budget cap: $50 Claude, 6h wallclock (per original pilot brief).

## Why This Matters

The Voyager-shaped methodology is the canonical answer to two open questions left by Phase 12:

1. **Cross-dataset abstract knowledge accumulation** — does training-on-3 teach the model patterns that transfer to held-out 2 without GATE-06 leakage?
2. **Train/test separation for TLR design space** — current Phase 12 evaluates on ALL 5 datasets always; train/test methodology gives a stronger generalization claim.

Both questions are central to the v2.2 milestone. The infrastructure built here (axiom prompts + skill-learned variant + training script) is reusable; only the LLM-burn portion is deferred.

## Recommendation

Carry to **v2.2 first plan**. By then either:
- Claude usage budget restores, or
- Equivalent train-test pattern is testable on gpt-5.4 (cheaper) first as a feasibility check before paying Claude cost.
