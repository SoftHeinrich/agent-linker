# Voyager-TLR Train-Test Pilot — DEFERRED

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
