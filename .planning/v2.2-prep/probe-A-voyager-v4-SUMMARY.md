---
phase: v2.2-PROBE-WAVE
probe: A
mechanism: Voyager v4 multi-role training (R1-R5)
backend: gpt-5.4
dataset: mediastore
date: 2026-06-01
verdict: PROBE_FAIL — R5 rejects 100% of R3 proposals
status: complete-negative
tags: [voyager, v4, multi-role, abstraction-validator, gate-falsification, negative-result]
key-files:
  created:
    - scripts/voyager_train_tlr_v4.py
    - results/v2_2_probes/A_voyager_v4/probe_mediastore/probe_summary.json
    - results/v2_2_probes/A_voyager_v4/probe_mediastore/r3_proposed_patterns.json
    - results/v2_2_probes/A_voyager_v4/probe_mediastore/r5_abstraction_verdicts.json
    - results/v2_2_probes/A_voyager_v4/probe_mediastore/r4_categorical_signal.json
metrics:
  iter0_F1: 0.9508
  iter1_F1: null
  delta_F1_axiom_to_v4: null
  r5_accept_count: 0
  r5_reject_count: 2
  r5_reject_rate: 1.0
  total_llm_calls: 5
falsification_hit: r5_rejected_100pct
---

# Probe A — Voyager v4 Multi-Role Architecture (FAIL)

## One-Liner

Voyager v4's R5 Abstraction Validator rejected 100% of R3 Skill Distillator's proposals on mediastore gpt-5.4 — the R3 prompt instructs the distillator to use textbook SE placeholders (parser, lexer, broker, dispatcher, scheduler, controller, queue, monitor, pipeline), but R5 marks those terms as style-dependent and rejects every pattern. This hits the v4 architecture proposal's explicit falsification criterion: "R5 rejects 80%+ of R3's proposals (R3 can't produce abstract enough patterns)." Probe FAILED.

## Setup

- **Variant**: New 5-role harness in `scripts/voyager_train_tlr_v4.py` per `voyager-v4-architecture-proposal.md`.
- **Backend**: gpt-5.4 (per v2.2 directive: gpt-5.4 first to conserve Claude budget).
- **Dataset**: mediastore only (probe scope per v2.2 directive).
- **Skill banks**: BOTH empty at start (fresh-start per directive).
- **Roles**:
  - R1+R2: SLinker13SkillLearned (axiom prompts + skill bank wrap)
  - R3: skill distillator (categorical-only input from R4)
  - R4: feedback judge (gold + abstracted component IDs ONLY; NO doc text, NO component names)
  - R5: abstraction validator (textbook-style library; 5 architectural styles)

## Results

| Metric | Value | Notes |
| --- | --- | --- |
| Iter 0 F1 (axiom-only baseline) | 0.9508 | gpt-5.4 mediastore axiom prompts; matches expected baseline class |
| R3 proposed patterns | 2 | One per role (linker, validator) |
| R5 ACCEPT | 0 | — |
| R5 REJECT | 2 | 100% reject rate |
| Iter 1 F1 | n/a | Skipped — empty skill banks after R5 |
| Δ F1 axiom→v4 | n/a | Cannot measure without iter 1 |
| LLM calls (probe) | 5 | Iter 0 linker + R4 + R3 + R5 × 2 |

## R5 Rejection Pattern

Both rejections cite the SAME root cause: the R3-proposed pattern uses textbook SE role vocabulary (controller, dispatcher, broker, queue, monitor, scheduler, parser, lexer, pipeline), which R5 considers "style-dependent" because some architectural styles (pipe-and-filter, event-sourced) do not necessarily contain those roles.

Sample R5 verdict:

> "The pattern is not style-neutral because it explicitly relies on role terms such as controller, dispatcher, broker, queue, monitor, scheduler, parser, lexer, and pipeline, which are unevenly applicable across the listed architectural styles." (style_dependency: Pipe-and-filter)

## Structural Issue

The R3 prompt explicitly instructs the distillator: "Use textbook SE vocabulary ONLY (lexer, parser, scheduler, broker, dispatcher, controller, queue, monitor, pipeline). NEVER use any name resembling a specific project component."

The R5 prompt asks: "Would this pattern produce the same accept/reject decisions across [microservice mesh, event-sourced, layered monolith, pipe-and-filter, hexagonal]?"

These two prompts are MUTUALLY INCONSISTENT. R3's safe vocabulary is, by R5's standard, NOT style-neutral. The only way to satisfy both would be patterns using ONLY abstract logical predicates like "an alias whose token overlaps multiple components" — but then patterns become so generic that they cease to encode the error category at all.

## Hypothesis Falsified

Per `voyager-v4-architecture-proposal.md` falsification criteria, v4 FAILS if any of:
- 3-split mean held-out lift < 0.5pp (NOT YET TESTED — probe was 1 split, mediastore only)
- **R5 rejects 80%+ of R3's proposals (R3 can't produce abstract enough patterns)** — **HIT (100%)**
- Per-iter cost > 4× v2 without proportional outer-pass reduction (NOT REACHED — probe ended at R5)
- Split 3 still regresses (NOT TESTED)

The 80%+ R5 rejection criterion was the architectural-feasibility gate. v4 fails it on the first try.

## Publishable Negative Result

The R3 vs R5 prompt inconsistency is itself the v4 finding. Two interpretations:

1. **The proposal's R5 design is too strict.** The 5-textbook-style library is too aggressive a transferability test for TLR patterns. A practical R5 should test against a smaller or differently-curated style set.

2. **The proposal's R3 design is under-specified.** The "use textbook vocabulary" instruction is not actually a path to style-neutral patterns — even textbook role vocabulary is style-dependent. R3 would need to emit patterns at a strictly higher abstraction layer (e.g., "an alias whose token overlaps N components"), which loses the error-category specificity that R4 surfaces.

The v4 architecture proposal cannot reconcile (1) and (2) without re-architecting either R3 or R5. **Recommendation: do NOT pursue v4 in v2.2.** Document as publishable negative — role separation alone, with the proposal's R3/R5 prompts, does not converge.

## Cost

- Total LLM calls: 5
- Iter 0 linker on mediastore: ~25 s
- R4 + R3 + 2× R5 = 4 short calls, ~15 s
- Estimated cost: < $1 (gpt-5.4)

## Verdict

**PROBE_FAIL** — do not Range test. Roll up as a publishable v4 architectural insight: R3 vs R5 prompts are mutually inconsistent on the proposal's textbook-style design.

## Open Followups (not in scope for v2.2)

- If user wants to salvage v4: revise R5 to test transferability against TLR-specific error classes rather than architectural styles.
- Alternative: drop R5 entirely (v4-without-R5 = v4 stripped of the abstraction validator). The category-only R4 feedback is still a v3-improvement claim that might survive without R5.
