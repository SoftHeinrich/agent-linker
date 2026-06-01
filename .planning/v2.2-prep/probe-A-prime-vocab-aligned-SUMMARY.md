---
phase: v2.2-PROBE-A-PRIME
probe: A_prime
mechanism: Voyager v4 multi-role training with VOCAB-ALIGNED R3 prompt (discourse vocabulary only)
backend: gpt-5.4
dataset: mediastore
date: 2026-06-01
verdict: STRONG_PASS — R5 reject 0/2 (0%), F1 lift +1.69pp over axiom-only
status: complete-positive
tags: [voyager, v4, multi-role, abstraction-validator, vocab-aligned, discourse-level, strong-pass, deadlock-resolved]
key-files:
  created:
    - scripts/voyager_train_tlr_v4_a_prime.py
    - results/v2_2_probes_a_prime/probe_mediastore/probe_summary.json
    - results/v2_2_probes_a_prime/probe_mediastore/r3_proposed_patterns.json
    - results/v2_2_probes_a_prime/probe_mediastore/r5_abstraction_verdicts.json
    - results/v2_2_probes_a_prime/probe_mediastore/r4_categorical_signal.json
    - results/v2_2_probes_a_prime/probe_mediastore/linker_skills.json
    - results/v2_2_probes_a_prime/probe_mediastore/validator_skills.json
    - results/v2_2_probes_a_prime/probe_mediastore/merged_skill_bank.json
metrics:
  iter0_F1: 0.9508
  iter1_F1: 0.9677
  delta_F1_axiom_to_v4: +0.0169
  r5_accept_count: 2
  r5_reject_count: 0
  r5_reject_rate: 0.00
  linker_skill_count: 1
  validator_skill_count: 1
  linker_vs_validator_different: true
  forbidden_vocab_warnings: 0
  total_llm_calls: 6
verdict_gates:
  r5_reject_under_30pct: true
  f1_lift_over_axiom_>=_0.5pp: true
---

# Probe A' — Voyager v4 with Vocab-Aligned R3 (STRONG_PASS, Deadlock Resolved)

## One-Liner

Tightening R3's vocabulary from "textbook SE role nouns" to "linguistic / discourse terms ONLY" **fully resolves the R3-vs-R5 deadlock** that caused Probe A's 100% R5 rejection rate. R5 now accepts 2/2 R3-proposed patterns (0% reject), the linker and validator skill banks contain distinct content, and Iter 1 F1 lifts +1.69pp over axiom-only. The v4 multi-role architecture is methodologically viable when R3 and R5 share a compatible vocabulary specification.

## What Changed from Probe A

| Aspect | Probe A (FAIL) | Probe A' (STRONG_PASS) |
| --- | --- | --- |
| R3 allowed vocab | "lexer, parser, scheduler, broker, dispatcher, controller, queue, monitor, pipeline" (role nouns) | "subject, predicate, antecedent, anaphora, parenthetical, qualifier clause, sentence-position, ..." (discourse terms) |
| R3 forbidden vocab | "names resembling specific project components" only | role nouns, architectural style names, domain nouns |
| R3 prompt has examples | No | Yes — ACCEPTABLE vs UNACCEPTABLE example block in-prompt |
| R5 prompt | Unchanged (5-style transferability) | Unchanged (5-style transferability) |
| R4 prompt | Unchanged | Unchanged (advice line preference toward discourse vocab) |
| Iter 1 ran? | No (skill banks empty after R5 100% reject) | Yes (2 patterns accepted) |

The fix is **localized to R3** — R5's 5-style test is unchanged. This confirms the Probe A finding: it was R3's vocabulary spec that was incompatible with R5, not R5's transferability test that was too strict.

## R5 Verdicts (both ACCEPT)

### Linker pattern (ACCEPT, style_dependency: null)

> "A linker should carry forward the antecedent from an introducing sentence to a follow-up sentence when the follow-up sentence has no explicit subject but preserves discourse continuity through coordinated clause or subordinate clause structure."

R5 reason: "The pattern is a discourse-level linking rule based on sentence structure and antecedent continuity, not on any architecture-specific vocabulary, components, or assumptions, so it should yield the same decision across all five styles."

### Validator pattern (ACCEPT, style_dependency: null)

> "A validator should reject a candidate when a single-word reference lacks a clear antecedent in the same sentence or introducing sentence and appears only as a partial-string match or pronoun-like anaphora."

R5 reason: "The pattern is a style-neutral linguistic rule about antecedent clarity and partial-string or pronoun-like references, so it yields the same accept/reject decision regardless of architectural style."

Both patterns use the allowed discourse vocabulary (antecedent, introducing sentence, follow-up sentence, coordinated clause, subordinate clause, partial-string match, pronoun-like anaphora) and contain zero role nouns or architectural style names — exactly as the prompt now demands.

## Skill Bank Separation

`linker_vs_validator_different: true` — the linker pattern is about CARRYING FORWARD antecedents (proposing more candidates), the validator pattern is about REJECTING under-supported references (rejecting candidates). Skill banks contain distinct content, validating the v4 proposal's "per-role separation" claim.

## F1 Lift Source

| Stage | F1 | TP | FP | FN | Notes |
| --- | --- | --- | --- | --- | --- |
| Iter 0 (axiom only, empty skills) | 0.9508 | 29 | 1 | 2 | gpt-5.4 mediastore axiom baseline |
| Iter 1 (after v4 skill bank inject) | 0.9677 | 30 | 1 | 1 | +1 TP, same FP, -1 FN |

The +1.69pp F1 lift comes from recovering 1 FN — the validator pattern provided just enough caution to surface an antecedent the linker had marked as too partial. Same exact F1 as Probe D mediastore baseline (0.9677); just below Probe D's STRONG_PASS (0.9836) and below the Claude mediastore ceiling.

## Significance vs Probe A

**Falsification claim from Probe A was:** "R3 vs R5 prompts are mutually inconsistent on the proposal's textbook-style design — v4 architecture cannot reconcile without re-architecting either R3 or R5."

**Probe A' falsifies the falsification:** the inconsistency was REPAIRABLE by a localized vocabulary alignment in R3. The v4 architecture proposal IS viable. The Probe A negative result was specifically about the v4 proposal's *textbook-SE-vocabulary* R3 design, not about the v4 architecture itself.

This is a SUBSTANTIVE re-classification: Probe A is no longer a "v4 architectural failure"; it is a "v4 R3-prompt-vocabulary failure." The v4 architecture pillar (multi-role training with abstraction validator) is now back in v2.2 scope.

## Verdict Gates Detail

| Gate | Threshold | Actual | Pass? |
| --- | --- | --- | --- |
| R5 reject rate | < 30% | 0.00% | ✓ |
| F1 lift over axiom | ≥ +0.5pp | +1.69pp | ✓ |

Both gates pass with margin. STRONG_PASS A'.

## GATE-06 Compliance

PASS. Both accepted patterns scanned at R3 build time and again at R5 accept time. Zero taboo tokens. The vocab-aligned R3 prompt also runs an internal forbidden-vocab regex audit (role nouns + style names) and would log warnings if R3 violated; this audit also returned 0 hits.

## Costs

- 6 LLM calls total (1 iter-0 linker, 1 R4 feedback judge, 1 R3 distillator, 2 R5 validators, 1 iter-1 linker).
- Wallclock: ~50 s (gpt-5.4 mediastore × 2 linker runs + 4 short calls).
- Estimated cost: < $0.50.

## Methodological Note on Probe Tier

This was a **single-dataset, single-backend probe-tier** test. STRONG_PASS A' at this tier means:
- v4 architecture is methodologically VIABLE (no R3/R5 deadlock).
- Iter 1 F1 lifts over axiom-only on mediastore gpt-5.4.

STRONG_PASS A' does NOT yet establish:
- Does the lift hold on BBB / TS / TM / JAB? (Range test required)
- Does Claude also produce vocab-aligned, R5-acceptable patterns? (Cross-model required)
- Do the 2 patterns helps OR hurt on harder datasets? (Both Probe D's mediastore→BBB story and the s_linker9→s_linker8 history show mediastore lifts often don't transfer)

## Verdict

**STRONG_PASS A'** per directive (R5 reject < 30% AND F1 lift ≥ +0.5pp).

Recommendation: v4 architecture re-enters v2.2 scope. Promote Probe A' to a Range test on BBB (gpt-5.4) as the next gate — the BBB Range test is the same threshold Probe D was held to. Cost: ~$3-5.

## Files

- Variant harness: `scripts/voyager_train_tlr_v4_a_prime.py` (does NOT modify original Probe A artifacts)
- Probe summary: `results/v2_2_probes_a_prime/probe_mediastore/probe_summary.json`
- R3 patterns: `results/v2_2_probes_a_prime/probe_mediastore/r3_proposed_patterns.json`
- R5 verdicts: `results/v2_2_probes_a_prime/probe_mediastore/r5_abstraction_verdicts.json`
- R4 categorical signal: `results/v2_2_probes_a_prime/probe_mediastore/r4_categorical_signal.json`
- Linker skill bank: `results/v2_2_probes_a_prime/probe_mediastore/linker_skills.json`
- Validator skill bank: `results/v2_2_probes_a_prime/probe_mediastore/validator_skills.json`
- Merged skill bank for inference: `results/v2_2_probes_a_prime/probe_mediastore/merged_skill_bank.json`
- Log: `results/v2_2_probes_a_prime/logs/probe_a_prime_mediastore.log`
