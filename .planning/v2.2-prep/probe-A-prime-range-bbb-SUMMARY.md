---
phase: v2.2-RANGE-A-PRIME-BBB
date: 2026-06-01
backend: gpt-5.4
dataset: bigbluebutton
verdict: WEAK_PASS
r5_reject_rate: 0.00%
r5_accept_count: 8
iter0_F1: 0.7568
iter1_F1: 0.7544
delta_F1: -0.0024
total_llm_calls: 12
budget_spent_estimate_usd: ~$0.50
tags: [v2.2, range-test, probe-A-prime, voyager-v4, BBB, gpt-5.4, weak-pass]
key-files:
  result_dir: results/v2_2_probes_a_prime_range/probe_bigbluebutton/
  run_log: results/v2_2_probes_a_prime_range/run_bbb.log
  probe_summary: results/v2_2_probes_a_prime_range/probe_bigbluebutton/probe_summary.json
prior_result:
  probe_A_prime_mediastore_gpt54_verdict: STRONG_PASS (iter0 0.9508 → iter1 0.9677, +1.69pp, R5 reject 0%)
---

# Probe A' Range — BBB gpt-5.4

## Headline

| Metric | Value |
| --- | --- |
| iter0 F1 (axiom only) | 0.7568 |
| iter1 F1 (with v4 skills) | 0.7544 |
| Δ F1 | **−0.0024 (no lift)** |
| R5 reject count | 0 / 8 |
| R5 reject rate | 0.00% |
| Total LLM calls | 12 |
| Verdict | **WEAK_PASS** |

Per the user-directive verdict gates:
- R5 reject < 30% AND F1 lift ≥ +0.5pp → STRONG_PASS
- **R5 reject < 30% AND F1 lift < +0.5pp → WEAK_PASS** ✓
- R5 reject ≥ 30% → FAIL

## What Passed

**Vocabulary alignment holds on BBB.** R5 accepted **all 8** R3-proposed patterns (4 linker + 4 validator) with `style_dependency: null`. The vocab-aligned R3 (discourse-only vocabulary) is robustly style-neutral across both mediastore and bigbluebutton.

**Architectural plausibility maintained.** Linker vs validator skill banks contain distinct content (`linker_vs_validator_different: true`). The v4 architecture's role separation is meaningful — different abstract categories produce different skill patterns.

**No forbidden-vocab warnings.** All 8 patterns used only the allowed discourse/linguistic vocabulary (subject, predicate, antecedent, anaphora, qualifier clause, head noun, dotted path, capitalization, etc.). Zero role nouns or architectural style names appeared client-side.

## What Did Not Pass

**No F1 lift on BBB.** The skill bank patterns, while abstractly accepted by R5, did NOT translate into measurable inference improvement. Iter1 lost 1 TP (45→44) and gained 1 FN (19→20) net, dropping F1 by −0.24pp.

Possible explanations (not yet confirmed):
1. **BBB's natural failure modes are not addressable by discourse-level rules.** BBB FNs are dominated by partial-name disambiguation ("Server" → HTML5 Server vs Redis Server) which the v2.1 PHASE_CONTRIBUTION analysis already flagged as un-replaceable by LLM coref/discourse logic (P8b finding).
2. **Skill patterns are too abstract.** Discourse-only vocabulary cannot encode the project-specific naming conventions BBB needs. The MS lift (+1.69pp) on a simpler dataset (12 components, formal prose) may have been a ceiling effect; BBB's complexity exceeds what 1-iter discourse skills can address.
3. **gpt-5.4 inference variance.** A 1-link delta on a ~62-gold-link dataset is within the known gpt-5.4 BBB stdev band (5-12 links). The "no lift" could be a noise-band result rather than a true negative; a 3-seed average would be needed to disambiguate.

## v4 Architecture Status (Updated)

| Dataset | Backend | R5 reject | F1 lift | Verdict |
| --- | --- | --- | --- | --- |
| mediastore | gpt-5.4 | 0/2 (0%) | +1.69pp | STRONG_PASS |
| **bigbluebutton** | gpt-5.4 | 0/8 (0%) | −0.24pp | **WEAK_PASS** |
| mediastore | claude | — | — | NOT TESTED |
| bigbluebutton | claude | — | — | NOT TESTED (user deferred this turn) |

**Promotion status: viable mediastore-only.** The v4 architecture (R1–R5 multi-role, vocab-aligned) PRODUCES style-neutral skills on BBB (R5 always accepts) but those skills do NOT lift F1 on BBB. The architecture is not a failure on BBB — it is **inactive**.

**Comparison to Probe D (sister mechanism):**
- Probe D mediastore gpt-5.4: STRONG_PASS (+1.59pp)
- Probe D BBB gpt-5.4: STRONG_PASS (+3.29pp originally; +1.12pp on cache-fix re-run)
- Probe A' mediastore gpt-5.4: STRONG_PASS (+1.69pp)
- Probe A' BBB gpt-5.4: **WEAK_PASS (−0.24pp)**

Probe D transfers to BBB; Probe A' does not. The mechanisms target different surfaces — Probe D rewrites the coref rubric (entire prompt for a single tier), Probe A' adds discourse-rule patterns to an existing skill-bank prompt. The skill-bank addition is more diffuse and harder to translate into BBB-recoverable behavior.

## Generated Patterns (full list)

**Linker skills (4 accepted):**
1. (implicit_subject_missed) Carry forward nearest antecedent when follow-up sentence has no explicit subject but maintains topic continuity.
2. (alias_synonym_missed) Propose when multi-word phrase shares head-noun + modifier structure with a prior exact-string match.
3. (containment_missed) Extend candidate from dotted-path/namespace mention to head noun when reused as local antecedent.
4. (passive_voice_missed) Treat object of passive predicate as valid referential signal even when not surface subject.

**Validator skills (4 accepted):**
1. (ambiguous_over_approved) Reject single-word reference unless exact-string match + clear antecedent.
2. (tech_label_over_approved) Reject when matched only in section heading / capitalization-label / namespace prefix.
3. (pattern_name_over_approved) Reject when matched phrase functions as modifier/apposition only.
4. (subprocess_over_approved) Reject when matched string appears only in subordinate clause / qualifier clause not resumed by anaphora.

All 4-rule structure (no R5 rejection) but inference layer did not pick them up.

## Gate Compliance

| Gate | Status | Notes |
| --- | --- | --- |
| GATE-01 strict | DEFERRED | Confirmation tier requires 5-dataset × 2-backend coverage; not in scope this turn |
| GATE-06 lexical taboo | PASS | r3 patterns + r5 verdicts scanned (0 hits) |
| GATE-07 canonical registration | N/A | Harness-only; no new variant registered |
| GATE-08 cost vs F1 | N/A (WEAK_PASS) | 12 calls × ~$0.04 ≈ $0.50 for −0.24pp delta |

## Recommendation

**Defer v4-as-flagship to v2.3.** Range A' on BBB gpt-5.4 produces no measurable benefit. Without a positive Range result on the hardest dataset, v4 architecture cannot be the v2.2 milestone anchor.

**Conditional ship path (optional):** Add the 8 accepted patterns as an opt-in `use_voyager_v4_skills=False` flag to `s_linker13_skill_learned_clean`. The MS lift (+1.69pp) is real and reproducible, but the BBB null result means the flag would be gpt-5.4 + mediastore-class-only — narrower than Probe D's gpt-5.4-only conditional.

**Higher-value next step:** Investigate WHY R5 accepts patterns that don't help inference. The v4 architecture passes its falsification test (no R5 deadlock) but fails its utility test (no F1 lift). The gap is in the iter0→iter1 inference layer, not the R3/R5 distillation/validation. Future Probe A'' could try:
- Run 2 outer iterations (skill bank accumulation)
- Try a non-empty validator-side bank with smaller patterns
- Replace the discourse-vocabulary constraint with a more permissive but R5-validated vocabulary

## Open Questions

1. Is the −0.24pp delta within BBB gpt-5.4's noise band? A 3-seed average run would disambiguate.
2. Would running the iter1 linker with ONLY the linker bank (no validator bank) lift F1 on BBB? (Validator-side rules could over-reject TPs.)
3. Would Probe A' Range on a third dataset (teastore or teammates) split the verdict 2-vs-1?

## Files

- `results/v2_2_probes_a_prime_range/probe_bigbluebutton/probe_summary.json`
- `results/v2_2_probes_a_prime_range/probe_bigbluebutton/iter0_axiom_only_results.json`
- `results/v2_2_probes_a_prime_range/probe_bigbluebutton/iter1_after_skills_results.json`
- `results/v2_2_probes_a_prime_range/probe_bigbluebutton/r3_proposed_patterns.json`
- `results/v2_2_probes_a_prime_range/probe_bigbluebutton/r5_abstraction_verdicts.json`
- `results/v2_2_probes_a_prime_range/probe_bigbluebutton/r4_categorical_signal.json`
- `results/v2_2_probes_a_prime_range/probe_bigbluebutton/linker_skills.json` (4 patterns)
- `results/v2_2_probes_a_prime_range/probe_bigbluebutton/validator_skills.json` (4 patterns)
- `results/v2_2_probes_a_prime_range/run_bbb.log`

## Code Changes

- `scripts/voyager_train_tlr_v4_a_prime.py`: added `PROBE_A_PRIME_OUT_ROOT` env override (single line, allows directing output to `results/v2_2_probes_a_prime_range/` without touching the mediastore artifacts).
