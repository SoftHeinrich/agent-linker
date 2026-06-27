---
spike: 004
name: nogap-validator-ab
validates: "Given s_linker20_union at reasoning effort 0 (CLAUDE_DISABLE_THINKING=1), when the validation gates use a layered validator (Mode 5 justification scaffold + Mode 1 claim-rubric + Mode 2 taboo-audited trap-list, Mode 4 skeptic on coref survivors only), then macro-F1 recovers toward the thinking-on baseline (~92.8) without spending the ~10 implicit (name_in_text=False) true links"
verdict: PROPOSED
related: [003-llm-mention-classifier]
tags: [validator, no-reasoning, false-positive-filter, ab-test, prompt-schema]
---

# Spike 004: No-reasoning validator A/B (close the gap)

## What This Validates

**Given** `s_linker20_union` run at reasoning effort 0 (`CLAUDE_DISABLE_THINKING=1`)
loses ~3.4 macro-F1 vs thinking-on, almost entirely as false positives (28
nothink-only FPs vs 10 FNs; coref FPs 7→27, entity 25→35),
**when** the Phase 4 (entity) and Phase 5 (coref) validators are replaced with a
layered validator that relocates the deleted reasoning into the prompt/output,
**then** effort-0 macro-F1 climbs toward the thinking-on baseline (~92.8) while the
implicit-link false negatives (`name_in_text=False`) stay flat.

## Hypothesis

Extended thinking was functioning as a false-positive filter at the validation
gates. The same discrimination can be recovered without thinking tokens by
(a) forcing a per-candidate justification field into the output (Mode 5),
(b) gating on an *architectural-claim* rubric rather than name presence (Mode 1),
(c) a short, taboo-audited trap-pattern checklist (Mode 2), and
(d) an adversarial skeptic pass on coref survivors only (Mode 4).

## Method

1. **Baselines already on disk** — no new runs needed for the control:
   - thinking-on: `results/v2.6.5_s20union_sonnet/` (macro 92.8)
   - effort-0 plain: `results/v2.6.5_s20union_sonnet_nothink_20260627/` (macro 89.4)
2. Implement the layered validator behind a variant/flag (keep s_linker20_union
   intact; add an experimental sibling).
3. Run effort-0 + layered validator, N=3, in a fresh non-colliding results dir
   (mirror the dated-folder discipline used for the nothink sweep).
4. Compare against both baselines.

## Success Criteria (the one measurable)

- **Primary:** macro-F1 ≥ ~92.0 at effort-0 (recovers the bulk of the 3.4 gap).
- **Guardrail (must hold):** implicit `name_in_text=False` FNs do **not** increase
  vs the plain effort-0 run — the precision win must not be paid for in recall.
- **Diagnostic:** coref FP count falls from 27 back toward ≤10; teammates FP falls
  from 41 toward ≤20.

## Cost note

Mode 5 + Mode 4 trade thinking tokens for output tokens / an extra pass. Record the
output-token and latency delta so the "cheaper than thinking-on" claim is grounded,
not assumed.

## Open risks

- Mode 1 phrased as "name present" would kill implicit true links — guardrail above
  exists to catch this regression.
- Trap-list (Mode 2) must pass BENCHMARK_TABOO — see the linked todo before drafting.
