# Research Questions

Open questions surfaced during exploration. Each should be resolvable by a spike,
an experiment, or a literature/doc pass.

## Q1 — Can prompt-relocated reasoning substitute for extended thinking as a false-positive filter?

**Raised:** 2026-06-27 (explore: next-gen validator modes)

Disabling extended thinking on `s_linker20_union` (Sonnet) cost ~3.4 macro-F1,
almost entirely as false positives (coref FPs 7→27, entity 25→35) — i.e. thinking
was acting as an FP filter at the validation gates, not as a recall driver.

- Does forcing chain-of-thought **into the response** (a per-candidate justification
  field, billed as output tokens) recover the FP-filtering that the deleted thinking
  block did?
- At equal F1, what is the cost tradeoff: **output tokens (CoT-in-output) + optional
  extra skeptic pass** vs **thinking tokens**? Is effort-0 + smart validator actually
  cheaper / faster than thinking-on, or just differently billed?
- Does the substitution hold the recall guardrail — specifically the implicit
  (`name_in_text=False`) true links that an over-aggressive validator tends to drop?

**How to resolve:** spike `004-nogap-validator-ab` (A/B against the two sweeps already
on disk), logging output-token and latency deltas alongside macro-F1.
