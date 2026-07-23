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

**RESOLVED (2026-06-27) — PARTIAL. See `spikes/004-nogap-validator-ab/RESULTS.md`.**
- As a *false-positive filter*: YES. The effort-0 layered validator (Mode 5 justification +
  Mode 1 architectural-claim rubric) matches thinking-on's FP profile EXACTLY (entity 25 /
  coref 7), with the implicit-link guardrail held (implicit-FN 59 = nothink).
- As a *macro-F1 substitute*: NO. Best effort-0 config reaches macro 90.8 — recovers only
  +1.1 of the 3.1-pt gap (target was ~92.0).
- Why: a control (re-validate the SAME nothink candidates with thinking ON) recovers the
  gap and exceeds production (mediastore 93.9→96.0, teastore →100). Since a validator can
  only reject, the F1 rise proves thinking ALSO re-approves true candidates the cheap
  validator wrongly rejected — not just FP removal. The effort-0 prompt replicates the
  FP-removal half (FP parity) but only ~1/3 of the total benefit. The gap lives at the
  validation gates and is fully recoverable WITH thinking there.
- Cost: claude CLI exposes no token usage; latency proxy — effort-0 ≈14–16 s/call,
  thinking-on validation ≈35–40 s/call.
