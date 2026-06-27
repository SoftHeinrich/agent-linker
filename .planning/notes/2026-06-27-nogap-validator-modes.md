---
date: "2026-06-27 15:45"
promoted: false
---

## Next-gen validator modes: closing the no-reasoning gap

**Context.** Ran `s_linker20_union` on Sonnet with extended thinking disabled
(`CLAUDE_DISABLE_THINKING=1`, i.e. reasoning effort 0), N=3, vs the existing
thinking-on Sonnet sweep. Results in `results/v2.6.5_s20union_sonnet_nothink_20260627/`
vs `results/v2.6.5_s20union_sonnet/`.

- Macro-F1: **89.4 (nothink) vs 92.8 (thinking-on)** — a ~3.4-point drop.
- The drop is a **precision collapse, not recall**: keying every error by
  (sentence, component): **28 nothink-only false positives vs only 10 nothink-only
  false negatives**. FP-by-source: coref **7 → 27**, entity **25 → 35**.
- teammates dominates: FP 16 → 41 (+25), which is essentially the whole regression
  and explains its −8.8 F1. It is the dataset richest in package-enumeration and
  test-scaffolding prose.
- 30 false negatives are **shared by both** sweeps (mostly bigbluebutton's recall
  floor, 54–60 missed either way) — a variant limitation, NOT a thinking effect.

**Conclusion.** Extended thinking was acting as a **false-positive filter** — a
rejection/discrimination function at the validation gates. Disabling it doesn't
blind the linker (recall barely moves); it makes it **undiscriminating**. So a
better validator can in principle substitute for raw reasoning budget. Goal A:
make effort-0 + smart validator ≈ thinking-on (recover ~3.4 macro-F1 at
no-thinking cost/latency).

### Where the FPs slip through

| Gate | Phase | Failure flavor | Example FPs (×runs) |
|------|-------|----------------|---------------------|
| Entity | Phase 4 V3 twopass | enumeration / test-scaffolding | `s174-177→Test Driver` "x.logic contains component test cases"; `s197/198→Client` "client.remoteapi…" |
| Coref | Phase 5 coref validate | overview header / pronoun / **negation** | `s195→Client` "Package overview contains client.util…" (3/3); `s3→UI` "Given above is an overview…"; `s26→UI` "ui.website **is not** a Java package"; `s21→Reencoding` "**This** can result in…" |

### The unifying principle

With `MAX_THINKING_TOKENS=0`, the deliberation that filtered FPs wasn't made
cheaper — it was deleted. Every mode relocates it to one of three places:
**into rules** (Modes 1–2), **into the visible output** billed as answer tokens
(Mode 5), or **into an extra pass** (Mode 4).

### Modes analyzed

1. **Evidence-assertion rubric (general, taboo-safe).** Per candidate, require the
   span that *asserts an architectural fact*; reject bare name-mentions. Catches
   overview-header + bare-pronoun FPs. **Sharp risk:** the 10 lost FNs are almost
   all `name_in_text=False` *implicit* true links — an "assertion span required"
   rule kills exactly those. Must be phrased around **architectural claims, not
   name presence**, or it trades 28 FPs for 10+ FNs and nets zero.

2. **Trap-pattern rejecter (targeted).** Explicit red-flag list — overview/header,
   negation, unresolved pronoun, package/module enumeration, test-scaffolding —
   reject when sole evidence matches a trap. Catches the *most*, including the
   systematic teammates FPs and the negation case (only mode that reads "is not a"
   as a reject signal). Most overfit-prone; patterns are **linguistic/structural,
   not benchmark terms**, so BENCHMARK_TABOO-safe but needs an explicit audit.

4. **Adversarial skeptic pass.** Second pass that only tries to refute survivors;
   default-reject on passing mentions. Broad, framing-derived discrimination.
   Same recall tension as Mode 1, amplified; +1 full pass cost. Reserve for coref
   survivors (where thinking helped most), not all candidates.

5. **Reasoning relocated into output ("re-enable reasoning from our side") — build
   first.** Thinking tokens are 0 but *output* tokens are not. Force a one-line
   justification field per candidate (evidence span + verdict) **before** the
   keep/reject. This reconstructs the deleted thinking block as answer tokens,
   makes failures auditable, and **carries** every other mode (Mode 1 = the
   justification's required content; Mode 2 = what it must check for). Cheapest way
   to buy back the compute we turned off; taboo-neutral.

### Recommendation: one layered validator, not four

1. **Mode 5 scaffold** — forced per-candidate justification field.
2. **Mode 1 as the rubric** that justification must satisfy — phrased as
   "architectural claim," not "name present," to protect implicit links.
3. **Mode 2 trap-list** as a short, taboo-audited checklist the justification clears.
4. **Mode 4 only on coref survivors** — spend the extra pass where FPs 7→27.

**The one measurable that decides it:** does this recover precision *without*
spending the 10 implicit-link (`name_in_text=False`) FNs? Clean A/B against the two
sweeps already on disk.

See spike `004-nogap-validator-ab`, research question (CoT-in-output as FP filter),
and todo (taboo-audit the trap list).
