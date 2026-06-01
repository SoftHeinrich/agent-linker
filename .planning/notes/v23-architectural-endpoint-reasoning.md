---
title: v2.3 Architectural Endpoint — Why Voyager-Bank (A) Over Runtime-Rubric (B) Over Hybrid (C)
date: 2026-06-01
context: /gsd:explore output from v2.3 pre-kickoff ideation
related:
  - .planning/v2.3-prep/v2.3-KICKOFF-SEED.md
  - .planning/v2.2-prep/v2.2-SCOPE-DECISION.md
  - .planning/v2.2-prep/probe-D-upstream-SUMMARY.md
  - memory: user_voyager_preference.md, feedback_prefer_gpt_backend.md
tags: [v2.3, architectural-decision, voyager, runtime-rubric, slot-asymmetry, durable-reasoning]
---

# v2.3 Architectural Endpoint — Decision Reasoning

This note captures the WHY behind the v2.3 architectural-endpoint pick so future paper-writing, milestone audits, and reviewer responses don't have to reconstruct it.

## The Three Endpoints Considered

After v2.2 close, the natural v2.3 architectural endpoints were:

- **(A) Voyager-bank canonical** — `s_linker14_voyager` ships a Voyager v4-trained skill bank that injects distilled rules into all 9 axiom prompt slots. Cross-doc transfer is the learning mechanism. The bank is the swappable content over the frozen `prompts_v3_axiom.py` skeleton. Honors the v2.2-deferred commitment.
- **(B) Runtime-rubric canonical** — `s_linker14_min` ships axiom skeleton + N runtime rubrics (one per LLM-using slot). Each rubric built per-(text_stem, comp_hash, backend, model) by one LLM call, cached on disk. NO Voyager bank. Per-doc adaptation is the mechanism. Natural endpoint of Probe D + alias-rubric spike + extraction-rubric spike all passing.
- **(C) Hybrid** — runtime rubrics in some slots (e.g. coref, alias) + bank-distilled rules in others (e.g. judge). Trades architectural complexity for coverage. The hedged middle.

## Decision: (A) Mainline; (B) Contingency; (C) Ruled Out

Picked (A). (B) acceptable ONLY as a fallback IF v4 fails AND user reconsiders endpoint AND ALL LLM-using slots get runtime conversion simultaneously. (C) eliminated.

## The Principle That Decided It — Slot Asymmetry Is Ugly

User-articulated principle: in a linker that makes multiple LLM calls, **the LLM-using slots must use the same mechanism class**. Either ALL slots are static rules, ALL slots are bank-injected (v4), or ALL slots are runtime rubrics (B). Mixing — Probe D-style runtime in one slot, static rules in others — is ugly by construction. The asymmetry is the ugliness.

Why this matters:

1. **Architectural coherence** — a linker is a system; its LLM stages should be governed by one mechanism class. Mixed-mechanism systems are harder to reason about, harder to defend in publication, and harder to maintain.
2. **Reviewer defensibility** — "we use runtime rubrics for coref, bank-distilled rules for judge, and static rules for validation" reads as ad-hoc. "We use trained Voyager banks across all slots" reads as a principled design choice.
3. **Cherry-picking suspicion** — partial-runtime conversion looks like the team tried each mechanism on each slot and kept whichever happened to win, which is exactly the kind of overfitting reviewers flag.

This principle resolves (C) immediately and reframes (B): the alias-rubric and extraction-rubric spike candidates (named in the earlier seed) cannot be run in isolation. They were ORTHOGONAL to v4 in the seed; under the new principle, they are CONTINGENT on (B) being chosen as the wholesale architectural direction. Since (A) is mainline, the spikes are demoted.

## Why (A) Over (B) Even Outside the Asymmetry Argument

Even setting the slot-asymmetry principle aside:

1. **Cross-doc generalization** is the harder, more publishable research contribution. Voyager v4 trains across projects to produce transferable skills; runtime rubrics built from one doc cannot generalize by design. The v2.3 thesis "multi-role training produces transferable skills" is research-grade; "per-doc rubrics work" is engineering.
2. **Voyager v4 already has 2 datasets of evidence** (mediastore STRONG_PASS +1.69pp; BBB WEAK_PASS via Probe A' vocab fix). v2.3 inherits this evidence. (B) endpoint would start fresh with two unfunded spikes.
3. **Static-prompt-elegance is preserved equally** — both (A) and (B) inject content into the frozen `prompts_v3_axiom.py` skeleton. Bank-vs-rubric is content-source, not skeleton-changing.
4. **Budget alignment** — v4 was the v2.2 deferred commitment with budget classes ($40-80) already planned. Spike-first (B) would consume budget exploring an alternative the user has now demoted.

## Why (B) Stays as a Documented Contingency

If v4 fails the 0.87 macro floor on gpt-5.4 AND user reconsiders, (B) becomes the wholesale fallback (ALL slots → runtime rubrics, simultaneously, no single-slot exceptions). The alias-rubric and extraction-rubric spikes get run together, plus coref-rubric retention, to produce a 3+ slot uniform runtime linker. This is documented in the seed under "Spike Candidates (DEMOTED — Contingency Only)".

## Promotion Bar Reasoning

3-tier bar on gpt-5.4 macro F1:

| Tier | Threshold | Rationale |
|---|---|---|
| STRONG | ≥ 0.9173 | trim1 is the current gpt-5.4 best-in-class mechanism. v4 needs to beat it to claim shipping value. |
| WEAK | [0.87, 0.9173) | Below trim1 but still positive contribution; ships with documented caveat. 0.87 chosen as deliberately lenient to allow architectural exploration — v4 may have research value at this tier even if not production-ready. |
| FAIL | < 0.87 | Below v2.0 gpt-5.4 baseline 0.9077 by >3.7pp; below s_linker13_clean 0.9077 by >3.7pp. At this point v4 is worse than all known alternatives. Switch to Compact-B. |

## Dual-Artifact Policy Reasoning

The 0.87 WEAK floor conflicts with standing GATE-01 (carried from v2.1: gpt-5.4 macro F1 ≥ 0.8977 absolute). Resolution: TWO artifacts in v2.3.

- `s_linker13_min` retains `canonical=True`. Still GATE-01-blessed. Production reference. Carries forward v2.2 numbers (Claude 0.9506, gpt-5.4 0.9069).
- `s_linker14_voyager` (or whatever v4 is named) ships with `experimental=True`. Subject only to the 0.87 floor. Research-grade alternative.

GATE-01 applies only to `canonical=True` artifacts. `experimental=True` artifacts are explicitly NOT bound by GATE-01. This means v2.3 can publish v4 findings (positive or negative) without violating the cross-model floor. If a future milestone wants to promote v4 to canonical, it must clear GATE-01 at that time — that's a separate milestone decision.

## Implications for v2.3 Plan-Phase

When v2.3 plan-phase runs from this seed + note:

1. Main phase = train v4 with vocab-aligned R3 on 3-5 datasets, gpt-5.4 only.
2. Promotion verdict = computed against the 3-tier bar.
3. Artifact registration = `s_linker14_voyager` as `experimental=True`; no canonical promotion attempted.
4. Dual-track preservation = `s_linker13_min` canonical entry untouched.
5. Fallback trigger = if v4 macro < 0.87, plan-phase auto-spawns Compact-B implementation phase.

## What This Note Does NOT Decide

- Specific v4 training methodology details (R3 prompt content, R5 abstraction-style library, R1 actor prompt) — defer to plan-phase.
- Budget allocation per dataset — defer to plan-phase.
- Spike contingency activation criteria (if v4 fails, how is the (B) reconsideration triggered) — out of scope, will be handled if/when v4 fails.
- v2.4 cross-model promotion of v4 to canonical — separate milestone.

## Memory References

- `[[user-voyager-preference]]` — static-prompt-elegance hard rule.
- `[[feedback-prefer-gpt-backend]]` — backend policy that makes Claude verification out-of-scope.
