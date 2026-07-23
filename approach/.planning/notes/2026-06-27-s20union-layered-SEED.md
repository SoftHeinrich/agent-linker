---
date: "2026-06-27"
type: seed
status: captured-not-in-active-scope
spiked: ["004-nogap-validator-ab", "005-upstream-candidate-gap"]
variant: s_linker20_union_layered
promotable_to: milestone
orthogonal_to: v2.6.6   # different ablation axis (no-REASONING, not no-KNOWLEDGE)
---

# SEED — `s_linker20_union_layered`: no-reasoning layered validator (ship candidate)

> **Why this is a seed, not a phase.** This came out of spikes 004/005 and is on a
> *different ablation axis* than the active v2.6.6 milestone (which is no-**knowledge**
> RQ3/RQ4 eval infra on the frozen full-knowledge s20_union). Capturing here so it is
> discoverable and promotable to its own milestone without derailing v2.6.6. The variant
> code is already shipped and registered; nothing is at risk of being lost. GATE-01 holds
> (canonical files untouched; the variant is an opt-in sibling).

## The finding (one line)

At reasoning effort 0 (`CLAUDE_DISABLE_THINKING=1`), a layered validator that **relocates
the deleted thinking into output tokens** recovers thinking-on's *precision* exactly, at
zero implicit-recall cost — on **both** backends.

## Config (the winning one = v4)

- **Mode 5** — forced per-candidate justification field (evidence span + verdict) *before*
  keep/reject. Reconstructs the deleted thinking block as billed answer tokens.
- **Mode 1** — the justification must satisfy an **architectural-claim** rubric ("does the
  sentence make a claim about this component", NOT "is the name present") — load-bearing for
  protecting implicit (`name_in_text=False`) true links.
- **entity-lenient / coref-strict** asymmetry — approve a *named* entity mention unless
  code-path/negation/different-entity/generic; hold coref to the strict claim rubric.
- **Rejected:** Mode 2 (rule traps — strip ~3× more true links than false, FPs share
  sentences with TPs) and Mode 4 (coref skeptic — net-negative at scale, over-rejects
  recall-bound datasets).

## Numbers (vs each backend's no-reasoning baseline)

| backend | baseline (nothink) macro | layered v4 macro | Δ | FP profile | implicit recall |
|---------|--------------------------|------------------|---|------------|-----------------|
| Sonnet  | 89.7 | **90.8** | **+1.1** | entity 25 / coref 7 = thinking-on exactly | held (impl-FN 59 = nothink) |
| gpt-5.4 | 89.4 | **93.2** (N=3) | **+3.8** | coref FP 13→2 | held (impl-FN ~flat) |

- gpt gain is larger because gpt reasoning is net-negative (the prompt is gpt's only lever)
  and gpt's no-reasoning baseline had coref-heavy FPs the coref-strict rubric cleans.
- thinking-on Sonnet is 92.8 — so v4 recovers +1.1 of the 3.1-pt gap. PARTIAL on macro,
  EXACT on the FP-filter half.

## Ship condition

Ship as the **opt-in** `s_linker20_union_layered` if the goal is **thinking-on precision at
no-thinking cost/latency** (it delivers that exactly; effort-0 ≈14–16 s/call vs thinking-on
≈35–40 s/call). It does NOT fully recover macro-F1 — don't sell it as a macro win on Sonnet.

## Why it falls short — and why extraction can't close the rest (spike 005)

The 3.1-pt Sonnet gap decomposes: **~2.0 at the validation gates** (recoverable WITH thinking
there — it both removes FPs *and* re-approves wrongly-rejected true candidates) + **~1.1
upstream** at extraction. The effort-0 prompt banks ~1.1 of the 2.0 gate points (the
FP-filter half), not the recall-recovery half.

Spike 005 then asked whether the **same mechanism** can recover the upstream extraction loss.
**No** — and the prize is smaller than it looked:
- The extraction-bound gap = 6.2% of gold (36 instances), but only **25 distinct links, 68%
  single-run** (= thinking-on extraction *variance*, not a stable effort-0 deficit).
- Mechanism split: 32% literal-skip, 24% coref, **44% non-verbatim inference**.
- The Mode-5 trick relocates *discrimination* reasoning; it cannot make the model *propose* a
  mapping it never surfaced. **You can't justify a candidate you never proposed.**

**The clean, paper-worthy asymmetry:** thinking does **precision/discriminator** work at the
gates (reasoning-relocation reconstructs it → shipped) and **recall/generator** work at
extraction (a justification field cannot reconstruct it). Reasoning-relocation substitutes
for thinking-as-discriminator, not thinking-as-generator.

## Paper-result framing (candidate RQ)

A cost/quality result for the eval or approach paper: *prompt-relocated reasoning recovers a
no-reasoning model's false-positive discrimination at a fraction of the latency, but cannot
recover its candidate-generation recall.* Dual-backend evidence (Sonnet +1.1, gpt +3.8, zero
implicit-recall cost). Resonates with the eval/ Ch2 metric-critique finding on gold
inconsistency (teammates penalizes enumeration mentions bbb rewards as bare headings).

## Promotion criteria (when to turn this into a milestone)

Promote if any of: (a) the paper wants a no-reasoning cost/quality RQ; (b) a latency/cost
budget makes effort-0 the target backend config; (c) the gpt +3.8 gain becomes the ship path
for a gpt-default deployment. Otherwise leave parked — it does not block v2.6.6.

## Artifacts

- Variant: `src/llm_sad_sam/linkers/experimental/s_linker20_union_layered.py` (registered)
- Spike 004: `.planning/spikes/004-nogap-validator-ab/` (RESULTS.md, harness/)
- Spike 005: `.planning/spikes/005-upstream-candidate-gap/` (RESULTS.md, harness/extraction_mechanism.py)
- Origin note: `.planning/notes/2026-06-27-nogap-validator-modes.md`
- Research question (resolved): `.planning/research/questions.md` Q1
