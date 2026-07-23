---
spike: 005
name: upstream-candidate-gap
validates: "Of the ~1.1 macro-F1 points that spike 004 showed are UPSTREAM (unreachable by re-validating nothink's candidates), how many come from true links that thinking-on EXTRACTS as candidates but nothink never generates — vs links present in nothink's pool but wrongly rejected? And can spike 004's Mode-5 reasoning-relocation mechanism recover them at extraction?"
verdict: COMPLETE
verdict_date: "2026-06-27"
result: "Step 1: 36 extraction-bound gold instances (6.2% of gold), bbb-dominated. Step 2 ($0 mechanism inspection): those are 25 DISTINCT links, 68% single-run (= thinking-on extraction variance, not a stable effort-0 deficit). Mechanism split: 32% literal-skip, 24% coref-anaphora, 44% indirect-INFERENCE. The Mode-5 mechanism (relocates discrimination reasoning) cannot reach the 44% inference bucket — you cannot justify a candidate never proposed. Clean asymmetry: thinking = precision/discriminator at the gates (reconstructable, shipped) vs recall/generator at extraction (NOT reconstructable by a justification field). Recommendation: do NOT run the bbb LLM extraction probe; stop. Realizable robust upside ~3-8 links, mostly inference + noise."
related: [004-nogap-validator-ab]
tags: [recall, candidate-generation, extraction, coref-discovery, no-reasoning]
---

# Spike 005: the upstream candidate gap

## Why

Spike 004 decomposed the no-reasoning gap (nothink 89.7 → thinking-on 92.8) into
**~2.0 pts at the validation gates** (recoverable by thinking there) + **~1.1 pts
upstream** (re-validating nothink's candidates with thinking tops out at macro 91.7).
The "upstream" residual is candidate generation: thinking-on extracts true links that
nothink's effort-0 extraction/coref-discovery never proposes.

## Step 1 (FREE — cache only)

For every cell, compare the **candidate pools** (the (sentence, component) pairs the
pipeline GENERATES before validation):
- nothink pool = layer3 entity `candidates` ∪ layer4 `coref_raw`
- thinking-on pool = same from the thinking-on cache
- vs gold.

A validator can only approve candidates in the pool, so `|gold ∩ pool| / |gold|` is the
**recall ceiling** of that pool. The decomposition:
- gold links in nothink's pool but wrongly rejected → **validator-recoverable** (spike 004).
- gold links in thinking-on's pool but NOT nothink's → **extraction-bound** (this spike).
- gold links in neither → unreachable by either backend's candidates.

Output: `harness/candidate_gap.py` → per-dataset ceilings + the extraction-bound link list.
Decides whether a thinking-on extraction replay (Step 2, LLM) is worth running.
