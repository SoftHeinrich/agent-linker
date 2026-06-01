---
phase: v2.2-PROBE-WAVE
probe: D
mechanism: Runtime coref rubric replaces static COREF_RULES (upstream-tier rule removal, EXT-upstream)
backend: gpt-5.4
dataset: mediastore
date: 2026-06-01
verdict: STRONG_PASS — F1 +1.59pp vs gpt-5.4 mediastore anchor; matches Claude-Sonnet baseline F1
status: complete-positive
tags: [upstream-rule-removal, coref, runtime-rubric, ext-upstream, strong-pass]
key-files:
  created:
    - src/llm_sad_sam/linkers/experimental/s_linker14_probe_d_upstream_clean.py
    - results/v2_2_probes/D_upstream/s_linker14_probe_d_upstream_clean_mediastore_results.json
    - results/v2_2_probes/D_upstream/cache/mediastore__e569e96ce812.json
metrics:
  F1: 0.9836
  P: 1.0000
  R: 0.9677
  tp: 30
  fp: 0
  fn: 1
  anchor_F1_gpt54: 0.9677
  anchor_F1_claude: 0.9836
  delta_F1_vs_gpt54: +0.0159
  delta_F1_vs_claude: +0.0000
---

# Probe D — Upstream-Tier Rule Removal (Coref) — STRONG PASS

## One-Liner

Replacing the static `COREF_RULES` constant with an LLM-built per-dataset coref rubric lifted mediastore gpt-5.4 F1 by **+1.59pp (0.9677 → 0.9836)**, achieving PERFECT precision (P=1.0, 0 FPs) and matching the Claude-Sonnet baseline F1 exactly. The runtime rubric eliminated the 1 coref FP that the static rules emitted — the v2.2 probe wave's strongest single-mechanism signal.

## Setup

- **Variant**: `s_linker14_probe_d_upstream_clean.py` — fork of `s_linker13_clean_v3`; overrides `_coref_cases_in_context` to call a 1-shot rubric builder once per dataset, cache the result, and inject the rubric in place of the static `COREF_RULES` in every coref-batch prompt.
- **Mechanism class**: Identical to trim9 (runtime seed-disambiguation rubric, Phase 12 Plan 12-12), but applied to the coref tier instead of the seed tier.
- **Backend**: gpt-5.4
- **Dataset**: mediastore
- **Cache**: `results/v2_2_probes/D_upstream/cache/mediastore__e569e96ce812.json` (~1700 char rubric, GATE-06 clean)
- **Anchors**:
  - 0.9677 (s_linker13_min on gpt-5.4 mediastore; primary anchor per v2.2 directive)
  - 0.9836 (s_linker13_clean on Claude Sonnet mediastore; cross-model reference)

## Results

| Metric | Probe D | gpt-5.4 anchor | Claude anchor |
| --- | --- | --- | --- |
| F1 | **0.9836** | 0.9677 | 0.9836 |
| Δ F1 | — | **+0.0159** | **+0.0000** |
| P  | 1.0000 | — | — |
| R  | 0.9677 | — | — |
| TP | 30 | — | — |
| FP | 0 | — | — |
| FN | 1 | — | — |
| Elapsed (s) | 47.96 | — | — |

**Verdict**: STRONG_PASS (delta ≥ +0.5pp threshold; in fact +1.59pp vs the gpt-5.4 anchor).

## Mechanism Behaviour

- Source breakdown: seed=27, entity=3, coref=0. **The coref tier emitted ZERO links** under Probe D's runtime rubric — the rubric's rule 2 ("Do NOT resolve when more than one component appears in the antecedent window... Pronouns such as this, that, these, and those are especially likely to summarize an action or outcome instead of naming a component") is more conservative than the static `COREF_RULES`, suppressing the 1 coref FP without losing any TPs (the 2 baseline coref TPs were apparently lower-confidence, and the baseline emitted them PLUS the 1 FP for net +1 FP).
- The Probe B and Probe C variants ALSO had the static coref rule active and produced 1 coref FP each.
- Probe D is the ONLY variant where the coref tier emitted no FP — the runtime coref rubric's increased conservatism is the precise mechanism.

## Cross-Model Significance

The Probe D F1 of 0.9836 on gpt-5.4 mediastore EQUALS the Claude-Sonnet baseline (also 0.9836). This is the FIRST mediastore variant in the v2.x line where gpt-5.4 matches Claude on this dataset. The cross-model gap on mediastore — historically Claude 0.9836 vs gpt-5.4 0.9677 = 1.59pp — is FULLY CLOSED by this probe.

If the effect replicates on Range (BBB) + cross-model (Claude), Probe D would be the strongest candidate for s_linker14_min promotion in v2.2.

## GATE-06

PASS. The cached coref rubric was scanned at build time; no taboo tokens found. The rubric uses only abstract SE vocabulary; the example pattern in rule 4 ("The Parser validates input. It reports syntax errors.") uses an explicit textbook placeholder (Parser).

## Cost

- 1 rubric-builder call + N standard coref-batch calls.
- Cache hit on subsequent runs (free).
- Total estimated cost: ~$0.30-$0.50 (gpt-5.4).

## Verdict

**STRONG_PASS** — recommend **immediate Range test** on bigbluebutton (the hard dataset). Two acceptance criteria for Range survival:

1. F1 on BBB ≥ baseline − 1pp (per v2.2 Probe → Range gate).
2. F1 lift on BBB ≥ 0 (no regression on the hard dataset). A neutral BBB result still keeps mediastore's perfect-precision gain.

If BBB also strong-passes: immediate cross-model probe (Claude on BBB). If both pass: this becomes the v2.2 PROMO-v2.2 anchor — `s_linker14_min` = `s_linker13_min + runtime coref rubric`.

## Mechanism Generality

Probe D is analogous to trim9 (Phase 12 Plan 12-12) at the coref tier. Trim9 shipped at +0.77pp Claude on the seed-disambiguation tier. Probe D shows +1.59pp gpt-5.4 on the coref tier. Both apply the same general mechanism (runtime per-dataset LLM-built rubric replaces a static constant). If Range confirms, the v2.2 finding is: **runtime LLM-built rubrics generalize across pipeline tiers AND across backends.**

## Related Files

- Cached coref rubric: `results/v2_2_probes/D_upstream/cache/mediastore__e569e96ce812.json` (~1700 chars)
- Per-probe log: `results/v2_2_probes/D_upstream/logs/run_D_mediastore.log`
