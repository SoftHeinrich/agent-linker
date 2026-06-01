---
phase: v2.2-PROBE-WAVE
probe: C
mechanism: Self-Refine / Reflexion 2-iter loop on alias judge
backend: gpt-5.4
dataset: mediastore
date: 2026-06-01
verdict: WEAK_PASS — F1 matches anchor exactly; cost roughly doubled on contested mappings
status: complete-neutral
tags: [self-refine, reflexion, alias-judge, weakness-class, neutral-result]
key-files:
  created:
    - src/llm_sad_sam/linkers/experimental/s_linker14_probe_c_selfrefine_clean.py
    - results/v2_2_probes/C_selfrefine/s_linker14_probe_c_selfrefine_clean_mediastore_results.json
    - results/v2_2_probes/C_selfrefine/iter_counts/mediastore.json
metrics:
  F1: 0.9677
  P: 0.9677
  R: 0.9677
  tp: 30
  fp: 1
  fn: 1
  anchor_F1: 0.9677
  delta_F1: +0.00004
  iter0_verdicts: 5
  iter0_contested: 4
  iter1_called: true
  total_judge_calls: 2
---

# Probe C — Self-Refine on Alias Judge (WEAK PASS)

## One-Liner

The 2-iter Self-Refine mechanism (iter 0 verifier emits `{verdict, weakness_class}`; iter 1 refines mappings with `weakness_class != "none"`) matched the gpt-5.4 mediastore anchor F1 of 0.9677 exactly. The mechanism does not regress, but the F1 gain on this dataset is zero — 4 out of 5 alias mappings triggered iter 1 (80% contested rate), meaning the cost roughly doubles for negligible F1 improvement.

## Setup

- **Variant**: `s_linker14_probe_c_selfrefine_clean.py` — fork of `s_linker13_clean_v3`; overrides `_learn_document_knowledge_enriched` with 2-iter Reflexion loop on the alias-judge call.
- **Iter 0 (verifier)**: judges each mapping with structured output `{term, verdict, weakness_class}`. Weakness classes: "ambiguous" | "weak_evidence" | "none".
- **Iter 1 (refine)**: re-judges ONLY mappings with `weakness_class != "none"`. Sees all iter-0 verdicts for context; revises only contested.
- **Cap**: 2 iters total (1 verify + at most 1 refine).
- **Backend**: gpt-5.4
- **Dataset**: mediastore
- **Anchor**: 0.9677 (s_linker13_min on gpt-5.4 mediastore)

## Results

| Metric | Probe C | Anchor | Δ |
| --- | --- | --- | --- |
| F1 | 0.9677 | 0.9677 | **+0.00004** |
| P  | 0.9677 | — | — |
| R  | 0.9677 | — | — |
| TP | 30 | — | — |
| FP | 1 | — | — |
| FN | 1 | — | — |
| Elapsed (s) | 27.57 | — | — |
| Iter-0 verdicts emitted | 5 | — | — |
| Iter-0 contested | 4 | — | — |
| Iter-1 called | Yes | — | — |
| Total judge calls | 2 (vs 1 baseline) | — | — |

**Verdict**: WEAK_PASS — within +0.5pp threshold, but no positive lift.

## Mechanism Behaviour

- 5 alias mappings reached the judge stage; iter 0 flagged 4 as contested (80% contested rate).
- Iter 1 fired and revised the 4 contested mappings.
- The final approved set was identical to the anchor's approved set (F1 unchanged).
- Total alias-judge LLM calls: 2 (one verifier + one refine) vs anchor's 1.

## Cost vs Benefit

- ~2× judge cost on this dataset (4/5 mappings contested → iter 1 fires).
- F1 gain: 0.004pp (well within run-to-run variance band).
- GATE-08 violation candidate: cost > 2× without proportional F1 gain.

The cost/F1 trade-off on mediastore (the easiest dataset) is unfavorable. On harder datasets (BBB, teammates), the contested rate may be higher AND there may be more genuine ambiguity for iter 1 to resolve — so the mediastore-only probe is potentially under-sampling the mechanism's benefit. However, per v2.2-prep "sample tier" doctrine, the Probe tier kill threshold is F1 vs anchor; a +0.004pp result IS a WEAK_PASS (not a STRONG_PASS).

## GATE-06

PASS. No prompts in this variant added benchmark-specific content. Verifier + refine prompts use only abstract SE vocabulary and the static `DOC_KNOWLEDGE_JUDGE_EXAMPLES` + `DOC_KNOWLEDGE_JUDGE_RULES` inherited from prompts_v3 (themselves GATE-06 audited in Phase 12-06).

## Verdict

**WEAK_PASS** — survives the kill threshold but does not show a STRONG_PASS lift. Recommendation: **defer to Range test** ONLY if user explicitly approves spending the budget. Per v2.2-prep "Probe → Range" gate, weak passes are user-decision; if Range adds bigbluebutton (the hard dataset), we may see the mechanism's actual value. If Range still shows ~0 lift, kill.

## Open Questions

1. Does the contested-rate stay around 80% on harder datasets, or does it drop / rise meaningfully? (Probe was n=1 dataset)
2. Does the `weakness_class` taxonomy ("ambiguous" | "weak_evidence" | "none") cluster differently across backends? (gpt-5.4 may emit "weak_evidence" more aggressively than Claude.)
3. Is there a STRONG_PASS configuration with iter cap = 3 and stricter weakness-class targeting? (Not tested.)

## Related Files

- Iter counts: `results/v2_2_probes/C_selfrefine/iter_counts/mediastore.json`
- Per-probe log: `results/v2_2_probes/C_selfrefine/logs/run_C_mediastore.log`
