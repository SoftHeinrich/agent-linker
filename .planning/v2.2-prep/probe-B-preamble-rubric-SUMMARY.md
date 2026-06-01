---
phase: v2.2-PROBE-WAVE
probe: B
mechanism: Problem-statement preamble + cached per-dataset alias-judge rubric
backend: gpt-5.4
dataset: mediastore
date: 2026-06-01
verdict: FAIL — F1 -5.24pp vs gpt-5.4 mediastore anchor
status: complete-negative
tags: [erdos, preamble, cached-rubric, alias-judge, regression]
key-files:
  created:
    - src/llm_sad_sam/linkers/experimental/s_linker14_probe_b_preamble_clean.py
    - results/v2_2_probes/B_preamble_rubric/s_linker14_probe_b_preamble_clean_mediastore_results.json
    - results/v2_2_probes/B_preamble_rubric/cache/mediastore__e569e96ce812.json
metrics:
  F1: 0.9153
  P: 0.9643
  R: 0.8710
  tp: 27
  fp: 1
  fn: 4
  anchor_F1: 0.9677
  delta_F1: -0.0524
---

# Probe B — Problem-statement Preamble + Cached Rubric (FAIL)

## One-Liner

Replacing the static `DOC_KNOWLEDGE_JUDGE_RULES` with an LLM-built per-dataset rubric (and prefixing the canonical TLR problem statement) regressed mediastore gpt-5.4 F1 by 5.24pp (0.9677 → 0.9153). The cached rubric was substantively reasonable and passed GATE-06 cleanly, but it produced a different set of alias-judge verdicts that propagated 4 false negatives into the final link set.

## Setup

- **Variant**: `s_linker14_probe_b_preamble_clean.py` — fork of `s_linker13_clean_v3`; overrides `_learn_document_knowledge_enriched` to (a) prepend `TLR_PROBLEM_PREAMBLE` and (b) substitute a per-dataset cached rubric for `DOC_KNOWLEDGE_JUDGE_RULES`.
- **Backend**: gpt-5.4
- **Dataset**: mediastore
- **Cache**: results/v2_2_probes/B_preamble_rubric/cache/mediastore__e569e96ce812.json (1334 chars rubric, GATE-06 clean)
- **Anchor**: 0.9677 (s_linker13_min on gpt-5.4 mediastore; see `.planning/milestones/v2.1-phases/13-promotion-wrap/13-01-SUMMARY.md`)

## Results

| Metric | Probe B | Anchor | Δ |
| --- | --- | --- | --- |
| F1 | 0.9153 | 0.9677 | **-0.0524** |
| P  | 0.9643 | — | — |
| R  | 0.8710 | — | — |
| TP | 27 | — | — |
| FP | 1 | — | — |
| FN | 4 | — | — |
| Elapsed (s) | 40.87 | — | — |

**Verdict**: FAIL (delta < -1.0pp threshold).

## Failure Mode Analysis

- Sources: seed=26, coref=2; FP-by-source: coref=1.
- The drop is in RECALL (R=0.8710 vs anchor recall implied ~0.97+).
- 4 FNs vs the expected 1 FN at the anchor — the rubric is approving fewer aliases than the static rules.

This is consistent with the Probe B mechanism being **conservatively over-restrictive** on its rubric build. Inspection of the cached rubric (rule 2): "reject truncated forms built from highly reusable tokens such as storage, access, management, adapter, facade, cache, database, file, media, load balancer, or packaging when ambiguity remains." This rule, while abstract, is too aggressive — it likely rejects valid aliases derived from these tokens.

## GATE-06

PASS. The cached rubric was scanned at build time; no taboo tokens found. The rubric uses only abstract SE vocabulary and explicitly names trial categories (e.g., "load balancer") that happen to match a subset of generic words — these are not benchmark-derived (they appear in the rubric-builder prompt's "what to specify" section as suggestions).

## Cost

- ~1 alias-judge call → expanded to ~2 (extraction + judge) plus 1 rubric-build call.
- Total cost: ~$0.50-$1 (gpt-5.4).

## Verdict

**FAIL** — do not Range test as currently designed. The mechanism is mechanically sound; the failure is in the rubric content. Two possible salvage paths (not in v2.2 scope):

1. **Cap the rubric's reject-list aggressiveness**: amend the rubric-builder prompt to require a justification for every "reject this token class" rule. May or may not eliminate the over-restriction.
2. **Preserve the static rules and ADD the preamble only**: separates the preamble effect from the rubric effect. Would isolate which sub-mechanism caused the regression.

The current Probe B composition (preamble + rubric) is a -5pp net regression on the easiest dataset (mediastore). Probe wave decision: kill the variant; document negative result.

## Related Files

- Cached rubric: `results/v2_2_probes/B_preamble_rubric/cache/mediastore__e569e96ce812.json` (1334 chars)
- Per-probe log: `results/v2_2_probes/B_preamble_rubric/logs/run_B_mediastore.log`
