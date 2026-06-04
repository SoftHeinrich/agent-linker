---
created: 2026-06-04T04:41:14.388Z
title: Replace Framing C intersection (L3) with union — Phase 4 is the sufficient gate
area: tooling
files:
  - src/llm_sad_sam/linkers/experimental/s_linker17f.py
  - scripts/test_l3_contribution.py
---

## Problem

Framing C runs a 2-pass extraction consensus (L3): keeps only the intersection of pass1 and pass2.
Empirical analysis across all 5 datasets (`scripts/test_l3_contribution.py`) shows L3 is harmful or redundant:

| Dataset | L3 rejects | TPs killed | FPs killed | Phase 4 catches FPs? | Verdict |
|---|---|---|---|---|---|
| MediaStore | 0 | 0 | 0 | — | No effect |
| TeaStore | 3 | 0 | 3 (100%) | Yes (100%) | Redundant |
| Teammates | 15 | 4 (27%) | 11 (73%) | No (0%) | Mixed |
| BigBlueButton | 5 | 5 (100%) | 0 (0%) | No | **Actively hurts** |
| JabRef | 2 | 0 | 2 (100%) | Yes (100%) | Redundant |

BBB is the decisive case: L3 rejects 5 candidates, all 5 are true positives — pure recall loss with
zero FP benefit. TeaStore and JabRef: redundant with Phase 4. Only Teammates shows independent
FP suppression, but at 27% TP cost.

## Solution

In `_extract_framing_c_candidates` (s_linker17f.py and any successor), replace:
```python
intersected = {key: pass1[key] for key in pass1 if key in pass2}
```
with union:
```python
intersected = {key: pass1.get(key, pass2[key]) for key in set(pass1) | set(pass2)}
```

Phase 4 unified validation is the correct sole quality gate for all union candidates.
L3 also removes one of the two Framing C extraction passes (saves ~half the LLM calls for Framing C).

Create s_linker17g as the next variant with this fix. Expected improvement: +5 recall on BBB,
minimal precision cost (Phase 4 will handle the FPs that L3 was catching on Teammates).
