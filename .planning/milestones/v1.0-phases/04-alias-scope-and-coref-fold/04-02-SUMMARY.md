---
phase: 04-alias-scope-and-coref-fold
plan: 02
subsystem: linker-variants
tags: [s_linker13f, ablation, alias-coref-fold, var-06, gate-01-pass]

requires:
  - phase: 04-alias-scope-and-coref-fold
    provides: "13e parent variant (VAR-05 satisfied)"
provides:
  - "s_linker13f standalone variant — 13e with `_has_strong_alias_mention` folded into the coref prompt evidence schema"
  - "Completion of all six structural rule removals across the 13-series chain (modulo VAR-04 retired in Phase 3)"
  - "GATE-01 full-sweep pass with macro F1 IMPROVING over 12c by +1.04pp"

requirements-completed: [VAR-06]
duration: ~60 min
completed: 2026-05-29
---

# Phase 04 / Plan 02 — SUMMARY (VAR-06 ships)

**Status: GATE-01 PASS.** 13f folds `_has_strong_alias_mention` into the coref prompt as inline evidence, removes the structural predicate, and holds the dual floor — actually IMPROVING macro F1 over 12c.

## What Shipped (Task 1)

- `src/llm_sad_sam/linkers/experimental/s_linker13f.py` — standalone copy of `s_linker13e.py` with:
  - `_VARIANT_NAME = "s_linker13f"`
  - `_has_strong_alias_mention` deleted.
  - Coref prompt extended with an `antecedent_via_alias` evidence field carrying the alias-context inline (no structural prefilter).
  - Structured module docstring (REMOVED_FROM: 13e; RULES_REMOVED).
  - Registered in `run_ablation.py`. BENCHMARK_TABOO audit clean.
- Commit: `4996981 feat(04-02): add s_linker13f — fold _has_strong_alias_mention into coref prompt (antecedent_via_alias)`

## Hard-Tier (Task 2, single per D-35b)

JSON: `results/ablation_results/ablation_20260529_204652.json`

| Dataset       | F1_12c | F1_13e (parent) | F1_13f | Δ vs 12c | Δ vs 13e | gate |
|---------------|-------:|----------------:|-------:|---------:|---------:|------|
| teammates     |  0.938 |  0.939 |  0.922 |  -0.016  |  -0.017  | marginal band (proceed, flag) |
| bigbluebutton |  0.844 |  0.822 (centroid) |  0.842 |  -0.002  |  +0.020  | auto-approve |

TM marginal-band flag noted. Standing policy: marginal proceeds to full sweep, full sweep is the decisive gate. Auto-approved Task 3 checkpoint.

## Full Sweep (Task 4, GATE-01)

JSON: `results/ablation_results/ablation_20260529_215932.json`

| Dataset       | F1_12c | F1_13e | F1_13f | Δ vs 12c | Δ vs 13e | floor | status |
|---------------|-------:|-------:|-------:|---------:|---------:|------:|--------|
| mediastore    |  0.984 |  0.984 |  0.984 |   0.000  |   0.000  | 0.964 | PASS |
| teastore      |  0.963 |  0.963 |  1.000 |  +0.037  |  +0.037  | 0.943 | PASS |
| teammates     |  0.938 |  0.939 |  0.947 |  +0.009  |  +0.008  | 0.918 | PASS |
| bigbluebutton |  0.844 |  0.804 |  0.821 |  -0.023  |  +0.017  | 0.784 | PASS |
| jabref        |  0.973 |  1.000 |  1.000 |  +0.027  |   0.000  | 0.953 | PASS |
| **MACRO**     | **0.9405** | **0.9380** | **0.9509** | **+0.0104** | **+0.0129** | 0.93 | **PASS** |

**GATE-01 PASS.** Macro F1 0.9509 IMPROVES on 12c by 1.04pp. All per-dataset floors met (BBB 0.037 above the 6pp floor; all others within or above 2pp).

The TM hard-tier marginal band (-0.016) did not reproduce in the full sweep (TM 13f = 0.947, ΔTM = +0.009) — consistent with documented Claude run-to-run variance band.

## VAR-06 Outcome

**SATISFIED.** Coref-fold replacement of `_has_strong_alias_mention` produces a stronger linker than the original structural predicate. The LLM uses inline alias context productively without overcorrecting.

## Phase 4 Status

Both plans complete:
- 04-01 (VAR-05, 13e) — PASS, macro 0.9380
- 04-02 (VAR-06, 13f) — PASS, macro 0.9509 (best in the 13-series chain)

13f is the new best variant by macro F1 across the entire ablation chain to date.

## Commands Executed

```bash
# Task 1 (committed 4996981)
# Task 2 (hard-tier):
nohup python run_ablation.py --variants s_linker13f --datasets teammates bigbluebutton > /tmp/13f_hardtier.log 2>&1 &
# → ablation_20260529_204652.json (TM 0.922 marginal, BBB 0.842 auto)
# Task 3: auto-approved
# Task 4 (full sweep):
nohup python run_ablation.py --variants s_linker13f > /tmp/13f_fullsweep.log 2>&1 &
# → ablation_20260529_215932.json (macro 0.9509)
```

## BENCHMARK_TABOO Audit

PASS. `ANTECEDENT_ALIAS_GUIDE` uses safe SE-textbook placeholders.

## Pickle Cache Hygiene (D-07)

`results/phase_cache/s_linker13f/{mediastore,teastore,teammates,bigbluebutton,jabref}/` all present. No leakage.

---
*Phase: 04-alias-scope-and-coref-fold*
*Plan: 02*
*Completed: 2026-05-29 (VAR-06 satisfied; macro 0.9509 — best in chain)*
