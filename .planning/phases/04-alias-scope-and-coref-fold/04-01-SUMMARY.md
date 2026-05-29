---
phase: 04-alias-scope-and-coref-fold
plan: 01
subsystem: linker-variants
tags: [s_linker13e, ablation, alias-scope, var-05, gate-01-pass, dual-hard-tier]

requires:
  - phase: 02-ambiguity-cleanup
    provides: "13c parent variant (Phase 2 winner under loosened gate)"
provides:
  - "s_linker13e standalone variant — 13c with alias-discovery prompt emitting `scope: global|local` field, replacing `_is_strong_alias` + `_get_strong_alias_mappings`"
  - "Dual hard-tier evidence that VAR-05 (widest blast radius) is stable under cache-cleared variance"
  - "GATE-01 full-sweep pass under standing 6pp BBB tolerance"

requirements-completed: [VAR-05]
duration: ~90 min (Task 1 code + dual hard-tier + full sweep)
completed: 2026-05-29
---

# Phase 04 / Plan 01 — SUMMARY (VAR-05 ships)

**Status: GATE-01 PASS.** s_linker13e replaces `_is_strong_alias` + `_get_strong_alias_mappings` with an LLM-emitted `scope: global|local` field on the alias-discovery prompt, and holds the dual floor on a full 5-project sweep.

## What Shipped (Task 1)

- `src/llm_sad_sam/linkers/experimental/s_linker13e.py` — standalone copy of `s_linker13c.py` with:
  - `_VARIANT_NAME = "s_linker13e"`
  - New `AliasEntry` frozen dataclass `(component: str, scope: str)`
  - Alias-discovery prompt extended to emit `scope` field; 6 read sites migrated to consult `entry.scope == "global"` directly.
  - `_is_strong_alias` and `_get_strong_alias_mappings` deleted.
  - Structured module docstring (REMOVED_FROM, RULES_REMOVED).
  - Registered in `run_ablation.py`. BENCHMARK_TABOO audit clean.
- Commit: `bf0185a feat(04-01): add s_linker13e — LLM-emitted alias scope, remove _is_strong_alias + _get_strong_alias_mappings`

## Dual Hard-Tier Results (Tasks 2 + 3, per D-35a)

| Run        | JSON | TM F1 | BBB F1 | ΔTM vs 12c | ΔBBB vs 12c | Per-run gate |
|------------|------|------:|-------:|-----------:|------------:|--------------|
| RUN 1      | ablation_20260529_181005.json | 0.938 | 0.826 | 0.000 | -0.018 | PASS |
| RUN 2      | ablation_20260529_193240.json | 0.938 | 0.818 | 0.000 | -0.026 | PASS |
| **Centroid** | — | **0.938** | **0.822** | **0.000** | **-0.022** | — |

Inter-run variance band check (D-37c, threshold |ΔTM| ≤ 0.02 AND |ΔBBB| ≤ 0.04):
- |TM run1 − TM run2| = 0.000 ≤ 0.02 ✓
- |BBB run1 − BBB run2| = 0.008 ≤ 0.04 ✓
- **PASS — variance is well-controlled.**

GATE-05 checkpoint: auto-approved per standing policy.

Note: An initial RUN 2 attempt failed with `RuntimeError: LLM query failed after 3 retries: Claude request timed out` — transient API timeout, unrelated to 13e code. Re-launched cache-cleared and succeeded.

## Full Sweep (Task 5, GATE-01)

JSON: `results/ablation_results/ablation_20260529_201324.json`

| Dataset       | F1_12c | F1_13c (parent) | F1_13e | Δ vs 12c | Δ vs 13c | floor | status |
|---------------|-------:|----------------:|-------:|---------:|---------:|------:|--------|
| mediastore    |  0.984 |  1.000 | 0.984 |  0.000 | -0.016 | 0.964 | PASS |
| teastore      |  0.963 |  0.964 | 0.963 |  0.000 | -0.001 | 0.943 | PASS |
| teammates     |  0.938 |  0.938 | 0.939 | +0.001 | +0.001 | 0.918 | PASS |
| bigbluebutton |  0.844 |  0.782 | 0.804 | -0.040 | +0.022 | 0.784 | PASS |
| jabref        |  0.973 |  0.973 | 1.000 | +0.027 | +0.027 | 0.953 | PASS |
| **MACRO**     | **0.9405** | **0.9314** | **0.9380** | **-0.0025** | **+0.0066** | 0.93 | **PASS** |

**GATE-01 PASS.** Macro F1 0.9380 clears 0.93 floor. All per-dataset floors met (BBB lands exactly 0.020 above the 6pp floor; MS/TS/TM/JAB within 2pp).

**13e improves over 13c on BBB by +2.2pp.** The scope-field replacement is actually MORE stable than the structural predicates it replaces — likely because the LLM can disambiguate `global` vs `local` alias intent in context, whereas the regex-based `_is_strong_alias` was a coarser filter that overcorrected on BBB.

## VAR-05 Outcome

**SATISFIED.** Widest-blast-radius rule removal validated under both dual-hard-tier variance check AND full-sweep GATE-01.

## Phase 4 Status

- Plan 04-01 complete. 13e ships.
- 13f is the next variant (Plan 04-02). Parent = 13e per D-31 (since 13e passed).

## Commands Executed

```bash
# Task 1: code (committed bf0185a)
# Task 2: hard-tier RUN 1
nohup python run_ablation.py --variants s_linker13e --datasets teammates bigbluebutton > /tmp/13e_hardtier_r1.log 2>&1 &
# → ablation_20260529_181005.json

# Task 3: hard-tier RUN 2 (cache cleared)
rm -rf results/phase_cache/s_linker13e/{teammates,bigbluebutton}
nohup python run_ablation.py --variants s_linker13e --datasets teammates bigbluebutton > /tmp/13e_hardtier_r2.log 2>&1 &
# (first attempt: transient timeout; retried)
# → ablation_20260529_193240.json

# Task 4 checkpoint: auto-approved

# Task 5: full sweep
nohup python run_ablation.py --variants s_linker13e > /tmp/13e_fullsweep.log 2>&1 &
# → ablation_20260529_201324.json
```

## BENCHMARK_TABOO Audit

PASS. Inline `ALIAS_SCOPE_SCHEMA` uses safe SE-textbook placeholders (TaskScheduler / Scheduler / Dispatcher / Broker / Parser / Lexer).

## Pickle Cache Hygiene (D-07)

`results/phase_cache/s_linker13e/{mediastore,teastore,teammates,bigbluebutton,jabref}/` all present. No leakage to other variant namespaces.

---
*Phase: 04-alias-scope-and-coref-fold*
*Plan: 01*
*Completed: 2026-05-29 (VAR-05 satisfied)*
