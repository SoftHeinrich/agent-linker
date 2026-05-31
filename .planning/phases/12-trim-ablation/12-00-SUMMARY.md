---
plan: 12-00
phase: 12
title: gpt-5.4 baseline sweep — SKIPPED BY DESIGN
status: skipped
completed: 2026-05-31
requirements: [PROMPT-02]
---

# Plan 12-00 — gpt-5.4 Baseline Sweep — SUMMARY (SKIPPED)

## Decision

**Plan 12-00 SKIPPED by explicit user decision** (2026-05-31). The reuse of `s_linker13`'s existing gpt-5.4 anchor as the cross-model reference for `s_linker13_clean` and downstream trim variants is accepted as the v2.1 cross-model baseline source.

## Rationale

- **Structural-parity argument.** Phase 10 SUMMARY (10-03) established that `s_linker13_clean` is a byte-identical helper-extraction refactor of `s_linker13`. All extracted helpers in `helper_v3.py` are AST-identical to their `s_linker13.py` originals (verified by Plan 10-02 code review and `git diff` checks). Prompts are unchanged (`prompts_v2.py` imports). Therefore the cross-model F1 anchor for `s_linker13_clean` is the same as `s_linker13`'s anchor within LLM run-to-run variance.
- **v2.0 anchor.** `s_linker13` on gpt-5.4: **macro F1 = 0.9077** (v2.0 CROSS evidence; see `.planning/milestones/v2.0-MILESTONE-AUDIT.md`). This is the codified GATE-01 cross-model baseline (T = 1.0pp, absolute floor 0.8977, codified in Phase 10 Plan 10-04).
- **Cost-driven decision.** A fresh gpt-5.4 5-dataset baseline sweep would consume ~1-2 hours of wallclock and real API cost. The marginal precision gain over re-using the s_linker13 anchor does not justify the cost given the byte-identical helper extraction.

## Caveat (Honest Risk Disclosure)

Helper extraction is structural; per-run LLM stochasticity still applies. Phase 10's parity sweep on Claude Sonnet recorded per-dataset diffs of 0.0–1.7pp between `s_linker13` and `s_linker13_clean` — all within Claude's run-to-run variance band. We assume gpt-5.4 exhibits similar variance behavior. **If a Phase 12 trim variant fails the cross-model gate by a margin smaller than expected gpt-5.4 variance (~1-2pp), Phase 13 (Promotion & Wrap) MUST re-establish a fresh `s_linker13_clean` gpt-5.4 baseline before declaring rejection final.** This caveat is recorded here for milestone-level audit.

## Acceptance Used Going Forward

For every Phase 12 trim variant `s_linker13_<trim_id>_clean`:
- **Claude Sonnet acceptance** (GATE-01): macro F1 ≥ 0.93 AND BBB drop ≤ 6pp vs Claude `s_linker13_clean` baseline (Phase 10 sweep) AND other-dataset drop ≤ 2pp.
- **gpt-5.4 cross-model acceptance** (GATE-01 cross-model): macro F1 ≥ 0.8977 (absolute floor) AND macro F1 within ≤ 1.0pp of 0.9077 (the reused `s_linker13` v2.0 CROSS anchor).

## Files (Created)

- `.planning/phases/12-trim-ablation/12-00-SUMMARY.md` (this file)
- No `results/phase_cache_gpt54/s_linker13_clean/*` checkpoints created (by design).
- No `results/ablation_results/12_00_gpt54_baseline/*` JSON created (by design).

## Frozen-File Safety

No source files touched. `git diff --quiet` against all v2.0 frozen files: clean.

## Requirements Status

- **PROMPT-02 partial**: The cross-model gate this plan was meant to establish IS established (by reuse). Downstream plans 12-03 / 12-04 / 12-05 can apply it directly.

## Decision Authority

User decision logged in conversation 2026-05-31. Skip option chosen explicitly over the "Run fresh gpt-5.4 baseline sweep" alternative.
