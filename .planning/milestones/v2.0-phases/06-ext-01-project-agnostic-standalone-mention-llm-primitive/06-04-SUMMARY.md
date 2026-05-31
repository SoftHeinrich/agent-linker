---
phase: 06
plan: 06-04
status: partial-pivot
date: 2026-05-30
requirements: [EXT-01]
verdict: gate-05-fail → design-pivot to alias-aware variants
---

# Plan 06-04 — Partial: GATE-05 Negative Result, Design Pivot

## Status

Plan 06-04 executed Task 1 (GATE-05 hard-tier dev loop) only. Tasks 2-7 superseded by a user-directed design pivot: pure-LLM standalone-mention primitive is empirically insufficient on high-abbreviation datasets (BBB). Phase 6 is being replanned around alias-aware sub-variants that feed already-gathered project knowledge (alias map, coref antecedents, link map, Spike-003 piggyback) into the standalone-mention LLM call.

## GATE-05 Hard-Tier Results (TM + BBB vs s_linker13 parent)

| Variant | TM F1 | TM Δ | TM ≥ 0.9374? | BBB F1 | BBB Δ | BBB ≥ 0.8890? | GATE-05 |
|---|---:|---:|:---:|---:|---:|:---:|:---:|
| s_linker13 (parent) | 0.9474 | — | — | 0.8990 | — | — | — |
| s_linker13g_pre | 0.9381 | -0.0093 | YES (by 0.0007) | 0.8108 | **-0.0882** | **NO** | **FAIL** |
| s_linker13g_sem | 0.9217 | **-0.0257** | **NO** | 0.8108 | **-0.0882** | **NO** | **FAIL** |

Source data: `results/ablation_results/ablation_ext01_hardtier.json`, `ablation_20260530_115900.json`, `ablation_20260530_121014.json` (gitignored; on disk for review).

## Root Cause (Verified, Not Wiring Bug)

BBB recall loss = 17 FNs identical across both independent sub-variants, all `name_in_text: false`, concentrated on HTML5 Client (6) / HTML5 Server (5) abbreviation references plus other coref-required mentions. Pipeline integrity checks pass: both variants ran the full Tier-1/2/3 DAG, ILinker3 seed produced, doc_knowledge aliases discovered (`bbb-html5 → HTML5 Server`), all checkpoints saved. FP set diverges between variants (consistent with prompt differences in the standalone-mention judge); TP/FN set does not (rules out wiring bug).

The LLM standalone-mention primitive, given only `(comp_name, sentence)`, cannot recognize that a sentence referencing "HTML5 Client" or "the client" is a standalone mention of the BBB `HTML5 Client` component when that mapping lives in the alias/coref tier. The regex baseline accidentally caught some of these via exact substring match on the alias surface form, but the LLM treats the sentence on its own terms and rejects the standalone-mention claim.

This matches the v1.0 VAR-04 retirement pattern (negative result from removing a structural rule without a sufficient LLM substitute on a high-abbreviation dataset).

## Empirical Finding (preserves D-01 / D-02 / D-04 evidence)

Pure-LLM standalone-mention substitution — across the literal/semantic axis AND across the dotted-path-handling axis — is **insufficient** to clear GATE-05 on BBB without supplementary knowledge inputs. This finding is the empirical justification for the design pivot recorded in CONTEXT.md as D-07.

## Design Pivot (user-directed, 2026-05-30)

**User direction:** "design more variants, maybe pure llm not work, but llm + some already gathered knowledge/ data like alias, links, corefs"

**Decision (new D-07..D-10 added to CONTEXT.md):**

- **D-07 — Knowledge sources:** Feed *all* already-gathered project knowledge into the new standalone-mention LLM call: alias map (from doc_knowledge phase), coref antecedents, running link map, Spike-003 entity-extraction piggyback context.
- **D-08 — Variant matrix:** Build **4 new sub-variants** = {pre, sem} × {alias-only context, full-knowledge context (alias + coref + linkmap)}.
- **D-09 — Empirical floor:** Pure-LLM pair (`s_linker13g_pre`, `s_linker13g_sem`) becomes the rejected-baseline floor for the ablation table. New variants must clear GATE-05 on BBB (≥ 0.8890) where pure-LLM failed.
- **D-10 — Phase 6 replan:** Plans 06-05 onward designed around D-07/08/09. Plans 06-01 / 06-02 / 06-03 artifacts (prompts, sibling files, DIFF-MATRIX) retained as input data.

## Disposition

- Plans 06-01, 06-02, 06-03: COMPLETE, artifacts retained as input for the replan.
- Plan 06-04: PARTIAL (Task 1 only). Tasks 2-7 voided. This SUMMARY closes out 06-04.
- Phase 6: REPLAN. New plans (06-05+) will be authored by the orchestrator's planner to execute the D-07/08/09 design.

## Files (on disk)

- `src/llm_sad_sam/linkers/experimental/s_linker13g_pre.py` — pure-LLM pre-filter variant (rejected baseline)
- `src/llm_sad_sam/linkers/experimental/s_linker13g_sem.py` — pure-LLM LLM-only variant (rejected baseline)
- `src/llm_sad_sam/linkers/experimental/prompts_v2.py` — `STANDALONE_MENTION_RULES_PRE_FILTERED` + `STANDALONE_MENTION_RULES_LLM_ONLY` constants (will be extended for D-07 alias-aware prompts)
- `run_ablation.py` — registers `s_linker13g_pre` + `s_linker13g_sem` (will add D-07 variants)
- `results/ablation_results/ablation_ext01_hardtier.json` — GATE-05 failure record (gitignored)
- `BENCHMARK_TABOO.md` — anti-pattern section added per Plan 06-03 user adjudication
