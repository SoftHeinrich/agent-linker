---
phase: 08
status: complete-no-op
date: 2026-05-31
requirements: [COMBINE-01, COMBINE-02, COMBINE-03]
verdict: retro-designation (s_linker13 IS the COMBINE artifact)
---

# Phase 8 — COMBINE — `s_linker14` Stack-or-Unify

## Verdict

**No `s_linker14.py` file built.** Phase 8 closes as a documentation-only phase. `s_linker13` is retro-designated as the COMBINE artifact. COMBINE-01/02/03 satisfied via retro-designation.

## Why no new variant

Phase 8's original framing assumed the 3 rule-removal LLM primitives in scope (Spike-001 trailing-words + scope:global|local alias field + alias-coref-fold) lived as separate Tier-1 LLM calls in `s_linker13` and that "unification" meant folding them into a single call (e.g. into `_extract_entities_enriched` per Spike-003 pattern). CONTEXT.md D-03 encoded this hypothesis.

**Research (08-RESEARCH.md) found this was anachronistic.** Code audit shows all 3 primitives are already unified inside `_learn_document_knowledge_enriched` (s_linker13.py:366-466). They were folded there during the v1.0 chain:

- Spike-001 trailing-words → folded in 13a (`_split_component_name → LLM trailing-word` rule removed)
- scope:global|local alias field → folded in 13e (`_is_strong_alias + _get_strong_alias_mappings → LLM scope field`)
- alias-coref-fold → folded in 13f (`_has_strong_alias_mention → coref antecedent_via_alias` fold)

Entity extraction (Tier 2) CONSUMES the merged + scope-filtered alias map at line 814; it does not produce trailing-words or scope output. There is therefore no separate "stack baseline" to compare against — the stack has already been unified.

The remaining unification space inside `s_linker13` (= prompt-structure tightening within the already-unified `_learn_document_knowledge_enriched` call + consolidating 3 RULES_REMOVED docstring entries into 1) is a cleanup exercise without empirical content. User decision (2026-05-31): close Phase 8 as no-op rather than build a same-call-shape variant.

## EXT-01 disposition

Phase 8's original scope also included EXT-01 standalone-mention primitive. Phase 6 closed empty (06-SUMMARY.md). EXT-01 is excluded from the COMBINE artifact definition.

## COMBINE Requirement Closure

| REQ | Status | Evidence |
|-----|:---:|---|
| COMBINE-01 | RETRO-SATISFIED | s_linker13 integrates all v1.0+v2.0 rule-removal primitives (EXT-01 excluded as Phase 6 negative). Standalone file, no inheritance. Structured docstring `RULES_REMOVED` lists the 6 v1.0 cumulative removals. Registered `canonical=True` in `run_ablation.py`. |
| COMBINE-02 | RETRO-SATISFIED | s_linker13 macro F1 = 0.9506 ≥ 0.93 (GATE-01); per-dataset F1 logged in `ablation_20260529_215932.json`. Dual-floor cleared. |
| COMBINE-03 | SATISFIED THIS PHASE | New row in `ABLATION-TABLE.md` (v1.0-phases/05-promote-and-ablation-artifact/) under "v2.0 COMBINE addendum" + 6 v2.0 EXT-01 rejected-baseline rows. Stack-vs-unify provenance string: **"unified during v1.0 chain construction"**. |

## Stack-vs-unify provenance string (per COMBINE-01, GATE-07)

> `s_linker13` is the v2.0 COMBINE artifact. The 3 rule-removal LLM primitives in scope (Spike-001 trailing-words + scope:global|local alias field + alias-coref-fold) were already unified inside `_learn_document_knowledge_enriched` during the v1.0 chain (13a + 13e + 13f). No Tier-1-to-Tier-2 fold remained available. EXT-01 standalone-mention was excluded (Phase 6 closed empty). No new `s_linker14.py` file was created — Phase 8 retro-designates `s_linker13` as the COMBINE artifact.

## GATE-06 unit re-audit

Not applicable — no new prompts were combined. The doc_knowledge prompt was already audited as a unit during v1.0 Phase 5 PROMO; per-primitive audits in v1.0 13a/13e/13f cover the rule-removal prompts individually. v2.0 EXT-01 prompts (06-GATE-06-AUDIT.md) are audited but not shipped (Phase 6 negative).

## Cost/quality signal (Phase 9 input — tagged for grep)

## COMBINE cost/quality signal (Phase 9 input)

- **No call-count delta.** s_linker14 was not built; the unification was retroactively recognized in s_linker13.
- **Macro F1:** 0.9506 (s_linker13, Claude Sonnet, 5-dataset full sweep, v1.0 final artifact).
- **Per-dataset:** MS 0.984, TS 1.000, TM 0.947, BBB 0.821, JAB 1.000.
- **Phase 9 (CROSS) implications:** CROSS-01 (s_linker13 on GPT-5.2) AND CROSS-02 (s_linker14 on GPT-5.2) collapse to one evaluation arm — both run on the same `s_linker13.py` file. The CROSS-02 line item is documentation-only; the actual GPT-5.2 sweep is single-variant.

## Files Modified

- `.planning/milestones/v1.0-phases/05-promote-and-ablation-artifact/ABLATION-TABLE.md` — added v2.0 EXT-01 addendum (6 rejected rows) + v2.0 COMBINE addendum (1 retro-designation row) + 2 explanatory paragraphs
- `.planning/phases/08-combine-s-linker14-stack-or-unify-combined-llm-primitives/08-SUMMARY.md` (this file)
- `.planning/phases/08-combine-s-linker14-stack-or-unify-combined-llm-primitives/08-CONTEXT.md` (existing — D-01..D-10 captured before research-driven reframe)
- `.planning/phases/08-combine-s-linker14-stack-or-unify-combined-llm-primitives/08-RESEARCH.md` (existing — surfaced the anachronism)

## Files NOT Created (intentional)

- `src/llm_sad_sam/linkers/experimental/s_linker14.py` — would be a same-call-shape sibling of `s_linker13.py`, no empirical content. Skipped per user decision.
- No new entries in `run_ablation.py` `CANONICAL_VARIANTS` / `VARIANT_SPECS` — `s_linker13` retains `canonical=True`.

## ABLATION-TABLE.tex

The `.tex` mirror is NOT updated in this phase — `render_ablation.py` regenerates it from the `.md` source (per v1.0 PROMO-03 convention). The next regen will pick up the new rows automatically.
