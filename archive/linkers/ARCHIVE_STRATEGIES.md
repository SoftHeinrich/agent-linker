# Archived Linker Strategies

This document describes the strategy and key findings for each archived linker version.
Files are numbered chronologically (archiv01–archiv43).

---

## Pre-Agent Era (TransArc Refined)

| # | Original Name | Strategy | F1 |
|---|---------------|----------|----|
| 01 | transarc_refined_linker_v44 | V41 base + multi-component extraction. Removed complex Component Wiki; kept simpler doc knowledge learning. 10-phase pipeline with document knowledge learning & judge validation. | — |
| 02 | transarc_refined_linker_v45 | Added discourse-aware coreference resolution. Entity tracking (recent mentions, paragraph topic, active subject). Implicit reference detection phase. | — |
| 03 | transarc_refined_linker_v46 | Automatic linguistic pattern learning — replaced all hardcoded pronouns/transitions/verbs with LLM-learned document-adaptive patterns. | — |

## Early Agent Linkers (V2–V5)

| # | Original Name | Strategy | F1 |
|---|---------------|----------|----|
| 04 | agent_linker_ablation | Base AgentLinker extended with configurable ablation flags: coref modes (discourse/debate/discourse_judge/adaptive), implicit modes, sliding-batch entity extraction. | — |
| 05 | agent_linker_v2 | Qualitative judgments (approve/reject + certainty: high/low) instead of numeric confidence thresholds. No Phase 0 threshold learning. Also reviews TransArc for ambiguous components. | — |
| 06 | agent_linker_v2_ablation | V2 with adaptive coref/implicit mode switching and semantic filter options (embedding, tfidf, lexical, selective). | — |
| 07 | agent_linker_v3 | Self-contained linker: qualitative pipeline, source-priority dedup, adaptive coref (discourse vs debate), post-hoc semantic filters, Phase 9 judges ambiguous TransArc. | — |
| 08 | agent_linker_v4 | NO data leakage: removed all hardcoded word lists & thresholds. Everything learned from input data. Embedding threshold derived from TransArc similarity distribution. Judge stabilization variants. | — |
| 09 | agent_linker_v5 | Consolidated best approach. Judge as rejection filter (not gate), case-index keying, word-boundary checks on synonyms, batched implicit/recovery validation. | — |

## V6 Exploration Phase

| # | Original Name | Strategy | F1 |
|---|---------------|----------|----|
| 10 | agent_linker_v6 | Fixed dot-filter bug + removed dataset-specific judge examples. Sentence-ending periods no longer trigger package-path filter. Generic examples only. | **91.4%** (best non-deterministic) |
| 11 | agent_linker_v6b | V6 + abbreviation guard. Rejects "abbreviation + different-noun" matches (e.g., "GAE server" != "GAE Datastore"). Applied at Phase 5, 7, 8b. | — |
| 12 | agent_linker_v6c | V6 + 3 deterministic boundary filters (package-path, generic-word, weak-partial) + abbreviation guard. Consistently improves TM by +4-6pp, no regression on MS/TS/JAB. | ~86% |

## Voting Experiments (Failed)

| # | Original Name | Strategy | F1 | Outcome |
|---|---------------|----------|----|---------|
| 13 | agent_linker_v6_vote | Multi-run majority voting (N=3 runs). Accept link if majority agrees. | **87.5%** | **FAILED**: Kills borderline TPs appearing in only 1/3 runs |
| 14 | agent_linker_v6_phase_vote | Phase-level voting: deterministic phases (0-4) once, noisy phases (5-10) vote N times. ~2x cost. | **89.7%** | **FAILED**: Noisy phase voting kills borderline TPs |

## Pattern Learning Experiments (Failed)

| # | Original Name | Strategy | F1 | Outcome |
|---|---------------|----------|----|---------|
| 15 | agent_linker_v7 | NEW Phase 2b: Confusion Pattern Learning — LLM identifies component names that could cause FPs. Judge uses learned patterns as rejection rules. | **89.2%** | **FAILED**: Over-filtering causes recall regression |
| 16 | agent_linker_v8a | Union-accept voting with tiebreaker judge: take UNION of N runs, strict judge ONLY for unstable links. | **87.1%** | **FAILED**: Union brings too many FPs |
| 17 | agent_linker_v8b | V6 + soft learned hints: confusion patterns injected as context notes (not rejection rules). | **83.1%** | **FAILED**: Soft hints destabilize judge completely |

## Feature-Driven Variants

| # | Original Name | Strategy | F1 |
|---|---------------|----------|----|
| 18 | agent_linker_v9 | Multi-run majority voting over full V6 pipeline (N=3, cleaner impl). | — |
| 19 | agent_linker_v9t | V5 + 6 error-driven features (A-F): dot-filter fix, abbreviation guard, coref distance filter, generic word filter, ambiguous recovery, section-heading safe. Feature B eliminated ALL 9 GAE Datastore FPs. | **83.6%** (gains on TM offset by LLM variance) |

## Generic-Name-Aware Era (V11–V16)

| # | Original Name | Strategy | F1 |
|---|---------------|----------|----|
| 20 | agent_linker_v11 | Generic-name-aware pipeline: deterministic single-word generic classification, Phase 3b stoplist, generic entity validation (require capitalized mention), generic coref filter, remove dead-weight Phases 8+10, V6C boundary filters, generic-informed judge. | **84.9%** (TM +8pp but TS -14pp) |
| 21 | agent_linker_v12 | V11 + contextual stoplist + contextual weak-partial filter. Only block auto-partial if word is also a single-word component in THIS model. | **82.0%** |
| 22 | agent_linker_v13 | V6 base + intersection generic filtering (only truly ambiguous generics) + TransArc-protected boundary filters. | **86.1%** |
| 23 | agent_linker_v14 | Deterministic Phase 1 (structural rules, no LLM), decoupled ambiguity from judge, structured judge criteria (3-point checklist), enriched prompts. Had Phase 6 bug. | **86.7%** |
| 24 | agent_linker_v15 | **BREAKTHROUGH**: Fully deterministic Phase 1 (skip LLM call entirely). Split Phase 6 validation (union for non-generic, intersect for generic). TransArc immunity in Phase 9. Enriched Phase 3/5 prompts. TM best ever: 91.7%. | **88.8%** (best deterministic) |
| 25 | agent_linker_v16 | Per-mention generic check (no global labels). Each mention checked individually. Alias-safe bypass: if any synonym/partial appears in sentence text, skip judge review. Saved 16+ links across MS/TS/BBB/TM. | **88.1%** |
| 26 | agent_linker_w16 | V16 variant + 3 targeted fixes: Phase 5b parent-overlap guard, Phase 7 link-aware antecedent, Phase 3 deterministic generic overrides. | — |

## Prompt Engineering & Hybrid Judge Era (V17–V23e)

| # | Original Name | Strategy | F1 |
|---|---------------|----------|----|
| 27 | agent_linker_v17 | V16 + togglable ablation flags: intersect_validation, enable_implicit, enable_fn_recovery, nuanced_transarc, judge prompt variants J1-J6. | — |
| 28 | agent_linker_v18 | Consolidated best ablation settings (intersect validation + adaptive context judge + show match text). Hardcoded best from V17 experiments. | **90.7%** |
| 29 | agent_linker_v19 | V18 + Phase 7 antecedent-citation prompt, balanced Phase 5 prompt, word-boundary validation v2, 2-pass intersect + evidence post-filter for generic names. | — |
| 30 | agent_linker_v20 | V19 + 3 combined fixes: Phase 3 generic override (rescue "Database"), Phase 8b partial context guard, Phase 5b parent-overlap guard. | — |
| 31 | agent_linker_v20a | V19 + Phase 3 component-name generic override. Rescue CamelCase/acronym/component-mapping terms rejected as generic. Fixes MS "Database" variance. | — |
| 32 | agent_linker_v20b | V19 + Phase 8b partial-injection context guard. Block "UI name" compound nouns. Eliminated 4 TS FPs (S11/12/13/15 -> WebUI). | — |
| 33 | agent_linker_v20c | V19 + Phase 5b parent-overlap guard. Skip sub-type when parent component already linked. | — |
| 34 | agent_linker_v20d | V20 + Phase 5 smaller batches (50 vs 100) + longer timeout (240s vs 150s). Fixes BBB timeout. | — |
| 35 | agent_linker_v20e | V20 + Phase 7 link-aware antecedent check. Existing links count as antecedents. Widens window 2->3. | — |
| 36 | agent_linker_v21 | V20d promoted — 4 fixes combined (A: generic override, B: partial guard, C: parent-overlap, D: smaller batches). | ~94.9% (non-deterministic, high variance) |
| 37 | agent_linker_v22 | V21 + enhanced Phase 9 judge with 4 explicit exclusion rules: abstraction, incidental, subject, name collision. | — |
| 38 | agent_linker_v23 | V22 + removed alias-safe bypass + 4-rule Phase 6 prompt. Unified Phase 6/9 criteria. | — |
| 39 | agent_linker_v23a | V22 + alias-aware judge context (tell judge which alias matched) + 4-rule Phase 6. Informed decisions instead of bypass. | — |
| 40 | agent_linker_v23b | V22 + selective alias bypass (keep for validated/partial_inject, remove for coref/entity/recovered) + 4-rule Phase 6. | — |
| 41 | agent_linker_v23c | V22 + short-alias bypass (only <=6 char aliases bypass judge) + 4-rule Phase 6. | — |
| 42 | agent_linker_v23d | V22 + 4-rule Phase 6 only. Kept alias-safe bypass in Phase 9. Minimal change. | — |
| 43 | agent_linker_v23e | **ALL-TIME BEST**: Hybrid batch/union judge. Batch context for components with 3+ alias links, union-voted individual judge for rest. Only 23 FP total (16 TransArc, 6 validated, 1 coref), 5 FN. | **94.1%** (avg of 93.7%, 94.5%) |

---

## Key Lessons Learned

1. **Majority voting fails** (archiv13-14, archiv18): Kills borderline TPs more than it removes FPs.
2. **LLM-learned patterns as rejection rules fail** (archiv15): Over-filtering destroys recall.
3. **Soft hints destabilize** (archiv17): Making the judge aware of ambiguity causes over-caution.
4. **Generic-risk words are dangerous** (archiv20-21): "Persistence", "Registry", "Model" are valid component names — global labeling hurts.
5. **Deterministic Phase 1 is critical** (archiv24): Non-deterministic LLM classification in Phase 1 causes cascading variance.
6. **TransArc links have 90%+ precision** — never judge them (archiv24+).
7. **Per-mention checks beat global labels** (archiv25): "Logic handles" = reference, "business logic" = generic.
8. **Hybrid batch/union judge is the breakthrough** (archiv43): Alias groups with 3+ links deserve collective context; singles need 2-pass vote consensus.

## Performance Evolution

```
V6  (archiv10): 91.4% — best non-deterministic single run
V15 (archiv24): 88.8% — best deterministic
V16 (archiv25): 88.1% — per-mention generic checks
V18 (archiv28): 90.7% — consolidated best ablation settings
V23e(archiv43): 94.1% — ALL-TIME BEST (hybrid batch/union judge)
```
