---
id: 260601-v25-training-infra-axiom-expansion
created: 2026-06-01
priority: high
milestone: v2.5
blocks: nothing (future work)
resolves_phase: 26
---

# v2.5 Candidate: Infrastructure Fixes + Expand Axiom Scope to All Complex Prompts

## Part A — Infrastructure Bugs (fix before any re-run)

### A-1 — Oracle Cache Contamination (NOT YET FIXED)

**File:** `scripts/voyager_train_tlr_v4_beta.py` line 455

Oracle cache key = `{text_stem}_{comp_hash}_{backend}_{model}_oracle_iter{iter_num}` — no bank state.
Mainline range populates cache first; subsequent splits get stale mainline oracle outputs.
Split3 TM oracle analyzed mainline bank (F1=0.8264) when split3 actual TM F1 was 0.7874.

**Fix:** Add `bank_content_hash` to oracle key, same as L already does:
```python
bch = _bank_content_hash(bank)
ck = _cache_key(text_path, project, backend_str, model_str, f"oracle_iter{iter_num}_{bch}")
```

### A-2 — Cache Truncation Policy (FIXED + GENERAL RULE)

**Was:** `prompt[:200].encode()` as hash — truncated before oracle data → all splits at same iter_num got same D output.
**Fix already applied:** line 664 now uses `hashlib.md5(prompt.encode()).hexdigest()[:12]` (full prompt hash).

**General rule going forward:** NEVER truncate prompt for cache key. Always hash the full prompt. Cache misses are cheap; cache collisions silently corrupt results.

**Open empirical question:** How many cache hits actually occur in practice? May be very few given different bank states per pass/split. GPT provider may also have its own prompt cache (prefix caching) making local disk cache redundant for identical prompts. **Empirically measure cache hit rate in v2.5 first pass before investing in cache complexity.** If hit rate < 5%, consider dropping local O/D cache entirely and relying on provider-side prefix caching.

### A-3 — Probation Variance for High-Noise Datasets

BBB ±3pp LLM variance swamps probation delta (−0.005 to −0.015). Single L re-run cannot distinguish real +0.5pp from noise.

**Fix options (pick one):**
- Average 2–3 L runs per probation check for any split containing BBB
- Raise minimum commit threshold: `probation_delta >= 0.005` (not just `> 0`)
- Exclude BBB from train set for probation measurement, use it only for final eval

### A-4 — ENTITY_EXTRACTION_RULES Never Proposed

Despite BBB having 10+ missed exact-match extractions per pass, D never proposed ENTITY_EXTRACTION_RULES patterns. DOC_KNOWLEDGE_EXTRACTION_RULES over-proposed (5/5 passes).

**Fix:** In D prompt, explicitly list "high-priority underfilled slots" with description so D is steered toward underserved areas. Example: after listing bank state, add "Slots with zero patterns that are high priority: ENTITY_EXTRACTION_RULES (exact-match extraction failures), AMBIGUITY_FEW_SHOT (calibration examples needed)."

## Goal

All complex/behavioral LLM prompts in `s_linker14_voyager.py` + `ilinker3.py` must become learnable bank slots. Currently the Voyager training loop covers 9 axiom slots; 6 additional high/medium-complexity prompts are fully or partially static and invisible to training.

## Full Prompt Audit (2026-06-01)

Files audited: `s_linker14_voyager.py`, `prompts_v3_axiom.py`, `ilinker3.py`

| # | Name / Location | Complexity | Axiomed? | Slot(s) | Learnable? | Proposed Slot |
|---|----------------|------------|----------|---------|------------|---------------|
| 1 | `AMBIGUITY_RULES` — prompts_v3_axiom.py:41 | HIGH | YES | `AMBIGUITY_RULES` | YES | *(already a slot)* |
| 2 | `AMBIGUITY_FEW_SHOT` — prompts_v3_axiom.py:38 | HIGH | YES | `AMBIGUITY_FEW_SHOT` | YES | *(slot exists but starts EMPTY — needs population)* |
| 3 | `DOC_KNOWLEDGE_EXTRACTION_RULES` — prompts_v3_axiom.py:50 | HIGH | YES | `DOC_KNOWLEDGE_EXTRACTION_RULES` | YES | *(already a slot)* |
| 4 | `DOC_KNOWLEDGE_JUDGE_RULES` — prompts_v3_axiom.py:60 | HIGH | YES | `DOC_KNOWLEDGE_JUDGE_RULES` | YES | *(already a slot)* |
| 5 | `DOC_KNOWLEDGE_JUDGE_EXAMPLES` — prompts_v3_axiom.py:55 | HIGH | YES | `DOC_KNOWLEDGE_JUDGE_EXAMPLES` | YES | *(slot exists but starts EMPTY — needs population)* |
| 6 | `ENTITY_EXTRACTION_RULES` — prompts_v3_axiom.py:69 | HIGH | YES | `ENTITY_EXTRACTION_RULES` | YES | *(already a slot)* |
| 7 | `VALIDATION_RULES` — prompts_v3_axiom.py:74 | HIGH | YES | `VALIDATION_RULES` | YES | *(already a slot)* |
| 8 | `COREF_RULES` — prompts_v3_axiom.py:83 | HIGH | YES | `COREF_RULES` | YES | *(already a slot)* |
| 9 | `SEED_DISAMBIGUATION_RULES` — prompts_v3_axiom.py:92 | HIGH | YES | `SEED_DISAMBIGUATION_RULES` | YES | *(already a slot)* |
| 10 | **`ALIAS_SCOPE_SCHEMA`** — s_linker14_voyager.py:162 | HIGH | **NO** | — | **YES** | `ALIAS_SCOPE_RULES` |
| 11 | **`ANTECEDENT_ALIAS_GUIDE`** — s_linker14_voyager.py:183 | MEDIUM | **NO** | — | **YES** | `ANTECEDENT_ALIAS_RULES` |
| 12 | `_classify_components()` scaffold — s_linker14_voyager.py:422 | LOW | PARTIAL | `AMBIGUITY_FEW_SHOT`, `AMBIGUITY_RULES` | MAYBE | low priority |
| 13 | `_learn_document_knowledge_enriched()` prompt1 scaffold — s_linker14_voyager.py:456 | MEDIUM | PARTIAL | `DOC_KNOWLEDGE_EXTRACTION_RULES` | YES | → covered by `ALIAS_SCOPE_RULES` |
| 14 | `_learn_document_knowledge_enriched()` prompt2 (judge) scaffold — s_linker14_voyager.py:504 | LOW | PARTIAL | `DOC_KNOWLEDGE_JUDGE_EXAMPLES`, `DOC_KNOWLEDGE_JUDGE_RULES` | NO | framing-only |
| 15 | `_run_single_extraction_pass()` scaffold — s_linker14_voyager.py:745 | MEDIUM | PARTIAL | `ENTITY_EXTRACTION_RULES` | MAYBE | slot coverage sufficient |
| 16 | **Generic filter prompt** — s_linker14_voyager.py:849 | HIGH | **NO** | — | **YES** | `GENERIC_WORD_USAGE_RULES` |
| 17 | `_run_validation_pass()` scaffold — s_linker14_voyager.py:930 | LOW | PARTIAL | `VALIDATION_RULES` | MAYBE | low priority |
| 18 | **`_classify_specific_terminals()` prompt** — s_linker14_voyager.py:977 | MEDIUM | **NO** | — | **YES** | `COREF_TERMINAL_SPECIFICITY_RULES` |
| 19 | `_coref_cases_in_context()` scaffold — s_linker14_voyager.py:1031 | MEDIUM | PARTIAL | `COREF_RULES` | YES | → covered by `ANTECEDENT_ALIAS_RULES` |
| 20 | **`ILinker3._prompt_extract()`** — ilinker3.py:108 | HIGH | **NO** | — | **YES** | `SEED_EXTRACTION_RULES` |
| 21 | **`ILinker3._prompt_actor()`** — ilinker3.py:124 | HIGH | **NO** | — | **YES** | `SEED_ACTOR_RULES` |

## Coverage Gap Summary — v2.5 New Slots Needed

### HIGH priority (fully static, no slot injection)

| Proposed Slot | Source Prompt | Why Critical |
|--------------|---------------|--------------|
| `SEED_EXTRACTION_RULES` | `ILinker3._prompt_extract()` ilinker3.py:108 | Seed extraction drives ~80% of recall. Rules for valid mention forms (exact/synonym/partial, dotted-path exclusion) are fully static. Root cause of low training yield in v2.3/v2.4. |
| `SEED_ACTOR_RULES` | `ILinker3._prompt_actor()` ilinker3.py:124 | Pass B (actor framing) determines which seed links survive intersection. Single-word-name caution and architectural relevance rules are fixed. Also the likely source of Claude vs GPT seed divergence. |
| `GENERIC_WORD_USAGE_RULES` | Generic filter prompt s_linker14_voyager.py:849 | Decides whether a lowercase name is a component reference or generic English. Highest-stakes classification in the validation pipeline. Entirely hardcoded. |
| `ALIAS_SCOPE_RULES` | `ALIAS_SCOPE_SCHEMA` s_linker14_voyager.py:162 | "global vs local" alias scope classification — multi-word, CamelCase, all-caps, single-lowercase rules. Injected verbatim into alias extraction prompt. Should be learnable. |

### MEDIUM priority (partially static)

| Proposed Slot | Source Prompt | Why Matters |
|--------------|---------------|-------------|
| `ANTECEDENT_ALIAS_RULES` | `ANTECEDENT_ALIAS_GUIDE` s_linker14_voyager.py:183 | Injected alongside COREF_RULES. Defines when antecedent_via_alias=true. Edge cases (ambiguous alias forms) would benefit from learned examples. |
| `COREF_TERMINAL_SPECIFICITY_RULES` | `_classify_specific_terminals()` s_linker14_voyager.py:977 | Classifies terminal words of multi-word component names as specific vs generic. Controls "the X" role-reference coref. Shares the same generic-vs-specific judgment problem as GENERIC_WORD_USAGE_RULES but for a different surface form. |

### Existing empty slots (need calibration patterns, not new slots)

| Slot | Issue |
|------|-------|
| `AMBIGUITY_FEW_SHOT` | Starts empty in prompts_v3_axiom.py. No floor behavior in axiom-only mode. |
| `DOC_KNOWLEDGE_JUDGE_EXAMPLES` | Starts empty. Same problem. |

## v2.5 Implementation Plan (draft)

1. Add 6 new slot names to `prompts_v3_axiom.py`: `SEED_EXTRACTION_RULES`, `SEED_ACTOR_RULES`, `GENERIC_WORD_USAGE_RULES`, `ALIAS_SCOPE_RULES`, `ANTECEDENT_ALIAS_RULES`, `COREF_TERMINAL_SPECIFICITY_RULES`.
2. In `ilinker3.py`: refactor `_prompt_extract()` and `_prompt_actor()` to accept an injected rules block (empty string = current behavior). Wire injection from the bank loader.
3. In `s_linker14_voyager.py`: replace inline `ALIAS_SCOPE_SCHEMA`, `ANTECEDENT_ALIAS_GUIDE`, generic filter prompt, and `_classify_specific_terminals()` prompt with f-string injection from corresponding bank slots (empty string = current hardcoded text as fallback).
4. Update `voyager_train_tlr_v4_beta.py`: expand Oracle/Distillator prompts to recognize and propose patterns for the 6 new slots. Update bank schema to 15 slots.
5. Populate `AMBIGUITY_FEW_SHOT` and `DOC_KNOWLEDGE_JUDGE_EXAMPLES` with seed examples so axiom-only mode has a floor.
6. Re-run v2.5 β training — new Oracle visibility into ILinker failures should yield substantially more committed patterns.

## Net bank size: 9 → 15 slots
