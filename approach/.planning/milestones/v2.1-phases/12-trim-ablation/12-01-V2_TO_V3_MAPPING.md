---
phase: 12-trim-ablation
plan: 01
artifact: v2_to_v3_prompt_mapping
requirements: [PROMPT-01, PROMPT-04]
---

# prompts_v2 → prompts_v3 Mapping (Phase 12 Step 0)

Per-constant disposition for every top-level prompt constant in
`src/llm_sad_sam/linkers/experimental/prompts_v2.py`. Step 0 is a lossless
registration delete: every kept constant is byte-equal to its v2 counterpart;
the 7 dropped constants are unused by `s_linker13_clean` and survive in
`prompts_v2.py` only for back-compat with frozen sibling variants.

## Table — All 16 prompts_v2 Top-Level Constants

| constant_name                                                  | v2_lines | v3_status         | rationale                                                                                                                                                                                                  |
| -------------------------------------------------------------- | -------- | ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| AMBIGUITY_FEW_SHOT                                             | 14-47    | kept (byte-equal) | Active in `s_linker13_clean` Tier 1 model-analysis ambiguity classifier; 4 calibration examples carry information density that Claude Sonnet leverages (Phase 11 survey).                                  |
| AMBIGUITY_RULES                                                | 50-64    | kept (byte-equal) | Active in `s_linker13_clean` Tier 1 model-analysis; defines architectural vs ambiguous decision rule with WHAT-vs-WHICH/HOW-vs-WHAT-KIND framing.                                                          |
| DOC_KNOWLEDGE_EXTRACTION_RULES                                 | 71-84    | kept (byte-equal) | Active in `s_linker13_clean` Tier 1 alias discovery; defines what an abbreviation/synonym is.                                                                                                              |
| DOC_KNOWLEDGE_JUDGE_EXAMPLES                                   | 87-121   | kept (byte-equal) | Active in `s_linker13_clean` Tier 1 alias judge; 7 calibrated examples (5 APPROVE, 2 REJECT) tune the boundary.                                                                                            |
| DOC_KNOWLEDGE_JUDGE_RULES                                      | 124-139  | kept (byte-equal) | Active in `s_linker13_clean` Tier 1 alias judge; ordered decision rules with "when in doubt APPROVE" approve-bias.                                                                                         |
| WORD_USAGE_PROMPT                                              | 146-172  | **dropped**       | Legacy ≤ s_linker12c word-usage classifier; not imported by `s_linker13_clean`. Survives in `prompts_v2.py` only for older variants still in the registry. Survey §0 row "WORD_USAGE_PROMPT: dead in 13+". |
| ENTITY_EXTRACTION_RULES                                        | 179-191  | kept (byte-equal) | Active in `s_linker13_clean` Tier 2 entity extraction; 6 inclusion + 2 exclusion rules.                                                                                                                    |
| VALIDATION_RULES                                               | 194-205  | kept (byte-equal) | Active in `s_linker13_clean` Tier 2 validation reviewer.                                                                                                                                                   |
| COREF_RULES                                                    | 212-222  | kept (byte-equal) | Active in `s_linker13_clean` Tier 2 coreference resolution.                                                                                                                                                |
| STANDALONE_MENTION_RULES_PRE_FILTERED                          | 229-238  | **dropped**       | EXT-01 sub-variant (a) — used only by `s_linker13g_pre`. Deferred to v2.2+ per STATE.md "Deferred Items"; not imported by `s_linker13_clean`.                                                              |
| STANDALONE_MENTION_RULES_LLM_ONLY                              | 241-255  | **dropped**       | EXT-01 sub-variant (b) — used only by `s_linker13g_sem`. Deferred to v2.2+; not imported by `s_linker13_clean`.                                                                                            |
| STANDALONE_MENTION_RULES_PRE_FILTERED_ALIAS_AWARE              | 271-286  | **dropped**       | EXT-01 alias-aware (a) — used only by `s_linker13g_pre_alias`. Deferred; not imported by `s_linker13_clean`.                                                                                               |
| STANDALONE_MENTION_RULES_LLM_ONLY_ALIAS_AWARE                  | 289-310  | **dropped**       | EXT-01 alias-aware (b) — used only by `s_linker13g_sem_alias`. Deferred; not imported by `s_linker13_clean`.                                                                                               |
| STANDALONE_MENTION_RULES_PRE_FILTERED_FULL_KNOWLEDGE           | 313-334  | **dropped**       | EXT-01 full-knowledge (a) — used only by `s_linker13g_pre_full`. Deferred; not imported by `s_linker13_clean`.                                                                                             |
| STANDALONE_MENTION_RULES_LLM_ONLY_FULL_KNOWLEDGE               | 337-365  | **dropped**       | EXT-01 full-knowledge (b) — used only by `s_linker13g_sem_full`. Deferred; not imported by `s_linker13_clean`.                                                                                             |
| SEED_DISAMBIGUATION_RULES                                      | 372-390  | kept (byte-equal) | Active in `s_linker13_clean` Tier 2 seed disambiguation; also lifted as `SLinker13Clean.SEED_DISAMBIGUATION_RULES` classvar (line 143) byte-equal to module-level constant.                                |

**Total:** 16 prompts in `prompts_v2.py` → 9 kept (byte-equal in `prompts_v3.py`) + 7 dropped (not imported by `s_linker13_clean`).

## Verification — Byte-Equality of Kept Constants

```bash
python -c "from llm_sad_sam.linkers.experimental import prompts_v2 as v2, prompts_v3 as v3; \
names=['AMBIGUITY_FEW_SHOT','AMBIGUITY_RULES','DOC_KNOWLEDGE_EXTRACTION_RULES', \
       'DOC_KNOWLEDGE_JUDGE_EXAMPLES','DOC_KNOWLEDGE_JUDGE_RULES','ENTITY_EXTRACTION_RULES', \
       'VALIDATION_RULES','COREF_RULES','SEED_DISAMBIGUATION_RULES']; \
mismatch=[n for n in names if getattr(v2,n)!=getattr(v3,n)]; assert not mismatch, mismatch; \
print('byte-equal')"
# byte-equal
```

## Acceptance

- **PROMPT-01** (REQUIREMENTS.md): `prompts_v3.py` ships side-by-side with `prompts_v2.py`,
  this mapping table documents kept/dropped for every constant, only active constants kept.
- **PROMPT-04** (REQUIREMENTS.md): Byte-equality of kept constants means Step 0 introduces
  zero new benchmark-derived phrasing. The narrow 9-name benchmark-component probe
  (`tests/test_prompts_v3.py::test_no_benchmark_taboo_terms`) returns zero matches. Full
  reviewer-defensibility lexical sweep is Plan 12-06.
- **Phase 12 CONTEXT decisions** (`.planning/phases/12-trim-ablation/12-CONTEXT.md`):
  Step 0 — "free win, no LLM cost. Create prompts_v3.py containing only the 9 prompts
  actively imported by s_linker13_clean. Drop the 7 dead constants. Net deletion:
  ~150 LOC / ~36 rules. No ablation needed."
- **Test surface:** `tests/test_prompts_v3.py` (5 tests — clean import, kept present,
  dropped absent, byte-equal to v2, no benchmark-component tokens).
- **Sibling:** `tests/test_s_linker13_clean_v3_registration.py` (6 tests — import,
  variant name, standalone class, CANONICAL_VARIANTS membership, canonical=False,
  class_name/module fields).

## Downstream Impact

Plans 12-03, 12-04, 12-05 can safely import their kept constants from `prompts_v3`
and embed trim-specific overrides as variant-class attributes inside their own
`s_linker13_<trim>_clean.py`. They do NOT need to mutate `prompts_v3.py`.
