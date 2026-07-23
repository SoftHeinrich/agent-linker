# Phase 31 Prompt Audit Report

**Scope**: All static prompt strings in `ilinker4.py` and `s_linker14_voyager.py`.
**Classification**: (a) Structural scaffolding — stays inline. (b) Inline behavioral rule — must migrate to bank slot.

---

## ilinker4.py

### `_prompt_extract`

| Content | Classification | Action |
|---------|---------------|--------|
| `ARCHITECTURE COMPONENTS: {comp_block}` | (a) Structural | No change |
| `DOCUMENT: {doc_block}` | (a) Structural | No change |
| `TASK: For each sentence, find architecture components EXPLICITLY mentioned or referenced.` | (a) Structural (task framing) | No change |
| `Valid: exact name, synonym, abbreviation, or unambiguous partial name in the sentence text.` | (a) Structural (axiom-level boundary rule, abstract, not benchmark-specific) | No change |
| `Invalid: names inside dotted paths, generic English words, or no clear textual evidence.` | (a) Structural (axiom-level boundary rule) | No change |
| `LEARNED PATTERNS + {seed_extraction_rules}` | Bank slot SEED_EXTRACTION_RULES | ✅ Already injected |
| `Return JSON: {...}` | (a) Structural | No change |
| `Precision is critical.` | (a) Structural (general task calibration) | No change |

**Verdict**: Zero (b) items. All behavioral content comes through SEED_EXTRACTION_RULES slot injection.

### `_prompt_actor`

| Content | Classification | Action |
|---------|---------------|--------|
| `ARCHITECTURE COMPONENTS: {comp_block}` | (a) Structural | No change |
| `DOCUMENT: {doc_block}` | (a) Structural | No change |
| `TASK: For each sentence, find components that are ARCHITECTURALLY RELEVANT...` | (a) Structural | No change |
| `Report ALL participating components... "X connects to Y" → both X and Y.` | (a) Structural (axiom-level multi-participant rule) | No change |
| `CAUTION with single-word names (e.g., "Scheduler", "Dispatcher")...` | (a) Structural (axiom-level ambiguity note; "Scheduler"/"Dispatcher" are safe SE textbook terms, GATE-06 clean) | No change |
| `Rules: Must be explicitly named... Skip pronouns. Skip dotted paths. Skip generic word usage.` | (a) Structural (axiom-level extraction constraints) | No change |
| `LEARNED PATTERNS + {seed_actor_rules}` | Bank slot SEED_ACTOR_RULES | ✅ Already injected |
| `Return JSON: {...}` | (a) Structural | No change |

**Verdict**: Zero (b) items. All behavioral content comes through SEED_ACTOR_RULES slot injection.

---

## s_linker14_voyager.py

### Module-level constants

| Constant | Classification | Action |
|----------|---------------|--------|
| `ALIAS_SCOPE_SCHEMA` | Mixed: structural schema + axiom-level scope definitions | Wrapped with `ALIAS_SCOPE_RULES` slot ✅ |
| `ANTECEDENT_ALIAS_GUIDE` | Mixed: behavioral (true/false rules + examples) | Wrapped with `ANTECEDENT_ALIAS_RULES` slot ✅ |

### `_classify_components` prompt

| Content | Classification | Action |
|---------|---------------|--------|
| `Classify these software architecture component names.` | (a) Structural | No change |
| `NAMES: {names}` | (a) Structural | No change |
| `{self._AMBIGUITY_FEW_SHOT}` | Bank slot `AMBIGUITY_FEW_SHOT` (axiom + bank) | ✅ |
| `NOW CLASSIFY THE NAMES ABOVE.` | (a) Structural | No change |
| `Return JSON: {"architectural":..., "ambiguous":...}` | (a) Structural | No change |
| `{self._AMBIGUITY_RULES}` | Bank slot `AMBIGUITY_RULES` (axiom + bank) | ✅ |

**Verdict**: Zero (b) items remaining.

### `_learn_document_knowledge_enriched` — prompt 1

| Content | Classification | Action |
|---------|---------------|--------|
| `Find all alternative names used...` | (a) Structural | No change |
| `{self._DOC_KNOWLEDGE_EXTRACTION_RULES}` | Bank slot | ✅ |
| `{self._ALIAS_SCOPE_RULES}` | Bank slot | ✅ |
| `Return JSON: {"abbreviations":..., "synonyms":...}` | (a) Structural | No change |

**Verdict**: Zero (b) items remaining.

### `_learn_document_knowledge_enriched` — prompt 2 (judge)

| Content | Classification | Action |
|---------|---------------|--------|
| `JUDGE: Review these component name mappings for correctness.` | (a) Structural | No change |
| `{self._DOC_KNOWLEDGE_JUDGE_EXAMPLES}` | Bank slot | ✅ |
| `{self._DOC_KNOWLEDGE_JUDGE_RULES}` | Bank slot | ✅ |
| `Return JSON: {"approved": [...]}` | (a) Structural | No change |

**Verdict**: Zero (b) items remaining.

### `_run_seed_validation` prompt

| Content | Classification | Action |
|---------|---------------|--------|
| `REFERENCE DISAMBIGUATION for component "{comp_name}"` | (a) Structural | No change |
| `COMPONENT PROFILE: {profile}` | (a) Structural | No change |
| `{anchor_section}` | (a) Structural (runtime-built, not static) | No change |
| `CASES TO VERIFY: {case_lines}` | (a) Structural | No change |
| `{self._SEED_DISAMBIGUATION_RULES}` | Bank slot | ✅ |
| `Return JSON: {"disambiguations": [...]}` | (a) Structural | No change |

**Verdict**: Zero (b) items remaining.

### `_run_single_extraction_pass` prompt

| Content | Classification | Action |
|---------|---------------|--------|
| `Extract ALL references to software architecture components...` | (a) Structural | No change |
| `{self._ENTITY_EXTRACTION_RULES}` | Bank slot | ✅ |
| `Return JSON: {"references": [...]}` | (a) Structural | No change |

**Verdict**: Zero (b) items remaining.

### `_validate_with_evidence` — generic word filter prompt

| Content | Classification | Action |
|---------|---------------|--------|
| `CONTEXTUAL WORD USAGE: Does the word refer to the architecture component...` | (a) Structural (task framing) | No change |
| `{anchor_section}` | (a) Structural (runtime-built) | No change |
| `For each case, determine: - COMPONENT: ... - GENERIC: ...` | (a) Structural (defines classification schema) | No change |
| `Key distinction: A component reference names a specific system entity...` | (a) Structural (axiom-level distinction, abstract and not benchmark-specific) | No change |
| `{self._slot_text("GENERIC_WORD_USAGE_RULES")}` | Bank slot | ✅ |
| `Return JSON: {"results": [...]}` | (a) Structural | No change |

**Verdict**: Zero (b) items remaining. "Key distinction" paragraph is axiom-level abstract guidance, not a benchmark-derived rule.

### `_run_validation_pass` prompt

| Content | Classification | Action |
|---------|---------------|--------|
| `Validate component references... {focus}` | (a) Structural (focus is dynamic, not static) | No change |
| `{self._VALIDATION_RULES}` | Bank slot | ✅ |
| `Return JSON: {"validations": [...]}` | (a) Structural | No change |

**Verdict**: Zero (b) items remaining.

### `_classify_specific_terminals` prompt

| Content | Classification | Action |
|---------|---------------|--------|
| `Architecture components have multi-word names. Identify which terminal words...` | (a) Structural (task framing) | No change |
| `A terminal word is GENERIC if it could refer to any component in any system...` | (a) Structural (axiom-level, abstract classification criteria) | No change |
| `A terminal word is SPECIFIC if it is distinctive or unusual enough...` | (a) Structural (axiom-level) | No change |
| `Also mark a terminal as GENERIC if multiple components in this list share the same terminal word...` | (a) Structural (axiom-level disambiguation rule) | No change |
| `{self._slot_text("COREF_TERMINAL_SPECIFICITY_RULES")}` | Bank slot | ✅ |
| `Return JSON: {"specific": [...], "generic": [...]}` | (a) Structural | No change |

**Verdict**: Zero (b) items remaining. All classification criteria are axiom-level (system-general, not dataset-specific).

### `_coref_cases_in_context` prompt

| Content | Classification | Action |
|---------|---------------|--------|
| `Resolve anaphoric references (pronouns and role-referential noun phrases)...` | (a) Structural | No change |
| `--- Case {i+1}: S{case['sent'].number} ---` / context blocks | (a) Structural (runtime-built) | No change |
| `{self._COREF_RULES}` | Bank slot | ✅ |
| `{self._ANTECEDENT_ALIAS_RULES}` | Bank slot | ✅ |
| `Return JSON: {"resolutions": [...]}` | (a) Structural | No change |

**Verdict**: Zero (b) items remaining.

---

## Summary

**Total static prompts audited**: 12 (ilinker4: 2, s_linker14_voyager: 10)

**Inline behavioral rules requiring migration**: **0**

All behavioral rules in `ilinker4.py` and `s_linker14_voyager.py` are either:
1. Injected through named bank slots (15 slots, all properly wired), or
2. Axiom-level abstract guidance (system-general, not benchmark-specific) classified as structural scaffolding.

The only substantive change in Phase 31 is:
- **ILinker3Injected** (which prepended slot content before base prompts) → **ILinker4** (slot content injected inside prompts at natural position, before `Return JSON:`)
- This makes `SEED_EXTRACTION_RULES` and `SEED_ACTOR_RULES` genuine first-class bank slots with correct injection position.

**GATE-06 status**: All new/modified prompt text clean. "Scheduler"/"Dispatcher" in ILinker4 are SE textbook examples (not benchmark components). No benchmark vocabulary introduced.

**Frozen artifacts**: `ilinker1.py`, `ilinker2.py`, `ilinker3.py`, `s_linker13.py`, `s_linker13_min.py`, `prompts_v2.py`, `data_types_v2.py`, `document_loader_v2.py`, `pcm_parser_v2.py` — all byte-equal to v2.5 state. ✅
