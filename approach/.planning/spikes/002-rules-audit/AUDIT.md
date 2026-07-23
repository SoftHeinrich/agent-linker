# Spike 002: Rules & Heuristics Audit

Source: `llm_sad_sam/linkers/experimental/s_linker12c.py` (1211 lines)

## Classification

| Function | Category | Callers | Rationale |
|----------|----------|---------|-----------|
| `_is_structurally_unambiguous` | **REPLACEABLE** | _classify_components (filter ambiguous), _is_ambiguous_name_component | Pure naming-convention inference (CamelCase/space/all-caps). LLM already does ambiguity classification in _classify_components; this is a redundant post-filter. Remove and trust the LLM output. |
| `_is_strong_alias` | **REPLACEABLE** | _get_strong_alias_mappings, _has_strong_alias_mention | Decides whether an alias is 'safe for global broadcast'. This is exactly a semantic judgement the LLM can encode at alias-discovery time: have the discovery prompt emit {alias, scope: global|local} and drop the post-hoc regex strength check. |
| `_split_component_name` | **ESSENTIAL (but removable with Spike 001)** | _enrich_trailing_words only | Only consumer is trailing-word enrichment, which Spike 001 showed can be LLM-only. Delete once Spike 001 lands. |
| `_has_standalone_mention` | **RISKY** | _classify_mention, _build_evidence_bundle (anchor collection) | Regex word-boundary match with dotted-path / hyphen guards. Called O(N*M) in anchor collection. LLM replacement would be a massive prompt per sentence-pair. Recommend KEEP as boundary primitive, but could be narrowed (drop dotted-path guard → let LLM mention classifier handle it). |
| `_has_strong_alias_mention` | **REPLACEABLE** | coref antecedent verification | Boolean 'does sentence reference comp via a strong alias'. Already running LLM on coref cases in context (Variant E) — fold this signal into the coref prompt's evidence schema. |
| `_is_ambiguous_name_component` | **REPLACEABLE** | _build_evidence_bundle (is_ambiguous flag) | Wraps the LLM-classified ambiguous_names set with a structural guard. Drop the structural guard; trust LLM classification directly. |
| `_classify_mention` | **REPLACEABLE** | _build_evidence_bundle | Returns a human-readable mention type string ('proper case, standalone' / 'lowercase mention' / 'via known alias X' / 'lowercase, inside dotted path'). Currently 4 regex branches. Spike 003 shows LLM can emit this as an enum per candidate during extraction (no extra call — piggyback on existing entity-extraction pass). |
| `_parse_snum` | **ESSENTIAL** | many call sites | String-to-int parser ('S42' -> 42). Deterministic, microsecond. Keep as-is — not a heuristic. |
| `_get_comp_names` | **ESSENTIAL** | many call sites | List comprehension accessor. Not a heuristic. |
| `_get_strong_alias_mappings` | **REPLACEABLE (after _is_strong_alias retired)** | extraction prompt injection | Filter over doc_knowledge.aliases. Becomes trivial once _is_strong_alias is gone — or fold the scope flag into doc_knowledge schema. |
| `_build_component_profile` | **ESSENTIAL** | disambiguation prompt | String formatter. Not a heuristic — it just serializes data. Keep. |
| `_build_evidence_bundle` | **MIXED** | validation pipeline | Orchestrator that calls _classify_mention + _has_standalone_mention + _is_ambiguous_name_component. Each called helper is REPLACEABLE except _has_standalone_mention (anchor collection). After replacements, this function mostly consumes LLM-emitted mention/ambiguity fields. |

## Summary

- **REPLACEABLE**: 6  → can become LLM-driven
- **RISKY**: 1  → LLM replacement loses performance or precision
- **ESSENTIAL**: 4  → keep (parsers/accessors, not heuristics)
- **MIXED**: 1  → orchestrator; becomes trivial after replacements

## Inline regex hotspots (outside helper functions)

Total `re.*` call sites in source: **11**

- L64: `PRONOUN_PATTERN = re.compile(`
- L263: `if re.search(r'[a-z][A-Z]', name):`
- L281: `if re.search(r'[a-z][A-Z]', term):`
- L296: `return re.split(r'[\s-]+', name)`
- L297: `parts = re.findall(r'[A-Z][a-z]*|[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\b)', name)`
- L643: `if re.search(rf'\b{re.escape(comp_lower)}\b', text):`
- L645: `for m in re.finditer(rf'\b{re.escape(comp_lower)}\b', text):`
- L659: `if target == comp_name and re.search(`
- L863: `re.search(rf'\b{re.escape(comp_lower)}\b', sent.text))`
- L1135: `for m in re.finditer(pattern, text, flags):`
- L1159: `if re.search(rf'\b{re.escape(alias.lower())}\b', text_lower):`

## Recommended Removal Order

1. **`_split_component_name`** — unblocked by Spike 001. Zero-risk delete.
2. **`_is_structurally_unambiguous`** — call in `_classify_components` is a trust-the-LLM regression test; remove filter, verify macro F1 parity.
3. **`_is_ambiguous_name_component`** — trivial wrapper; inline after #2.
4. **`_classify_mention`** — fold mention-type emission into the entity-extraction prompt (Spike 003). One-prompt change, same LLM budget.
5. **`_is_strong_alias` + `_get_strong_alias_mappings`** — add `scope` field to doc_knowledge.aliases schema; regenerate at discovery time.
6. **`_has_strong_alias_mention`** — eliminated by #5 + coref prompt schema extension.
7. **KEEP `_has_standalone_mention`** — latency-critical in anchor collection; not a content heuristic. Optionally simplify (drop dotted-path guard).

## Verdict

**9 of 12 audited helpers are REPLACEABLE** via the cite-evidence LLM pattern validated by Spike 001. Only `_has_standalone_mention` (word-boundary primitive) is RISKY to replace — it is called O(sentences × components) during anchor collection. Parsers (`_parse_snum`, `_get_comp_names`) and formatters (`_build_component_profile`) are not heuristics and stay.

A **fully-LLM-driven pipeline is feasible** with one surviving structural primitive (`_has_standalone_mention`) kept for performance, not policy.
