# Pipeline Dependency Analysis

Reference material for the approach section. Documents the actual data
dependencies between S-Linker12e pipeline phases, informing the paper's
DAG-based presentation.

## Phase-to-Variable Dependency Matrix

| Phase | Reads (instance vars) | Reads (local data) | Produces |
|---|---|---|---|
| Model Analysis | — | components | model_knowledge |
| Document Knowledge (incl. trailing-word) | — | sentences, components | doc_knowledge |
| Seed Extraction (ILinker3) | — | sentences, components | raw_seed_links |
| Seed Validation | model_knowledge, doc_knowledge | raw_seed_links, sent_map | seed_links |
| Entity Pipeline (extraction) | doc_knowledge | sentences, components | candidates |
| Entity Pipeline (validation) | model_knowledge, doc_knowledge | candidates, sent_map | validated |
| Coreference | doc_knowledge | sentences, components, sent_map | coref_links |
| Merge (dedup) | — | seed_links, entity_links, coref_links | final links |

## Dependency DAG (edges = "must complete before")

```
Model Analysis ──────┬──→ Seed Validation
                     ├──→ Entity Pipeline (generic-mention detection)
                     └──→ Entity Pipeline (validation, for ambiguity info)

Document Knowledge ──┬──→ Seed Validation (alias table, component profiles)
                     ├──→ Entity Pipeline (aliases in extraction prompts)
                     ├──→ Entity Pipeline (validation, for alias info)
                     └──→ Coreference (alias verification of antecedents)

Seed Extraction ─────┬──→ Seed Validation (raw seed links to disambiguate)

Entity Pipeline (extraction) ──→ Entity Pipeline (validation) ──→ Merge

Seed Validation ──→ Merge
Entity Pipeline ──→ Merge
Coreference ──────→ Merge
```

## Parallel Groups (maximum parallelism schedule)

| Slot | Running | Bottleneck |
|---|---|---|
| Tier 1 | Model Analysis ∥ Document Knowledge ∥ Seed Extraction | ILinker3 (multiple LLM batches) |
| Tier 2 | Seed Validation ∥ Entity Pipeline ∥ Coreference | Entity extraction (largest batch count) |
| Tier 3 | Merge (dedup) | Instant, deterministic |

## Critical Path

Two chains race to the merge point:

**Chain A (entity path):**
Document Knowledge → Entity Extraction (×2 parallel passes) → Entity Validation (generic filter + 2-pass intersection) → Merge

**Chain B (seed path):**
Seed Extraction → Seed Validation (per-component disambiguation) → Merge

Entity extraction is typically the longest-running step due to dual-pass
consensus over all document batches. Seed validation groups by component
and issues one LLM call per component, which is usually faster.

Coreference runs independently and is rarely on the critical path.

## Mapping: Code Methods → Paper Sections

| S-Linker12e Method | Paper Section |
|---|---|
| `_analyze_model` / `_classify_components` | §3.2.1 Architectural Model Understanding |
| `_learn_document_knowledge` | §3.2.2 Architectural Document Understanding (core discovery + judge) |
| `_enrich_trailing_words` | §3.2.2 Architectural Document Understanding (trailing-word enrichment) |
| `_run_seed` (ILinker3) | §3.2.3 Seed Extraction |
| `_run_seed_validation` | §3.3.1 Seed Reference Disambiguation |
| `_run_entity_pipeline` | §3.3.2 Contextual Reference Agent |
| `_extract_entities` | §3.3.2 Contextual Reference Agent (dual-pass consensus) |
| `_filter_generic_mentions` | §3.3.2 Contextual Reference Agent (generic-mention detection) |
| `_validate_candidates` | §3.3.2 Contextual Reference Agent (two-pass intersection) |
| `_run_coreference` / `_coref_cases_in_context` | §3.3.3 Anaphoric Reference Agent |
| dedup loop in `link()` | §3.4 Link Consolidation (first-seen dedup) |

## Removed from S-Linker12e (present in earlier versions)

| Phase | Removed in | Reason |
|---|---|---|
| Subprocess Term Learning | S-Linker6 | Zero F1 contribution |
| Targeted Recovery | S-Linker6 | Zero net gain |
| Convention-Aware Boundary Filter | S-Linker7 | ICSE simplification |
| Evidence Filter | S-Linker7 | Subsumed by validation |
| Implementation Variant Filtering | S-Linker9 | Zero text mentions |
| CamelCase Synonym Injection | S-Linker9 | Zero dependents |
| CamelCase Rescue Override | S-Linker9 | Never fires |
| Code-first Auto-approval | S-Linker10 | Replaced by evidence-stratified voting |
| Count>=3 Enrichment Threshold | S-Linker10 | Replaced by LLM word usage classification |
| Evidence-Stratified Voting | S-Linker12e | Replaced by pure intersection |
| Abbreviated Reference Agent | S-Linker12e | Trailing words feed alias table → entity extraction |
| Partial-Ref. Refinement (separate step) | S-Linker12e | Merged into document analysis |

## Key Differences from approach4.tex (S-Linker10 era)

1. **Layer 1**: 3 parallel analyses (was 2 + sub-tier 1b). Trailing-word enrichment
   is now inside `_learn_document_knowledge`, not a separate step requiring both
   model and doc analysis to complete first.

2. **Seed extraction in Layer 1**: ILinker3 runs in parallel with knowledge
   acquisition (was conceptually Layer 2 "starting early").

3. **Seed validation (new)**: Per-component disambiguation with component profile,
   anchor sentences, and COMPONENT/OTHER classification. Replaces evidence-stratified
   revalidation of seed links.

4. **Entity validation simplified**: Pure intersection (both passes must agree) for
   ALL candidates. No alias-based stratification (was: union for alias-confirmed,
   intersection for exact-name).

5. **No abbreviated reference agent**: Trailing-word aliases now feed directly into
   entity extraction prompts via the alias table. No separate deterministic scanning
   + partial validation step.

6. **Consolidation simplified**: Just priority-ordered dedup (seed > entity > coref).
   No boundary filter, no evidence filter.
