# Pipeline Dependency Analysis

Reference material for the approach section. Documents the actual data
dependencies between S-Linker10 pipeline phases, informing the paper's
DAG-based presentation.

## Phase-to-Variable Dependency Matrix

| Phase | Reads (instance vars) | Reads (local data) | Produces |
|---|---|---|---|
| Model Analysis | — | components | model_knowledge |
| Document Knowledge | — | sentences, components | doc_knowledge |
| Partial-Ref. Enrichment (LLM) | doc_knowledge | sentences, components | mutates doc_knowledge |
| Explicit Ref. Extraction | — | text_path, model_path | seed_links, seed_set |
| Contextual Ref. Discovery | model_knowledge, doc_knowledge | sentences, components | candidates |
| Validation (evidence-stratified) | model_knowledge, doc_knowledge | candidates | validated |
| Anaphoric Ref. Resolution | doc_knowledge | sentences, components | coref_links |
| Abbreviated Ref. Matching | doc_knowledge | sentences, seed_set, validated, coref_links | partial_candidates |
| Partial Validation | model_knowledge, doc_knowledge | partial_candidates | partial_validated |
| Merge (dedup) | — | seed, entity, coref, partial | final links |

## Dependency DAG (edges = "must complete before")

```
Model Analysis ──────┬──→ Contextual Ref. Discovery
                     ├──→ Validation
                     └──→ Partial Validation

Document Knowledge ──┬──→ Partial-Ref. Enrichment (LLM)
                     ├──→ Contextual Ref. Discovery
                     ├──→ Validation
                     ├──→ Anaphoric Ref. Resolution (alias checks)
                     ├──→ Abbreviated Ref. Matching
                     └──→ Partial Validation

Partial-Ref. Enrichment ──→ All of Tier 2 (enriched doc_knowledge)

Explicit Ref. Extraction ──→ Abbreviated Ref. Matching (dedup)

Contextual Ref. Discovery ──→ Validation

Validation ──────────┬──→ Abbreviated Ref. Matching (dedup)
                     └──→ Merge

Anaphoric Ref. Resolution ┬──→ Abbreviated Ref. Matching (dedup)
                          └──→ Merge

Abbreviated Ref. Matching ──→ Partial Validation ──→ Merge
```

## Parallel Groups (maximum parallelism schedule)

| Slot | Running | Bottleneck |
|---|---|---|
| Tier 1 | Model Analysis ∥ Document Knowledge ∥ Explicit Ref. Extraction | ILinker2 (multiple LLM batches) |
| Tier 1 (cont.) | Generic Partials Derivation | Fast, deterministic (after Model Analysis) |
| Tier 1.5 | Partial-Ref. Enrichment (LLM word usage) | Per-partial LLM calls (~2-4) |
| Tier 2 | Contextual Discovery Pipeline (extract→validate) ∥ Anaphoric Resolution | Entity extraction (largest batch count) |
| Tier 2.5 | Abbreviated Ref. Matching → Partial Validation | Sequential (needs entity+coref done) |
| Tier 3 | Merge (dedup) | Instant, deterministic |

## Critical Path

Two chains race to the merge point:

**Chain A (contextual path):**
Model Analysis → Generic Partials → Enrichment → Contextual Discovery → Validation → Merge

**Chain B (anaphoric path):**
Document Knowledge → Enrichment → Anaphoric Resolution → Merge

Anaphoric resolution and the contextual discovery pipeline are on separate
branches that converge only at the merge point. This is why they can
run in parallel, and this is the key insight that the paper's DAG
figure communicates.

## Mapping: Code Methods → Paper Sections

| S-Linker10 Method | Paper Section |
|---|---|
| `_analyze_model` | §3.2.1 Name Ambiguity Classification |
| `_learn_document_knowledge_enriched` | §3.2.2 Alternative Name Discovery (core discovery) |
| `_enrich_multiword_partials` | §3.2.2 Alternative Name Discovery (LLM word usage enrichment) |
| `_run_seed` (ILinker2) | §3.3.1 Explicit Reference Extraction |
| `_extract_entities_enriched` | §3.3.2 Contextual Reference Discovery (same-prompt intersection) |
| `_validate_intersect` | §3.3.2 Contextual Reference Discovery (evidence-stratified voting) |
| `_coref_cases_in_context` | §3.3.3 Anaphoric Reference Resolution |
| `_inject_partial_candidates` | §3.3.4 Abbreviated Reference Matching |
| dedup loop in `link()` | §3.4 Link Consolidation (first-seen dedup) |

## Removed from S-Linker10 (present in earlier versions)

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
