# Pipeline Dependency Analysis

Reference material for the approach section. Documents the actual data
dependencies between pipeline phases, informing the paper's DAG-based
presentation.

## Phase-to-Variable Dependency Matrix

| Phase | Reads (instance vars) | Reads (local data) | Produces |
|---|---|---|---|
| Model Analysis | — | components | model_knowledge |
| Generic Set Derivation | model_knowledge | components | GENERIC_COMPONENT_WORDS, GENERIC_PARTIALS |
| Document Profiling | — | sentences, components | _is_complex, pronoun_ratio |
| Document Knowledge | — | sentences, components | doc_knowledge |
| Doc Knowledge Enrichment | GENERIC_PARTIALS, doc_knowledge | sentences, components | mutates doc_knowledge |
| Partial Usage Classification | doc_knowledge | sentences | _activity_partials |
| Pattern Learning | model_knowledge | sentences, components | learned_patterns |
| Seed Extraction | — | text_path, model_path | seed_links, seed_set |
| Entity Extraction | model_knowledge, doc_knowledge | sentences, components | candidates |
| Targeted Recovery | doc_knowledge | sentences, seed_links, candidates | appends to candidates |
| Validation | model_knowledge, learned_patterns, doc_knowledge, GENERIC_COMPONENT_WORDS | candidates | validated |
| Coreference | model_knowledge, learned_patterns, doc_knowledge | sentences, components | coref_links |
| Partial Injection | doc_knowledge | sentences, seed_set, validated, coref_links | partial_links |
| Boundary Filter | model_knowledge, doc_knowledge, _activity_partials | preliminary | filtered preliminary |
| Evidence Filter | doc_knowledge | filtered preliminary, seed_set | final links |

## Dependency DAG (edges = "must complete before")

```
Model Analysis ──────┬──→ Pattern Learning
                     ├──→ Doc Knowledge Enrichment
                     ├──→ Entity Extraction
                     ├──→ Validation
                     ├──→ Coreference
                     ├──→ Boundary Filter
                     └──→ Merge (parent-overlap guard)

Document Profiling ──────→ (diagnostic logging only; no downstream consumers)

Document Knowledge ──┬──→ Doc Knowledge Enrichment
                     ├──→ Entity Extraction
                     ├──→ Targeted Recovery
                     ├──→ Validation
                     ├──→ Coreference (alias checks)
                     ├──→ Partial Injection
                     └──→ Evidence Filter (alias lookup)

Pattern Learning ────┬──→ Validation
                     └──→ Coreference (subprocess filter)

Doc Knowledge Enrichment ─→ Partial Usage Classification

Partial Usage Classification ────→ Boundary Filter (_activity_partials)

Document Knowledge ──────────────→ Boundary Filter (alias context)

Seed Extraction ─────┬──→ Targeted Recovery (coverage check)
                     ├──→ Partial Injection (dedup)
                     └──→ Evidence Filter (seed provenance check)

Entity Extraction ───┬──→ Targeted Recovery (coverage check)
                     └──→ Validation

Validation ──────────┬──→ Partial Injection (dedup)
                     └──→ Merge

Coreference ─────────┬──→ Partial Injection (dedup)
                     └──→ Merge

Partial Injection ───────→ Merge
Merge ───────────────────→ Boundary Filter
Boundary Filter ─────────→ Evidence Filter
```

## Parallel Groups (maximum parallelism schedule)

| Slot | Running | Bottleneck |
|---|---|---|
| T1 | Model Analysis, Document Profiling, Document Knowledge, Seed Extraction | Seed (ILinker2 = multiple LLM batches) |
| T1b | Generic Set Derivation (from model_knowledge) | Fast, deterministic |
| T2 | Pattern Learning, Doc Knowledge Enrichment | Both short |
| T2b | Partial Usage Classification | Per-partial LLM calls |
| T3 | Entity Pipeline (extract→guard→recover→validate) ∥ Coreference | Coreference (full pass) |
| T3b | Partial Injection | Needs Validation + Coreference |
| T4 | Merge + Boundary Filter + Evidence Filter | Sequential (deterministic evidence filter is instant) |

## Critical Path

Two chains race to the merge point:

**Chain A (entity path):**
Model Analysis → Pattern Learning ─┐
Document Knowledge → Enrichment ──→ Entity Extraction → Targeted Recovery → Validation ──→ Merge
Seed Extraction ──────────────────→ Targeted Recovery

**Chain B (coreference path):**
Model Analysis → Pattern Learning ┬→ Coreference ──→ Merge
Document Knowledge → Enrichment ──┘

Phase 7 (Coreference) and Phase 5→5b→6 (Entity path) are on separate
branches that converge only at the merge point. This is why they can
run in parallel, and this is the key insight that the paper's DAG
figure should communicate.

## Mapping: Code Phases → Paper Sections

| S-Linker Method | Paper Section |
|---|---|
| `_compute_complexity` | §3.2.2 Document Analysis (profiling paragraph) |
| `_analyze_model` | §3.2.1 Model Analysis |
| `_compute_generic_sets` | §3.2.1 Model Analysis (generic partial words) |
| `_learn_document_knowledge_enriched` | §3.2.2 Document Analysis (alternative names) |
| `_enrich_multiword_partials` | §3.2.2 Document Analysis (enrichment paragraph) |
| `_classify_partial_usage` | §3.2.2 Document Analysis (partial usage classification) |
| `_learn_patterns_with_debate` | §3.2.3 Pattern Learning |
| `_run_seed` | §3.3.1 Seed Extraction |
| `_extract_entities_enriched` | §3.3.2 Entity Extraction and Validation (extraction) |
| `_targeted_recovery` | §3.3.2 Entity Extraction and Validation (targeted recovery) |
| `_validate_intersect` | §3.3.2 Entity Extraction and Validation (validation) |
| `_coref_cases_in_context` | §3.3.3 Coreference Resolution |
| `_inject_partial_references` | §3.3.4 Partial Reference Injection |
| `_combine_links` | §3.4.1 Priority-Based Merge |
| `_apply_boundary_filters` | §3.4.2 Convention-Aware Boundary Filter |
| keep_coref filter (inline in `link()`) | §3.4.3 Evidence Filter |
