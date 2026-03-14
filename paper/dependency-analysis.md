# Pipeline Dependency Analysis

Reference material for the approach section. Documents the actual data
dependencies between pipeline phases, informing the paper's DAG-based
presentation.

## Phase-to-Variable Dependency Matrix

| Phase | Reads (instance vars) | Reads (local data) | Produces |
|---|---|---|---|
| Model Analysis | — | components | model_knowledge |
| Generic Set Derivation | model_knowledge | components | GENERIC_COMPONENT_WORDS, GENERIC_PARTIALS |
| Document Knowledge | — | sentences, components | doc_knowledge |
| Doc Knowledge Enrichment | GENERIC_PARTIALS, doc_knowledge | sentences, components | mutates doc_knowledge |
| Partial Usage Classification | doc_knowledge | sentences | _activity_partials |
| Subprocess Term Learning | model_knowledge | sentences, components | learned_patterns (subprocess_terms only) |
| Explicit Ref. Extraction | — | text_path, model_path | explicit_links, explicit_set |
| Contextual Ref. Discovery | model_knowledge, doc_knowledge | sentences, components | candidates |
| Targeted Recovery | doc_knowledge | sentences, explicit_links, candidates | appends to candidates |
| Validation | model_knowledge, learned_patterns, doc_knowledge | candidates | validated |
| Anaphoric Ref. Resolution | model_knowledge, learned_patterns, doc_knowledge | sentences, components | anaphoric_links |
| Abbreviated Ref. Matching | doc_knowledge | sentences, explicit_set, validated, anaphoric_links | abbreviated_links |
| Boundary Filter | model_knowledge, doc_knowledge, _activity_partials | preliminary | filtered preliminary |
| Evidence Filter | doc_knowledge | filtered preliminary, explicit_set | final links |

## Dependency DAG (edges = "must complete before")

```
Model Analysis ──────┬──→ Subprocess Term Learning
                     ├──→ Doc Knowledge Enrichment
                     ├──→ Contextual Ref. Discovery
                     ├──→ Validation
                     ├──→ Anaphoric Ref. Resolution
                     ├──→ Boundary Filter
                     └──→ Merge (parent-overlap guard)

Document Knowledge ──┬──→ Doc Knowledge Enrichment
                     ├──→ Contextual Ref. Discovery
                     ├──→ Targeted Recovery
                     ├──→ Validation
                     ├──→ Anaphoric Ref. Resolution (alias checks)
                     ├──→ Abbreviated Ref. Matching
                     └──→ Evidence Filter (alias lookup)

Subprocess Term Learning ┬──→ Validation (subprocess exclusion)
                         └──→ Anaphoric Ref. Resolution (subprocess filter)

Doc Knowledge Enrichment ─→ Partial Usage Classification

Partial Usage Classification ────→ Boundary Filter (_activity_partials)

Document Knowledge ──────────────→ Boundary Filter (alias context)

Explicit Ref. Extraction ┬──→ Targeted Recovery (coverage check)
                         ├──→ Abbreviated Ref. Matching (dedup)
                         └──→ Evidence Filter (provenance check)

Contextual Ref. Discovery ┬──→ Targeted Recovery (coverage check)
                          └──→ Validation

Validation ──────────┬──→ Abbreviated Ref. Matching (dedup)
                     └──→ Merge

Anaphoric Ref. Resolution ┬──→ Abbreviated Ref. Matching (dedup)
                          └──→ Merge

Abbreviated Ref. Matching ──→ Merge
Merge ──────────────────────→ Boundary Filter
Boundary Filter ────────────→ Evidence Filter
```

## Parallel Groups (maximum parallelism schedule)

| Slot | Running | Bottleneck |
|---|---|---|
| L1 | Model Analysis, Document Knowledge, Explicit Ref. Extraction | Explicit (ILinker2 = multiple LLM batches) |
| L1b | Generic Set Derivation (from model_knowledge) | Fast, deterministic |
| L2 | Subprocess Term Learning, Doc Knowledge Enrichment | Both short |
| L2b | Partial Usage Classification | Per-partial LLM calls |
| L3 | Contextual Discovery Pipeline (extract→recover→validate) ∥ Anaphoric Resolution | Anaphoric (full pass) |
| L3b | Abbreviated Ref. Matching | Needs Validation + Anaphoric Resolution |
| L4 | Merge + Boundary Filter + Evidence Filter | Sequential (evidence filter is instant) |

## Critical Path

Two chains race to the merge point:

**Chain A (contextual path):**
Model Analysis → Subprocess Term Learning ─┐
Document Knowledge → Enrichment ──→ Contextual Discovery → Targeted Recovery → Validation ──→ Merge
Explicit Extraction ──────────────→ Targeted Recovery

**Chain B (anaphoric path):**
Model Analysis → Subprocess Term Learning ┬→ Anaphoric Resolution ──→ Merge
Document Knowledge → Enrichment ──────────┘

Anaphoric resolution and the contextual discovery pipeline are on separate
branches that converge only at the merge point. This is why they can
run in parallel, and this is the key insight that the paper's DAG
figure communicates.

## Mapping: Code Methods → Paper Sections

| S-Linker5 Method | Paper Section |
|---|---|
| `_analyze_model` | §3.2.1 Model Analysis |
| `_compute_generic_sets` | §3.2.1 Model Analysis (generic partial words) |
| `_learn_document_knowledge_enriched` | §3.2.2 Document Analysis (alternative name discovery) |
| `_enrich_multiword_partials` | §3.2.2 Document Analysis (statistical enrichment) |
| `_classify_partial_usage` | §3.2.2 Document Analysis (usage classification) |
| `_learn_subprocess_terms` | §3.2.3 Pattern Learning (subprocess debate) |
| `_run_seed` (ILinker2) | §3.3.1 Explicit Reference Extraction |
| `_extract_entities_enriched` | §3.3.2 Contextual Reference Discovery (two-pass intersection) |
| `_targeted_recovery` | §3.3.2 Contextual Reference Discovery (targeted recovery) |
| `_validate_candidates` | §3.3.2 Contextual Reference Discovery (3-step validation) |
| `_coref_cases_in_context` | §3.3.3 Anaphoric Reference Resolution |
| `_inject_partial_references` | §3.3.4 Abbreviated Reference Matching |
| `_combine_links` (inline) | §3.4.1 Priority-Based Merge |
| `_apply_boundary_filters` | §3.4.2 Convention-Aware Boundary Filter |
| keep_coref filter (inline) | §3.4.3 Evidence Filter |
