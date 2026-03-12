# Pipeline Dependency Analysis

Reference material for the approach section. Documents the actual data
dependencies between pipeline phases, informing the paper's DAG-based
presentation.

## Phase-to-Variable Dependency Matrix

| Phase | Reads (instance vars) | Reads (local data) | Produces |
|---|---|---|---|
| Model Analysis | — | components | model_knowledge, GENERIC_COMPONENT_WORDS, GENERIC_PARTIALS |
| Document Profiling | — | sentences, components | doc_profile, _is_complex |
| Document Knowledge | — | sentences, components | doc_knowledge |
| Doc Knowledge Enrichment | GENERIC_PARTIALS | sentences, components, doc_knowledge | mutates doc_knowledge |
| Pattern Learning | model_knowledge | sentences, components | learned_patterns |
| Seed Extraction | — | text_path, model_path | transarc_links, transarc_set |
| Entity Extraction | model_knowledge, doc_knowledge | sentences, components | candidates |
| Targeted Recovery | doc_knowledge | sentences, transarc_links, candidates | appends to candidates |
| Validation | model_knowledge, learned_patterns, doc_knowledge, GENERIC_COMPONENT_WORDS | candidates | validated |
| Coreference | _is_complex, model_knowledge, learned_patterns, doc_knowledge | sentences, components | coref_links |
| Partial Injection | doc_knowledge | sentences, transarc_set, validated, coref_links | partial_links |
| Boundary Filter | model_knowledge | preliminary, transarc_set | filtered preliminary |
| Judge Review | _is_complex, model_knowledge, doc_knowledge | preliminary, transarc_set | final links |

## Dependency DAG (edges = "must complete before")

```
Model Analysis ──────┬──→ Pattern Learning
                     ├──→ Doc Knowledge Enrichment
                     ├──→ Entity Extraction
                     ├──→ Validation
                     ├──→ Coreference
                     ├──→ Boundary Filter
                     ├──→ Judge Review
                     └──→ Merge (parent-overlap guard)

Document Profiling ──┬──→ Coreference (mode selection)
                     └──→ Judge Review (adaptive context)

Document Knowledge ──┬──→ Doc Knowledge Enrichment
                     ├──→ Entity Extraction
                     ├──→ Targeted Recovery
                     ├──→ Validation
                     ├──→ Coreference (alias checks)
                     ├──→ Partial Injection
                     └──→ Judge Review

Pattern Learning ────┬──→ Validation
                     └──→ Coreference (subprocess filter)

Seed Extraction ─────┬──→ Targeted Recovery (coverage check)
                     ├──→ Partial Injection (dedup)
                     ├──→ Boundary Filter (immunity)
                     └──→ Judge Review (immunity)

Entity Extraction ───┬──→ Targeted Recovery (coverage check)
                     └──→ Validation

Validation ──────────┬──→ Partial Injection (dedup)
                     └──→ Merge

Coreference ─────────┬──→ Partial Injection (dedup)
                     └──→ Merge

Partial Injection ───────→ Merge
Merge ───────────────────→ Boundary Filter
Boundary Filter ─────────→ Judge Review
```

## Parallel Groups (maximum parallelism schedule)

| Slot | Running | Bottleneck |
|---|---|---|
| T1 | Model Analysis, Document Profiling, Document Knowledge, Seed Extraction | Seed (ILinker2 = multiple LLM batches) |
| T2 | Pattern Learning, Doc Knowledge Enrichment | Both short |
| T3 | Entity Extraction ∥ Coreference | Coreference (2x full pass) |
| T4 | Targeted Recovery | Needs Seed + Entity outputs |
| T5 | Validation | Needs Targeted Recovery + Patterns |
| T6 | Partial Injection | Merge point: needs Validation + Coreference |
| T7 | Merge + Boundary Filter | Sequential |
| T8 | Judge Review | Sequential |

## Critical Path

Two chains race to the merge point:

**Chain A (entity path):**
Model Analysis → Pattern Learning ─┐
Document Knowledge → Enrichment ──→ Entity Extraction → Targeted Recovery → Validation ──→ Merge
Seed Extraction ──────────────────→ Targeted Recovery

**Chain B (coreference path):**
Document Profiling ───────────────┐
Model Analysis → Pattern Learning ┼→ Coreference ──→ Merge
Document Knowledge → Enrichment ──┘

Phase 7 (Coreference) and Phase 5→5b→6 (Entity path) are on separate
branches that converge only at the merge point. This is why they can
run in parallel, and this is the key insight that the paper's DAG
figure should communicate.

## Mapping: Code Phases → Paper Sections

| Code Phase | Paper Section |
|---|---|
| Phase 0 | §3.2 Document Analysis (profiling paragraph) |
| Phase 1 | §3.2 Model Analysis |
| Phase 2 | §3.2 Pattern Learning |
| Phase 3, 3b | §3.2 Document Analysis (alternative names paragraph) |
| Phase 4 | §3.3 Seed Extraction |
| Phase 5, 5b | §3.3 Entity Extraction and Validation (extraction) |
| Phase 6 | §3.3 Entity Extraction and Validation (validation) |
| Phase 7 | §3.3 Coreference Resolution |
| Phase 8b | §3.3 Partial Reference Injection |
| Combine | §3.4 Priority-Based Merge |
| Phase 8c | §3.4 Convention-Aware Boundary Filter |
| Phase 9 | §3.4 Judicial Review |
