# Architecture Research

**Domain:** Iterative ablation of a Python LLM-pipeline (rule removal, per-variant F1 regression testing)
**Researched:** 2026-04-21
**Confidence:** HIGH — all findings from direct code inspection; no external sources needed

---

## System Overview

The milestone adds `s_linker13a` through (at most) `s_linker13f`, each removing one logical group of rules from `s_linker12c`. The diagram below shows the stable skeleton (untouched across variants) and the mutable cells that change per variant.

```
┌──────────────────────────────────────────────────────────────────────────┐
│  run_ablation.py  (harness — no changes needed)                          │
│    CANONICAL_VARIANTS += ["s_linker13a", "s_linker13b", ...]             │
│    VARIANT_SPECS    += {module, class_name, description}                 │
└───────────────────────────────┬──────────────────────────────────────────┘
                                 │ build_linker() / importlib.import_module
┌───────────────────────────────▼──────────────────────────────────────────┐
│  s_linker13x.py  (standalone variant, one file per rule removal)         │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │  Tier 1: Knowledge Acquisition (parallel — _run_parallel)           │ │
│  │  ┌────────────────┐  ┌──────────────────────────┐  ┌─────────────┐ │ │
│  │  │ _analyze_model │  │ _learn_document_knowledge │  │ _run_seed   │ │ │
│  │  │                │  │  (alias discovery +       │  │ (ILinker3)  │ │ │
│  │  │ [MUTABLE]      │  │   trailing-word backstop) │  │ [STABLE]    │ │ │
│  │  │ removes        │  │  [MUTABLE] adds scope     │  │             │ │ │
│  │  │ _is_struct_    │  │   field; removes _split,  │  │             │ │ │
│  │  │ unambiguous    │  │   _is_strong_alias        │  │             │ │ │
│  │  └────────────────┘  └──────────────────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                 │                                        │
│           model_knowledge       │ doc_knowledge                          │
│           (ambiguous_names)     │ (aliases: {term -> comp, scope})       │
│                                 │                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │  Tier 2: Link Recovery (parallel — _run_parallel)                   │ │
│  │  ┌────────────────┐  ┌──────────────────────────┐  ┌─────────────┐ │ │
│  │  │ Seed           │  │ Entity pipeline           │  │ Coref       │ │ │
│  │  │ validation     │  │                           │  │             │ │ │
│  │  │ [MUTABLE]      │  │ [MUTABLE]                 │  │ [MUTABLE]   │ │ │
│  │  │ _classify_     │  │ mention_type from LLM     │  │ _has_strong │ │ │
│  │  │ mention →      │  │ enum (Spike 003); drops   │  │ _alias_     │ │ │
│  │  │ LLM enum       │  │ inline re.search calls    │  │ mention →   │ │ │
│  │  │                │  │                           │  │ scope field │ │ │
│  │  └────────────────┘  └──────────────────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                 │                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │  Tier 3: Consolidation (dedup — STABLE across all variants)         │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  Shared helpers (STABLE — never change):                                 │
│    _parse_snum  _get_comp_names  _build_component_profile                │
│    _has_standalone_mention  _run_parallel  checkpoint/logging            │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Component Boundaries

| Component | Responsibility | Changes in 13-series | Communicates With |
|-----------|---------------|----------------------|-------------------|
| `run_ablation.py` | CLI harness: load variant, run on datasets, compute metrics, print table | Add entries to `CANONICAL_VARIANTS` + `VARIANT_SPECS`; nothing else | All `s_linker13x` via `build_linker()` |
| `s_linker13x.py` | Self-contained pipeline variant; one file = one rule removal | All rule changes live here; duplicate boilerplate is intentional | `prompts_v2.py`, `ilinker3.py`, `data_types_v2.py`, `llm_client.py` |
| `prompts_v2.py` | Prompt constants shared by all v2-stack linkers | Add `MENTION_TYPE_SCHEMA` constant for Spike 003 prompt extension; add `ALIAS_SCOPE_SCHEMA` for alias-scope field | All 13-series variants via `from prompts_v2 import ...` |
| `data_types_v2.py` | Core dataclasses (`SadSamLink`, `DocumentKnowledge`, `ModelKnowledge`, `CandidateLink`) | Add `scope` field to alias representation (see alias-scope section below) | All variant files |
| `ilinker3.py` | Seed extractor (ILinker3 two-pass) | No changes — seed layer is out of scope for this milestone | `s_linker13x._run_seed` |
| `LLMClient` | Backend abstraction (Claude, OpenAI, CHECKPOINT) | No changes | All LLM calls inside variant files |

---

## File/Variant Layout

One standalone file per rule removal, no inheritance. This is the deliberate project policy.

```
src/llm_sad_sam/linkers/experimental/
├── s_linker12c.py          baseline (read-only reference)
├── s_linker13a.py          removal 1: _split_component_name  (Spike 001)
├── s_linker13b.py          removal 2: _is_structurally_unambiguous post-filter
├── s_linker13c.py          removal 3: _is_ambiguous_name_component wrapper
├── s_linker13d.py          removal 4: _classify_mention + 4 regex branches  (Spike 003)
├── s_linker13e.py          removal 5: _is_strong_alias + _get_strong_alias_mappings
├── s_linker13f.py          removal 6: _has_strong_alias_mention  (fold into coref)
└── s_linker13.py           winner (promoted from best 13x that holds F1 >= 93%)
```

Each `s_linker13x.py` is created by copying the immediately preceding variant and applying exactly one change. Copying from predecessor (not from 12c) preserves the cumulative removal chain and prevents accidentally reverting a previously removed rule.

---

## Spike Integration: Where Each LLM Call Slots Into the DAG

### Spike 001 — LLM Trailing-Word Enrichment (unblocks 13a)

Spike 001's `fully_llm_driven()` function replaces `_enrich_trailing_words`'s structural candidate-gate. It slots into the existing Tier 1 call inside `_learn_document_knowledge_enriched`, exactly where `self._enrich_trailing_words(knowledge, sentences, components)` is today.

**Change is local to `_learn_document_knowledge_enriched`:**

- Delete `_split_component_name` (the CamelCase/space/hyphen splitter).
- Replace the structural candidate-filter loop inside `_enrich_trailing_words` with a single LLM call using the `LLM_ONLY_PROMPT` pattern from the spike.
- Keep the lightweight post-guardrail (evidence sentence must contain alias, must not contain full component name) — this is not a gate, it is a post-condition sanity check.
- Prompt cost: net zero (structural gate was free, verify was 1 call; proposed is 1 combined call).

No new DAG tier is needed. The call stays inside the Tier 1 `doc_knowledge` branch.

### Spike 003 — LLM Mention-Type Enum (unblocks 13d)

Spike 003 adds `mention_type` as a field on each extracted candidate, emitted by the existing entity-extraction pass (`_run_single_extraction_pass`). No new LLM call.

**Change is a prompt-schema extension inside `_run_single_extraction_pass`:**

- Extend the JSON schema in the extraction prompt to include `"mention_type"` (enum: `proper_case | lowercase | dotted_path | via_alias | indirect`) and optional `"alias_used"`.
- Add `format_mention(mention_type, alias_used)` formatter (copied from spike; produces the same strings `_classify_mention` produced).
- In `_build_evidence_bundle`, replace the call to `self._classify_mention(...)` with `format_mention(candidate.mention_type, candidate.alias_used)`.
- Delete `_classify_mention` and its 4 regex branches.

The extraction pass is Tier 2 entity pipeline, which is already running in parallel with seed validation and coref. The schema extension adds no new parallelism requirements.

---

## Alias-Scope Schema Extension (Data Flow for Removals 13e/13f)

`_is_strong_alias` currently makes a per-alias decision at consumption time (in `_get_strong_alias_mappings` and `_has_strong_alias_mention`). The replacement moves this decision upstream to the alias-discovery prompt, where the LLM already has full context.

### Where `doc_knowledge` is produced

`_learn_document_knowledge_enriched` (Tier 1, `doc_knowledge` branch) produces `DocumentKnowledge`. Its `aliases` dict is currently `{term: comp_name}` — a flat mapping with no scope annotation.

### Schema change

Extend the alias-discovery prompt (prompt1 inside `_learn_document_knowledge_enriched`) to emit:

```json
{
  "abbreviations": {"short_form": {"component": "FullName", "scope": "global"}},
  "synonyms":      {"specific_name": {"component": "FullName", "scope": "local"}}
}
```

`scope: global` means the alias is safe to broadcast to entity extraction and coref (equivalent to the old `_is_strong_alias` returning True). `scope: local` means restrict to contexts where the full component name appears nearby.

### How to propagate scope without touching the upstream seed layer

Option A (preferred): Store scope in `DocumentKnowledge` alongside aliases.

Change `data_types_v2.py`:

```python
@dataclass
class AliasEntry:
    component: str
    scope: str  # "global" | "local"

@dataclass
class DocumentKnowledge:
    aliases: dict[str, AliasEntry] = field(default_factory=dict)
    # legacy flat fields kept for backward compat with pre-13e variants
    abbreviations: dict[str, str] = field(default_factory=dict)
    synonyms: dict[str, str] = field(default_factory=dict)
    partial_references: dict[str, str] = field(default_factory=dict)
```

All callers of `doc_knowledge.aliases` in the 13e variant file update from `doc_knowledge.aliases[term]` to `doc_knowledge.aliases[term].component`. The filter `_get_strong_alias_mappings()` becomes a one-liner: `[f"{t}={e.component}" for t, e in aliases.items() if e.scope == "global"]` — and can then be inlined and the function deleted.

**Important**: This schema change is local to the 13e variant file. The pre-13e variants (`12c`, `13a`–`13d`) use the old flat dict and must not be touched. `data_types_v2.py` cannot be modified directly without affecting all variants. The correct approach is to define `AliasEntry` inline inside `s_linker13e.py` (or in a small `_alias_types.py` helper that only 13e+ import). This keeps the change self-contained per the standalone-file policy.

Option B (alternative): Keep the flat `aliases` dict; add a parallel `alias_scope` dict (`{term: "global"|"local"}`). Simpler for the transition but leaves a messier API.

Recommendation: Use Option A inline definition. Define `AliasEntry` at the top of `s_linker13e.py` alongside the other dataclasses. This avoids modifying shared infrastructure and keeps the variant fully standalone.

### Coref integration for `_has_strong_alias_mention` (13f)

After 13e, the coref antecedent-verification check (`_has_strong_alias_mention`) can be folded into the coref prompt's evidence schema. The coref prompt already receives the component name and context sentences. Add a field to the resolution output: `"antecedent_via_alias": "AliasWord" | null`. The code check becomes: `if resolution.antecedent_via_alias and (resolution.antecedent_via_alias in doc_knowledge.aliases)`. This removes the regex scan over aliases in `_has_strong_alias_mention`.

---

## Ablation Harness: Runner Integration

### No new driver needed

`run_ablation.py` already handles per-variant scoring, CSV export, and summary tables. The only change required is two additions per new variant:

1. Append to `CANONICAL_VARIANTS` list.
2. Add entry to `VARIANT_SPECS` dict with `module`, `class_name`, `description`.

No flags, no config files, no per-variant script. The harness design scales to the 13-series without modification.

### Running a single variant during development

```bash
python run_ablation.py --variants s_linker13a --datasets teammates bigbluebutton
```

This is already supported. Use teammates + bigbluebutton first (hard tier) before running all 5 datasets.

### Full 5-dataset sweep on a promoted candidate

```bash
python run_ablation.py --variants s_linker12c s_linker13a s_linker13b ... --datasets mediastore teastore teammates bigbluebutton jabref
```

Running 12c alongside each 13x variant in the same sweep gives the comparison row for the ablation table automatically.

---

## Checkpoint/Caching Strategy

**Recommendation: Keep existing per-phase pickles; do not add per-rule ablation infrastructure.**

Rationale:

The ablation unit for this milestone is the linker variant, not the individual rule. The project decision (from PROJECT.md) is "ablation unit = linker variant, not individual rule". V30c-style per-rule offline ablation (test_heuristics.py loading phase checkpoints and toggling rule flags) was valuable when the goal was to measure rule contribution in isolation. Here the goal is different: each variant is independently evaluated against gold standards, so the signal of interest is the full-pipeline F1 delta, not per-rule microscopy.

Per-phase checkpoints inside each variant remain useful for:
- Diagnosing which tier caused a regression when a variant drops F1.
- Resuming a run interrupted mid-pipeline on a slow dataset (teammates).
- Cross-variant checkpoint migration to confirm that a rule removal did not corrupt shared state.

The existing `_save_phase` / `_checkpoint_dir` infrastructure in `s_linker12c` carries forward to all 13-series variants unchanged (they copy the full checkpoint/logging block). The checkpoint directory name should be updated to reflect the variant: change the hardcoded `"s_linker12c"` string in `_checkpoint_dir` to `self._variant_name` (a class-level constant added to each variant file, e.g., `_VARIANT_NAME = "s_linker13a"`).

Do not build a feature-flag system or shared-state ablation harness. The standalone-file policy exists precisely to avoid shared mutable infrastructure that obscures what each variant is actually doing.

---

## Suggested Build Order

The order follows Spike 002's ranked removal plan, with two modifications noted.

| Step | Variant | Rule(s) Removed | Dependency | Risk |
|------|---------|-----------------|------------|------|
| 1 | `s_linker13a` | `_split_component_name` (Spike 001) | None — Spike 001 fully validated | LOW — 1 helper, 1 call site in `_enrich_trailing_words` |
| 2 | `s_linker13b` | `_is_structurally_unambiguous` post-filter in `_classify_components` | 13a (establishes new baseline) | LOW — remove 3-line conditional in `_classify_components`; trust LLM output directly |
| 3 | `s_linker13c` | `_is_ambiguous_name_component` wrapper | 13b (ambiguity classification now clean) | LOW — inline the `ambiguous_names` set lookup; drop structural guard |
| 4 | `s_linker13d` | `_classify_mention` + 4 regex branches (Spike 003) | 13c (evidence bundle callers are clean) | LOW — prompt-schema extension in extraction pass, zero new LLM calls; parity proven by Spike 003 |
| 5 | `s_linker13e` | `_is_strong_alias` + `_get_strong_alias_mappings` | 13d (alias consumption is simplified) | MEDIUM — requires alias-scope schema in `_learn_document_knowledge`; changes how strong aliases flow to extraction and coref |
| 6 | `s_linker13f` | `_has_strong_alias_mention` | 13e (scope field makes this trivially replaceable) | LOW — fold antecedent alias check into coref prompt field |

**Deviation from Spike 002 order**: Spike 002 listed `_classify_mention` as step 4 and `_is_strong_alias` + `_get_strong_alias_mappings` as step 5. That order is preserved here. However, note that `_has_strong_alias_mention` (step 6) depends on the alias scope field introduced in step 5; it cannot move earlier without that dependency.

**Deferred**: `_has_standalone_mention` stays. Spike 002 classified it RISKY (O(N×M) anchor collection; LLM replacement would require a prompt-per-sentence-pair at prohibitive cost). Keep as the one surviving structural primitive.

### Why this order unblocks the next step

- `13a` removes `_split_component_name`, which is only called from `_enrich_trailing_words`. Once removed, `_enrich_trailing_words` is fully LLM-driven and structurally simplified. No other removal depends on `_split_component_name` being gone, but doing it first validates the Spike 001 integration under real pipeline conditions.
- `13b` relies on the LLM ambiguity classification output from `_classify_components` being trustworthy on its own. It is safest to run `13a` first because the new trailing-word enrichment in `13a` feeds the alias set used in subsequent extraction. If `13a` regresses, that regression must be understood before attributing results to `13b`.
- `13c` is a mechanical wrapper removal (`_is_ambiguous_name_component` just calls `_is_structurally_unambiguous` and checks `ambiguous_names`). `13b` removes `_is_structurally_unambiguous`; `13c` then removes the now-trivial wrapper that used to call it.
- `13d` changes the extraction prompt schema to emit `mention_type`. The evidence bundle now reads an LLM field rather than calling `_classify_mention`. The `EvidenceBundle.mention_type` field was already a string; no change to `_build_evidence_bundle`'s callers. Safe after `13c` because ambiguity classification is clean.
- `13e` requires touching `_learn_document_knowledge_enriched` (alias-discovery prompt) and all alias consumers. This is the widest-blast-radius change in the series. Doing it after `13d` means entity extraction is already consuming LLM-emitted mention types, which simplifies the alias-scope consumer logic.
- `13f` is only meaningful after `13e` has established the scope-annotated alias structure; otherwise `_has_strong_alias_mention` still needs to filter aliases by strength.

---

## Rollback Plan

If a variant regresses below the 93% macro F1 floor:

1. **Do not promote** — the variant is still created and committed as `s_linker13x.py` for the ablation record, but is not added to the primary results table as a promoted variant.
2. **Report the regression** — document the per-dataset delta, which datasets were hit hardest, and the failure mode (FP increase vs FN increase vs both).
3. **Do not attempt to patch the failed variant** — per the standalone-file policy, patch attempts would produce `s_linker13x_v2.py`, which is a different variant.
4. **Continue the series** — subsequent removals may still be attempted if the failed removal can be logically skipped. Example: if `13b` (_is_structurally_unambiguous) fails, `13c` (wrapper removal) cannot proceed because its dependency (`13b`'s clean ambiguity output) is broken. But `13d` (mention classification) does not depend on `13b` — it could still be attempted off `12c` as a fork. In practice, the ordered chain means a regression at step N blocks all subsequent steps that depend on N's output; document the block explicitly.
5. **Floor definition**: 93% macro F1 over all 5 datasets with Claude Sonnet backend. A variant that holds 93.5% on 4 datasets but drops to 89% on bigbluebutton fails the floor (bigbluebutton is the hardest dataset and most likely to surface rule-removal regressions).

---

## Anti-Patterns

### Anti-Pattern 1: Feature-Flag Variant

**What people do:** Add a `remove_split_name: bool = True` flag to `s_linker12c` and run the same file with different flag combinations.

**Why it is wrong:** The standalone-file policy exists to prevent exactly this. Feature flags make it impossible to read a variant file and know what it does without tracing all flag combinations. Each ablation step also modifies prompt logic, not just whether a function is called — flag-toggling cannot represent prompt-schema changes like Spike 003's mention-type enum extension.

**Do this instead:** Copy the predecessor file, make the single change, name it `s_linker13x.py`. The file is self-documenting.

### Anti-Pattern 2: Modifying `data_types_v2.py` For the 13-Series

**What people do:** Add `scope` field to `DocumentKnowledge.aliases` in `data_types_v2.py` to make the alias-scope extension available globally.

**Why it is wrong:** `data_types_v2.py` is shared by all variants from `s_linker11` onward. Modifying the shared dataclass changes the import contract for `s_linker12c`, `12d`, `12e`, and all `13a`–`13d` files that use the flat-dict `aliases`. Even if backward-compatible, it risks silent breakage (a caller that uses `doc_knowledge.aliases["term"]` and expects a `str` now gets an `AliasEntry`).

**Do this instead:** Define `AliasEntry` inline at the top of `s_linker13e.py`. Use `DocumentKnowledge` with its existing `aliases: dict[str, str]` for variants before `13e`; introduce the scope-aware structure only inside `13e` and `13f`.

### Anti-Pattern 3: Per-Rule Checkpoint Ablation Infrastructure

**What people do:** Build `test_13x_rules.py` scripts that load phase checkpoints from `13a`, swap out one function, and re-score without re-running the full pipeline (mirroring V30c's `test_heuristics.py`).

**Why it is wrong:** The ablation unit is the variant, not the rule. Per-rule offline ablation was valuable in the V30d era when the goal was isolating individual heuristic contributions. Now the goal is a cumulative removal chain where each variant's F1 is the signal. Per-rule scripts add maintenance burden and can produce misleading results (a rule that looks neutral in isolation may interact with other changes).

**Do this instead:** Run `run_ablation.py --variants s_linker12c s_linker13a --datasets teammates bigbluebutton` for fast signal. Full 5-dataset sweeps for every promoted variant.

### Anti-Pattern 4: Modifying 12c to "Fix" Regressions Found in 13-Series

**What people do:** Discover a regression in `s_linker13b`, conclude that `s_linker12c` has the same root problem, and update `s_linker12c`.

**Why it is wrong:** `s_linker12c` is the ICSE baseline. Modifying it after the fact invalidates prior results. `s_linker12c` is read-only.

**Do this instead:** If a systemic issue is found, document it in the variant's commit message and in the ablation table notes. New fixes belong in `s_linker14+`.

---

## Integration Points

### `prompts_v2.py` Extensions

Two prompt constants need to be added for the 13-series:

| Constant | Used by | Purpose |
|----------|---------|---------|
| `MENTION_TYPE_SCHEMA` | `s_linker13d` extraction pass | Schema hint for LLM to emit `mention_type` enum per candidate |
| `ALIAS_SCOPE_SCHEMA` | `s_linker13e` alias-discovery prompt | Schema extension asking LLM to annotate each alias with `scope: global|local` |

These constants follow the same pattern as existing `prompts_v2.py` constants: safe SE textbook examples only (no benchmark component names), organized by tier. Add them to the Tier 1 model analysis / doc knowledge section.

### `EvidenceBundle` Dataclass

The `EvidenceBundle` dataclass is defined locally inside each variant file (not shared). In `s_linker13d`, update it to reflect that `mention_type` is now an LLM-emitted enum string rather than a regex-computed string — the external representation is identical, so no callers of `_format_evidence` need updating.

In `s_linker13e/f`, if `AliasEntry` is defined locally, the `EvidenceBundle` does not need to change. The alias scope filtering happens at alias-consumption sites (`_get_strong_alias_mappings`, `_has_strong_alias_mention`), not in the evidence bundle.

### `run_ablation.py` Registration

Registering a new variant requires exactly two edits:

```python
# 1. Append to CANONICAL_VARIANTS list (preserves display order in summary table)
CANONICAL_VARIANTS = [
    ...
    "s_linker12e",
    "s_linker13a",   # add here
]

# 2. Add to VARIANT_SPECS dict
VARIANT_SPECS = {
    ...
    "s_linker13a": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker13a",
        class_name="SLinker13a",
        description="S-Linker13a: 12c - _split_component_name (Spike 001 LLM trailing-word)",
    ),
}
```

The description should state what was removed and reference the relevant spike, following the pattern visible in `s_linker12c`'s description ("12b - dead Tier 2, intersection voting").

---

## Data Flow Summary

```
PCM .repository   documentation text
      │                   │
      ▼                   ▼
parse_pcm_repository   load_sentences
      │                   │
      └──────────┬─────────┘
                 │
        ┌────────▼────────┐
        │   Tier 1        │  ← all three branches parallel
        │   model │ doc   │    model: _classify_components
        │  knowl. │ knowl.│    doc:   _learn_document_knowledge_enriched
        │         │ seed  │    seed:  _run_seed (ILinker3)
        └────────┬────────┘
                 │
     ModelKnowledge.ambiguous_names
     DocumentKnowledge.aliases (+ scope in 13e+)
     raw_seed_links
                 │
        ┌────────▼────────┐
        │   Tier 2        │  ← all three branches parallel
        │ seed  │ entity  │    seed_val:  _run_seed_validation
        │  val  │pipeline │    entity:    _run_entity_pipeline
        │       │ coref   │               (_extract_entities_enriched
        │       │         │                → _validate_with_evidence)
        └────────┬────────┘    coref:     _run_coreference
                 │
     seed_links + entity validated CandidateLinks + coref_links
                 │
        ┌────────▼────────┐
        │   Tier 3        │
        │   dedup         │
        └────────┬────────┘
                 │
          list[SadSamLink]
                 │
     run_ablation.py → metrics → CSV
```

**Scope of alias propagation** (the critical path for 13e/13f):

```
_learn_document_knowledge_enriched  →  DocumentKnowledge.aliases
                                              │
                  ┌───────────────────────────┼──────────────────────────┐
                  ▼                           ▼                          ▼
   _get_strong_alias_mappings     _has_strong_alias_mention    _classify_mention
   (injected into extraction       (coref antecedent check)    (evidence bundle)
    prompt as KNOWN ALIASES)
```

After 13e, `_get_strong_alias_mappings` reads `alias.scope == "global"` instead of calling `_is_strong_alias(alias)`. After 13f, `_has_strong_alias_mention` is replaced by an LLM-emitted field in the coref resolution output. `_classify_mention` is already gone in 13d.

---

## Sources

- Direct inspection: `s_linker12c.py` (1211 lines, 2026-04-21)
- Direct inspection: `.planning/spikes/001-llm-trailing-words/spike.py` (VALIDATED)
- Direct inspection: `.planning/spikes/002-rules-audit/AUDIT.md` (VALIDATED)
- Direct inspection: `.planning/spikes/003-llm-mention-classifier/spike.py` (VALIDATED)
- Direct inspection: `run_ablation.py` (687 lines, variant registry + harness)
- Direct inspection: `.planning/codebase/ARCHITECTURE.md`, `STRUCTURE.md`, `CONCERNS.md`
- Direct inspection: `.planning/PROJECT.md` (milestone requirements + key decisions)

---
*Architecture research for: iterative rule-removal ablation on s_linker12c (fully-LLM-driven SAD-SAM linker)*
*Researched: 2026-04-21*
