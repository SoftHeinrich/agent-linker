# Phase 8: COMBINE — `s_linker14` Stack-or-Unify — Research

**Researched:** 2026-05-31
**Domain:** LLM-pipeline refactor — folding rule-removal primitives into unified prompts
**Confidence:** HIGH (codebase is small, fully read; no external library lookups required)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01 Unification scope = the 3 rule-removal LLM primitives only**: Spike-001 trailing-words detection (currently inside the doc-knowledge LLM prompt + judge — VAR-01), `scope: global|local` alias field (currently inside the doc-knowledge LLM prompt — VAR-05), and alias-coref-fold (currently the `antecedent_via_alias` field in the coref prompt — VAR-06). EXT-01 is dropped (Phase 6 closed empty). All other s_linker13 LLM stages (seed validation, two-pass entity validation, ambiguity classification, generic-word filter, judge) stay stacked as-is — they are verification / judgment stages, not rule replacements, and are out of COMBINE scope.
- **D-02 Stack-vs-unify decision is empirical**, not pre-locked. Plan phase will audit s_linker13's call-graph for each primitive and propose one unified design + one stacked baseline. Both compete on a single full sweep.
- **D-03 Default unified design hypothesis** (planner may refine after seeing the call graph):
  - Spike-001 trailing-words → fold into `_extract_entities_enriched` (Spike-003 piggyback pattern, zero net LLM cost).
  - Scope-field assignment → fold into the same unified entity-extraction call (already adjacent to alias output).
  - Alias-coref-fold → STAYS stacked (data dependency: needs the full merged alias map first).
  - Result: 2 unified calls + 1 stacked call = net −1 LLM call topology.
- **D-04 Ship rule = dual-floor with relaxed v2.0 budget**:
  - macro F1 ≥ 0.93 (GATE-01)
  - BBB ≤ `s_linker12c` BBB + 6pp tolerance
  - Other datasets ≤ `s_linker12c` per-dataset + 2pp tolerance
  - GATE-05 still applies: TM regression > 1pp vs `s_linker13` parent → no full sweep, re-work.
  - Failure mode: if `s_linker14` regresses > 6pp BBB vs s_linker12c, Phase 8 closes with `s_linker13` declared the COMBINE winner (negative-on-unify outcome still satisfies COMBINE-01..03 traceability).
- **D-05 Stack baseline = `s_linker13` itself** (macro F1 0.9509 from v1.0 ablation). No separate "stacked s_linker14" build — saves one full sweep. Unify candidate = new `s_linker14.py`.
- **D-06 Cost/quality signal in 08-SUMMARY.md** must include: per-(component, dataset) LLM call count for stack vs unify; wall-clock latency per dataset; per-dataset + macro F1 deltas; prompt-length / token-count comparison; stack-vs-unify winner + rationale string (becomes the GATE-07 `RULES_REMOVED` provenance entry per COMBINE-01).
- **D-07 Promotion rule**: `s_linker14.py` standalone, copy-fork from `s_linker13.py` (no inheritance — user preference + project convention). Registered in `run_ablation.py` `CANONICAL_VARIANTS` + `VARIANT_SPECS` with `canonical=True` ONLY after dual-floor PASS AND GATE-06 unit audit PASS. If unify loses, `s_linker14.py` may still be committed as a rejected baseline (canonical=False) for the ablation table.
- **D-08 Structured docstring (GATE-07)**:
  - `RULES_REMOVED` = cumulative list carried unchanged from `s_linker13` (no new removals in Phase 8).
  - Stack-vs-unify provenance string ("unified" or "stacked" + 1-sentence rationale citing the D-06 signal) — per COMBINE-01.
  - `REMOVED_FROM` = `s_linker13`.
- **D-09 GATE-06 re-audit as a UNIT** — combined prompt may surface leakage that per-phase audits missed. Audit recorded in `08-SUMMARY.md` as a dedicated section. BOTH the `BENCHMARK_TABOO.md` mechanical scan AND the reviewer-defensibility check required.
- **D-10 Ablation table update** — new row in `ABLATION-TABLE.md` and `ABLATION-TABLE.tex` for the winner. Row includes the stack-vs-unify provenance string. Previous v1.0 rows preserved.

### Claude's Discretion

- **Exact unified prompt design** for the Spike-001 + scope-field + entity-extraction fold. Must use safe SE textbook examples (BENCHMARK_TABOO.md). Spike-003 pattern is the reference shape.
- **Data-flow refactor inside `s_linker14`'s DAG** to absorb Spike-001's output into the entity-extraction (or doc-knowledge) batched-scan output. Tier-1 sequencing may need adjustment.
- **Fallback policy on LLM failure for the unified call** — default: approve-bias per existing pattern; planner may deviate with rationale.
- **Token-budget management** for the longer unified prompts (expected +30-50% prompt length vs single-purpose calls).
- **Per-call-site rewiring** inside `s_linker14` so consumers of the previously-separate signals read the unified output.

### Deferred Ideas (OUT OF SCOPE)

- EXT-01 standalone-mention LLM primitive — closed empty Phase 6.
- EXT-02 drop dotted-path guard — auto-skipped per ROADMAP gating.
- EXT-04 variance-band tightening — deferred to v2.1+.
- GPT-5.2 cross-model run — Phase 9 (CROSS-01..03).
- New rule removals beyond the v1.0+v2.0 chain — out of v2.0 thesis scope.
- Unifying verification stages (seed disambiguation, two-pass validation, coref antecedent verification) — explicitly NOT rule replacements per Phase 8 boundary.

</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| COMBINE-01 | `s_linker14.py` integrates all v1.0+v2.0 LLM rule-removal primitives — stacked or unified — with stack-vs-unify decision documented post-EXT-01 with cost/quality rationale. | Reduced scope: EXT-01 dropped → COMBINE-01 acts on the 3 surviving primitives (Spike-001 trailing-words, scope-field, alias-coref-fold). Section "Primitive Call-Graph Audit" gives exact line numbers + topology. Section "Default Unified Design" specifies the prompt + data-flow refactor. Section "Stack-vs-Unify Comparison Methodology" specifies what to log for COMBINE-01's provenance string. |
| COMBINE-02 | `s_linker14` passes dual floor on Claude Sonnet across all 5 datasets. | Section "Validation Architecture" gives the existing `run_ablation.py` harness pattern + GATE-05 hard-tier-first dev loop. Section "Baselines & Floors" gives the concrete numeric floors derived from `s_linker12c` and `s_linker13` ablation rows. |
| COMBINE-03 | Ablation row added to `ABLATION-TABLE.md` / `.tex` with provenance string for stack-vs-unify. | Section "Ablation Table Update" describes the exact existing row schema (8 columns) and the provenance-string slot — to be appended to the existing v1.0 table, not regenerated. |

</phase_requirements>

## Summary

Phase 8 builds `s_linker14.py` — a standalone copy-fork of `s_linker13.py` — that tries to **unify** three previously-stacked rule-removal LLM primitives into a smaller set of LLM calls. The candidate competes head-to-head against `s_linker13` (the stack baseline) on a single full 5-project sweep. If unify wins (or ties within tolerance) it ships as `canonical=True`; if it regresses past the dual-floor tolerance, `s_linker13` retains COMBINE designation and `s_linker14` is committed as a rejected baseline for the ablation table.

The codebase is small and fully understood: `s_linker13.py` is 1198 lines, 7 named LLM calls in a 3-tier DAG; all three target primitives live in identifiable, contiguous code regions. The default unified design hypothesis (D-03) is **structurally compatible** with the call graph — Spike-001 trailing-words and scope-field already live in the same `_learn_document_knowledge_enriched` prompt body (lines 366-466), and alias-coref-fold (`antecedent_via_alias`, lines 115-130 + 1063-1097) genuinely has a downstream data dependency on the merged alias set and so cannot fold further upstream without re-architecting Tier-1 sequencing.

**Primary recommendation:** Treat D-03 as a verified default, not a hypothesis. Spike-001 + scope-field are **already partially unified** inside `_learn_document_knowledge_enriched` (one prompt emits abbreviations + synonyms + scope in a single call) — the planner's actual unification work is *cleaner consolidation of the trailing-word few-shot guidance into that one prompt* + *making the consolidation explicit in the docstring and provenance string*. Alias-coref-fold stays where it is. The expected "net −1 LLM call" claim in D-03 is misleading because the calls are already partially folded; the realistic delta is "same call count, tighter prompt structure, single provenance entry instead of three." This needs to be flagged for the planner to set correct expectations.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Trailing-word alias discovery (Spike-001 / VAR-01) | Tier 1 — Knowledge Acquisition (`_learn_document_knowledge_enriched`) | — | Already lives in the doc-knowledge LLM prompt: `DOC_KNOWLEDGE_JUDGE_EXAMPLES` Example 2 ("Dispatcher" → TaskDispatcher), Example 4 ("Table" → SymbolTable), and `DOC_KNOWLEDGE_JUDGE_RULES` rule 1 bullet 2 explicitly auto-approve trailing words. No standalone "trailing-words" LLM call exists in s_linker13. |
| Scope-field assignment (VAR-05) | Tier 1 — Knowledge Acquisition (`_learn_document_knowledge_enriched`) | — | Already lives in the same doc-knowledge prompt: `ALIAS_SCOPE_SCHEMA` (s_linker13.py:94-112) is concatenated into the prompt at line 377, and the output records `scope: "global"|"local"` per alias. Scope is consumed by `_extract_entities_enriched` (line 814: `if entry.scope == "global"`). |
| Alias-coref-fold (VAR-06) | Tier 2 — Link Recovery (`_coref_cases_in_context`) | Tier 1 (consumes the merged alias map) | `ANTECEDENT_ALIAS_GUIDE` (s_linker13.py:115-130) is concatenated into the coref prompt at line 1065. The coref call at line 1095-1097 uses the `antecedent_via_alias` flag together with `_has_standalone_mention(comp, ant_sent.text)` to decide whether the antecedent is acceptable. Needs the doc-knowledge aliases map already merged (Tier-1 dependency). |
| Entity extraction (the natural fold-target named in D-03) | Tier 2 — Link Recovery (`_extract_entities_enriched`) | — | Spike-003 piggyback pattern lives here. Already enriched with global-scope alias mappings (s_linker13.py:813-820). NOT currently carrying trailing-words or scope output — those land upstream in `_learn_document_knowledge_enriched`. |

**Why this matters for planning:** The "fold Spike-001 + scope-field into `_extract_entities_enriched`" wording in D-03 is anachronistic for the current state of `s_linker13.py`. Trailing-words and scope-field are *already* emitted by the doc-knowledge call (Tier 1), not by entity extraction (Tier 2). Unification, if pursued, means **tightening the doc-knowledge prompt** (a single, more structured prompt block), not **moving signals from Tier 1 to Tier 2**. Tier-2 entity extraction *consumes* these signals (global-scope alias filter at line 814); it does not produce them.

## Standard Stack

### Core

| Library / Asset | Version / Location | Purpose | Why Standard |
|-----------------|---------------------|---------|--------------|
| `s_linker13.py` (parent) | `src/llm_sad_sam/linkers/experimental/s_linker13.py`, 1198 lines | Copy-fork base for `s_linker14.py` (no inheritance per project convention) | v1.0 final canonical artifact, macro F1 0.9509; integrates all 3 target primitives stacked. `_VARIANT_NAME = "s_linker13"` (line 136) must be changed to `"s_linker14"` in the fork (used for checkpoint namespacing at line 1162, fail-fast assert at 1165). |
| `prompts_v2.py` | `src/llm_sad_sam/linkers/experimental/prompts_v2.py`, 390 lines | Hosts all prompt constants. New unified prompt constant(s) append here. | Existing convention: every prompt body is a module-level string constant. Plan 06 already added 6 STANDALONE_MENTION_RULES_* constants; same pattern for Phase 8. |
| `run_ablation.py` | `run_ablation.py`, 778 lines | Variant registration (GATE-07 enforcement point) + sweep harness. | `CANONICAL_VARIANTS` list (lines 40-86) + `VARIANT_SPECS` dict (lines 88-359) are the registration surface. `DATASETS` dict (lines 372-403) drives the full sweep. |
| `ILinker3` | `src/llm_sad_sam/linkers/experimental/ilinker3.py` (imported s_linker13.py:54) | Seed extraction (Tier 1 parallel) — out of COMBINE scope. | Used unchanged. `s_linker14` instantiates it the same way (line 181). |
| `prompts_v2` constants currently consumed | `AMBIGUITY_FEW_SHOT`, `AMBIGUITY_RULES`, `DOC_KNOWLEDGE_JUDGE_EXAMPLES`, `DOC_KNOWLEDGE_JUDGE_RULES`, `DOC_KNOWLEDGE_EXTRACTION_RULES`, `ENTITY_EXTRACTION_RULES`, `VALIDATION_RULES`, `COREF_RULES` (s_linker13.py:55-60) | Six imports already; the unified-prompt design appends 1–2 new constants. | Established pattern. No restructuring of the prompts_v2.py header organization is needed. |

`[VERIFIED: codebase read 2026-05-31]`

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Copy-fork standalone `s_linker14.py` | Inheritance / mixin from `s_linker13.py` | REJECTED — user preference + project convention (PROJECT.md: "user prefers duplicated standalone files over inheritance chains"). All s_linker13a..13f variants are standalone copy-forks. Inheritance breaks the ablation invariant "one rule = one self-contained file." |
| Single new "MEGA_PROMPT" constant | One unified constant + one explicit `ALIAS_SCOPE_SCHEMA`-style guide | The Phase 6 alias-aware constants (`STANDALONE_MENTION_RULES_*_ALIAS_AWARE`) use the `{KNOWN_ALIASES_BLOCK}`-injection pattern with `.replace()` (NOT `.format()`) per `prompts_v2.py:265-268` — because the JSON template at the end uses literal braces. Plan must follow this same string-injection pattern, not `.format()`. |
| Tier-1 sequencing change to feed alias map into entity extraction earlier so alias-coref-fold could be unified upstream | Keep alias-coref-fold stacked | D-03 confirms stacked; the coref call NEEDS the full merged alias map (post-judge approval at line 451-453) plus the `antecedent_via_alias` flag (line 1096), which is structurally a Tier-2-after-Tier-1 dependency. Unifying further breaks the current DAG parallelism. |

`[VERIFIED: codebase read 2026-05-31]`

**Installation:** No new dependencies. Phase 8 is pure code refactor inside an existing Python package.

**Version verification:** Not applicable — no external library targets.

## Architecture Patterns

### System Architecture Diagram (s_linker13 → s_linker14)

```
                          link(text, model)
                                 |
                                 v
                  [ Tier 1 — Knowledge Acquisition ]
                  _run_parallel({model, doc_knowledge, seed})
                                 |
        +------------------------+------------------------+
        |                        |                        |
        v                        v                        v
  _analyze_model       _learn_document_knowledge_     _run_seed
  (LLM ambiguity        enriched (LLM extraction      (ILinker3 — out of
   classification)       + LLM judge, emits alias      COMBINE scope)
                         + scope: global|local         (in s_linker14
                         + trailing-word judgments     unchanged)
                         in ONE prompt+judge pair)
                                 |
                          *** UNIFICATION POINT 1 ***
                          (Spike-001 trailing-words +
                           VAR-05 scope-field already
                           folded here — Phase 8 task =
                           tighten prompt, single
                           provenance entry)
                                 |
                                 v
                       doc_knowledge.aliases : {term -> AliasEntry(component, scope)}
                                 |
                                 v
                  [ Tier 2 — Link Recovery (parallel) ]
                  _run_parallel({seed_val, entity, coref})
                                 |
        +------------------------+------------------------+
        |                        |                        |
        v                        v                        v
  _run_seed_           _run_entity_pipeline       _run_coreference
  validation           |                          |
  (LLM disambig —      _extract_entities_         _coref_cases_in_
   verification, NOT   enriched (dual-pass,       context (LLM with
   rule replacement)   intersection voting)       ANTECEDENT_ALIAS_GUIDE)
                       |                          |
                       (consumes global-scope     *** STAYS STACKED ***
                       aliases — global filter    Spike-001-style fold
                       at line 814)               (VAR-06 alias-coref)
                                                  uses antecedent_via_alias
                                                  field set by LLM. Needs
                                                  the merged alias map.
                                 |
                                 v
                       _validate_with_evidence
                       (LLM 2-pass intersection —
                        verification, NOT rule
                        replacement)
                                 |
                                 v
                  [ Tier 3 — Link Consolidation ]
                       dedup (seed > entity > coref)
                                 |
                                 v
                          list[SadSamLink]
```

**Key insight:** In `s_linker13.py` as it stands, **the three primitives are already at minimum-call topology** — there is no standalone "Spike-001 trailing-words" LLM call to eliminate. The Spike-001 work landed during VAR-01 (13a) by folding the trailing-word judgments INTO the existing doc-knowledge extraction+judge call pair. Phase 8 unification is therefore a **prompt-design tightening exercise**, not a call-topology reduction. The planner should set ship-rule expectations accordingly. `[VERIFIED: codebase read 2026-05-31 — no separate trailing-words call exists in s_linker13.py]`

### Recommended Project Structure

```
src/llm_sad_sam/linkers/experimental/
├── s_linker13.py            # parent — UNCHANGED
├── s_linker14.py            # NEW — copy-fork target (this phase's deliverable)
└── prompts_v2.py            # APPEND new unified constant(s) at end of file
                             # — do not reorder existing constants

run_ablation.py              # APPEND "s_linker14" to CANONICAL_VARIANTS list
                             # + add new VARIANT_SPECS dict entry

.planning/phases/08-combine-s-linker14-stack-or-unify-combined-llm-primitives/
├── 08-CONTEXT.md            # exists
├── 08-RESEARCH.md           # this file
├── 08-PLAN.md               # planner output
├── 08-GATE-06-AUDIT.md      # NEW — unit re-audit per D-09
├── 08-SUMMARY.md            # NEW — must include `## COMBINE cost/quality signal` block per D-06
└── 08-NN-SUMMARY.md         # per-plan summaries

ABLATION-TABLE.md            # APPEND row per D-10 (NOT regenerate)
ABLATION-TABLE.tex           # APPEND row per D-10 (NOT regenerate)
```

**Note:** `ABLATION-TABLE.md` and `.tex` currently live at `.planning/milestones/v1.0-phases/05-promote-and-ablation-artifact/`. Per the CONTEXT.md `<canonical_refs>` and D-10 wording, the v2.0 deliverable may either (a) append a row to the existing v1.0 file or (b) generate a v2.0-scoped file in the Phase 8 directory. The planner should pick one explicitly. `[VERIFIED: filesystem search 2026-05-31 — no top-level ABLATION-TABLE.md exists; only the v1.0 location]`

### Pattern 1: Copy-Fork with Standalone `_VARIANT_NAME` Namespacing

**What:** Duplicate the parent file, change only the identifier-level surface markers, leave behavior identical until the rule-removal refactor lands.

**When to use:** Every new variant in this project (13a..13f all follow this — verified by `run_ablation.py:280-322`).

**Example (the minimum diff at fork time):**

```python
# Source: s_linker13.py:1, 133-134, 136 [VERIFIED]
class SLinker13:
    """LLM-driven SAD-SAM traceability — canonical promotion of s_linker13f (Phase 5)."""
    _VARIANT_NAME = "s_linker13"

# After copy-fork to s_linker14.py:
class SLinker14:
    """LLM-driven SAD-SAM traceability — COMBINE unified primitives (Phase 8).

    REMOVED_FROM: s_linker13
    RULES_REMOVED: [<carry s_linker13's list verbatim — no new removals>]
    STACK_VS_UNIFY: "unified" | "stacked"  # per D-08 provenance string
    """
    _VARIANT_NAME = "s_linker14"  # MUST update — guards checkpoint namespace
```

The `_checkpoint_dir` method (s_linker13.py:1159-1170) contains a `assert self._VARIANT_NAME in d` fail-fast guard that explicitly protects against forgetting this rename. `[VERIFIED: s_linker13.py:1165-1168]`

### Pattern 2: Append-Only Prompt Constant in `prompts_v2.py`

**What:** New prompt constants are appended at the end of `prompts_v2.py`; existing constants are not reordered.

**When to use:** Every new prompt body. Phase 6 added 6 constants (STANDALONE_MENTION_RULES_*) at lines 229-371 following this pattern.

**Example (current Phase 6 form):**

```python
# Source: prompts_v2.py:265-268 [VERIFIED]
# These four constants extend the Plan 06-01 STANDALONE_MENTION_RULES_PRE_FILTERED
# and STANDALONE_MENTION_RULES_LLM_ONLY constants with knowledge blocks injected
# at call time by the linker (Plan 06-06). The blocks are substituted via
# `prompt.replace("{KNOWN_ALIASES_BLOCK}", ...)` — NOT `.format(...)` — because
# the JSON template at the end uses literal braces.
```

**Critical pattern note:** Use `.replace()` for placeholder injection, NEVER `.format()` — the JSON output schema uses literal `{...}` braces. This is non-obvious and project-specific; planner must enforce in code review.

### Pattern 3: Approve-Biased Fallback on LLM Failure

**What:** When `self.llm.extract_json(...)` returns None after retry, keep all candidates (do not reject silently).

**When to use:** Every LLM call in `s_linker13`.

**Example:**

```python
# Source: s_linker13.py:561-569 [VERIFIED]
for attempt in range(2):
    data = self.llm.extract_json(self.llm.query(prompt, timeout=120))
    if data and data.get("disambiguations"):
        break
    if attempt == 0:
        print(f"    [{comp_name}] Empty response, retrying...")
if not data:
    verified.extend(valid_seeds)  # Keep all on failure (approve-biased)
    continue
```

This pattern shows up at: `_classify_components` (line 352-358), `_learn_document_knowledge_enriched` (lines 389-394, 445-451, fallback to `set(all_mappings.keys())` if judge fails — line 451), `_run_seed_validation` (lines 561-569), `_extract_entities_enriched` extraction pass (lines 775-783), `_validate_with_evidence` (lines 932-940), `_run_validation_pass` (lines 1016-1029), `_coref_cases_in_context` (lines 1072-1080).

The unified Phase 8 call must follow the same shape. Recommended default: on LLM failure, fall back to the pre-Phase-8 separated outputs (i.e. fail-safe to the stack behavior). Practically this means the unified prompt must be wrapped in a try/fallback such that if the unified JSON is malformed, the planner explicitly falls back to the stacked semantics — NOT silently dropping signals.

### Pattern 4: Tier-1 Parallel + Tier-2 Parallel DAG

**What:** Both Tier 1 (model / doc_knowledge / seed) and Tier 2 (seed_val / entity / coref) use `_run_parallel` (s_linker13.py:240-244, 269-276). Inside each tier, the three branches are independent.

**When to use:** Tier-1 unification preserves this parallelism (doc_knowledge already independent of model + seed). Tier-2 unification would NOT, because alias-coref-fold consumes the merged alias map produced by Tier 1.

### Anti-Patterns to Avoid

- **Inheritance from `SLinker13`** — violates project convention. Sibling files compete; winner gets `canonical=True`.
- **`.format()` on prompts containing JSON literal braces** — silent format-string error or KeyError on the `{` in `{"results": ...}`. Use `.replace()` for placeholder injection. `[VERIFIED: prompts_v2.py:265-268 explicit guidance]`
- **Forgetting `_VARIANT_NAME` rename** — checkpoint cache cross-contamination. Mitigated by fail-fast assert at s_linker13.py:1165-1168.
- **Adding new rule-removal primitives** — explicitly out of scope. Phase 8 is "finishing the v1.0+v2.0 chain, not adding to it" (CONTEXT.md `<specifics>`).
- **Unifying verification stages** (seed_val, two-pass validation, coref antecedent verification) — these are judgment/verification, NOT rule replacements; explicitly out of scope per Phase 8 boundary.
- **Per-component regex flags / casing tables in code** — banned per `BENCHMARK_TABOO.md` "Tailored Code Anti-Patterns" section. Already a Phase-6 lesson.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Prompt placeholder injection | `str.format()` | `str.replace("{PLACEHOLDER}", value)` | JSON templates in prompts use literal braces; `.format()` breaks. Project convention (prompts_v2.py:265-268). |
| LLM call retry / fallback | Custom retry loop per call site | Existing 2-attempt + approve-bias pattern (s_linker13.py example at 561-569) | Pattern is uniform across all 7 LLM call sites; deviation would break maintainability. |
| Checkpoint namespacing | Manual cache-dir paths | `_checkpoint_dir(text_path)` with `_VARIANT_NAME` (s_linker13.py:1159-1170) | Built-in fail-fast assert protects against rename-forgetting. |
| Stack-vs-unify A/B harness | Custom comparison runner | `run_ablation.py --variants s_linker13 s_linker14 --datasets <all>` | Existing harness already runs the 5-project sweep + reports macro F1 per variant per dataset. Existing JSON output format is what feeds ABLATION-TABLE rendering. |
| Per-row ablation table rendering | Manual markdown editing | `render_ablation.py` (referenced in v1.0 ABLATION-TABLE.md header: "Generated by `render_ablation.py` (Phase 5, PROMO-03)") | Existing infrastructure. Planner must confirm `render_ablation.py` lives under `scripts/` or analogous and is invoked the same way Phase 5 invoked it. |
| Component-specific casing handling | Per-component regex flags (`re.IGNORECASE`) or per-component casing tables | LLM primitive that handles casing as natural-language detail | Banned per `BENCHMARK_TABOO.md` "Anti-pattern: Case-mismatch regex baselines". |

**Key insight:** The COMBINE phase's "build" surface is small (one new linker file, 1-2 new prompt constants, one new ablation row). Everything else is reuse of existing infrastructure. The planning risk is **not** in tooling — it is in (a) prompt design that survives the GATE-06 unit re-audit, and (b) honest cost/quality reporting per D-06.

## Runtime State Inventory

> Required: Phase 8 includes a file rename / copy-fork.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | **Checkpoint pickle cache** at `./results/phase_cache/{_VARIANT_NAME}/{dataset}/` per s_linker13.py:1162. `s_linker14`'s `_VARIANT_NAME = "s_linker14"` will create a NEW namespace; no migration needed; no collision with `s_linker13`'s cache. **Sweep JSON** outputs at `results/ablation_results/ablation_*.json` (named by timestamp, not by variant) — new sweep will produce a new file; no rename/migration needed. **Phase-log JSON** at `./results/llm_logs/{_VARIANT_NAME}_{dataset}_{ts}.json` (s_linker13.py:1192-1194) — new namespace, no collision. | None — the per-variant namespacing pattern is already in place and isolates `s_linker14` from `s_linker13` automatically. |
| Live service config | LLM backend selection via env vars `CLAUDE_MODEL`, `OPENAI_MODEL_NAME` (s_linker13.py:170-171). `s_linker14` inherits the same defaults. No service-side state to update. | None. |
| OS-registered state | None — pure-Python library code; no daemons, no scheduled jobs. | None — verified by absence of any systemd/Task Scheduler/pm2 references in tree. |
| Secrets / env vars | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` (read by `LLMClient` via standard backends). `s_linker14` reads them the same way `s_linker13` does — no rename. | None. |
| Build artifacts / installed packages | `pip install -e ".[dev,openai]"` per CLAUDE.md. Adding `s_linker14.py` to the package needs no re-install (editable mode picks it up). `prompts_v2.py` append-only edit needs no re-install. | None — verified by editable install convention. |

## Common Pitfalls

### Pitfall 1: Treating "fold Spike-001 + scope-field into `_extract_entities_enriched`" literally

**What goes wrong:** Planner reads D-03 as "move trailing-word detection + scope output FROM Tier 1 TO Tier 2 entity extraction." This is structurally wrong — both already live in Tier 1 (`_learn_document_knowledge_enriched`, s_linker13.py:366-466).

**Why it happens:** D-03 is written as a hypothesis, and the entity-extraction call is the most prominent Spike-003 piggyback example. But Spike-003's piggyback target was `_classify_mention`, not trailing-words — the trailing-words primitive piggybacked on doc-knowledge during VAR-01.

**How to avoid:** Read `_learn_document_knowledge_enriched` (s_linker13.py:366-466) before designing the unified prompt. Observe that `DOC_KNOWLEDGE_JUDGE_EXAMPLES` Example 2 (line 94-97) and Example 4 (line 103-106) — and `DOC_KNOWLEDGE_JUDGE_RULES` rule 1 bullet 2 (line 128) — already encode the trailing-word logic.

**Warning signs:** A planner task says "add trailing-word detection to ENTITY_EXTRACTION_RULES" or "move the trailing-word example from DOC_KNOWLEDGE_JUDGE_EXAMPLES to ENTITY_EXTRACTION_RULES." Both are wrong directions.

### Pitfall 2: Cross-prompt combination leakage (the D-09 surface)

**What goes wrong:** Individual prompts pass `BENCHMARK_TABOO.md` scan. The UNIFIED prompt — concatenated — surfaces a combination that doesn't (e.g., a trailing-word example + an alias example that together evoke a benchmark component).

**Why it happens:** Examples chosen for separate prompts independently are independently clean; their juxtaposition can echo a benchmark pattern. E.g. `Dispatcher` (compiler domain) + `Scheduler` (OS domain) is fine in isolation; pairing them with a "X handles requests asynchronously" example near another OS-domain example might inadvertently sketch a BBB-style architecture.

**How to avoid:** D-09 unit audit means running `BENCHMARK_TABOO.md` scan on the **entire concatenated prompt text** that the LLM receives, not on each constant separately. Use the same word-bounded scan that Phase 6 used (per `06-GATE-06-AUDIT.md`). Plus reviewer-defensibility check on the concatenated body.

**Warning signs:** New unified prompt body mixes examples from multiple safe domains (compiler + OS + e-commerce) in a way that sketches an architecture pattern; or the new few-shot examples reuse the same fake component name across multiple cases (`Dispatcher` appears 3 times) — that's a defensibility flag even if not a TABOO hit.

### Pitfall 3: Misleading "net −1 LLM call" claim in D-03

**What goes wrong:** Cost/quality signal block (D-06) reports "+0 LLM calls" for the unified design and the planner / reviewer treats it as "unification didn't work." The actual COMBINE deliverable is **prompt-structure consolidation**, not call-count reduction — because Spike-001 and VAR-05 were never separate LLM calls in the first place.

**Why it happens:** D-03 wording assumes a Tier-1-with-separate-trailing-words-call topology that hasn't existed since s_linker13a (VAR-01 already folded it).

**How to avoid:** Phase the D-06 signal block to report **prompt token-count + structural complexity (rule count, example count, output-schema branches)** as the unification metric — not raw LLM call count. The cost/quality "win" should be framed as "single prompt with single judge, single provenance line in docstring" vs "currently three rules implicit across two prompts that happen to share a body."

**Warning signs:** Planner writes "unified design saves N LLM calls" when N is 0. Reframe to "unified design halves the number of GATE-06 audit surfaces" or "single provenance string vs three."

### Pitfall 4: Token-budget pushed past Claude Sonnet's prompt-quality sweet spot

**What goes wrong:** Unified prompt grows to +30-50% length; Claude Sonnet output quality on the doc-knowledge call degrades (judge approves more synonyms, scope assignments get noisier, F1 regresses by 1-3pp).

**Why it happens:** Long prompts dilute the signal of individual rules. From MEMORY.md: "V35 prompt simplification experiments — ALL FAILED... Claude Sonnet prompts are at local optimum. Every simplification regresses." The reverse (over-expansion) is similarly risky.

**How to avoid:** Measure prompt-character-count and judge approval-rate per dataset for the unified vs stacked prompts BEFORE running the full sweep. If unified prompt is >50% longer than the original `_learn_document_knowledge_enriched` body, refactor toward structured sub-sections (e.g., explicit "PART 1 — Trailing Words", "PART 2 — Scope Field") rather than free-flowing rules concatenation.

**Warning signs:** Pre-sweep token count check shows unified prompt >50% above `s_linker13`'s. Same-call output decisions diverge from `s_linker13`'s on a 10-sentence smoke test.

### Pitfall 5: Claude Sonnet variance band (±4pp on BBB) masking real regression

**What goes wrong:** A single full sweep shows `s_linker14` BBB at 0.85, `s_linker13` parent at 0.89, planner ships. Re-run two days later: `s_linker14` BBB at 0.81, `s_linker13` at 0.86. Both moved; the delta survives; but neither absolute number was actionable.

**Why it happens:** Documented LLM variance band: "same model gives different behaviour across days, ±4pp on BBB" (Phase 6 06-SUMMARY.md "Variance Context"). Single-run F1 is noisy at the 1-2pp level.

**How to avoid:** Phase 6's lesson: **same-session baseline.** Run `s_linker13` and `s_linker14` back-to-back in the same sweep, compare deltas to same-session baselines, not to the historical 0.9509. The GATE-05 hard-tier-first dev loop already enforces this for TM/BBB.

**Warning signs:** Cost/quality block (D-06) reports `s_linker14` F1 vs the v1.0 0.9509 figure rather than vs the same-session `s_linker13` rerun. Reject and demand same-session.

### Pitfall 6: Ablation-table row regenerated, not appended

**What goes wrong:** Planner runs `render_ablation.py` over `results/ablation_results/` and clobbers the v1.0 table.

**Why it happens:** `render_ablation.py` may rebuild the whole table from results JSONs. If `s_linker14` JSON is added, all rows may be re-rendered.

**How to avoid:** Inspect `render_ablation.py` behavior BEFORE running. Either (a) restrict it to append-mode (if supported), or (b) snapshot the v1.0 file and verify that the regenerated v1.0 rows are byte-identical before promoting.

**Warning signs:** Render script has no `--append` or `--variants` flag. Planner must add a guard / explicit snapshot diff step.

## Code Examples

Verified patterns from `s_linker13.py` (the parent file):

### Spike-001 + scope-field unification — the EXISTING (stacked) form

```python
# Source: s_linker13.py:366-466 [VERIFIED]
def _learn_document_knowledge_enriched(self, sentences, components):
    """Discover aliases (abbreviations and synonyms) via LLM + judge."""
    comp_names = [c.name for c in components]
    doc_lines = [s.text for s in sentences]

    prompt1 = f"""Find all alternative names used for these components in the document.

COMPONENTS: {', '.join(comp_names)}

{DOC_KNOWLEDGE_EXTRACTION_RULES}    # contains trailing-words rule [Spike-001]

{ALIAS_SCOPE_SCHEMA}                 # contains scope: global|local rules [VAR-05]

DOCUMENT:
{chr(10).join(doc_lines)}

Return JSON:
{{
  "abbreviations": [{{"term": "short_form", "component": "FullComponent", "scope": "global"}}],
  "synonyms":      [{{"term": "specific_alternative_name", "component": "FullComponent", "scope": "local"}}]
}}
JSON only:"""

    # ... [extraction call] ...

    # Then a JUDGE call uses DOC_KNOWLEDGE_JUDGE_EXAMPLES and DOC_KNOWLEDGE_JUDGE_RULES
    # which BOTH explicitly cover the trailing-word case (Examples 2 & 4) and the
    # generic-rejection case.
```

**Observation:** Spike-001 (trailing-words) and VAR-05 (scope-field) are ALREADY co-resident in this one call pair (extractor + judge). The unification work is to consolidate `DOC_KNOWLEDGE_EXTRACTION_RULES` + `ALIAS_SCOPE_SCHEMA` + the relevant `DOC_KNOWLEDGE_JUDGE_*` content into a tighter single prompt body — and / or to explicitly factor out a `TRAILING_WORDS_GUIDE` constant for naming clarity in the docstring + provenance string.

### Alias-coref-fold — the EXISTING form (stays stacked)

```python
# Source: s_linker13.py:115-130, 1063-1097 [VERIFIED]

# Tier-2 coref prompt:
ANTECEDENT_ALIAS_GUIDE = """For each resolution, also set `antecedent_via_alias`:
- true:  the antecedent quote refers to the component by an ALIAS ...
- false: the antecedent quote refers to the component by its CANONICAL NAME ...
"""

# Used inside _coref_cases_in_context:
prompt += f"""{COREF_RULES}

{ANTECEDENT_ALIAS_GUIDE}

Return JSON:
{{"resolutions": [{{"case": 1, "sentence": N_INTEGER, "pronoun": "it",
   "component": "Name", "antecedent_sentence": M_INTEGER,
   "antecedent_text": "exact quote with component name",
   "antecedent_via_alias": false}}]}}

Only include resolutions you are CERTAIN about. JSON only:"""

# Acceptance gate uses BOTH the standalone-mention rule AND the alias-fold field:
if not (self._has_standalone_mention(comp, ant_sent.text) or
        res.get("antecedent_via_alias", False)):
    continue
```

**Observation:** This is the VAR-06 fold landed in s_linker13f. The fold replaced the separate `_has_strong_alias_mention` helper call with the in-prompt `antecedent_via_alias` field. It cannot be folded further upstream (e.g. into entity extraction) because the alias map must be FULLY MERGED (post-judge, line 451-463) before the coref call sees it.

### `_VARIANT_NAME` namespacing fail-fast (must update on fork)

```python
# Source: s_linker13.py:1159-1170 [VERIFIED]
def _checkpoint_dir(self, text_path):
    cache_dir = os.environ.get("PHASE_CACHE_DIR", "./results/phase_cache")
    ds = os.path.splitext(os.path.basename(text_path))[0]
    d = os.path.join(cache_dir, self._VARIANT_NAME, ds)
    # D-07: fail-fast if the directory does not embed the variant name.
    # Guards against subclasses or accidental edits that drop the namespace.
    assert self._VARIANT_NAME in d, (
        f"_checkpoint_dir must contain _VARIANT_NAME "
        f"('{self._VARIANT_NAME}' not in '{d}')"
    )
    os.makedirs(d, exist_ok=True)
    return d
```

**Implication:** s_linker14.py must set `_VARIANT_NAME = "s_linker14"` at class scope. Forgetting this fails loudly on the first checkpoint write — the assert will fire. `[VERIFIED]`

### `run_ablation.py` registration pattern

```python
# Source: run_ablation.py:79, 316-322 [VERIFIED]

# 1) Append to CANONICAL_VARIANTS list:
CANONICAL_VARIANTS = [
    ...
    "s_linker13",   # canonical promotion of 13f (Phase 5)
    "s_linker13g_pre", ...
    "s_linker14",   # NEW — COMBINE deliverable
]

# 2) Add to VARIANT_SPECS dict:
VARIANT_SPECS = {
    ...
    "s_linker13": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker13",
        class_name="SLinker13",
        description="S-Linker13: canonical promotion of s_linker13f (Phase 5) — 6 rules removed cumulatively from 12c",
        canonical=True,
    ),
    ...
    "s_linker14": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker14",
        class_name="SLinker14",
        description="S-Linker14: COMBINE — unified rule-removal LLM primitives (Phase 8) — cumulative 6 rules + unified provenance",
        canonical=True,   # ONLY after dual-floor + GATE-06 PASS (per D-07)
    ),
}
```

`canonical=True` ships only after pass. If unify loses, leave the dict entry but set `canonical=False`.

## Baselines & Floors

Numeric floors derived directly from the v1.0 `ABLATION-TABLE.md` `s_linker12c` row + `s_linker13` row.

| Project | s_linker12c F1 | s_linker13 F1 | GATE-01 floor (12c − tolerance) | GATE-05 floor (parent − 1pp) |
|---------|---------------:|--------------:|--------------------------------:|-----------------------------:|
| MS (MediaStore) | 0.984 | 0.984 | 0.964 (2pp tolerance) | 0.974 |
| TS (TeaStore) | 0.963 | 1.000 | 0.943 (2pp) | 0.990 |
| TM (Teammates) | 0.938 | 0.947 | 0.918 (2pp) | 0.937 |
| BBB (BigBlueButton) | 0.844 | 0.821 | 0.784 (6pp) | 0.811 |
| JAB (JabRef) | 0.973 | 1.000 | 0.953 (2pp) | 0.990 |
| **macro** | **0.9404** | **0.9506** | **≥ 0.93** (hard floor) | — |

`[VERIFIED: source = .planning/milestones/v1.0-phases/05-promote-and-ablation-artifact/ABLATION-TABLE.md rows 7 + 14]`

**Hard-tier-first gates (GATE-05) for the Phase 8 dev loop:**
- TM regression > 1pp vs `s_linker13` parent (0.947) → no full sweep, re-work. Floor: 0.937.
- BBB regression > 1pp vs `s_linker13` parent (0.821) → re-work. Floor: 0.811.

**Dual-floor ship gates (GATE-01) for full sweep:**
- macro F1 ≥ 0.93.
- BBB ≥ 0.784 (s_linker12c 0.844 − 6pp).
- Each of {MS, TS, TM, JAB} ≥ s_linker12c per-dataset baseline − 2pp.

**Variance band caveat:** Phase 6 06-SUMMARY.md documents ±4pp BBB run-to-run variance on the SAME variant on the SAME dataset. Same-session baseline (run `s_linker13` and `s_linker14` back-to-back in one sweep) is mandatory for the cost/quality block per Pitfall 5.

`[VERIFIED: 06-SUMMARY.md "Variance Context" section]`

## Project Constraints (from CLAUDE.md and MEMORY.md)

### From repo-level CLAUDE.md (`/mnt/hostshare/ardoco-home/llm-sad-sam-v45/CLAUDE.md`)

- Active surface: 3 ilinkers + s_linker..s_linker11a + prompts.py / prompts_v2.py + core data types + pcm_parser_v2. Archived families under `archive/`.
- Build: `pip install -e ".[dev,openai]"`; `python run_ablation.py`; `pytest`.
- `run_ablation.py` is the entry point for variant sweeps and supports `--list-variants`.
- Default model policy: Claude Sonnet (no Opus).
- `ilinker3` is wrapped by an adapter (not a standalone historical pipeline).

### From auto-MEMORY.md (operator-level preferences)

- Always Claude Sonnet (set in `run_ablation.py` and linker constructors).
- No dataset-specific examples in prompts — data leakage. Safe SE textbook domains (compiler, OS, e-commerce).
- User prefers standalone linker files (duplicate code intentionally, not inheritance chains).
- **Hard rule: zero hardcoded/tailored values in prompts OR logic.** v2.0 thesis depends on this.
- LLM variance is real: ±4pp BBB across runs. Same-session baseline mandatory.
- V32 / V31 prompt history confirms: **simplification regresses Claude.** Phase 8 unification must NOT be a "simplification" exercise — it must preserve the rule density that Claude Sonnet currently relies on. Restructure for clarity / provenance, do not delete examples.

### From `/mnt/hostshare/ardoco-home/CLAUDE.md` (upper-project ARDoCo guidance)

Not directly applicable — that file governs the Java ARDoCo framework, not the Python `llm-sad-sam-v45` linker package. Listed here for completeness only.

## State of the Art

This phase concerns an internal refactor; there is no external "state of the art" to track. Internal state:

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Separate `_split_component_name` regex + `_enrich_trailing_words` structural gate + LLM verify | Trailing-word judgment folded into `_learn_document_knowledge_enriched` (single extractor + judge) | v1.0 VAR-01 (s_linker13a, 2026-05-28) | Spike-001 validated; macro F1 -0.0041 vs 12c (within band) |
| Separate `_is_strong_alias` + `_get_strong_alias_mappings` structural rules | Scope field `global\|local` emitted by alias-discovery prompt | v1.0 VAR-05 (s_linker13e, 2026-05-29) | macro F1 -0.0025 vs 12c |
| Separate `_has_strong_alias_mention` helper called by coref consumer | `antecedent_via_alias` field in coref output schema | v1.0 VAR-06 (s_linker13f, 2026-05-29) | macro F1 +0.0102 vs 12c — best in v1.0 chain |

**Deprecated / outdated:** None within this codebase. Out-of-chain `s_linker13d` (VAR-04 Spike-003 mention-classifier) is retired and documented as a publishable negative result.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `render_ablation.py` exists and can be invoked to update `ABLATION-TABLE.md` / `.tex` per the Phase 5 PROMO-03 pattern. | Don't Hand-Roll, Pitfall 6 | If absent or differently-invoked, planner must build append logic manually. Mitigation: planner verifies tool exists during plan-checking. `[ASSUMED — header of v1.0 ABLATION-TABLE.md cites it, but the script file was not opened during this research session]` |
| A2 | `ABLATION-TABLE.md` / `.tex` should be **appended to** at the v1.0 location (`.planning/milestones/v1.0-phases/05-promote-and-ablation-artifact/`), not regenerated as a v2.0-scoped file. | Recommended Project Structure | D-10 wording is ambiguous between "new row in ABLATION-TABLE.md" (current file) and a v2.0 successor. Mitigation: planner asks user in plan-checking step OR creates a v2.0-scoped copy and points to it from PROJECT.md / MILESTONES.md. `[ASSUMED]` |
| A3 | The doc-knowledge prompt at `s_linker13.py:366-466` is the empirically-tuned local optimum — restructuring (vs deleting content) preserves macro F1 within ±2pp. | Pitfall 4, Recommendation | MEMORY.md V35 evidence supports this for prompt simplification specifically; restructuring without semantic loss is untested. Mitigation: hard-tier smoke test before full sweep. `[ASSUMED — MEMORY V35 evidence is suggestive but not identical experiment]` |
| A4 | A "stacked s_linker14" build is unnecessary because `s_linker13` IS the stack baseline (D-05). | Phase 1 framing | If a reviewer / downstream agent later wants a literal `s_linker14_stacked.py` to keep the naming symmetric, this assumption would force a no-op file. D-05 explicitly says no separate stacked file. `[CITED: 08-CONTEXT.md D-05]` |
| A5 | Generic LLM verification calls (seed disambiguation, two-pass validation, generic-word filter, ambiguity classification) are NOT rule replacements and stay stacked. | Phase boundary, Pitfall (anti-pattern) | If user later disputes this, COMBINE scope expands. Currently CITED from D-01. `[CITED: 08-CONTEXT.md D-01]` |
| A6 | Token-count proxy for the cost/quality signal can be computed with `len(prompt_string)` / 4 (char→token approximation) for both Claude Sonnet and reporting purposes. | D-06 cost/quality signal | Crude — real tokenization would differ by ~10-20%. Adequate for relative comparison (stacked-vs-unified delta), not for absolute pricing. Mitigation: planner labels the metric as "characters" or "approx tokens" honestly. `[ASSUMED]` |

## Open Questions

1. **Where does the ablation row land?**
   - What we know: D-10 says "new row in `ABLATION-TABLE.md` and `ABLATION-TABLE.tex`."
   - What's unclear: The only existing files are under `.planning/milestones/v1.0-phases/05-promote-and-ablation-artifact/`. v2.0 may want its own scoped artifact.
   - Recommendation: Planner adds a Plan 08-XX task to make this choice explicit, defaulting to "append to v1.0 file" if no objection.

2. **Is `render_ablation.py` append-safe?**
   - What we know: Phase 5 PROMO-03 used it to generate the v1.0 table.
   - What's unclear: Behavior on partial input / regeneration semantics.
   - Recommendation: Planner adds a "verify render_ablation.py behavior on s_linker14 input before sweep" task. If destructive, snapshot the v1.0 file first.

3. **What is the concrete unified-prompt design hypothesis to plan against?**
   - What we know: D-03 default hypothesis points to `_extract_entities_enriched` as the fold target. Research finds the actual current state is already-folded inside `_learn_document_knowledge_enriched`.
   - What's unclear: Whether the user prefers (a) tighten doc-knowledge prompt + flag in docstring, or (b) literally move trailing-word + scope into entity-extraction (which is a bigger surgery and risks regression).
   - Recommendation: Planner should propose BOTH (a) and (b) in 08-01-PLAN.md and pick (a) as the default with rationale, leaving (b) as the explicit fallback if (a) cannot be defended as "unification" to a reviewer. May warrant a planner ↔ user checkpoint.

4. **GATE-06 unit-audit operational mechanics.**
   - What we know: D-09 requires the unified prompt body to be scanned as a unit + reviewer-defensibility check.
   - What's unclear: Whether existing `06-GATE-06-AUDIT.md` mechanical scan tooling (word-bounded grep over BENCHMARK_TABOO terms) lives in a script or was done manually.
   - Recommendation: Planner reads `06-GATE-06-AUDIT.md` to recover the exact methodology used; if it's a manual grep checklist, replicate. If it's a script, reuse.

5. **What is the concrete `STACK_VS_UNIFY` provenance-string slot in the docstring?**
   - What we know: D-08 says the docstring records the choice + 1-sentence rationale citing the D-06 signal.
   - What's unclear: Does this go into the module-level docstring (alongside `REMOVED_FROM` / `RULES_REMOVED`) or into the class docstring?
   - Recommendation: Follow s_linker13.py:1-20 pattern — module-level docstring with structured tags. Add `STACK_VS_UNIFY: "unified" | "stacked"` and `STACK_VS_UNIFY_RATIONALE: "..."` as new tag lines.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3 | All linker code | ✓ (assumed — project is actively under dev) | per pyproject.toml | — |
| `pip` editable install | Adding `s_linker14.py` to the package surface | ✓ | — | — |
| Claude Sonnet via `LLMClient` (env `ANTHROPIC_API_KEY`) | Sweep + dev-loop runs | ✓ (assumed — used through v1.0+v2.0 to date) | — | None — required |
| `pytest` | Existing test invariants | ✓ | — | — |
| `render_ablation.py` | COMBINE-03 row update | ❓ (referenced in v1.0 ABLATION-TABLE.md header but not file-grepped this session) | — | Manual markdown row append + LaTeX table row append |
| Existing sweep harness (`run_ablation.py`) | COMBINE-02 dual-floor measurement | ✓ — file inspected; supports per-variant + per-dataset sweep | — | — |

**Missing dependencies with fallback:**
- `render_ablation.py` MAY be missing — fallback is manual markdown / LaTeX row append, which is straightforward given the existing 8-row table schema.

**Missing dependencies with no fallback:**
- None.

## Security Domain

Not applicable. This is a research code refactor with no auth, no session management, no input from untrusted users, no cryptography, and no external network surface beyond LLM API calls (which use existing credentials managed by `LLMClient`). No new attack surface introduced.

ASVS categories V2 (Auth), V3 (Session), V4 (Access Control), V5 (Input Validation), V6 (Cryptography): **N/A — no new attack surface.**

The only mildly-security-adjacent concern is the LLM API key (`ANTHROPIC_API_KEY`) — read by existing `LLMClient`, unchanged by this phase.

## Sources

### Primary (HIGH confidence)

- `/mnt/hostshare/ardoco-home/llm-sad-sam-v45/.planning/phases/08-combine-s-linker14-stack-or-unify-combined-llm-primitives/08-CONTEXT.md` — Phase 8 decisions D-01..D-10, scope, deferred ideas (read in full).
- `/mnt/hostshare/ardoco-home/llm-sad-sam-v45/.planning/REQUIREMENTS.md` — COMBINE-01/02/03 (read in full).
- `/mnt/hostshare/ardoco-home/llm-sad-sam-v45/.planning/STATE.md` — current position, standing gates (read in full).
- `/mnt/hostshare/ardoco-home/llm-sad-sam-v45/.planning/ROADMAP.md` — Phase 8 success criteria, standing gates (read in full).
- `/mnt/hostshare/ardoco-home/llm-sad-sam-v45/.planning/PROJECT.md` — Key Decisions, generality constraint, validated requirements (read in full).
- `/mnt/hostshare/ardoco-home/llm-sad-sam-v45/.planning/phases/06-ext-01-project-agnostic-standalone-mention-llm-primitive/06-SUMMARY.md` — Phase 6 close-empty disposition + EXT-01 cost/quality signal (Phase 8 input block).
- `/mnt/hostshare/ardoco-home/llm-sad-sam-v45/BENCHMARK_TABOO.md` — full taboo list + safe SE domains + Tailored Code Anti-Patterns.
- `/mnt/hostshare/ardoco-home/llm-sad-sam-v45/.planning/spikes/001-llm-trailing-words/README.md` — Spike-001 validation pattern.
- `/mnt/hostshare/ardoco-home/llm-sad-sam-v45/.planning/spikes/003-llm-mention-classifier/README.md` — Spike-003 piggyback pattern (D-03 reference shape).
- `/mnt/hostshare/ardoco-home/llm-sad-sam-v45/src/llm_sad_sam/linkers/experimental/s_linker13.py` — parent file, all 1198 lines mapped (lines 366-466 doc-knowledge, 480-592 seed validation, 711-839 entity pipeline, 841-999 validation, 1031-1101 coref, 1107-1170 helpers/checkpoint).
- `/mnt/hostshare/ardoco-home/llm-sad-sam-v45/src/llm_sad_sam/linkers/experimental/prompts_v2.py` — prompt constants (71-222 doc-knowledge + entity + coref bodies; 229-371 Phase 6 standalone-mention constants).
- `/mnt/hostshare/ardoco-home/llm-sad-sam-v45/run_ablation.py` — `CANONICAL_VARIANTS` (lines 40-86), `VARIANT_SPECS` (lines 88-359), `DATASETS` (lines 372-403).
- `/mnt/hostshare/ardoco-home/llm-sad-sam-v45/.planning/milestones/v1.0-phases/05-promote-and-ablation-artifact/ABLATION-TABLE.md` — existing 8-row ablation table format + baselines.
- `/mnt/hostshare/ardoco-home/llm-sad-sam-v45/.planning/config.json` — workflow flags (`nyquist_validation: false` — Validation Architecture section omitted).
- `/mnt/hostshare/ardoco-home/llm-sad-sam-v45/CLAUDE.md` — repo conventions (Claude Sonnet, standalone files, no leakage).
- `/home/dev/.claude/projects/-mnt-hostshare-ardoco-home-llm-sad-sam-v45/memory/MEMORY.md` — operator preferences + variance documentation + V35 simplification-regresses lesson.

### Secondary (MEDIUM confidence)

- `/mnt/hostshare/ardoco-home/llm-sad-sam-v45/.planning/milestones/v1.0-ROADMAP.md` — confirmed VAR-05 / VAR-06 origins for scope-field and alias-coref-fold (line 75-82).

### Tertiary (LOW confidence)

- None invoked. All claims in this RESEARCH.md are tagged either `[VERIFIED]` (codebase / file read this session), `[CITED]` (referenced directly from a Phase 8 / Phase 6 / v1.0 planning artifact), or `[ASSUMED]` (listed in Assumptions Log).

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — single internal codebase, no external library targets, all files read.
- Architecture (primitive call-graph audit): HIGH — every claimed line number verified against `s_linker13.py`.
- Pitfalls: HIGH for #1, #2, #3, #6 (directly grounded in codebase + CONTEXT/MEMORY); MEDIUM for #4, #5 (extrapolated from V35 MEMORY entry + Phase 6 variance documentation).
- Default unified design hypothesis (D-03 interpretation): MEDIUM — research finds D-03's wording anachronistic (the "fold into entity-extraction" target is the wrong tier for the current code). This is flagged as Pitfall #1 + Open Question #3; planner should confirm with user before proceeding.

**Research date:** 2026-05-31

**Valid until:** Stable internal codebase, no external dependencies. Valid until either (a) `s_linker13.py` is materially refactored outside this phase, or (b) the v2.0 milestone closes. Estimate: through Phase 9 close.
