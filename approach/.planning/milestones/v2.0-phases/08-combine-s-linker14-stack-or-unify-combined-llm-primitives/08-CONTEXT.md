# Phase 8: COMBINE — `s_linker14` Stack-or-Unify — Context

**Gathered:** 2026-05-31
**Status:** Ready for planning

<domain>
## Phase Boundary

Build `s_linker14.py` as the v2.0 deliverable that **unifies** the rule-removal LLM primitives accumulated through the s-linker variant chain into a coherent single design, OR documents that stacking is the empirically-supported choice.

The "stack-vs-unify" decision originally hinged on the EXT-01 cost/quality signal from Phase 6. With EXT-01 closed empty (no primitive shipped), Phase 8 exercises the decision on the **3 remaining rule-removal LLM primitives** already integrated in s_linker13:

1. Spike-001 trailing-words detection
2. scope:global|local alias field
3. alias-coref-fold

EXT-01 standalone-mention is NOT in scope (closed empty Phase 6).

In scope:
- Auditing where each of the 3 rule-removal primitives currently lives in s_linker13's call graph
- Designing a unified call shape that consolidates compatible primitives (Spike-001 trailing-words + scope-field as fold-into-entity-extraction candidates; alias-coref-fold as separate Tier evaluation)
- Building `s_linker14.py` (copy-fork from `s_linker13.py`, no inheritance) with the unified design
- Full 5-project sweep + dual-floor check
- Ablation row added to ABLATION-TABLE.md / .tex
- GATE-06 generality re-audit of unified prompts as a UNIT (combination can surface leakage individual audits missed)

Out of scope:
- Unifying verification stages (seed cite-evidence validation, Phase-3 ambiguous-name judge, coref antecedent verification) — those are NOT rule replacements per [[project-combine-scope]]
- EXT-01 standalone-mention (closed empty Phase 6)
- Cross-model GPT-5.2 evaluation (Phase 9)
- New rule-removal primitives (v2.0 scope is finishing the chain, not adding to it)

</domain>

<decisions>
## Implementation Decisions

### Unification Target

- **D-01:** **Unification scope = 3 rule-removal LLM primitives only.** Spike-001 trailing-words detection + scope:global|local alias field + alias-coref-fold. EXT-01 dropped (Phase 6 negative). All other s_linker13 stages (seed validation, Phase-3 judge, coref antecedent verification, doc_knowledge alias discovery, generic detection, convention filter) stay stacked as-is. Generic LLM calls are verification/judgment stages, not rule replacements — out of scope.

- **D-02:** **Stack-vs-unify decision is empirical, not pre-locked.** Plan phase will run a brief audit of the current s_linker13 call-graph locations of each primitive and propose one unified design + one stacked baseline (= s_linker13 itself). Both compete on a single full sweep. Winner ships as `s_linker14.py`.

- **D-03:** **Default unified design hypothesis** (planner may refine): Fold Spike-001 trailing-words detection INTO `_extract_entities_enriched` (Spike-003 pattern — zero net LLM cost). Fold scope-field assignment INTO `_extract_entities_enriched` (already adjacent to alias output). Keep alias-coref-fold as separate Tier-2 prompt (data dependency: needs the merged alias map first). Result: 2 unified calls + 1 stacked call = net -1 LLM call topology.

### Ship Rule

- **D-04:** **Ship if dual-floor passes — accept up to 6pp BBB regression vs s_linker13 parent.** Relaxed v2.0 cost budget. Unification justified as a generality/clarity win even with modest F1 cost; the rule-removal claim is the deliverable, not the F1 number.
  - Specifically: macro F1 ≥ 0.93 (GATE-01), BBB ≤ s_linker12c BBB + 6pp tolerance, other datasets ≤ s_linker12c per-dataset + 2pp tolerance.
  - GATE-05 hard-tier-first dev loop still applies — TM regression > 1pp vs s_linker13 parent → no full sweep, re-work.
  - If `s_linker14` regresses > 6pp BBB vs s_linker12c (i.e. fails dual-floor), Phase 8 closes with stacked s_linker13 declared the COMBINE winner (negative-on-unify outcome, still satisfies COMBINE-01..03 traceability).

### Empirical Comparison Method

- **D-05:** **Stack baseline = s_linker13 (already integrates the 3 primitives stacked).** No separate "stacked s_linker14" build — s_linker13's macro F1 (0.9509 from v1.0 final) is the stack baseline. Unify candidate is the new `s_linker14.py`. Direct head-to-head.

- **D-06:** **Cost/quality signal** must be captured in 08-SUMMARY.md (tagged `## COMBINE cost/quality signal` block for downstream reference). Minimum content:
  - LLM call count per (component, dataset) for `s_linker13` (stack baseline) vs `s_linker14` (unify candidate)
  - Wall-clock latency per dataset
  - Per-dataset + macro F1 delta
  - Prompt-length / token-count comparison for the unified vs stacked primitives
  - Stack-vs-unify winner + rationale string (will be the GATE-07 docstring `RULES_REMOVED` provenance entry per COMBINE-01)

### Canonical Promotion & Registration

- **D-07:** `s_linker14.py` is built as a standalone file (copy-fork from `s_linker13.py`, no inheritance — user preference + project convention). Registered in `run_ablation.py` `CANONICAL_VARIANTS` + `VARIANT_SPECS` with `canonical=True` only AFTER it passes the dual floor AND GATE-06 unit audit. If unify loses, `s_linker14.py` may still be committed as a rejected baseline (canonical=False) for the ablation table — `s_linker13` retains `canonical=True`.

- **D-08:** Structured docstring (GATE-07) records:
  - `RULES_REMOVED` = cumulative list of all v1.0+v2.0 rules removed (carried from `s_linker13`'s list; no new removals in Phase 8)
  - **Stack-vs-unify provenance string** = "unified" or "stacked" + 1-sentence rationale citing the D-06 cost/quality signal (per COMBINE-01)
  - `REMOVED_FROM` = `s_linker13` (immediate parent)

### Generality Audit (GATE-06)

- **D-09:** **Unified prompts get re-audited as a UNIT.** Per COMBINE-01 success criterion 4: the combined prompt may surface leakage that per-phase audits missed (e.g. a trailing-words example + an alias example in the same prompt might inadvertently echo a benchmark term combination). Audit recorded in `08-SUMMARY.md` as a dedicated section. Both BENCHMARK_TABOO mechanical scan AND reviewer-defensibility check required.

### Ablation Table Update

- **D-10:** New row in `ABLATION-TABLE.md` and `ABLATION-TABLE.tex` for the winner (`s_linker14` if unify wins, else explicit "s_linker13 retains COMBINE designation" annotation). Row includes the stack-vs-unify provenance string from D-08. Previous v1.0 ablation rows preserved. (COMBINE-03.)

### Claude's Discretion

Areas left to the planner / researcher:
- **Exact unified prompt design** for Spike-001+scope-field+entity-extraction fold. Must use safe SE textbook examples (BENCHMARK_TABOO.md). Spike-003 pattern (`_extract_entities_enriched` extension) is the reference shape.
- **Data-flow refactor** in the s_linker14 DAG to absorb Spike-001's output into the entity-extraction batched-scan output. Tier-1 sequencing may need adjustment.
- **Fallback policy** on LLM failure for unified call (default: approve-bias per existing pattern; planner may deviate with rationale).
- **Token-budget management** for the longer unified prompts (likely +30-50% prompt length vs single-purpose calls).
- **Per-call-site rewiring** in s_linker14 to consume the unified output where s_linker13 currently consumes 3 separate signals.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### v2.0 Milestone & Phase Definition
- `.planning/ROADMAP.md` — Phase 8 success criteria (COMBINE-01/02/03), standing GATE-01/05/06/07
- `.planning/REQUIREMENTS.md` — COMBINE-01 (s_linker14 integration), COMBINE-02 (dual-floor), COMBINE-03 (ablation row)
- `.planning/PROJECT.md` — Key Decisions, generality constraint
- `.planning/STATE.md` — current position, Phase 6 closed-empty status

### Phase 6 Outcome (informs Phase 8 scope reduction)
- `.planning/phases/06-ext-01-project-agnostic-standalone-mention-llm-primitive/06-SUMMARY.md` — Phase 6 closed empty, EXT-01 dropped from Phase 8 scope
- `.planning/phases/06-ext-01-project-agnostic-standalone-mention-llm-primitive/06-SUMMARY.md` tagged block `## EXT-01 cost/quality signal (Phase 8 input)` — confirms "EXT-01 NOT SHIPPED" and "Stack-vs-unify decision is unconstrained by EXT-01 (no primitive to stack)"

### Generality Audit Input
- `BENCHMARK_TABOO.md` — full taboo list + safe SE textbook domains + "Tailored Code Anti-Patterns" section (added Phase 6)

### Parent Baseline (= Stack Comparator)
- `src/llm_sad_sam/linkers/experimental/s_linker13.py` — v1.0 final artifact, macro F1 0.9509, already integrates the 3 primitives in stacked form. This is THE stack baseline.
- `src/llm_sad_sam/linkers/experimental/prompts_v2.py` — clean prompt constants; new unified prompts append here
- `run_ablation.py` — variant registration (GATE-07 enforcement point)
- `ABLATION-TABLE.md` / `ABLATION-TABLE.tex` — COMBINE-03 update target

### Rule-Removal Primitive Sources
- `.planning/spikes/001-llm-trailing-words/` — Spike-001 trailing-words detection (original cite-evidence pattern). Currently a Tier-1 LLM call in s_linker13.
- `.planning/spikes/003-llm-mention-classifier/README.md` — Spike-003 piggyback pattern (the unification reference for Spike-001 + entity-extraction fold)
- v1.0 ROADMAP `.planning/milestones/v1.0-ROADMAP.md` — full chain history including the scope:global|local alias field origin + alias-coref-fold origin

### v1.0 Pattern Reference
- `MILESTONES.md` §v1.0 — ablation-table outputs and final-artifact metrics (macro F1 0.9509 baseline for Phase 8 deltas)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`s_linker13.py` skeleton** — copy-fork template for `s_linker14.py`. 1198 lines. Project preference: no inheritance, no flags toggling structural behavior; sibling files compete; winner gets `canonical=True`.
- **`_extract_entities_enriched`** in s_linker13 — Spike-003 pattern; already reads `(comp_name, sentence, known_aliases)`. Natural extension point for folding in Spike-001 trailing-words AND scope-field assignment.
- **Cite-evidence + extract_json + retry-once + approve-bias** — established LLM call shape used throughout s_linker13. Unified prompt must follow the same shape.
- **`_VARIANT_NAME` + `_checkpoint_dir` namespacing** (s_linker13.py:1159, 1165) — s_linker14 must use a unique `_VARIANT_NAME` (`s_linker14`) to avoid cache cross-contamination.
- **`prompts_v2.py`** — new unified prompt constants belong here. Likely names: `UNIFIED_ENTITY_EXTRACTION_RULES` (extends existing extraction prompt with trailing-words + scope fields).
- **`run_ablation.py`** — `CANONICAL_VARIANTS` + `VARIANT_SPECS` registration (GATE-07).

### Established Patterns
- **One rule = one standalone variant file** — applies to s_linker14 as much as to the EXT-01 attempts.
- **Approve-biased fallback** on LLM failure — keeps recall when model is flaky. Unified call must match.
- **Structured variant docstring** with `REMOVED_FROM` / `RULES_REMOVED` + stack-vs-unify provenance string (D-08).
- **Dual-floor + hard-tier-first** — D-04 ship rule + GATE-05.
- **Per-variant `_checkpoint_dir`** — s_linker14 needs its own cache namespace.

### Integration Points (3 rule-removal primitives in s_linker13)
- **Spike-001 trailing-words detection** — currently a separate Tier-1 LLM call producing trailing-word labels per-sentence. Unify candidate: fold into `_extract_entities_enriched` output (Spike-003 pattern).
- **scope:global|local alias field** — currently assigned during doc_knowledge alias post-processing. Unify candidate: emit directly from the same unified entity-extraction call.
- **alias-coref-fold** — currently a Tier-2 pass that merges discovered aliases with coref antecedents. Likely STAYS stacked because of data-dependency (needs the full alias set built first). Planner to confirm.

</code_context>

<specifics>
## Specific Ideas

- **D-03 default hypothesis is empirical, not pre-locked** — the planner / researcher should audit the actual call-graph and may propose a different unification cut (e.g. only Spike-001 folded; both scope-field AND alias-coref-fold kept stacked). The 3 primitives are the SCOPE; the cut between unify and stack inside that scope is Claude's Discretion.
- **No new linker primitives** — Phase 8 is finishing the v1.0+v2.0 chain, not exploring new rule removals.
- **No EXT-01 / EXT-02 — Phase 6 closed empty, Phase 7 auto-skipped.** s_linker14 inherits `s_linker13`'s `_has_standalone_mention` and dotted-path guard unchanged.
- **Stack baseline is s_linker13 itself** — no separate "stacked s_linker14" file. Saves one full sweep.
- **D-09 unit audit is the differentiator vs per-phase GATE-06** — combined prompts can surface leakage that per-primitive audits miss. Must be a dedicated audit step, not a re-run of prior audits.

</specifics>

<deferred>
## Deferred Ideas

- **EXT-01 standalone-mention** — closed empty Phase 6 (see 06-SUMMARY.md).
- **EXT-02 drop dotted-path guard** — auto-skipped per ROADMAP gating.
- **EXT-04 variance-band tightening** — deferred to v2.1+.
- **GPT-5.2 cross-model run** — Phase 9.
- **New rule removals beyond the v1.0+v2.0 chain** — out of v2.0 thesis scope.

</deferred>

---

*Phase: 08-combine-s-linker14-stack-or-unify-combined-llm-primitives*
*Context gathered: 2026-05-31*
