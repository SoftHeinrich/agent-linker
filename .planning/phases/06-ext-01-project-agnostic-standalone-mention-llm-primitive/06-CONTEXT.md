# Phase 6: EXT-01 — Project-Agnostic Standalone-Mention LLM Primitive — Context

**Gathered:** 2026-05-30
**Status:** Ready for planning

<domain>
## Phase Boundary

Ship a new linker variant in which `_has_standalone_mention` (the last structural rule kept in `s_linker13`) is replaced by an LLM primitive that encodes **zero project-specific structure** — no Java packages, no dotted-path conventions, no BBB-style component names in prompts or logic.

In scope:
- Replacing `_has_standalone_mention` (s_linker13.py:1120-1147) and its callees
- Building two competing sub-variants and selecting one empirically (see decisions)
- Logging cost/quality signal that feeds the Phase 8 stack-vs-unify choice
- Recording a GATE-06 generality audit in SUMMARY.md

Out of scope:
- Dropping the dotted-path guard — deferred to Phase 7 (EXT-02), gated on Phase 6 pass
- Combining EXT-01 with other LLM primitives — that is Phase 8 (`s_linker14`)
- Cross-model GPT-5.2 evaluation — that is Phase 9

</domain>

<decisions>
## Implementation Decisions

### Semantic Scope of the LLM Primitive

- **D-01:** Do not pre-lock the semantic scope. Run an empirical study comparing three candidate primitives:
  - **Literal**: LLM mirrors regex semantics — "is `<comp_name>` a surface token here, not embedded in another identifier?"
  - **Semantic**: LLM judges "does this sentence reference the architectural component, not just contain the word?"
  - **Hybrid**: single call emits `{surface_mention: bool, architectural_ref: bool}`; caller chooses signal per call site.

- **D-02:** **Study method = offline anchor-collection diff → finalist sweep.** Replay `s_linker13`'s anchor-collection logic with each of the three primitives. Compute per-(component, dataset) diff vs the regex baseline (which sentences each variant picks as anchors). Drop any variant that diffs catastrophically vs the regex baseline (catastrophic = qualitatively obvious — research phase to operationalize the threshold). Surviving 1-2 variants get the full 5-project sweep. **No full-pipeline run during diff stage.**

- **D-03:** **Winner decided by macro F1 only.** Highest macro F1 from the finalist sweep wins. Ties broken by GATE-06 cleanliness (fewer/cleaner prompt examples, smaller call-graph footprint). Cost is not part of the winner-decision rule — it is captured separately as Phase 8 input (see D-06).

### Dotted-Path Handling (EXT-02 Hand-Off)

- **D-04:** **Phase 6 preserves dotted-path skip behavior** — EXT-02 is its own phase (Phase 7) and removes the guard there. Phase 6 ships **two sub-variants** that encode the skip in different ways and compete via the D-02 protocol:
  - **(a) Regex pre-filter + LLM judge** — keep the existing dotted/hyphen regex guards as a cheap pre-filter; LLM only judges sentences that survive the filter. EXT-02 then has a clean target (the pre-filter) to drop.
  - **(b) LLM-only with dotted-path encoded in prompt semantics** — no regex; the prompt teaches "a token embedded in a compound dotted identifier is not a standalone mention" using generic safe-domain examples (per BENCHMARK_TABOO.md). EXT-02 would then remove the prompt rule.
  - Both compete on the same diff → finalist sweep used for D-02. The semantic-scope choice (D-01) and the dotted-path choice (D-04) are evaluated as a single matrix.

- **D-05:** **Naming & promotion mirrors v1.0 13f→s_linker13 pattern.** Build the candidates as siblings (e.g. `s_linker13g_pre.py` and `s_linker13g_sem.py`); the winner is byte-copied to canonical `s_linker13g.py` (the EXT-01 deliverable). Loser stays in tree as a rejected artifact for the ablation table. Canonical file gets the structured docstring (`REMOVED_FROM` / `RULES_REMOVED`) and registration in `CANONICAL_VARIANTS` + `VARIANT_SPECS` per GATE-07.

### Claude's Discretion

Areas not discussed — Claude (researcher + planner) decides:
- **API shape & call topology** of the new LLM primitive (per-(comp, sent) call vs per-component batch vs piggyback on entity-extraction pass à la Spike 003 vs document-level enrichment map). The decision must serve D-02 (diff-able against regex baseline) and produce the D-06 cost signal. Spike 003 (`llm-mention-classifier`) is the natural pattern reference for the piggyback option.
- **Fallback policy** when the LLM returns malformed/empty (approve-bias matches existing patterns in `s_linker13._run_seed_validation`).
- **Anchor-section vs `has_exact_case`-flag split** — currently the same primitive serves both; the new primitive may keep them unified or split them.
- **Prompt-example domains** — must come from safe SE/textbook list in BENCHMARK_TABOO.md §"Safe SE Textbook Examples".

### Cost/Quality Signal for Phase 8

- **D-06:** Capture the **EXT-01 cost/quality signal** as a structured block in the Phase 6 SUMMARY.md, so Phase 8 can read it directly for the stack-vs-unify decision. Minimum content (planner may extend):
  - LLM call count delta vs `s_linker13` (per dataset, totals)
  - Wall-clock latency delta vs `s_linker13` (per dataset)
  - Per-dataset and macro F1 delta vs `s_linker13`
  - Notes on whether the call topology is naturally stackable with other LLM primitives or whether it suggests unifying into a single prompt
  - Tagged section header (e.g. `## EXT-01 cost/quality signal (Phase 8 input)`) so Phase 8 can grep for it deterministically

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### v2.0 Milestone & Phase Definition
- `.planning/ROADMAP.md` — Phase 6 success criteria (4 items), gating to Phase 7 (EXT-02), standing GATE-01/05/06/07
- `.planning/REQUIREMENTS.md` — EXT-01 (this phase), EXT-02 (Phase 7), GATE-06 generality constraint definition
- `.planning/PROJECT.md` — Key Decisions table (KEEP rationale, GATE-06 standing policy, stack-vs-unify deferred), Core Value, hard generality constraint
- `.planning/STATE.md` — current position, deferred items

### Generality Audit Input
- `BENCHMARK_TABOO.md` — full benchmark term list + safe SE textbook domains for prompt examples (GATE-06 §a)

### Spike Findings (Phase 6 design grounding)
- `.planning/spikes/002-rules-audit/AUDIT.md` — original RISKY classification of `_has_standalone_mention`, O(N×M) anchor-collection cost, recommended removal order
- `.planning/spikes/003-llm-mention-classifier/README.md` — pattern reference for piggybacking mention classification on entity-extraction prompt (a candidate API shape per Claude's Discretion)
- `.planning/spikes/001-llm-trailing-words/` — original cite-evidence LLM pattern that grounds all v1.0/v2.0 rule removals

### Parent Baseline & Replacement Site
- `src/llm_sad_sam/linkers/experimental/s_linker13.py` — parent variant, byte-equivalent to `s_linker13f` modulo class/banner
- `src/llm_sad_sam/linkers/experimental/s_linker13.py:1120-1147` — the rule being replaced (`_has_standalone_mention`)
- `src/llm_sad_sam/linkers/experimental/s_linker13.py:510,623,675,880,895,1095` — six call sites covering anchor collection, mention classification, evidence bundle, has-exact-case flag, coref antecedent verification
- `run_ablation.py` — `CANONICAL_VARIANTS` + `VARIANT_SPECS` registration (GATE-07 requirement)

### v1.0 Pattern Reference
- `.planning/milestones/v1.0-ROADMAP.md` — full v1.0 chain history including the 13f→`s_linker13` promotion pattern that D-05 mirrors
- `MILESTONES.md` §v1.0 — ablation-table outputs and final-artifact metrics (macro F1 0.9509 baseline for delta comparisons)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`s_linker13.py` skeleton** — standalone variant template; copy-fork pattern user prefers (no inheritance). 1198 lines.
- **Cite-evidence prompt pattern** — already used in `_run_seed_validation` (s_linker13.py:546-559) and elsewhere; structured JSON response + approve-biased fallback (line 568). New primitive should follow the same pattern.
- **`extract_json` + retry-once + approve-bias** — established LLM call shape (s_linker13.py:561-568). Reuse.
- **Spike 003 piggyback pattern** — `_extract_entities_enriched` prompt already reads `(comp_name, sentence, known_aliases)`; can be extended with a `mention_type` enum at zero net LLM cost. Candidate API shape.
- **`prompts_v2.py`** — clean prompt constants; new primitive's prompt belongs here per project convention.
- **`run_ablation.py`** — variant registration site; GATE-07 enforcement point.

### Established Patterns
- **One rule = one standalone variant file.** No inheritance, no flags toggling structural behavior. Sibling files compete; winner gets byte-copied to canonical name.
- **Approve-biased fallback** on LLM failure throughout `s_linker13` — keeps recall when the model is flaky. New primitive should match.
- **Structured variant docstring** with `REMOVED_FROM` / `RULES_REMOVED` (GATE-07). KEEP block also lists kept rules.
- **Dual-floor (GATE-01) + hard-tier-first (GATE-05)** — every full sweep must pass macro F1 ≥ 0.93, BBB tolerance ≤ 6pp vs `s_linker12c`, others ≤ 2pp; regress >1pp on TM/BBB vs parent → no full sweep.
- **Per-variant `_checkpoint_dir` namespacing** (s_linker13.py:1159) — new variants must keep `_VARIANT_NAME` in checkpoint paths (assertion at line 1165).

### Integration Points
- Six call sites in `s_linker13.py` for `_has_standalone_mention`: anchor collection (3×), `_classify_mention`, `has_exact_case` flag, coref antecedent verification. All need to consume the new primitive (or its outputs) — API shape (Claude's Discretion) decides whether they call one-by-one or read a precomputed map.
- `_classify_mention` (s_linker13.py:617-647) currently calls `_has_standalone_mention` and returns a human-readable string; Spike 003 shows this can be folded into entity-extraction at zero LLM cost.
- `_run_ablation.py` `CANONICAL_VARIANTS` + `VARIANT_SPECS` registration is the GATE-07 enforcement point — the canonical `s_linker13g.py` must land there before any sweep is considered final.
- SUMMARY.md is the Phase 6 → Phase 8 hand-off file; the D-06 cost/quality block lives there.

</code_context>

<specifics>
## Specific Ideas

- User explicitly chose **empirical comparison over a priori lock-in** for both gray areas discussed (D-01 + D-04). The phase's primary deliverable artifact is therefore a **decision derived from data**, not a pre-specified design. Planner must treat the study itself as a first-class plan, not an appendix.
- The semantic-scope study (D-01) and the dotted-path study (D-04) are evaluated as **one matrix** (3 semantic scopes × 2 dotted-path encodings = up to 6 cells), filtered by the D-02 anchor-diff stage. Planner may collapse cells that are obviously equivalent (e.g. literal-scope × prompt-encoded-dotted-path may be redundant).
- "Catastrophic diff" threshold (D-02) is intentionally left for the research phase to operationalize from data, not specified up front.
- D-06 cost signal must be **tagged for Phase 8 grep** — Phase 8 reads this section directly when picking stack vs unify.

</specifics>

<deferred>
## Deferred Ideas

- **API shape & call topology** for the new primitive (per-(comp, sent) vs per-component batch vs Spike-003 piggyback vs document-level map). Skipped in discussion → assigned to Claude's Discretion (constrained by D-02 diff-ability and D-06 cost-signal requirements).
- **Cost/quality metric set** beyond the D-06 minimum — planner may add token counts, p50/p95 latency, per-call-site breakdown.
- **Fallback policy** on LLM failure — defaulted to approve-bias per existing `s_linker13` pattern unless planner finds a reason to deviate.
- **Anchor vs has_exact_case split** — currently unified in regex; deferred to API-shape choice.
- **EXT-04** (emit-biased boundary prompting on alias-discovery; BBB variance band tightening) — already deferred to v2.1+ per ROADMAP/PROJECT.md.

</deferred>

---

*Phase: 06-ext-01-project-agnostic-standalone-mention-llm-primitive*
*Context gathered: 2026-05-30*
