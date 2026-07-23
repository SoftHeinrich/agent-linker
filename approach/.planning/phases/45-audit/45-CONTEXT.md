# Phase 45: AUDIT - Context

**Gathered:** 2026-06-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Produce a single read-only audit document, `s_linker20-PROMPT-AUDIT.md`, covering every static prompt fragment used by `s_linker19`:

- The 9 imported PROMPT CONSTANTS from `prompts_v5.py`: `AMBIGUITY_FEW_SHOT`, `AMBIGUITY_RULES`, `DOC_KNOWLEDGE_EXTRACTION_RULES`, `ALIAS_SCOPE_RULES`, `DOC_KNOWLEDGE_JUDGE_EXAMPLES`, `DOC_KNOWLEDGE_JUDGE_RULES`, `ENTITY_EXTRACTION_RULES`, `VALIDATION_RULES`, `COREF_RULES`.
- The 6 in-class f-string scaffolds in `s_linker19.py`: `_prompt_ambiguity`, `_prompt_doc_knowledge_extract`, `_prompt_doc_knowledge_judge`, `_prompt_extraction`, `_prompt_validation`, `_prompt_coref`.

Each item receives: current LOC, a generality verdict (`clean` / `domain-loaded` / `benchmark-leak`), and a list of line-level cut candidates + drop-whole-block candidates that Phase 46 will trial against the Phase 44 golden harness.

**In scope (Phase 45):**

- The single combined artefact `.planning/milestones/v2.6.4-*/s_linker20-PROMPT-AUDIT.md` (planner picks final location under the milestone tree per existing convention).
- Verdict assignment per item using the rubric in D-01/D-02.
- Cut-candidate enumeration with the structured row schema in D-08.
- Per-cut risk scoring (low / med / high) and harness annotation (which test modules gate each cut) per D-06.
- For every `benchmark-leak` finding: as many neutral rewordings as inspection suggests, split across two style families per D-04/D-05.
- F-string scaffolds: prose instructions audited; JSON-schema literals (`Return JSON: {...}`, `JSON only:`) skipped per D-03.

**Out of scope (Phase 45):**

- Any code change to `s_linker19.py`, `prompts_v5.py`, `s_linker13_min.py`, or any module they import. GATE-01 byte-equal must hold at phase close.
- Running the harness against trial cuts to validate them. The audit annotates harness coverage but does not execute trial cuts — that is Phase 46 (REQ-V264-05/06/07).
- Generating rewordings for `domain-loaded` items. Strict to the success-criterion wording: only `benchmark-leak` findings carry proposed rewordings; `domain-loaded` items are flagged and left for Phase 46's empirical lexical-neutralization loop. (D-04.)
- Creating `s_linker20.py`. That file does not exist until Phase 47.
- Claude-backend or cross-backend analysis. gpt-5.4 only (v2.3 standing policy).
- BBB recall closure, v2.7 work, v2.6 Phase 37 close — frozen / deferred per ROADMAP.

</domain>

<decisions>
## Implementation Decisions

### Verdict Rubric (the three categories)

- **D-01 — `domain-loaded` is pragmatic, not strict.** A constant or prose line is `domain-loaded` only when its domain vocabulary is *over-specified* — i.e., a universal noun (`entity`, `name`, `phrase`, `pronoun`) would carry the same meaning to the LLM. Examples that the user expects to come out as candidates: gratuitous uses of "software architecture component" / "anaphoric references" / "role-referential noun phrase" where the surrounding context already constrains the task. Examples that should stay `clean`: domain terms that are load-bearing for the SAD→SAM task (the "look general but still SAD/SAM-tuned" principle from PROJECT.md). The pragmatic rubric was chosen over a strict-jargon rubric to keep Phase 46's candidate list defensible — every `domain-loaded` flag must have a plausible universal-noun replacement.

- **D-02 — `benchmark-leak` detection = mechanical grep + manual review of universal-taboo hits.** For every word in every constant and prose line, run a token lookup against `BENCHMARK_TABOO.md` (all 5 dataset sections + the `Universal Taboo` list). Any hit is a candidate leak. Universal-taboo hits (e.g., `logic`, `storage`, `cache`, `UI`, `client`) then get a second-pass cross-dataset isolation check per v2.1 GATE-06 methodology: a token `t` only counts as a leak if, in this prompt context, it identifies a specific dataset's component rather than functioning as a generic SE noun. Per-dataset-section hits (e.g., `Scheduler` outside Universal Taboo, `RequestHandler`-style names) auto-classify as leak without manual review.

- **D-03 — F-string scaffold scope: prose only, skip JSON-schema literals.** Inside the 6 builder f-strings, audit instruction prose (e.g., `"Be conservative — only include resolutions you are CERTAIN about"` at `s_linker19.py:361`) under the same 3-verdict rubric as the imported constants. Skip JSON-schema literals (`Return JSON: {…}` blocks, `JSON only:` suffix) because they are byte-equality-critical for the parser path and not load-bearing for behavior. The decision keeps the audit focused on cuttable surface area.

### Few-Shot Block Treatment

- **D-04 — Few-shot example names get the strict `benchmark-leak` interpretation.** `AMBIGUITY_FEW_SHOT` and `DOC_KNOWLEDGE_JUDGE_EXAMPLES` use synthetic names (`Scheduler`, `RequestHandler`, `CacheLayer`, `TaskScheduler`, `Handler`). Any name with a hit in BENCHMARK_TABOO (keyword, alias, or universal list) is flagged `benchmark-leak`. `cache` (CacheLayer head noun) and any name overlapping with a per-dataset alias auto-classify. The few-shot blocks also appear as first-class **drop-whole-block** candidates in the cut list (per REQ-V264-06 they are tested as full-removal candidates in Phase 46 anyway), but the per-token leak audit still happens so that if a partial replacement is the survivor in Phase 46, the leak verdict carries.

### Rewording Scope and Style

- **D-05 — Reword `benchmark-leak` only.** Strict to the success criterion wording. Domain-loaded items are listed and characterized in the audit, but their rewordings are produced empirically by Phase 46 against the harness (REQ-V264-07). This preserves Phase 45's scope boundary and avoids wasted Phase 45 effort on rewordings that may fail byte-equality.

- **D-06 — Two rewording families per benchmark-leak finding, multiple variants per family, no fixed count.** For each leak:
  - **Family A (synthetic-neutral name swap):** keep the prompt shape, swap benchmark-overlapping names for synthetic names chosen to avoid ALL 5 dataset taboo lists. Generate as many variants as inspection plausibly supports.
  - **Family B (concept-only / name-stripped):** rewrite the affected segment to describe the rule abstractly without component-name examples. Generate as many variants as inspection plausibly supports.
  Each variant is a standalone row with its own `cut_id` (see D-08). Phase 46 tests in cheapness-then-coverage order and ships whichever (if any) survives byte-equality.

### Audit Depth and Predictive Annotations

- **D-07 — Annotated audit with D-03 (Phase 44) test-module cross-reference + per-cut risk tier; no harness execution.** Each cut row carries a `gated_by` column listing which Phase-44 test module(s) and which phase tag(s) gate the cut, using the locked mapping from Phase 44 §D-03:
  | Builder | Phase tag(s) |
  |---|---|
  | `_prompt_ambiguity` | `phase_1_model` |
  | `_prompt_doc_knowledge_extract` | `phase_1_doc_extract` |
  | `_prompt_doc_knowledge_judge` | `phase_1_doc_judge` |
  | `_prompt_extraction` | `phase_2_framing_c_pass1`, `phase_2_framing_c_pass2` |
  | `_prompt_validation` | `phase_4_twopass_p1`, `phase_4_twopass_p2`, `phase_5_coref_validation` |
  | `_prompt_coref` | `phase_5_coref` |
  Each cut also carries a 3-tier `risk` score (`low` / `med` / `high`) representing the prior probability of surviving byte-equality on all 97 Phase-44 snapshots. The risk score is opinionated reviewer judgment, not measured — Phase 46 will validate empirically. The annotation gives Phase 46 a "run THIS subset to verify THIS cut" recipe and lets it batch low-risk cuts first.

### Audit Doc Layout and Cut-Row Schema

- **D-08 — Layout: 5 sections by s19 pipeline phase; per-section subsections cover the imported constant(s) AND the builder that uses them.** Sections:
  1. Phase 1 — Ambiguity (`AMBIGUITY_FEW_SHOT`, `AMBIGUITY_RULES`, `_prompt_ambiguity`)
  2. Phase 1 — Doc-Knowledge (`DOC_KNOWLEDGE_EXTRACTION_RULES`, `DOC_KNOWLEDGE_JUDGE_EXAMPLES`, `DOC_KNOWLEDGE_JUDGE_RULES`, `ALIAS_SCOPE_RULES`, `_prompt_doc_knowledge_extract`, `_prompt_doc_knowledge_judge`)
  3. Phase 2 — Extraction (`ENTITY_EXTRACTION_RULES`, `_prompt_extraction`)
  4. Phase 4 — Validation (`VALIDATION_RULES`, `_prompt_validation`; `P1_FOCUS` + `P2_FOCUS` + `COREF_VALIDATION_FOCUS` covered here since they enter the same builder)
  5. Phase 5 — Coref (`COREF_RULES`, `ANTECEDENT_ALIAS_RULES`, `_prompt_coref`)
  This colocates the constant(s) with the builder that imports them — Phase 46 reasons about a builder-and-its-imports as a unit.

  Each item has a header table giving verdict and current LOC:
  ```
  | Item | Verdict | LOC |
  |---|---|---|
  | AMBIGUITY_FEW_SHOT | benchmark-leak | 7 |
  | AMBIGUITY_RULES    | clean          | 1 |
  | _prompt_ambiguity  | clean          | 18 |
  ```

  Cut candidates follow as a structured table (one row per candidate, including drop-whole-block and rewording variants):
  ```
  | cut_id | file:lines | trigger | before | after | risk | gated_by |
  |---|---|---|---|---|---|---|
  | CUT-AMB-01 | prompts_v5.py:30–36 | benchmark-leak (drop-block, REQ-V264-06) | AMBIGUITY_FEW_SHOT entire block | "" | high | tests/test_s_linker20_prompt_ambiguity.py @ phase_1_model |
  | CUT-AMB-02 | prompts_v5.py:30–36 | benchmark-leak (Family A: synthetic-neutral swap) | "Scheduler" examples | "OrderProcessor" examples (full rewrite) | med | tests/test_s_linker20_prompt_ambiguity.py @ phase_1_model |
  | CUT-AMB-03 | prompts_v5.py:30–36 | benchmark-leak (Family B: concept-only) | "Scheduler" examples | abstract rule restatement | med | tests/test_s_linker20_prompt_ambiguity.py @ phase_1_model |
  ```
  - `cut_id`: `CUT-{section-tag}-{NN}` — 1:1 referenced by Phase 46's `s_linker20-MINIMIZE-LOG.md`. Section tags: `AMB`, `DKX`, `DKJ`, `EXT`, `VAL`, `COR`.
  - `file:lines`: full repo-relative path + line range.
  - `trigger`: which verdict + which sub-case drove the candidate. For rewordings, includes the Family label.
  - `before` / `after`: short snippet (truncate with `…` over 80 chars per cell — the full proposed rewrite goes in a per-cut detail block below the table when the rewrite is long).
  - `risk`: `low` / `med` / `high` per D-07 reviewer judgment.
  - `gated_by`: test module path + `@` + phase tag(s). Multiple tags listed comma-separated (validation cuts gated by 3 tags).

### Claude's Discretion

- Final on-disk location of `s_linker20-PROMPT-AUDIT.md` (planner choice between `.planning/milestones/v2.6.4-…/` and `.planning/phases/45-audit/` — must be one of these per project convention).
- Whether the per-cut detail blocks (for long rewordings or multi-line before/after) live under each section or in an appendix at the end.
- Whether to include a top-of-doc summary table aggregating per-item verdicts before the per-section dives.
- Whether to capture audit reviewer-judgement notes (e.g., "this is a low-risk candidate because the parser ignores the suffix") inline per cut or in a closing rationale section.
- Whether risk tiers carry a short justification column or stay as the bare tag.
- Whether `P1_FOCUS`, `P2_FOCUS`, `COREF_VALIDATION_FOCUS` (small constants imported by `_prompt_validation`) get their own audit rows or fold into the validation builder's row. They were not in REQ-V264-03's enumerated 9 constants, but they are imported by `s_linker19` and gated by the same harness modules — planner picks whether to add them as bonus rows or hold for Phase 46.
- Whether the doc opens with a 2-sentence verdict-rubric recap or assumes the reader will follow the link to this CONTEXT.md.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase scope, requirements, gates

- `.planning/ROADMAP.md` §"Phase 45: AUDIT" — phase goal and 4 success criteria.
- `.planning/REQUIREMENTS.md` §"AUDIT" — REQ-V264-03 (per-constant audit, 9 constants enumerated) and REQ-V264-04 (per-builder audit, 6 builders enumerated, single combined artefact).
- `.planning/PROJECT.md` §"Constraints" and §"Key Decisions" — GATE-01 byte-equal of `s_linker19.py` + `s_linker13_min.py`; v2.1 GATE-06 cross-dataset isolation methodology; gpt-5.4 only (v2.3 standing); BENCHMARK_TABOO compliance.
- `.planning/STATE.md` — current milestone v2.6.4, Phase 45 second of six.
- `BENCHMARK_TABOO.md` (repo root) — the canonical taboo list for D-02 mechanical grep. Sections per dataset (MediaStore, TeaStore, Teammates, BigBlueButton, JabRef) + `Universal Taboo`. Required reading before any verdict assignment.

### Frozen source artefacts (read-only during this phase)

- `src/llm_sad_sam/linkers/experimental/prompts_v5.py` — defines all 9 PROMPT CONSTANTS + `P1_FOCUS`, `P2_FOCUS`, `COREF_VALIDATION_FOCUS`, `ANTECEDENT_ALIAS_RULES`. 124 LOC total. The full source of truth for the imported-constants half of the audit.
- `src/llm_sad_sam/linkers/experimental/s_linker19.py` — the 6 builder methods are at lines 264–378 (`_prompt_ambiguity` 264–282, `_prompt_doc_knowledge_extract` 284–302, `_prompt_doc_knowledge_judge` 304–319, `_prompt_extraction` 321–335, `_prompt_validation` 337–350, `_prompt_coref` 352–378). The `_TracingLLMClient` at line 121 + phase tags inside the run methods (561, 573, 646, 793, 835, 894) confirm the D-03 mapping.

### Phase 44 harness — the gate the audit annotates against

- `.planning/phases/44-harness/44-CONTEXT.md` — Phase 44 context, especially §D-03 (builder → phase-tag mapping) used verbatim in D-07. §D-01/D-02 (fixture sources, manifest) for cross-reference context.
- `.planning/phases/44-harness/44-VERIFICATION.md` — confirms harness is green (149 passed, 97 snapshots committed, GATE-01 holds). Audit may quote the snapshot counts when scoring risk.
- `tests/harness/MANIFEST.json` (referenced from `tests/harness/loader.py`) — pinned per-project fixture pairing. The audit's `gated_by` annotations resolve through this manifest.
- `tests/test_s_linker20_prompt_ambiguity.py`, `…_doc_extract.py`, `…_doc_judge.py`, `…_extraction.py`, `…_validation.py`, `…_coref.py` — the six golden-replay test modules referenced by `gated_by`. Audit does not modify them.
- `tests/harness/adapters.py`, `tests/harness/loader.py`, `tests/harness/replay_client.py`, `tests/harness/inputs.py`, `tests/harness/manifest.py` — the harness internals; read-only for Phase 45, but the audit's annotation column refers to them when explaining how a cut would be gated.

### Standing gates and methodology

- `.planning/codebase/CONCERNS.md` — context on the "variant proliferation" hygiene preference (standalone files, no inheritance). The audit doc itself is a planning artefact, not code, but the verdict rubric should not propose cuts that would push toward inheritance/shared modules.
- `.planning/milestones/v2.6.1-MILESTONE-AUDIT.md` and `.planning/milestones/v2.6.2-MILESTONE-AUDIT.md` — prior context for the s_linker17e breakthrough (92.3% gpt-5.4) that pins the v2.6.4 floor at 91.3%. The audit should not propose cuts that obviously gut the validated-coref or twopass behavior that drives that floor.

### Prior context (for pattern continuity)

- `.planning/phases/44-harness/44-PATTERNS.md` — Phase 44 pattern map; same shape (5-column table → row-per-item) carries over for Phase 45's cut table.
- `.planning/phases/44-harness/44-DISCUSSION-LOG.md` — Phase 44 discussion log. Sets the precedent for explicit ledger artefacts (the manifest in Phase 44 ↔ the per-cut MINIMIZE-LOG in Phase 46) — the cut_id schema in D-08 is the bridge.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- **Phase 44 builder → phase-tag mapping** (`.planning/phases/44-harness/44-CONTEXT.md` §D-03) — locked, reused verbatim by D-07 for the `gated_by` column. No rederivation needed.
- **`BENCHMARK_TABOO.md`** (`./BENCHMARK_TABOO.md`) — already curated per-dataset + universal sections. Token lookup against it is the canonical mechanical step for D-02.
- **Phase 44 harness snapshots** (`tests/test_s_linker20_prompt_*.ambr` files, 97 snapshots across 6 modules) — the empirical reality the risk tier in D-07 estimates against. Audit reads them passively (without running) to inform risk tagging — e.g., "this builder has 40 snapshots, low cross-project variance, so a wording tweak is low-risk".
- **`prompts_v5.py` module docstring** (lines 1–22) — already records the design rationale for `P1_FOCUS` and `COREF_VALIDATION_FOCUS`. The audit should not contradict the docstring's load-bearing claims (e.g., the "qualified-name identifier (e.g. X.Y.Z)" clause empirically catches code-path FPs); these claims become low-confidence-cut justifications in the risk column.

### Established Patterns

- **Standalone, structured ledger artefacts.** Phase 44 used `MANIFEST.json` as a single explicit ledger. Phase 45 follows the same pattern with a single `s_linker20-PROMPT-AUDIT.md`. Phase 46's `s_linker20-MINIMIZE-LOG.md` and Phase 47's `s_linker20.py` will continue the chain by `cut_id` reference.
- **Read-only on frozen artefacts.** Established by Phase 43 (replay) and Phase 44 (harness). Phase 45 carries this forward: zero code changes; GATE-01 must hold at phase close.
- **Per-builder isolation (no shared base classes).** Carries to the audit doc structure: each builder is audited as a unit alongside its imported constants, not via shared analysis layers.

### Integration Points

- **Phase 46 input.** The cut_id column is the integration contract: every row in Phase 46's `s_linker20-MINIMIZE-LOG.md` references a `cut_id` from this audit. No new cuts are introduced in Phase 46 unless flagged "EMERGENT" with a rationale.
- **Phase 47 ship.** The audit's verdict column drives which prompts get inlined in `s_linker20.py` (and in what form — original / synthetic / concept-only / dropped) once Phase 46 has produced the minimized text.
- **Phase 48 sweep.** GATE-06 re-verification on `s_linker20` uses the audit's leak inventory as the baseline check: every benchmark-leak finding must either have been cut or have its rewording survive GATE-06 grep on the shipped text.

</code_context>

<specifics>
## Specific Ideas

- **"Look general but still SAD/SAM-tuned" is the framing principle** (from PROJECT.md current-milestone goal). D-01's pragmatic rubric is the operationalization: behavior stays tuned, only the surface vocabulary is examined. Audit reviewers should ask "would a universal noun carry the same instruction here?" rather than "is this jargon?".
- **The Phase 44 harness has 97 snapshots, 149 passing tests, zero LLM calls** (44-VERIFICATION.md). Risk-scoring in D-07 should treat the snapshot-count distribution per builder as part of the prior — builders with more snapshots will catch more variance, so a wording change there is technically safer to attempt (more chances to fail) but more frequently fails (more diverse fixtures). Reviewer judgment per cut, not a formula.
- **Two rewording families, no count cap.** Per D-06, generate as many synthetic-neutral and concept-only variants as plausibly come up during inspection. Phase 46 chooses among them empirically; abundance here is cheap.
- **Drop-whole-block is first-class for the two few-shot constants.** REQ-V264-06 explicitly asks Phase 46 to try full removal first. The audit must include drop-block as the first `cut_id` under `AMBIGUITY_FEW_SHOT` and `DOC_KNOWLEDGE_JUDGE_EXAMPLES`.
- **JSON-schema literals are excluded from audit by D-03**, but if any schema-text glue looks load-bearing (e.g., the redundant `N_INTEGER` placeholder repeated in extraction + coref), the audit may flag it in a sidebar without proposing a cut — for Phase 46 visibility only.

</specifics>

<deferred>
## Deferred Ideas

- **Pilot trial cuts** (running 2–3 candidate cuts through the harness during Phase 45). Considered, rejected for Phase 45 in D-07 — execution is Phase 46's job. Revisit only if the audit doc cannot decide a risk tier from inspection alone.
- **Rewordings for `domain-loaded` items.** Phase 45 only catalogs them per D-05. Phase 46 (REQ-V264-07) does the empirical lexical-neutralization loop and produces rewordings under harness gating.
- **`P1_FOCUS` / `P2_FOCUS` / `COREF_VALIDATION_FOCUS` as standalone audit rows.** Not in the REQ-V264-03 enumeration. Flagged as Claude's discretion above — planner picks. Default: fold into the validation builder's row to stay strictly REQ-aligned.
- **Claude-backend audit.** Out of scope — v2.6.4 is gpt-5.4 only. Mirror audit is a v2.6.5 candidate iff v2.6.4 promotes.
- **Per-prompt minimization for `s_linker17e`.** Reviewers may ask for it if 17e remains the published champion. Deferred per `.planning/REQUIREMENTS.md` "Future Requirements".

### Reviewed Todos (not folded)

The `cross_reference_todos` step surfaced 6 low-relevance todos (mostly v2.6.1 axiom work, voyager-training redesign, and an unrelated flex-tier integration). None fit Phase 45's read-only audit scope. They remain in the global todo backlog without per-phase tagging.

</deferred>

---

*Phase: 45-AUDIT*
*Context gathered: 2026-06-07*
