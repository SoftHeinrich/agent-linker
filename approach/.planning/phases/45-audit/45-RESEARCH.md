# Phase 45: AUDIT — Research

**Researched:** 2026-06-07
**Domain:** Prompt-text audit — generality verdicts + benchmark-leak detection + cut-candidate enumeration for `s_linker19` / `prompts_v5.py`
**Confidence:** HIGH (all claims grounded in direct file reads of the frozen source artefacts)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01** — `domain-loaded` is pragmatic: flag only when a universal noun carries the same meaning; domain terms load-bearing for SAD→SAM stay `clean`.
- **D-02** — `benchmark-leak` detection = mechanical BENCHMARK_TABOO grep + manual second-pass for universal-taboo hits (cross-dataset isolation check per v2.1 GATE-06 methodology).
- **D-03** — F-string scaffold scope: audit prose instruction lines only; skip JSON-schema literals (`Return JSON: {…}`, `JSON only:`).
- **D-04** — Few-shot example names get strict benchmark-leak interpretation; drop-whole-block is a first-class cut_id for `AMBIGUITY_FEW_SHOT` and `DOC_KNOWLEDGE_JUDGE_EXAMPLES`.
- **D-05** — Rewording scope: `benchmark-leak` items only; `domain-loaded` items catalogued but not reworded in Phase 45.
- **D-06** — Two rewording families per benchmark-leak finding (Family A: synthetic-neutral name swap; Family B: concept-only). No count cap.
- **D-07** — Annotated audit with Phase 44 D-03 test-module cross-reference + per-cut risk tier; no harness execution.
- **D-08** — Layout: 5 sections by s19 pipeline phase, each colocating imported constants and their builder. Section tags: AMB, DKX, DKJ, EXT, VAL, COR. Cut schema: `cut_id | file:lines | trigger | before | after | risk | gated_by`.

### Claude's Discretion

- Final on-disk location of `s_linker20-PROMPT-AUDIT.md`: `.planning/milestones/v2.6.4-*/` or `.planning/phases/45-audit/`.
- Whether per-cut detail blocks for long rewordings go under each section or in an appendix.
- Whether a top-of-doc summary table aggregating per-item verdicts precedes the per-section dives.
- Whether inline reviewer-judgment notes appear per cut or in a closing rationale section.
- Whether risk tiers carry a short justification column or stay as bare tag.
- Whether `P1_FOCUS`, `P2_FOCUS`, `COREF_VALIDATION_FOCUS` get their own bonus audit rows or fold into their builder's row.
- Whether the doc opens with a 2-sentence verdict-rubric recap or links to CONTEXT.md.

### Deferred Ideas (OUT OF SCOPE)

- Pilot trial cuts during Phase 45 (execution is Phase 46).
- Rewordings for `domain-loaded` items (Phase 46 empirical loop, REQ-V264-07).
- `P1_FOCUS` / `P2_FOCUS` / `COREF_VALIDATION_FOCUS` as standalone rows — default is fold into validation builder row.
- Claude-backend audit (out of scope; v2.6.4 is gpt-5.4 only).
- Per-prompt minimization for `s_linker17e`.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| REQ-V264-03 | Per-constant audit report covers each of the 9 imported PROMPT CONSTANTS with columns: current LOC, generality verdict, size-cut candidates (line-level), drop-the-whole-block candidates. | Section 1 below gives exact LOC, first-pass verdict hints, and taboo-hit tokens for each constant. |
| REQ-V264-04 | Per-builder audit covers the 6 in-class f-string scaffolds with same columns as REQ-V264-03. Single combined artefact `s_linker20-PROMPT-AUDIT.md`. | Section 2 below confirms line ranges, identifies imported constants per builder, and splits prose vs JSON-schema lines. |
</phase_requirements>

---

## Summary

Phase 45 produces a single read-only documentation artefact (`s_linker20-PROMPT-AUDIT.md`) covering 9 imported PROMPT CONSTANTS from `prompts_v5.py` and 6 in-class f-string scaffolds from `s_linker19.py`. The work is pure text inspection: no code changes, no harness runs, zero LLM calls, GATE-01 byte-equal throughout.

The primary research finding is that `prompts_v5.py` is 124 LOC total and is already lean. The two few-shot constants (`AMBIGUITY_FEW_SHOT` at 7 LOC and `DOC_KNOWLEDGE_JUDGE_EXAMPLES` at 7 LOC) carry the clearest benchmark-leak risk — both use component names (`Scheduler`, `RequestHandler`, `CacheLayer`, `Handler`) that hit BENCHMARK_TABOO. The other 7 constants and all 6 builder scaffolds are mostly clean or only weakly domain-loaded; the universal-taboo second pass will need to adjudicate `storage`, `client`, `server`, `logic`, `validation`, `adapter`, `event`, `socket`, `layer` where they appear as generic SE prose.

The Phase 44 harness is fully green (149 passed, 97 snapshots across 6 `.ambr` files). Snapshot counts per builder are the key risk-tier prior: coref has 40 snapshots (most diverse), validation has 24, extraction has 18, doc-extract/doc-judge/ambiguity have 5 each (least diverse — lower empirical pressure on any single wording change).

**Primary recommendation:** Produce the audit as a single Wave-1 task per pipeline section (5 sections = 5 parallelizable sub-tasks), with a Wave-0 framing task that writes the header table and verdict-rubric recap. The BENCHMARK_TABOO grep runs as an inline step inside each section sub-task, not as a global pre-pass.

---

## 1. Concrete Inventory: 9 PROMPT CONSTANTS (`prompts_v5.py`)

All line numbers are from direct read of `src/llm_sad_sam/linkers/experimental/prompts_v5.py` (124 LOC total). [VERIFIED: direct file read]

### 1.1 Phase 1 — Ambiguity constants (lines 30–38)

#### AMBIGUITY_FEW_SHOT (lines 30–36, 7 LOC)

**Text shape:** Two parallel few-shot examples, each with Name / Sentence / Classification / Rationale. Both examples use `"Scheduler"` as the component name. No bullet rules. No JSON schema.

**First-pass categorisation hint:** `benchmark-leak` candidate. (This is a hint for the auditor; the verdict is assigned during Phase 45 work.)

**BENCHMARK_TABOO hits to expect:**

| Token | Section | Hit type |
|-------|---------|----------|
| `Scheduler` | Universal Taboo + BigBlueButton/TeaStore vicinity | Per D-04: "Scheduler" is in BENCHMARK_TABOO.md §"Safe SE Textbook Examples" as an OS-domain example — NOT taboo. However, auditor must confirm. More importantly, `scheduler-based` appears in Example 2. The word `scheduler` (lowercase) is not in Universal Taboo as a standalone entry, but `TaskScheduler` is used in `ANTECEDENT_ALIAS_RULES` examples (line 121–122) — different constant. |
| `worker threads` | — | Not taboo; generic SE. |
| `queues`, `dispatches`, `nodes` | — | Generic SE. |

**Critical note for auditor:** BENCHMARK_TABOO.md §"Safe SE Textbook Examples" explicitly lists `Scheduler` as safe for OS-domain examples. However D-04 mandates strict leak interpretation for few-shot names: any hit in ANY dataset's component/alias/keyword list triggers `benchmark-leak`. Auditor must check all 5 per-dataset sections, not just Universal Taboo. The `Scheduler` name does NOT appear as a component name in any of the 5 dataset sections. It is not taboo per the current BENCHMARK_TABOO.md — but the "Safe SE Textbook Examples" section lists it, confirming it was reviewed. This is a nuanced case: likely `clean` for the name itself, but the constant still qualifies as a **drop-whole-block candidate** per REQ-V264-06 regardless of verdict.

**Expected cut count:** 3 rows minimum (drop-block + Family A + Family B rewordings, if `benchmark-leak` verdict confirmed after auditor review of all 5 dataset sections).

#### AMBIGUITY_RULES (line 38, 1 LOC)

**Text shape:** Single sentence, two-clause rule (ARCHITECTURAL definition / AMBIGUOUS definition). No examples. No domain terms.

**First-pass categorisation hint:** `clean` candidate.

**BENCHMARK_TABOO hits to expect:** None obvious. "component" appears but is not in Universal Taboo. "mechanism", "role" are generic SE textbook terms.

**Expected cut count:** 0 rows (no cut candidates expected). If verdict is `clean`, no rows emitted.

---

### 1.2 Phase 1 — Doc-Knowledge constants (lines 47–60)

#### DOC_KNOWLEDGE_EXTRACTION_RULES (line 45, 1 LOC)

**Text shape:** Single sentence with four comma-separated clauses (introduced short forms / alternate names / words of multi-word names / reject ordinary-English-dominant terms).

**First-pass categorisation hint:** `clean` candidate.

**BENCHMARK_TABOO hits to expect:** `component` (repeated) — not in Universal Taboo. `ordinary English` — generic. No per-dataset hits expected.

**Expected cut count:** 0 rows likely.

#### DOC_KNOWLEDGE_JUDGE_EXAMPLES (lines 47–53, 7 LOC)

**Text shape:** Two few-shot examples (Example 1 / Example 2). Each has: Candidate / Component / Evidence / Judgment / Rationale. Uses component names `RequestHandler`, `CacheLayer`, `Handler`.

**First-pass categorisation hint:** `benchmark-leak` candidate.

**BENCHMARK_TABOO hits to expect:**

| Token | Section | Hit type |
|-------|---------|----------|
| `Handler` | — | Not in Universal Taboo as-is, but `RequestHandler` and `Handler` overlap conceptually with BBB's `HTML5 Server` handler pattern. Check per-dataset aliases carefully. |
| `CacheLayer` | MediaStore §Keywords | `cache` is Universal Taboo (MediaStore alias `Cache` component). `CacheLayer` head noun `cache` hits Universal Taboo directly. Auto-classify as benchmark-leak per D-02 rule: "Per-dataset-section hits auto-classify as leak without manual review." |
| `RequestHandler` | — | `processor` is in Universal Taboo (BBB Recording Processor alias word). `Handler` does not appear directly in Universal Taboo, but auditor should check BigBlueButton aliases section. |
| `the system` | Teammates/Universal | `common`, `client`, `storage` are universal-taboo Teammates keywords; "the system" is a generic reference — not a taboo hit. Universal-taboo second-pass would dismiss this as generic SE. |

**Expected cut count:** 3–5 rows (drop-block + Family A variants replacing `RequestHandler/CacheLayer/Handler` + Family B concept-only variants).

#### DOC_KNOWLEDGE_JUDGE_RULES (line 55, 1 LOC)

**Text shape:** Four clauses: valid alias definition / invalid (generic vocab) / invalid (whole-system naming) / invalid (different entity) / invalid (tier/platform grouping) / "when uncertain, prefer APPROVE".

**First-pass categorisation hint:** `domain-loaded` candidate for the clause "names an architectural tier or technology platform". The phrase "architectural tier" is potentially over-specified; "grouping" might be a universal noun substitute. Auditor should apply D-01 pragmatic test: "would a universal noun carry the same instruction?"

**BENCHMARK_TABOO hits to expect:**

| Token | Section | Hit type |
|-------|---------|----------|
| `platform` | BBB vicinity | Not in Universal Taboo as a standalone entry. Generic SE. |
| `tier` | — | Not taboo. |
| `entity` | — | Generic. Not taboo. |

**Expected cut count:** 0–1 rows (one possible `domain-loaded` candidate for "architectural tier" clause; no `benchmark-leak` expected).

#### ALIAS_SCOPE_RULES (lines 57–60, 4 LOC)

**Text shape:** Bullet-structured scope classification: "global" shape descriptors (multi-word, hyphenated, CamelCase, all-caps abbreviations, uppercase-initial names) / "local" shape descriptors (single all-lowercase ordinary-English) / exclusion rule for qualified-name fragments (X.Y.Z).

**First-pass categorisation hint:** `clean` candidate. All vocabulary is structural/typographic (CamelCase, all-caps, hyphenated) — no domain jargon, no benchmark component names.

**BENCHMARK_TABOO hits to expect:** None expected. "global" and "local" are not taboo. No component names, no dataset-specific vocabulary.

**Expected cut count:** 0 rows likely. Note: this constant is imported by BOTH `_prompt_doc_knowledge_extract` AND `_prompt_coref` (see cross-section dependency note in Section 4).

---

### 1.3 Phase 2 — Extraction constant (lines 67, 1 LOC)

#### ENTITY_EXTRACTION_RULES (line 67, 1 LOC)

**Text shape:** Single sentence: Include / Exclude / Favor-inclusion. The exclude clause references "code-level path" and "compound identifier".

**First-pass categorisation hint:** `clean` candidate.

**BENCHMARK_TABOO hits to expect:**

| Token | Section | Hit type |
|-------|---------|----------|
| `code-level path` | — | Generic SE. Not taboo. |
| `architectural intent` | — | Generic. Not taboo. |

**Expected cut count:** 0 rows likely.

---

### 1.4 Phase 4 — Validation constants (lines 94–95)

#### VALIDATION_RULES (line 94, 1 LOC)

**Text shape:** Three-clause rule: Approve (architectural participant + counterparts) / Reject (generic) / Reject (different entity or technique sharing name).

**First-pass categorisation hint:** `domain-loaded` candidate for "counterparts". Also, `validation` itself is a Universal Taboo word (Teammates — "input validation"). The word does NOT appear in the prompt text of `VALIDATION_RULES` as a content term — it IS the constant NAME, not a term in the rule text. The rule text uses "approve" / "reject" / "architectural participant" / "counterparts" / "technique" — auditor must grep the rule body (not the constant name) against taboo list.

**BENCHMARK_TABOO hits to expect:**

| Token | Section | Hit type |
|-------|---------|----------|
| `Approve` / `Reject` | — | Generic. Not taboo. |
| `counterparts` | — | Generic SE. Not taboo. |
| `participant` | — | Generic. Not taboo. |

**Expected cut count:** 0–1 rows (possible `domain-loaded` flag on "counterparts" if universal-noun replacement exists; `benchmark-leak` unlikely).

---

### 1.5 Phase 5 — Coref constants (lines 114–124)

#### COREF_RULES (line 114, 1 LOC of assignment, but the string spans multiple lines)

**Actual text inspection:** `COREF_RULES` is assigned at line 114 as a triple-quoted multi-line string. From direct read it occupies lines 114 as the assignment, but the string body continues. Let me clarify: the string `"""For each case, decide whether a pronoun or role-referential noun phrase...` starts at line 114 and the closing `"""` is also on line 114 (it is one long string literal written as a single statement). **LOC = 1** (single line in the file, though the text wraps in display).

**Text shape:** Dense single-paragraph instruction covering: pronoun resolution / role-referential noun phrases / two resolution conditions (name in context / section-established topic) / avoidance condition (two equally plausible antecedents) / alias definition / `antecedent_via_alias` flag instruction.

**First-pass categorisation hint:** `domain-loaded` candidate. Phrases like "role-referential noun phrase" and "section-established topic" are linguistics jargon that may be over-specified. Simpler alternatives exist ("a phrase that refers back to a component"). The "anaphoric reference" framing is load-bearing for the SAD→SAM task per the module docstring — auditor applies D-01 pragmatic test carefully.

**BENCHMARK_TABOO hits to expect:**

| Token | Section | Hit type |
|-------|---------|----------|
| `module` | — | Not in Universal Taboo as a component. Generic SE. |
| `service` | BBB indirect | Not in Universal Taboo directly. Generic SE. |
| `system` | — | Generic. |
| `component` | — | Generic. |
| `the module`, `the service` | — | Universal-taboo second-pass would dismiss as generic SE role-referential phrases. |
| `antecedent_via_alias` | — | A JSON key name, not prompt prose. Auditor applies D-03 and classifies as JSON-schema-adjacent; however it appears in prose context — auditor judgment call. |

**Expected cut count:** 1–2 rows (possible `domain-loaded` flags; `benchmark-leak` unlikely after second pass).

#### ANTECEDENT_ALIAS_RULES (lines 116–124, 9 LOC)

**Text shape:** Instruction for setting `antecedent_via_alias` boolean + two explicit examples using component names `TaskScheduler` and `scheduler`.

**First-pass categorisation hint:** `benchmark-leak` candidate for the examples. `TaskScheduler` is in BENCHMARK_TABOO.md §"Safe SE Textbook Examples" — explicitly listed as safe. But auditor must confirm against all 5 per-dataset sections. The word `scheduler` appears in examples. Per the same §Safe-SE analysis: OS-domain examples are listed as safe. However, `scheduler` lowercased is borderline — it matches no Universal Taboo entry directly.

**BENCHMARK_TABOO hits to expect:**

| Token | Section | Hit type |
|-------|---------|----------|
| `TaskScheduler` | Safe SE Textbook Examples | Explicitly listed as safe — OS-domain. |
| `scheduler` (lowercase) | — | Not in Universal Taboo. Not in any per-dataset component list. Likely clean after second pass. |
| `queues jobs` | — | Generic SE. |

**Expected cut count:** 0–2 rows depending on whether `TaskScheduler`/`scheduler` survive the strict per-dataset check. If clean, 0 rows. If auditor determines examples should be replaced for extra safety, Family A + Family B rewording rows.

---

### 1.6 Additional constants in `prompts_v5.py` (not in REQ-V264-03 enumeration)

These are **Claude's Discretion** items per CONTEXT.md §D-08 and `<deferred>`. Default treatment: fold into validation builder row.

| Constant | Lines | LOC | Text shape | Default treatment |
|----------|-------|-----|-----------|-------------------|
| `P1_FOCUS` | 80–86 | 7 (multi-line tuple) | Single-sentence architectural-participation question with `X.Y.Z` anchor clause. | Fold into `_prompt_validation` audit row |
| `P2_FOCUS` | 88–93 | 6 (multi-line tuple) | Single-sentence referential-specificity question. | Fold into `_prompt_validation` audit row |
| `COREF_VALIDATION_FOCUS` | 106–112 | 7 (multi-line tuple) | Single-sentence coref-validation question naming "pronoun", "it", "they", "the service", "role-referential phrase". | Fold into `_prompt_validation` audit row; `phase_5_coref_validation` tag applies |

**Benchmark-taboo note on `COREF_VALIDATION_FOCUS`:** The phrase "the service" appears — `service` is not in Universal Taboo but is common BBB vocabulary. Universal-taboo second-pass will dismiss it as generic. "role-referential phrase" is linguistics jargon — possible `domain-loaded` flag.

---

## 2. Concrete Inventory: 6 Builder F-String Scaffolds (`s_linker19.py`)

All line ranges verified by direct read of `s_linker19.py`. [VERIFIED: direct file read]

### Builder → Phase-Tag Mapping (from Phase 44 D-03, verbatim)

| Builder | Phase tag(s) in `llm_logs` |
|---------|---------------------------|
| `_prompt_ambiguity` | `phase_1_model` |
| `_prompt_doc_knowledge_extract` | `phase_1_doc_extract` |
| `_prompt_doc_knowledge_judge` | `phase_1_doc_judge` |
| `_prompt_extraction` | `phase_2_framing_c_pass1`, `phase_2_framing_c_pass2` |
| `_prompt_validation` | `phase_4_twopass_p1`, `phase_4_twopass_p2`, `phase_5_coref_validation` |
| `_prompt_coref` | `phase_5_coref` |

**Critical gotcha (from 44-CONTEXT.md §D-03):** `phase_5_coref_validation` uses `_prompt_validation` (not `_prompt_coref`). Its fixtures are in `test_s_linker20_prompt_validation.py`. Confirmed: `s_linker19.py:895` calls `self.llm.set_phase("phase_5_coref_validation")` inside `_validate_coref_links`, which calls `self._run_validation_pass(…, focus=COREF_VALIDATION_FOCUS)`.

---

### 2.1 `_prompt_ambiguity` (lines 264–282, 19 LOC)

**Confirmed line range:** 264 (decorator `@staticmethod`) through 282 (`JSON only:"`). The `return f"""…"""` block starts at 266.

**Imported constants inside this f-string:**
- `AMBIGUITY_FEW_SHOT` (line 270, interpolated as `{AMBIGUITY_FEW_SHOT}`)
- `AMBIGUITY_RULES` (line 280, interpolated as `{AMBIGUITY_RULES}`)

**Prose-vs-JSON split for D-03:**
- **Audit-relevant prose lines:**
  - Line 266: `"Classify these software architecture component names."` — "software architecture component" is potentially `domain-loaded` (universal noun "component names" may suffice).
  - Line 280 (AMBIGUITY_RULES interpolation): handled by the constant's own audit row.
- **Skip (JSON-schema literal) lines:**
  - Lines 274–279: `Return JSON: { "architectural": [...], "ambiguous": [...] }` block — D-03 exclusion.
  - Line 282: `JSON only:` suffix — D-03 exclusion.
- **Inert structural lines:** Lines 268 (`NAMES: {…}`), 272 (`NOW CLASSIFY THE NAMES ABOVE.`) — these are task directives, not instruction prose per se. Auditor judgment: "NOW CLASSIFY THE NAMES ABOVE" is a structural directive; `NAMES:` is a slot label. Neither is benchmark-loaded.

**Phase tag:** `phase_1_model` → `tests/test_s_linker20_prompt_ambiguity.py`

**Snapshot count for risk prior:** 5 snapshots (1 per project). Lowest diversity of all builders.

---

### 2.2 `_prompt_doc_knowledge_extract` (lines 284–302, 19 LOC)

**Confirmed line range:** 284 (`@staticmethod`) through 302 (`JSON only:`).

**Imported constants inside this f-string:**
- `DOC_KNOWLEDGE_EXTRACTION_RULES` (line 290, interpolated)
- `ALIAS_SCOPE_RULES` (line 292, interpolated)

**Prose-vs-JSON split for D-03:**
- **Audit-relevant prose lines:**
  - Line 286: `"Find all alternative names used for these components in the document."` — task directive; "components" is generic. Likely `clean`.
  - Line 290 (DOC_KNOWLEDGE_EXTRACTION_RULES interpolation): handled by constant's own row.
  - Line 292 (ALIAS_SCOPE_RULES interpolation): handled by constant's own row.
- **Skip (JSON-schema literal) lines:**
  - Lines 297–301: `Return JSON: { "abbreviations": [...], "synonyms": [...] }` block — D-03 exclusion.
  - Line 302: `JSON only:` suffix — D-03 exclusion.
- **Inert structural lines:** Lines 288 (`COMPONENTS: {…}`), 294–295 (`DOCUMENT:` / `{chr(10).join(doc_lines)}`).

**Phase tag:** `phase_1_doc_extract` → `tests/test_s_linker20_prompt_doc_extract.py`

**Snapshot count for risk prior:** 5 snapshots (1 per project). Note: 44-VERIFICATION.md documents 3 UserWarnings from teastore/teammates/bigbluebutton for prompt-version-drift — these are non-fatal but indicate some fixture staleness.

---

### 2.3 `_prompt_doc_knowledge_judge` (lines 304–319, 16 LOC)

**Confirmed line range:** 304 (`@staticmethod`) through 319 (`JSON only:`).

**Imported constants inside this f-string:**
- `DOC_KNOWLEDGE_JUDGE_EXAMPLES` (line 313, interpolated)
- `DOC_KNOWLEDGE_JUDGE_RULES` (line 315, interpolated)

**Prose-vs-JSON split for D-03:**
- **Audit-relevant prose lines:**
  - Line 306: `"JUDGE: Review these component name mappings for correctness."` — "JUDGE:" is a role prefix. "component name mappings" is task-specific vocabulary; `domain-loaded` candidate (universal alternative: "name mappings").
  - Lines 313, 315 (constant interpolations): handled by constant rows.
- **Skip (JSON-schema literal) lines:**
  - Lines 317–318: `Return JSON: {"approved": ["term1", "term2"]}` — D-03 exclusion.
  - Line 319: `JSON only:` suffix — D-03 exclusion.
- **Inert structural lines:** Lines 308–311 (`COMPONENTS:`, `PROPOSED MAPPINGS:`, `{chr(10).join(mapping_list)}`).

**Phase tag:** `phase_1_doc_judge` → `tests/test_s_linker20_prompt_doc_judge.py`

**Snapshot count for risk prior:** 5 snapshots (1 per project). Low diversity.

---

### 2.4 `_prompt_extraction` (lines 321–335, 15 LOC)

**Confirmed line range:** 321 (`@staticmethod`) through 335 (`JSON only:`).

**Imported constants inside this f-string:**
- `ENTITY_EXTRACTION_RULES` (line 329, interpolated)

**Prose-vs-JSON split for D-03:**
- **Audit-relevant prose lines:**
  - Line 323: `"Extract ALL references to software architecture components from this document."` — "software architecture components" is `domain-loaded` candidate (same pattern as `_prompt_ambiguity` opener). Universal alternative: "components" or "named elements".
  - Line 329 (ENTITY_EXTRACTION_RULES interpolation): handled by constant's row.
- **Skip (JSON-schema literal) lines:**
  - Lines 333–334: `Return JSON: {"references": [...]}` block — D-03 exclusion.
  - Line 335: `JSON only:` suffix — D-03 exclusion.
- **Inert structural lines:** Lines 325–326 (`COMPONENTS:`, conditional `KNOWN ALIASES:`), lines 331–332 (`DOCUMENT:`, sentence loop).

**Phase tags:** `phase_2_framing_c_pass1`, `phase_2_framing_c_pass2` → `tests/test_s_linker20_prompt_extraction.py`

**Snapshot count for risk prior:** 18 snapshots (5 projects × ~2 phase tags, with some projects having multiple batches). Medium diversity.

---

### 2.5 `_prompt_validation` (lines 337–350, 14 LOC)

**Confirmed line range:** 337 (`@staticmethod`) through 350 (`JSON only:`).

**Imported constants inside this f-string:**
- `VALIDATION_RULES` (line 341, interpolated as `{VALIDATION_RULES}`)
- `P1_FOCUS` or `P2_FOCUS` or `COREF_VALIDATION_FOCUS` injected via `focus` parameter at line 339 as `{focus}` — the builder receives focus as a caller-supplied argument.

**Key architectural note:** The `focus` parameter carries one of three constants at call time:
- `P1_FOCUS` → called from `_run_validation_pass(…, focus=P1_FOCUS)` at Phase 4 twopass P1
- `P2_FOCUS` → called at Phase 4 twopass P2
- `COREF_VALIDATION_FOCUS` → called at Phase 5 coref validation

This means `_prompt_validation` is a single template with three prompt variants in use. Audit of `P1_FOCUS`, `P2_FOCUS`, `COREF_VALIDATION_FOCUS` folds into this builder's section per D-08 default.

**Prose-vs-JSON split for D-03:**
- **Audit-relevant prose lines:**
  - Line 339: `f"Validate component references in a software architecture document. {focus}"` — the opener "Validate component references in a software architecture document." is a `domain-loaded` candidate for the same "software architecture" pattern. The `{focus}` slot carries imported constants (audited in their own rows under this section).
  - Line 341 (VALIDATION_RULES interpolation): handled by constant's row.
- **Skip (JSON-schema literal) lines:**
  - Lines 348–349: `Return JSON: {"validations": [...]}` block — D-03 exclusion.
  - Line 350: `JSON only:` suffix — D-03 exclusion.
- **Inert structural lines:** Lines 343 (`COMPONENTS:`), 345–346 (`CASES:`, case loop).

**Phase tags:** `phase_4_twopass_p1`, `phase_4_twopass_p2`, `phase_5_coref_validation` → `tests/test_s_linker20_prompt_validation.py`

**Snapshot count for risk prior:** 24 snapshots. Medium-high diversity. Note: validation cuts are gated by ALL 3 phase tags — most conservative gating of all builders.

---

### 2.6 `_prompt_coref` (lines 352–378, 27 LOC)

**Confirmed line range:** 352 (`@staticmethod`) through 378 (`return prompt`). This is the longest builder and uses a non-f-string construction: it builds `prompt` as a string and appends to it in a loop.

**Imported constants inside this builder:**
- `COREF_RULES` (line 370, appended as `{COREF_RULES}`)
- `ANTECEDENT_ALIAS_RULES` (line 372, appended as `{ANTECEDENT_ALIAS_RULES}`)

**Prose-vs-JSON split for D-03:**
- **Audit-relevant prose lines:**
  - Line 354: `"Resolve anaphoric references (pronouns and role-referential noun phrases) to architecture components."` — "anaphoric references" and "role-referential noun phrases" are `domain-loaded` candidates (linguistics jargon; universal alternatives: "pronouns and noun phrases that refer back"). "architecture components" is also a `domain-loaded` candidate.
  - Lines 358–364: The inline prose block: `"For each TARGET sentence below, identify any pronoun or role-referential\nnoun phrase that refers back to a component listed above. If a target\nsentence has no anaphoric reference to a listed component, return no\nresolution for it. Be conservative — only include resolutions you are\nCERTAIN about."` — this block contains multiple domain-loaded candidates (`anaphoric reference`, `role-referential noun phrase`, "resolution").
  - Line 361 (exact text per CONTEXT.md §D-03 example): `"Be conservative — only include resolutions you are CERTAIN about."` — functional behavioral instruction; auditor should assess whether "anaphoric reference" in the prior line is load-bearing.
  - Lines 370, 372 (constant interpolations): handled by COREF_RULES and ANTECEDENT_ALIAS_RULES rows.
- **Skip (JSON-schema literal) lines:**
  - Lines 374–376: `Return JSON: {"resolutions": [...]}` block with `N_INTEGER` / `M_INTEGER` placeholders — D-03 exclusion.
  - Line 377: `JSON only:` suffix — D-03 exclusion.
- **Inert structural lines:** Lines 365–369 (case loop: `--- Case {i+1}: S{…} ---`, `CONTEXT:`, `TARGET:` structural labels).

**Phase tag:** `phase_5_coref` → `tests/test_s_linker20_prompt_coref.py`

**Snapshot count for risk prior:** 40 snapshots. Highest diversity of all builders. Most cuts here have the broadest empirical validation surface.

---

## 3. BENCHMARK_TABOO.md Structure and Reading Recipe

**File location:** `BENCHMARK_TABOO.md` (repo root). [VERIFIED: direct file read]

### 3.1 Section Structure

| Section | Components listed | Keywords listed | Aliases listed |
|---------|-------------------|-----------------|----------------|
| MediaStore | 14 components | 8 keywords | 4 aliases |
| TeaStore | 11 components | 7 keywords | 4 aliases |
| Teammates | 8 components | 7 keywords | 4 aliases |
| BigBlueButton | 12 components + several sub-aliases | 13 keywords | many aliases |
| JabRef | 6 components | 6 keywords | 3 aliases |
| Universal Taboo | — | 29 terms (cross-dataset) | — |
| Safe SE Textbook Examples | 7 domains | — | — (safe list) |
| Tailored Code Anti-Patterns | — | — | — (anti-pattern descriptions) |

### 3.2 Universal Taboo Terms (29 entries, complete list)

`logic`, `UI`, `client`, `storage`, `common`, `model`, `database`, `DB`, `cache`, `registry`, `auth`, `server`, `persistence`, `facade`, `recording`, `cascade`, `conversion`, `validation`, `dedicated`, `preferences`, `config`, `internal`, `adapter`, `order`, `processor`, `event`, `socket`, `layer`

### 3.3 Relevance Linkage to Prompt Constants

| Universal Taboo term | Prompt text hit | Context | Second-pass verdict hint |
|----------------------|-----------------|---------|--------------------------|
| `validation` | Appears in constant NAME `VALIDATION_RULES`, not in rule body text | — | Body text does not contain the word "validation" — no hit in audit scope |
| `client` | `COREF_VALIDATION_FOCUS`: "…or similar role-referential phrase" — no; but `HTML5 Client` is a BBB component; does "client" appear? No — not in any of the 9 constants' text. | — | No body-text hit found |
| `server` | `COREF_VALIDATION_FOCUS` line 107: "the service" — "server" does not appear. | — | No hit |
| `storage` | Not in any constant body text | — | No hit |
| `logic` | Not in any constant body text | — | No hit |
| `layer` | Not in any constant body text (COREF_RULES mentions "role-referential" not "layer") | — | No hit |
| `adapter` | Not in any constant body text | — | No hit |
| `processor` | Not in any constant body text | — | No hit |
| `cache` | `DOC_KNOWLEDGE_JUDGE_EXAMPLES` line 51: `"CacheLayer"` — **direct hit** in component name example | MediaStore: `Cache` component, keyword `cache` | `benchmark-leak` (auto-classify per D-02 per-dataset rule) |
| `model` | `AMBIGUITY_FEW_SHOT` line 32: "queues jobs and dispatches them to worker threads" — `model` does not appear. `ENTITY_EXTRACTION_RULES`: does not appear. | — | No hit |
| `registry` | Not in any constant body text | — | No hit |
| `facade` | Not in any constant body text | — | No hit |
| `order` | `COREF_RULES`: "…in order" (as conjunction) — not a component reference context | TeaStore: OrderBasedRecommender keyword | Second-pass would dismiss — conjunction use, not component identification |
| `event` | `COREF_VALIDATION_FOCUS` mentions no "event". COREF_RULES: no. | — | No hit |
| `socket` | Not in any constant body text | — | No hit |
| `conversion` | Not in any constant body text | — | No hit |
| `recording` | Not in any constant body text | — | No hit |
| `auth` | Not in any constant body text | — | No hit |
| `persistence` | Not in any constant body text | — | No hit |
| `config` | Not in any constant body text | — | No hit |
| `internal` | Not in any constant body text | — | No hit |
| `dedicated` | Not in any constant body text | — | No hit |
| `common` | Not in any constant body text | — | No hit |
| `DB` | Not in any constant body text | — | No hit |
| `database` | Not in any constant body text | — | No hit |
| `cascade` | Not in any constant body text | — | No hit |
| `preferences` | Not in any constant body text | — | No hit |
| `UI` | Not in any constant body text | — | No hit |

**Summary:** Of 29 Universal Taboo terms, only `cache` (embedded in `CacheLayer` in `DOC_KNOWLEDGE_JUDGE_EXAMPLES`) produces a confirmed body-text hit. All other universal-taboo terms either do not appear in any constant body text or appear only in generic conjunctive/prepositional use that the second-pass cross-dataset isolation dismisses.

### 3.4 Per-Dataset Section Hits (name-level, not universal)

| Constant | Suspect token | Dataset section | Verdict hint |
|----------|--------------|-----------------|--------------|
| `AMBIGUITY_FEW_SHOT` | `"Scheduler"` | None of the 5 datasets lists `Scheduler` as a component, alias, or keyword. Safe SE Textbook confirms safe. | Likely `clean` for this name — auditor confirms all 5 sections. |
| `DOC_KNOWLEDGE_JUDGE_EXAMPLES` | `"RequestHandler"` | BBB aliases include `HTML5 Server` and `HTML5 Client` — `Handler` is not an explicit BBB alias entry. No per-dataset section lists `RequestHandler`. | Auditor should grep each of 5 per-dataset sections for "Handler". Likely `clean` for `RequestHandler` name itself. |
| `DOC_KNOWLEDGE_JUDGE_EXAMPLES` | `"CacheLayer"` | MediaStore: component `Cache`, keyword `cache`. `CacheLayer` contains `cache`. | `benchmark-leak` auto-classify per D-02. |
| `DOC_KNOWLEDGE_JUDGE_EXAMPLES` | `"Handler"` | No per-dataset component named "Handler" exactly. | Likely `clean` after full check — but auditor must complete the check. |
| `ANTECEDENT_ALIAS_RULES` | `"TaskScheduler"` | Confirmed safe in §Safe SE Textbook Examples (OS-domain: Scheduler, Dispatcher). | Likely `clean`. |

### 3.5 Mechanical Grep Pattern

For the audit's mechanical BENCHMARK_TABOO check, use:

```bash
# For each constant text body, check universal taboo single-token terms (case-insensitive):
grep -iw 'logic\|ui\|client\|storage\|common\|model\|database\|db\|cache\|registry\|auth\|server\|persistence\|facade\|recording\|cascade\|conversion\|validation\|dedicated\|preferences\|config\|internal\|adapter\|order\|processor\|event\|socket\|layer' <constant-text-file>

# For per-dataset multi-token phrases / component names:
grep -i 'RequestHandler\|CacheLayer\|TaskScheduler\|Scheduler\|Handler\|Recommender\|Persistence\|BigBlueButton\|HTML5\|FreeSWITCH\|JabRef\|GUI\|CLI' <constant-text-file>
```

**Case-sensitivity rule:** Universal Taboo terms should be matched case-insensitively (a `cache` hit in any casing is taboo). Per-dataset component names should also be matched case-insensitively (BENCHMARK_TABOO.md §"Tailored Code Anti-Patterns" highlights case-mismatch blind spots).

**Word-boundary vs substring:** Use `\b` word-boundary or `-w` flag for single-token universal taboo terms to avoid matching `reconfiguration` → `config` inside a larger word. Multi-word phrases require phrase-level grep: e.g., `grep -i "recording service"`.

### 3.6 Known False-Positive Shapes

Auditors will encounter these generic SE nouns that look like Universal Taboo hits but survive the second-pass cross-dataset isolation check:

| Token | False-positive shape | Why it survives |
|-------|---------------------|-----------------|
| `order` | "…in order to…" conjunction | Conjunctive use; does not identify TeaStore's OrderBasedRecommender in this context. |
| `server` | "the HTML5 Server" vs "the server" as generic | Generic "the server" in coref context is a role-referential phrase — does not identify BBB's HTML5 Server specifically. |
| `client` | "the client" as generic role | Does not identify Teammates' Client component specifically. |
| `logic` | "business logic" as generic SE term | Does not identify JabRef's logic component or Teammates' Logic component in most prompt contexts. |
| `model` | "the model" as generic ML/architecture term | Does not identify JabRef's model component in general discussion. |
| `layer` | "FreeSWITCH Event Socket Layer" alias word vs "layer" as architectural vocabulary | Generic "layer" (as in "layer of abstraction") does not identify BBB's FSESL alias. |
| `adapter` | "MediaStore UserDBAdapter" word vs "adapter" as a design pattern term | Generic adapter pattern term does not identify UserDBAdapter. |
| `processor` | "BBB Recording Processor" alias word vs "processor" as CPU term | CPU/component-level use does not identify Recording Processor alias. |

---

## 4. Phase 44 Harness Gating Details

### 4.1 Builder → Phase-Tag Mapping (verbatim from 44-CONTEXT.md §D-03)

| Builder | Phase tag(s) in `llm_logs` |
|---------|---------------------------|
| `_prompt_ambiguity` | `phase_1_model` |
| `_prompt_doc_knowledge_extract` | `phase_1_doc_extract` |
| `_prompt_doc_knowledge_judge` | `phase_1_doc_judge` |
| `_prompt_extraction` | `phase_2_framing_c_pass1`, `phase_2_framing_c_pass2` |
| `_prompt_validation` | `phase_4_twopass_p1`, `phase_4_twopass_p2`, `phase_5_coref_validation` |
| `_prompt_coref` | `phase_5_coref` |

Source: 44-CONTEXT.md §D-03; verified against `s_linker19.py` phase-tag set calls at lines 562, 574, (doc_judge is called from within `_learn_document_knowledge` which uses `phase_1_doc_extract` then `phase_1_doc_judge` via the code at lines 603–607), 646, 793–795, 836, 895.

**Note on doc_judge:** `self.llm.set_phase("phase_1_doc_judge")` is set by caller code (inside `_learn_document_knowledge`) before the judge call — not by a set_phase inside `_prompt_doc_knowledge_judge` itself. The phase tag is `phase_1_doc_judge` per the D-03 table and this is correct.

### 4.2 Per-Builder Snapshot Counts (from 44-VERIFICATION.md)

| Test module | Snapshot count | `.ambr` file size | Risk-tier implication |
|------------|---------------|------------------|-----------------------|
| `test_s_linker20_prompt_ambiguity.py` | 5 snapshots | ~1.8 KB | Lowest diversity. A wording change has few chances to fail → feels safer but fewer fixtures catch variance. |
| `test_s_linker20_prompt_doc_extract.py` | 5 snapshots | ~9.2 KB | Low diversity. Note: 3 of 5 have known prompt-version-drift UserWarnings (non-fatal). |
| `test_s_linker20_prompt_doc_judge.py` | 5 snapshots | ~2.2 KB | Low diversity. |
| `test_s_linker20_prompt_extraction.py` | 18 snapshots | ~48.7 KB | Medium diversity. 2 phase tags × up to several batches per project. |
| `test_s_linker20_prompt_validation.py` | 24 snapshots | ~34.1 KB | Medium-high diversity. 3 phase tags (including coref_validation). Most conservative gating. |
| `test_s_linker20_prompt_coref.py` | 40 snapshots | ~41.0 KB | Highest diversity. Broadest empirical pressure. |

Total: 97 snapshots. 149 tests pass (the difference: some tests assert multiple things beyond snapshot equality — e.g., byte-equality of the built prompt itself). [VERIFIED: 44-VERIFICATION.md]

### 4.3 Test Module Paths (verified on disk)

| Module | Path | Exists? |
|--------|------|---------|
| `test_s_linker20_prompt_ambiguity.py` | `tests/test_s_linker20_prompt_ambiguity.py` | Yes (63 lines) |
| `test_s_linker20_prompt_doc_extract.py` | `tests/test_s_linker20_prompt_doc_extract.py` | Yes (76 lines) |
| `test_s_linker20_prompt_doc_judge.py` | `tests/test_s_linker20_prompt_doc_judge.py` | Yes (51 lines) |
| `test_s_linker20_prompt_extraction.py` | `tests/test_s_linker20_prompt_extraction.py` | Yes (92 lines) |
| `test_s_linker20_prompt_validation.py` | `tests/test_s_linker20_prompt_validation.py` | Yes (101 lines) |
| `test_s_linker20_prompt_coref.py` | `tests/test_s_linker20_prompt_coref.py` | Yes (84 lines per disk, 85 per VERIFICATION.md) |

All 6 exist on disk. [VERIFIED: `ls` command]

**Snapshot files (also verified):** All 6 `.ambr` files exist under `tests/__snapshots__/`. [VERIFIED: direct `ls` + `wc -l`]

---

## 5. Cut-Row Schema Enumeration

### 5.1 Section Tags (from D-08)

| Section | Pipeline phase | Constants covered | Builder covered |
|---------|----------------|-------------------|-----------------|
| `AMB` | Phase 1 — Ambiguity | `AMBIGUITY_FEW_SHOT`, `AMBIGUITY_RULES` | `_prompt_ambiguity` |
| `DKX` | Phase 1 — Doc-Knowledge Extract | `DOC_KNOWLEDGE_EXTRACTION_RULES`, `ALIAS_SCOPE_RULES` | `_prompt_doc_knowledge_extract` |
| `DKJ` | Phase 1 — Doc-Knowledge Judge | `DOC_KNOWLEDGE_JUDGE_EXAMPLES`, `DOC_KNOWLEDGE_JUDGE_RULES` | `_prompt_doc_knowledge_judge` |
| `EXT` | Phase 2 — Extraction | `ENTITY_EXTRACTION_RULES` | `_prompt_extraction` |
| `VAL` | Phase 4 — Validation | `VALIDATION_RULES` (+ `P1_FOCUS`/`P2_FOCUS`/`COREF_VALIDATION_FOCUS` if bonus rows) | `_prompt_validation` |
| `COR` | Phase 5 — Coref | `COREF_RULES`, `ANTECEDENT_ALIAS_RULES` | `_prompt_coref` |

### 5.2 Expected Cut-Row Order per Section (from D-04/D-06)

For constants with a few-shot block (`AMBIGUITY_FEW_SHOT` under AMB, `DOC_KNOWLEDGE_JUDGE_EXAMPLES` under DKJ):
1. Drop-whole-block candidate first (per REQ-V264-06 and D-04)
2. Family A rewording variants (synthetic-neutral name swap, multiple variants)
3. Family B rewording variants (concept-only/name-stripped, multiple variants)

For non-few-shot constants with `benchmark-leak` verdict:
1. Line-level cut candidate(s) (specific cuttable span)
2. Family A rewording variants
3. Family B rewording variants

For `domain-loaded` items:
1. Single row flagging the domain-loaded span, no `after` rewordings (D-05)

For `clean` items:
- No rows emitted

### 5.3 Realistic Cut-Row Count Estimate per Section

| Section | Verdict expectations | Estimated cut rows | Sizing rationale |
|---------|---------------------|-------------------|-----------------|
| AMB | `AMBIGUITY_FEW_SHOT`: `benchmark-leak` candidate (block + 2+ rewording families); `AMBIGUITY_RULES`: `clean`; `_prompt_ambiguity` opener: `domain-loaded` candidate | **3–6 rows** | Low: "Scheduler" may survive taboo check; High: if strict interpretation, 3 drop-block + 2–3 Family A + 2–3 Family B |
| DKX | `DOC_KNOWLEDGE_EXTRACTION_RULES`: `clean`; `ALIAS_SCOPE_RULES`: `clean`; `_prompt_doc_knowledge_extract` opener: `clean` | **0–1 rows** | Low — mostly clean constants |
| DKJ | `DOC_KNOWLEDGE_JUDGE_EXAMPLES`: `benchmark-leak` confirmed (`CacheLayer`→`cache` hit); `DOC_KNOWLEDGE_JUDGE_RULES`: `domain-loaded` candidate; `_prompt_doc_knowledge_judge` opener: `domain-loaded` candidate | **4–8 rows** | Drop-block + 2+ Family A variants (replace CacheLayer/RequestHandler/Handler) + 2+ Family B variants + 1–2 domain-loaded flags |
| EXT | `ENTITY_EXTRACTION_RULES`: `clean`; `_prompt_extraction` opener: `domain-loaded` candidate ("software architecture components") | **0–2 rows** | Likely 1 domain-loaded flag on opener |
| VAL | `VALIDATION_RULES`: `domain-loaded` candidate; opener "in a software architecture document": `domain-loaded` candidate; `P1_FOCUS`/`P2_FOCUS`/`COREF_VALIDATION_FOCUS` (if bonus rows): possible `domain-loaded` flags | **1–4 rows** | Mostly domain-loaded flags; `benchmark-leak` unlikely |
| COR | `COREF_RULES`: `domain-loaded` candidates (linguistics jargon, 2–3 spans); `ANTECEDENT_ALIAS_RULES`: possible `benchmark-leak` or `clean` for `TaskScheduler`; `_prompt_coref` opener + inline prose (lines 354–364): `domain-loaded` candidates | **3–8 rows** | Highest LOC, most audit-relevant prose; linguistics jargon concentration here |

**Total estimated cut rows across all sections:** 11–29 rows. Median estimate: ~18 rows.

---

## 6. Sequencing / Parallelization Hints

### 6.1 Cross-Section Dependencies

**`ALIAS_SCOPE_RULES` dependency:** This constant is imported by BOTH `_prompt_doc_knowledge_extract` (Section DKX) AND indirectly referenced by the coref pipeline (though not directly interpolated in `_prompt_coref`). The audit document must produce a single canonical audit row for `ALIAS_SCOPE_RULES` under Section DKX (where the constant lives per D-08's colocate-with-builder rule), not duplicate it in COR. The COR section's auditor should note "ALIAS_SCOPE_RULES is audited under DKX" to avoid duplicated cut_ids.

**Confirmation:** Reading `prompts_v5.py` confirms `ALIAS_SCOPE_RULES` is used by `_prompt_doc_knowledge_extract` (line 292 in the builder). It is NOT directly imported into `_prompt_coref`. So no structural cross-section dependency for the audit — DKX owns it cleanly.

### 6.2 Section Parallelization

All 5 sections (AMB, DKX, DKJ, EXT, VAL, COR mapped to 5 D-08 document sections) are parallelizable after a Wave-0 framing task that writes the document header + verdict rubric + top-level summary table. Each section sub-task is self-contained: it reads its own constants from `prompts_v5.py`, reads its own builder from `s_linker19.py`, greps BENCHMARK_TABOO, and produces the header table + cut-candidate table for its section.

**Recommended wave structure:**

- **Wave 0 (1 task):** Write document skeleton — title block, scope note, `## Verdict Rubric` recap (if planner opts for inline recap per Claude's Discretion), and `## Summary Table` (one row per item, column: item | verdict | LOC). This task MUST complete before Wave 1 because Wave 1 tasks write into specific sections; the planner needs the document stub to exist.

- **Wave 1 (5 parallel tasks):** One task per D-08 section (AMB, DKX/DKJ combined or separate, EXT, VAL, COR). Each task:
  1. Reads its constants' text from `prompts_v5.py` (read-only).
  2. Reads its builder's lines from `s_linker19.py` (read-only).
  3. Runs BENCHMARK_TABOO grep against constant text bodies.
  4. Assigns verdicts.
  5. Writes the header table + cut-candidate table for the section.

**DKX and DKJ sharing:** Sections 2 and 3 (DKX and DKJ) share the Phase 1 pipeline phase. They can be assigned to the same Wave-1 task or separated. Separating is safer for parallelism: DKX owns `DOC_KNOWLEDGE_EXTRACTION_RULES` + `ALIAS_SCOPE_RULES` + `_prompt_doc_knowledge_extract`; DKJ owns `DOC_KNOWLEDGE_JUDGE_EXAMPLES` + `DOC_KNOWLEDGE_JUDGE_RULES` + `_prompt_doc_knowledge_judge`. The DKJ task has the most complex work (few-shot block + confirmed `CacheLayer` hit), so separating it is worth the overhead.

**Possible 6-task Wave-1 split:** AMB | DKX | DKJ | EXT | VAL | COR — one task per D-08 section.

### 6.3 BENCHMARK_TABOO Grep Step

The grep step runs **once per section, inline**, not as a global pre-pass. Each Wave-1 task greps the text bodies of its own constants. This is preferred because:
- The grep result must be immediately contextualized by the auditor (second-pass cross-dataset isolation) — it cannot be productively separated from the verdict-assignment step.
- The grep corpus per section is small (1–7 LOC each) — no performance argument for batching.
- A global pre-pass would produce a flat hit list that the auditor then re-distributes to sections anyway.

### 6.4 Risk-Tier Scoring Placement

**Trade-off:** Inline per cut row (adds reviewer-judgment inline) vs. a single dedicated post-verdicts pass.

**Planner's call.** Research finding: the risk-tier inputs are per-builder snapshot counts (Section 4.2 above), which are static — no per-cut computation needed. Inline is simpler: the auditor assigns risk at the moment of writing the cut row, using the snapshot count for the builder as a prior. A separate post-pass would only be warranted if risk-tier assignments are uncertain and the auditor wants to calibrate after seeing all cuts. The snapshot count table (Section 4.2) removes that uncertainty. **Recommendation: inline, one risk column per cut row.**

---

## 7. Anti-Patterns and Landmines

### 7.1 JSON-Schema Literal Exclusion (D-03)

Do NOT audit or propose cuts to the following patterns — they are byte-equality-critical for the parser:

- Any `Return JSON: {…}` block in any builder.
- Any `JSON only:` suffix line.
- The `N_INTEGER` / `M_INTEGER` placeholder tokens inside the JSON schema in `_prompt_coref` (lines 374–376).
- The `N_INTEGER` placeholder in `_prompt_extraction` (line 334).

If a schema text looks load-bearing or suspicious (e.g., redundant `N_INTEGER` in extraction + coref), the audit may flag it in a sidebar note under "Non-audit observations" WITHOUT proposing a cut — for Phase 46 visibility only (per CONTEXT.md §`<specifics>`).

### 7.2 No Inheritance Proposals (CONCERNS.md)

The audit doc is a planning artefact, not code, so this is primarily a concern for the planner reading it. Do NOT propose cuts that would result in:
- A shared prompt module that both `s_linker19` and `s_linker20` import.
- A base class or mixin that provides shared prompt logic.

All rewording variants in Family A and Family B should be formulated as self-contained text replacements that will be inlined into the standalone `s_linker20.py`. No "extract to shared module" suggestions.

### 7.3 No Rewordings for `domain-loaded` Items (D-05)

Phase 45 produces only the FLAG (with the domain-loaded span identified and a plausible universal-noun alternative mentioned informally). It does NOT produce formal `after` column content for `domain-loaded` rows. The `after` cell for a `domain-loaded` row should say `[Phase 46 empirical loop]` to avoid wasted effort.

### 7.4 Behavioral-Load-Bearing Phrases — Do Not Flag as Cuts

The module docstring at `prompts_v5.py:1–22` explicitly identifies load-bearing clauses:
- `"qualified-name identifier (e.g. a package- or member-access path X.Y.Z)"` — empirically validated to catch 2/3 code-path FPs on gpt-5.4 (experiment_dotted_path_rename.py). **Do not propose cutting this clause in `P1_FOCUS`.** If `P1_FOCUS` gets bonus audit rows, this clause must carry a `risk: high` annotation with a note citing the docstring justification.
- The "Be conservative — only include resolutions you are CERTAIN about" instruction at `_prompt_coref:361` — behavioral; auditor may flag as potentially cuttable but risk is high (no evidence this can be removed safely).
- The asymmetric single-pass design of `_prompt_validation` when called with `COREF_VALIDATION_FOCUS` — the docstring (lines 101–105) explains the empirical justification: "entity twopass leaks ~4 FPs on bigbluebutton coref". Any audit note on this architecture must NOT propose symmetrizing it; that is a code change, not a prompt cut.

### 7.5 GATE-01 Byte-Equal

Zero edits to: `src/llm_sad_sam/linkers/experimental/s_linker19.py`, `src/llm_sad_sam/linkers/experimental/prompts_v5.py`, `src/llm_sad_sam/linkers/experimental/s_linker13_min.py`. The audit document itself is a new file under `.planning/`; creating it does not affect GATE-01. Verified: GATE-01 is currently PASS per STATE.md and 44-VERIFICATION.md.

---

## 8. Claude's Discretion Items — Choices for the Planner

These are explicitly deferred to the planner per CONTEXT.md §`<decisions>` "Claude's Discretion":

| # | Item | Options | Research observation |
|---|------|---------|---------------------|
| CD-1 | Final on-disk location | (A) `.planning/milestones/v2.6.4-*/s_linker20-PROMPT-AUDIT.md` (directory does not yet exist — must be created) OR (B) `.planning/phases/45-audit/s_linker20-PROMPT-AUDIT.md` (directory already exists) | Prior milestones used `vX.Y.Z-phases/` sub-tree for per-phase artefacts (e.g., `v2.6.3-phases/`). Phase-specific artefacts that persist to Phase 49 are referenced from `ROADMAP.md` milestone entry. Both locations work; option A matches prior convention but requires directory creation. |
| CD-2 | Per-cut detail blocks | (A) Under each section (inline) OR (B) Appendix at end | Short rewordings fit inline. Long Family B concept-only rewrites benefit from an appendix. Hybrid (stub in table, full text in appendix keyed by cut_id) is workable. |
| CD-3 | Top-of-doc summary table | Include or omit | High value for Phase 46 planner orientation. Recommend include — adds ~15 rows to the doc header. |
| CD-4 | Reviewer-judgment notes | Inline per cut OR closing rationale section | Inline is faster to navigate during Phase 46 execution. Recommend inline. |
| CD-5 | Risk tier justification column | Bare tag OR with short justification | Short justification (1 phrase, e.g., "5 snapshots — low diversity") adds material value. Recommend include. |
| CD-6 | `P1_FOCUS`/`P2_FOCUS`/`COREF_VALIDATION_FOCUS` treatment | Bonus rows under VAL OR fold into builder row | Default per CONTEXT.md deferred: fold into builder row. Research note: all three are small (6–7 LOC each) and share the validation builder's gating — fold-in is simpler for Phase 46 execution. `COREF_VALIDATION_FOCUS` contains "role-referential phrase" (domain-loaded candidate) and "the service" (possible generic BBB proximity hit) — both need a flag even if folded in. |
| CD-7 | Verdict-rubric recap | 2-sentence inline OR link to CONTEXT.md | Inline recap makes the audit self-contained for Phase 46 executor. Recommend 2-sentence inline. |

---

## 9. Environment and Availability

No external dependencies beyond the existing repo. All sources are static files on disk.

| Dependency | Available | Notes |
|------------|-----------|-------|
| `prompts_v5.py` | Yes (read) | 124 LOC, read-only |
| `s_linker19.py` | Yes (read) | Builders at lines 264–378, read-only |
| `BENCHMARK_TABOO.md` | Yes (read) | 102 LOC, repo root |
| Phase 44 test modules | Yes (all 6 verified) | Read-only; no execution during Phase 45 |
| Phase 44 snapshot `.ambr` files | Yes (all 6 verified) | Read-only reference for risk-tier sizing |
| `.planning/milestones/v2.6.4-*/` directory | Does not exist yet | Must be created if CD-1 option A chosen |
| `.planning/phases/45-audit/` directory | Exists | Available immediately if CD-1 option B chosen |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | "Scheduler" is NOT in any of the 5 per-dataset component/alias/keyword sections of BENCHMARK_TABOO.md. | Section 1.1, 3.4 | If wrong, `AMBIGUITY_FEW_SHOT` escalates from tentatively `clean` name to confirmed `benchmark-leak` → 3+ additional cut rows in AMB. |
| A2 | "RequestHandler" and "Handler" are NOT in any of the 5 per-dataset component/alias/keyword sections. | Section 1.2, 3.4 | If wrong, DKJ cut-row count increases by 2–4 rows (both names need Family A + Family B treatments). |
| A3 | `ALIAS_SCOPE_RULES` is NOT imported by `_prompt_coref` directly (only by `_prompt_doc_knowledge_extract`). | Section 6.1 | If wrong, auditor must also add a note in COR section; single canonical audit row still lives in DKX. Low impact. |
| A4 | Line 38 in `prompts_v5.py` is the complete text of `AMBIGUITY_RULES` (1 LOC). | Section 1.1 | Direct read confirms. Risk = 0. [VERIFIED] |
| A5 | `phase_5_coref_validation` reuses `_prompt_validation` (not `_prompt_coref`). | Sections 2.5, 4.1 | Direct read confirms at `s_linker19.py:893–895`. Risk = 0. [VERIFIED] |

If this table is populated only with A1–A3 as unverified claims: these three can be resolved by a single grep against BENCHMARK_TABOO.md as the first action in the AMB and DKJ Wave-1 tasks.

---

## Open Questions

1. **"Scheduler" verdict in `AMBIGUITY_FEW_SHOT`.**
   - What we know: Confirmed safe in BENCHMARK_TABOO.md §"Safe SE Textbook Examples". Not in any per-dataset keyword list (from direct read of all 5 sections).
   - What's unclear: D-04 mandates strict few-shot leak interpretation, and the constant is a drop-whole-block candidate regardless. Does a "safe" name still count as `benchmark-leak` just because it appears in a few-shot example? D-04 says "any name with a hit" — if there's no hit, the name is not `benchmark-leak`. But the block is still a drop-whole-block candidate per REQ-V264-06.
   - Recommendation: Auditor should confirm the no-hit verdict for "Scheduler" against all 5 per-dataset sections and then assign verdict to the few-shot block as a whole. If no benchmark-leak tokens found, the block may be `clean` with a drop-whole-block candidate row (triggered by REQ-V264-06, not benchmark-leak verdict). The `trigger` column should then read "drop-block (REQ-V264-06, not benchmark-leak)".

2. **`COREF_VALIDATION_FOCUS` inclusion in VAL section.**
   - What we know: Planner default is fold into `_prompt_validation` builder row. The constant contains "role-referential phrase" (domain-loaded candidate) and "the service" (generic BBB proximity).
   - What's unclear: Should these domain-loaded flags appear as separate cut_id rows (requiring `VAL-NN` cut_ids) or as inline notes within the builder's row?
   - Recommendation: If the planner opts to fold it in, the builder row's cut-candidate table should include rows `CUT-VAL-NN` for the COREF_VALIDATION_FOCUS spans, clearly labeled with the constant name in the `file:lines` column. The constant is small enough (7 LOC) that this is workable within the builder row.

3. **`ANTECEDENT_ALIAS_RULES` examples (`TaskScheduler` / `scheduler`) — strict vs. permissive interpretation.**
   - What we know: Both names are in §"Safe SE Textbook Examples". Neither appears in any per-dataset component/alias/keyword list (from direct read).
   - What's unclear: D-04 applies strict leak interpretation to few-shot names. `ANTECEDENT_ALIAS_RULES` contains example component names, but it is not strictly a "few-shot" block in the AMBIGUITY_FEW_SHOT / DOC_KNOWLEDGE_JUDGE_EXAMPLES sense — it is a rule with examples.
   - Recommendation: Auditor applies D-04's strict interpretation to `ANTECEDENT_ALIAS_RULES` examples as well (they function like few-shot illustrations). If all names survive the per-dataset check, verdict is `clean` and no cut rows emitted.

---

## Sources

### Primary (HIGH confidence — direct file reads)

- `src/llm_sad_sam/linkers/experimental/prompts_v5.py` — read in full (124 LOC); all line numbers and text shapes verified.
- `src/llm_sad_sam/linkers/experimental/s_linker19.py` — builders at lines 264–378 and run methods at lines 561, 573, 644–646, 792–795, 835–836, 895 read directly.
- `BENCHMARK_TABOO.md` — read in full (102 LOC); all 6 sections and Universal Taboo list verified.
- `.planning/phases/44-harness/44-CONTEXT.md` — D-03 builder→phase-tag table read verbatim.
- `.planning/phases/44-harness/44-VERIFICATION.md` — snapshot counts (97 total), test results (149 passed) verified.
- `.planning/phases/45-audit/45-CONTEXT.md` — all decisions D-01 through D-08 read.
- `.planning/REQUIREMENTS.md` — REQ-V264-03, REQ-V264-04 text read.
- `.planning/ROADMAP.md` — Phase 45 success criteria (4 items) read.
- `tests/harness/fixtures/MANIFEST.json` — 5-project pairing verified.
- `ls tests/test_s_linker20_prompt_*.py` — all 6 modules confirmed on disk.
- `ls tests/__snapshots__/*.ambr` — all 6 snapshot files confirmed, line counts captured.

### Secondary (MEDIUM confidence)

- `.planning/STATE.md` — current milestone v2.6.4, GATE-01 PASS status.
- `.planning/PROJECT.md` §"Current Milestone: v2.6.4" — standing gates GATE-01/06/08 and floor 91.3%.

---

## Metadata

**Confidence breakdown:**
- Prompt constant inventory: HIGH — all text read directly from frozen source files.
- BENCHMARK_TABOO analysis: HIGH — BENCHMARK_TABOO.md read directly; specific hit claims cited to file sections.
- Harness gating details: HIGH — Phase 44 context and verification read directly; snapshot counts from 44-VERIFICATION.md.
- Cut-row estimates: MEDIUM — count estimates are reviewer judgment from LOC + first-pass categorization, not measured.
- Verdict hints: MEDIUM — first-pass only; marked explicitly as hints. Final verdicts are Phase 45 work product.

**Research date:** 2026-06-07
**Valid until:** Indefinite (all sources are frozen read-only artefacts for this milestone; BENCHMARK_TABOO.md is not expected to change during v2.6.4).
