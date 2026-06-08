---
phase: 45-audit
artifact_type: prompt-audit
milestone: v2.6.4
produced_by: 'Phase 45 (AUDIT)'
consumed_by: 'Phase 46 (MINIMIZE) — references cut_id rows in s_linker20-MINIMIZE-LOG.md'
gate_status: 'GATE-01 byte-equal: verified at phase close (45-08-PLAN)'
---

# s_linker20 Prompt Audit (v2.6.4 Phase 45)

## Verdict Rubric Recap

A prompt fragment is `domain-loaded` only when its domain vocabulary is over-specified — i.e. a universal noun ("entity", "name", "phrase", "pronoun") would carry the same instruction to the LLM, with load-bearing SAD/SAM terms staying `clean`. The `benchmark-leak` verdict is assigned by mechanical BENCHMARK_TABOO grep over every token, followed by a second-pass cross-dataset isolation check (v2.1 GATE-06 methodology): per-dataset section hits auto-classify as leaks while Universal Taboo hits require reviewer confirmation that the token identifies a specific dataset's component rather than functioning as a generic SE noun.

See `.planning/phases/45-audit/45-CONTEXT.md` §<decisions> for the locked D-01..D-08 schema.

## Scope

- The 9 imported PROMPT CONSTANTS from `prompts_v5.py` and the 6 in-class f-string builders in `s_linker19.py` (prose only per D-03).
- Verdict assignment and cut-candidate rows per D-08, with per-cut risk tiers (low/med/high) per D-07.
- Rewordings for `benchmark-leak` findings only (D-05); two style families per finding (Family A synthetic-neutral, Family B concept-only) per D-06.
- Back-reference notes for dual-use constants (e.g. `ALIAS_SCOPE_RULES` audited under DKX; cross-referenced from COR per 45-RESEARCH.md §6.1).

## Out of Scope

- Any code edits to `s_linker19.py`, `prompts_v5.py`, or `s_linker13_min.py` — GATE-01 byte-equality must hold at phase close (verified by 45-08-PLAN).
- JSON-schema literals inside f-string builders (the response-shape declaration and the `JSON only:` suffix) — excluded per D-03; byte-equality-critical for the parser path.
- Rewordings for `domain-loaded` items — deferred to Phase 46's empirical lexical-neutralization loop (REQ-V264-07).
- Harness execution against trial cuts — that is Phase 46 work (REQ-V264-05/06/07).

## Gating Reference (Phase 44 §D-03, verbatim)

| Builder | Phase tag(s) | Test module | Snapshot count |
|---|---|---|---|
| `_prompt_ambiguity` | `phase_1_model` | `tests/test_s_linker20_prompt_ambiguity.py` | 5 |
| `_prompt_doc_knowledge_extract` | `phase_1_doc_extract` | `tests/test_s_linker20_prompt_doc_extract.py` | 5 |
| `_prompt_doc_knowledge_judge` | `phase_1_doc_judge` | `tests/test_s_linker20_prompt_doc_judge.py` | 5 |
| `_prompt_extraction` | `phase_2_framing_c_pass1`, `phase_2_framing_c_pass2` | `tests/test_s_linker20_prompt_extraction.py` | 18 |
| `_prompt_validation` | `phase_4_twopass_p1`, `phase_4_twopass_p2`, `phase_5_coref_validation` | `tests/test_s_linker20_prompt_validation.py` | 24 |
| `_prompt_coref` | `phase_5_coref` | `tests/test_s_linker20_prompt_coref.py` | 40 |

`phase_5_coref_validation` reuses `_prompt_validation` (NOT `_prompt_coref`) per Phase 44 §D-03; its cuts are gated by `tests/test_s_linker20_prompt_validation.py`.

## Cut ID Scheme

- `CUT-AMB-NN` — Phase 1 Ambiguity cuts.
- `CUT-DKX-NN` — Phase 1 Doc-Knowledge Extract cuts.
- `CUT-DKJ-NN` — Phase 1 Doc-Knowledge Judge cuts.
- `CUT-EXT-NN` — Phase 2 Extraction cuts.
- `CUT-VAL-NN` — Phase 4 Validation cuts (includes `phase_5_coref_validation` reuse path).
- `CUT-COR-NN` — Phase 5 Coref cuts.

Numbering: zero-padded to 2 digits, restarts at `01` per section. Row schema (verbatim from D-08):

```
| cut_id | file:lines | trigger | before | after | risk | gated_by |
```

Cells exceeding 80 chars truncate with `…`; full Family B rewordings live in per-cut detail blocks below each section's cut table, introduced as `> **{cut_id} detail:**` immediately under the table for any cut whose `after` exceeds inline space.

### Drop-block convention (REQ-V264-06)

Per REQ-V264-06, `AMBIGUITY_FEW_SHOT` and `DOC_KNOWLEDGE_JUDGE_EXAMPLES` each receive a drop-whole-block row as their FIRST cut row regardless of verdict — Phase 46 tests full removal before partial replacement, and the audit records the drop candidate up front so the cut_id numbering downstream is stable.

## Verdict Summary

| Item | Type | LOC | Verdict | Cut Rows | Audited By |
|---|---|---|---|---|---|
| `AMBIGUITY_FEW_SHOT` | constant | 7 | clean | 1 | 45-02 (AMB) |
| `AMBIGUITY_RULES` | constant | 1 | clean | 0 | 45-02 (AMB) |
| `_prompt_ambiguity` | builder | 19 | domain-loaded | 1 | 45-02 (AMB) |
| `DOC_KNOWLEDGE_EXTRACTION_RULES` | constant | 1 | clean | 0 | 45-03 (DKX) |
| `ALIAS_SCOPE_RULES` | constant | 4 | clean | 0 | 45-03 (DKX) |
| `_prompt_doc_knowledge_extract` | builder | 19 | clean | 0 | 45-03 (DKX) |
| `DOC_KNOWLEDGE_JUDGE_EXAMPLES` | constant | 7 | benchmark-leak | 6 | 45-04 (DKJ) |
| `DOC_KNOWLEDGE_JUDGE_RULES` | constant | 1 | domain-loaded | 1 | 45-04 (DKJ) |
| `_prompt_doc_knowledge_judge` | builder | 16 | clean | 0 | 45-04 (DKJ) |
| `ENTITY_EXTRACTION_RULES` | constant | 1 | clean | 0 | 45-05 (EXT) |
| `_prompt_extraction` | builder | 15 | domain-loaded | 1 | 45-05 (EXT) |
| `VALIDATION_RULES` | constant | 1 | TBD | TBD | 45-06 (VAL) |
| `_prompt_validation` | builder | 14 | TBD | TBD | 45-06 (VAL) |
| `COREF_RULES` | constant | 1 | TBD | TBD | 45-07 (COR) |
| `_prompt_coref` | builder | 27 | TBD | TBD | 45-07 (COR) |

LOC values are inspection priors copied from 45-RESEARCH.md §1 and §2; Wave-1 plans confirm or correct each via their own read of the frozen source.

## Phase 1 — Ambiguity (AMB)

<!-- SECTION:AMB:START -->

### Items

| Item | Type | Verdict | LOC | Notes |
|---|---|---|---|---|
| `AMBIGUITY_FEW_SHOT` | constant | clean | 7 | Drop-block candidate per REQ-V264-06 regardless of verdict. Mechanical grep of every token (`Scheduler`, `queues`, `jobs`, `dispatches`, `worker`, `threads`, `scheduler-based`, `nodes`) against `BENCHMARK_TABOO.md` yields ZERO per-dataset hits; `Scheduler` is in the Safe SE Textbook Examples list (BENCHMARK_TABOO.md:63). |
| `AMBIGUITY_RULES` | constant | clean | 1 | One sentence. Manual token review confirms no per-dataset hits. The only Universal-Taboo overlap is `component`, which appears as the generic SE noun ("naming a specific component") — passes the v2.1 GATE-06 cross-dataset isolation check (does not identify any one project's component). No cut row. |
| `_prompt_ambiguity` (prose) | builder | domain-loaded | 19 total (3 prose lines: 266, 268, 272) | Line 266 opener `software architecture component names` is the domain-loaded candidate per D-01 — the COMPONENTS slot at line 268 already constrains universe; `component names` carries equivalent instruction. Lines 274–282 are JSON-schema and `JSON only:` suffix, excluded per D-03. |

> **AMB inventory note:** LOC counts confirmed against frozen source — `AMBIGUITY_FEW_SHOT` spans `prompts_v5.py:30–36` (7 lines), `AMBIGUITY_RULES` is `prompts_v5.py:38` (1 line), `_prompt_ambiguity` spans `s_linker19.py:264–282` (19 lines, of which lines 266, 268, 272 are prose and 274–282 are JSON-schema). No discrepancies vs the inspection priors copied into the top-of-doc Verdict Summary by 45-01.

### Cut Candidates

| cut_id | file:lines | trigger | before | after | risk | gated_by |
|---|---|---|---|---|---|---|
| CUT-AMB-01 | src/llm_sad_sam/linkers/experimental/prompts_v5.py:30-36 | drop-block (REQ-V264-06, not benchmark-leak) | `AMBIGUITY_FEW_SHOT` entire block (Examples 1+2 using `"Scheduler"`) | `""` | high — 5 snapshots; ambiguity classification is the leading Phase-1 prompt; removing the only few-shot is the most behavior-changing edit possible (per CD-5) | tests/test_s_linker20_prompt_ambiguity.py @ phase_1_model |
| CUT-AMB-02 | src/llm_sad_sam/linkers/experimental/s_linker19.py:266 | domain-loaded (`"software architecture component names"`) | `Classify these software architecture component names.` | `[Phase 46 empirical loop]` | low — the `NAMES: …` slot at line 268 already constrains scope; `software architecture` is pleonastic per D-01 | tests/test_s_linker20_prompt_ambiguity.py @ phase_1_model |

> AMBIGUITY_RULES: no cut rows (verdict: clean — Universal-Taboo grep plus manual review of the single sentence yields zero leaks; the only Universal-Taboo overlap, `component`, passes the v2.1 GATE-06 cross-dataset isolation check as a generic SE noun).

> AMBIGUITY_FEW_SHOT verdict is `clean` (no per-dataset taboo hits on any token, `Scheduler` is in the Safe SE Textbook Examples list). Per D-04/D-06 the rewording families (Family A synthetic-neutral, Family B concept-only) are emitted only when a constant escalates to `benchmark-leak` — that condition does not fire here, so no `CUT-AMB-03+` Family A/B rows are emitted. The drop-block CUT-AMB-01 is still mandated by REQ-V264-06 regardless of verdict and is the only AMBIGUITY_FEW_SHOT cut row.

> **Reviewer judgment (AMB).** CUT-AMB-01 risk is `high` because the entire Phase-1 ambiguity stage gates on exactly 5 snapshots and a single in-prompt example pair — removing the example flips the prompt from few-shot to zero-shot, which is the largest semantic delta available in this section. CUT-AMB-02 risk is `low` because the four-token prefix `software architecture` is provably non-load-bearing: the `NAMES:` slot scopes the task on the next prose line; the prefix only restates that scope. No `benchmark-leak` verdict was assigned (mechanical grep cleared every token against all 5 dataset sections), so the Phase-46 Family-A/B rewording slot stays empty for this section.

<!-- SECTION:AMB:END -->

## Phase 1 — Doc-Knowledge Extract (DKX)

<!-- SECTION:DKX:START -->

### Items

| Item | Type | Verdict | LOC | Notes |
|---|---|---|---|---|
| `DOC_KNOWLEDGE_EXTRACTION_RULES` | constant | clean | 1 | Single sentence (`prompts_v5.py:45`). Mechanical Universal-Taboo grep on the body returns one hit: `component` ("a single named component"). The v2.1 GATE-06 cross-dataset isolation check passes — `component` is used as a generic SE noun here, not identifying any specific dataset's component (same precedent as `AMBIGUITY_RULES` per 45-02). Load-bearing per D-01: `introduced short forms`, `alternate names`, `words of multi-word names`, and `ordinary English use dominates` are the task semantics (alias discovery cannot be re-stated without them). Per-dataset taboo hits: zero. |
| `ALIAS_SCOPE_RULES` | constant | clean | 4 | Four lines (`prompts_v5.py:57–60`). Universal-Taboo grep returns one hit: `component` (in `name the component` and `which component is being discussed`) — passes v2.1 GATE-06 isolation as a generic SE noun. The remaining vocabulary is structural/typographic (`CamelCase`, `all-caps abbreviations`, `hyphenated`, `qualified-name fragments`, `X.Y or X.Y.Z`) — none appears in any per-dataset taboo section. The `X.Y.Z` clause is empirically load-bearing for the code-path FP suppression (see `prompts_v5.py` module docstring lines 9–18, `experiment_dotted_path_rename.py`). Per-dataset taboo hits: zero. |
| `_prompt_doc_knowledge_extract` (prose) | builder | clean | 19 total (1 audit-relevant prose line: 286) | Opener at `s_linker19.py:286`: `Find all alternative names used for these components in the document.` The plural `components` does not match the Universal-Taboo `\bcomponent\b` token, but reviewer judgment treats it equivalently — generic SE noun, passes v2.1 GATE-06 isolation. `alternative names` is the actual task (universal nouns already in use per D-01 — no `domain-loaded` candidate available). Lines 297–302 are JSON-schema (`Return JSON: {…}` + `JSON only:` suffix) excluded per D-03. Per-dataset taboo hits: zero. |

> **DKX inventory note:** LOC counts confirmed against frozen source — `DOC_KNOWLEDGE_EXTRACTION_RULES` is `prompts_v5.py:45` (1 line), `ALIAS_SCOPE_RULES` spans `prompts_v5.py:57–60` (4 lines), `_prompt_doc_knowledge_extract` spans `s_linker19.py:284–302` (19 lines, of which line 286 is the only audit-relevant prose under D-03; line 288 is the `COMPONENTS:` slot, lines 290 and 292 are constant interpolations of the rows audited above, lines 294–295 are the `DOCUMENT:` slot, lines 297–302 are JSON-schema). No discrepancies vs the inspection priors copied into the top-of-doc Verdict Summary by 45-01.

### Cut Candidates

| cut_id | file:lines | trigger | before | after | risk | gated_by |
|---|---|---|---|---|---|---|

> DKX: no cut rows emitted (all three items `clean` per D-01/D-02). `DOC_KNOWLEDGE_EXTRACTION_RULES` and `ALIAS_SCOPE_RULES` carry no per-dataset taboo body-text hits; the single Universal-Taboo overlap (`component`) passes the v2.1 GATE-06 cross-dataset isolation check as a generic SE noun in both constants (same precedent as `AMBIGUITY_RULES` per 45-02). `_prompt_doc_knowledge_extract`'s opener at line 286 uses universal nouns (`components`, `alternative names`, `document`) and the JSON-schema lines 297–302 are out of scope per D-03. Phase 46 may skip DKX entirely.

> **ALIAS_SCOPE_RULES cross-section note:** This constant is audited HERE under DKX as its canonical home per 45-RESEARCH.md §6.1. `ALIAS_SCOPE_RULES` is imported by `_prompt_doc_knowledge_extract` (interpolated at `s_linker19.py:292`) and is NOT imported by `_prompt_coref`. The COR section (45-07-PLAN) will carry only a back-reference to this DKX row — no duplicate cut_ids will be issued under `CUT-COR-NN` for this constant.

> **Reviewer judgment (DKX).** All three items received `clean` verdicts with no cut rows — the lowest-yield section per 45-RESEARCH.md §5.3's prior estimate (0–2 cut rows) holds. Risk-tier discussion is therefore not applicable (no cut rows to score per CD-5). Phase 46 implication: the `phase_1_doc_extract` gate (5 snapshots in `tests/test_s_linker20_prompt_doc_extract.py`, 3 of which carry known prompt-version-drift `UserWarning`s per 44-CONTEXT §D-03) does not need to be exercised against any DKX-originated cut. Phase 46 may still touch this gate transitively if a cut in another section (e.g. EXT or COR) changes shared scaffolding, but no DKX cut drives Phase 46 work directly.

<!-- SECTION:DKX:END -->

## Phase 1 — Doc-Knowledge Judge (DKJ)

<!-- SECTION:DKJ:START -->

### Items

| Item | Type | Verdict | LOC | Notes |
|---|---|---|---|---|
| `DOC_KNOWLEDGE_JUDGE_EXAMPLES` | constant | benchmark-leak | 7 | Two few-shot examples at `prompts_v5.py:47–53`. Example 2 component `CacheLayer` contains the head noun `cache`, which appears in MediaStore §Components (`Cache`), MediaStore §Keywords (`cache`), AND the Universal Taboo list ("cache (MediaStore)"). Per D-02 per-dataset rule, this auto-classifies as `benchmark-leak` without manual review. Example 1's `RequestHandler`/`Handler` clear all 5 per-dataset sections by mechanical grep (zero whole-word hits in MediaStore / TeaStore / Teammates / BigBlueButton / JabRef) and `Handler` is not in the Universal Taboo list — so those names are individually clean, but the constant as a whole still triggers `benchmark-leak` via the `CacheLayer` hit alone. Per D-04 strict few-shot interpretation + REQ-V264-06, the block also receives a drop-block row. Per D-06, Family A (synthetic-neutral name swap) + Family B (concept-only) rewordings are emitted. |
| `DOC_KNOWLEDGE_JUDGE_RULES` | constant | domain-loaded ("architectural tier or technology platform") | 1 | Single dense sentence at `prompts_v5.py:55`. Mechanical grep on body tokens: `tier`, `platform`, `grouping`, `entity`, `equivalence`, `phrase`, `vocabulary`, `system`, `component`, `alias`, `APPROVE` — zero per-dataset hits; `component` is the only Universal Taboo overlap and passes v2.1 GATE-06 cross-dataset isolation as a generic SE noun (same precedent as `AMBIGUITY_RULES` per 45-02 and `DOC_KNOWLEDGE_EXTRACTION_RULES` per 45-03). The "architectural tier or technology platform that encompasses multiple elements" clause is `domain-loaded` per D-01 — a universal noun like "grouping that encompasses multiple elements" would carry the same instruction, since the load-bearing semantic is "names a multi-element collection, not a single component," not the specific words `tier`/`platform`. Per D-05, no rewording proposed; flagged for Phase 46 empirical loop (REQ-V264-07). |
| `_prompt_doc_knowledge_judge` (prose) | builder | clean | 16 total (1 audit-relevant prose line: 306) | Opener at `s_linker19.py:306`: `JUDGE: Review these component name mappings for correctness.` Per D-01 pragmatic rubric: `component`, `name`, `mappings`, `correctness` are all generic universal nouns; mechanical Universal-Taboo grep returns only the `component` overlap, which passes v2.1 GATE-06 isolation as a generic SE noun (same precedent as the DKX opener at line 286). Line 308 (`COMPONENTS:` slot), line 310 (`PROPOSED MAPPINGS:` slot), lines 313/315 (constant interpolations audited above), and lines 317–319 (JSON-schema `Return JSON: {…}` + `JSON only:` suffix) are excluded from rewrite per D-03. No cut row for the builder opener. |

> **DKJ inventory note:** LOC counts confirmed against frozen source — `DOC_KNOWLEDGE_JUDGE_EXAMPLES` spans `prompts_v5.py:47–53` (7 lines), `DOC_KNOWLEDGE_JUDGE_RULES` is `prompts_v5.py:55` (1 line), `_prompt_doc_knowledge_judge` spans `s_linker19.py:304–319` (16 lines, of which line 306 is the only audit-relevant prose under D-03; lines 308, 310 are input slots, lines 313, 315 are constant interpolations of the rows audited above, lines 317–319 are JSON-schema). No discrepancies vs the inspection priors copied into the top-of-doc Verdict Summary by 45-01.

> **DKJ benchmark-leak audit (mechanical grep results, verbatim):**
> - `grep -niw 'cache' BENCHMARK_TABOO.md` → 3 hits (line 7 MediaStore §Components `Cache`; line 9 MediaStore §Keywords `cache`; line 39 Universal Taboo `cache (MediaStore)`). **Confirms auto-classify** per D-02.
> - `grep -niwE 'Handler|RequestHandler|CacheLayer' BENCHMARK_TABOO.md` → 0 hits (whole-word match). Substring containment: `CacheLayer` contains `cache` → still a hit. `RequestHandler` / `Handler` contain no taboo substrings.
> - `grep -nw 'system' BENCHMARK_TABOO.md` → 0 hits. The few-shot prose `"the system"` is clean.
> - `grep -niwE 'platform|tier|entity|grouping' BENCHMARK_TABOO.md` → 0 hits. The DOC_KNOWLEDGE_JUDGE_RULES "architectural tier or technology platform" clause is not a leak — it is `domain-loaded` per D-01.
> - `grep -niwE 'component|name|mapping' BENCHMARK_TABOO.md` → only the standing `component` Universal-Taboo references; `name` and `mapping` zero whole-word hits. The `_prompt_doc_knowledge_judge` opener is clean (same generic-SE-noun precedent as `_prompt_doc_knowledge_extract`).

### Cut Candidates

| cut_id | file:lines | trigger | before | after | risk | gated_by |
|---|---|---|---|---|---|---|
| CUT-DKJ-01 | src/llm_sad_sam/linkers/experimental/prompts_v5.py:47-53 | benchmark-leak (drop-block, REQ-V264-06) | `DOC_KNOWLEDGE_JUDGE_EXAMPLES` entire block (Examples 1+2 using `RequestHandler`/`Handler`/`CacheLayer`) | `""` | high — 5 snapshots; the judge stage decides which alias proposals survive into Phase 2 extraction; the few-shot drives judge calibration (VALID/INVALID rationale shape); full removal is the most behavior-changing edit and per the v2.1 frontier-map finding proposer/judge boundaries are load-bearing | tests/test_s_linker20_prompt_doc_judge.py @ phase_1_doc_judge |
| CUT-DKJ-02 | src/llm_sad_sam/linkers/experimental/prompts_v5.py:47-50 | benchmark-leak (Family A: synthetic-neutral swap, Example 1) | `Candidate = "Handler", Component = "RequestHandler" …` | `[detail block CUT-DKJ-02 below — replaces RequestHandler→BookManager, Handler→Mgr]` | med — name-only swap preserves judge calibration shape (VALID rationale on parenthetical alias-definition still tested); synthetic candidate `BookManager`/`Mgr` grep-clears against all 5 per-dataset sections AND Universal Taboo AND Safe SE Textbook | tests/test_s_linker20_prompt_doc_judge.py @ phase_1_doc_judge |
| CUT-DKJ-03 | src/llm_sad_sam/linkers/experimental/prompts_v5.py:51-53 | benchmark-leak (Family A: synthetic-neutral swap, Example 2 — directly removes `CacheLayer` leak) | `Candidate = "the system", Component = "CacheLayer" …` | `[detail block CUT-DKJ-03 below — replaces CacheLayer→MailSender]` | med — name-only swap; INVALID rationale shape ("the system" overshoots referent) preserved; replacement `MailSender` grep-clears against all 5 per-dataset sections AND Universal Taboo AND Safe SE Textbook | tests/test_s_linker20_prompt_doc_judge.py @ phase_1_doc_judge |
| CUT-DKJ-04 | src/llm_sad_sam/linkers/experimental/prompts_v5.py:47-53 | benchmark-leak (Family A: synthetic-neutral swap, both examples — combined rewrite) | `Examples 1+2 (RequestHandler/Handler/CacheLayer/"the system")` | `[detail block CUT-DKJ-04 below — replaces with BookManager/Mgr + MailSender; single coherent component domain]` | med — same per-token risk shape as CUT-DKJ-02 and CUT-DKJ-03, but Phase 46 may prefer a single batch swap to test "judge survives name domain rotation"; one combined cut is cheaper to validate than two separate ones | tests/test_s_linker20_prompt_doc_judge.py @ phase_1_doc_judge |
| CUT-DKJ-05 | src/llm_sad_sam/linkers/experimental/prompts_v5.py:47-50 | benchmark-leak (Family B: concept-only, Example 1) | `Candidate = "Handler", Component = "RequestHandler" …` | `[detail block CUT-DKJ-05 below — name-stripped abstract rule shape]` | med-high — name-stripped form may break judge calibration (no concrete VALID anchor); lower fidelity than Family A; the parenthetical-definition pattern is still encoded but without a worked example | tests/test_s_linker20_prompt_doc_judge.py @ phase_1_doc_judge |
| CUT-DKJ-06 | src/llm_sad_sam/linkers/experimental/prompts_v5.py:51-53 | benchmark-leak (Family B: concept-only, Example 2) | `Candidate = "the system", Component = "CacheLayer" …` | `[detail block CUT-DKJ-06 below — name-stripped INVALID-shape rule]` | med-high — same shape as CUT-DKJ-05; the "names whole-system / different referent" failure mode is described abstractly without `CacheLayer` or `the system` as anchors | tests/test_s_linker20_prompt_doc_judge.py @ phase_1_doc_judge |
| CUT-DKJ-07 | src/llm_sad_sam/linkers/experimental/prompts_v5.py:55 | domain-loaded ("architectural tier or technology platform") | `…An alias is also invalid when it names an architectural tier or technology platform that encompasses multiple elements…` | `[Phase 46 empirical loop]` | med — the clause carries multi-element exclusion behavior (it's what catches a judge approving "the storage layer" as an alias for one specific component when the document uses it as a tier label); rewording risks losing this exclusion, but the rule-shape is universal-noun-replaceable per D-01 ("grouping that encompasses multiple elements") | tests/test_s_linker20_prompt_doc_judge.py @ phase_1_doc_judge |

> DOC_KNOWLEDGE_JUDGE_RULES otherwise clean: the rest of the sentence (`equivalence between a phrase and a single named component`, `generic vocabulary, names the whole system, or names a different entity`, `When uncertain, prefer APPROVE`) uses universal nouns and the standing-`component` overlap that passes v2.1 GATE-06 isolation. Only the tier/platform span carries the `domain-loaded` flag.

> _prompt_doc_knowledge_judge opener at `s_linker19.py:306` is `clean` (`component name mappings` is generic, same precedent as the DKX opener line 286). No cut row for the builder.

> **CUT-DKJ-02 detail (Family A, Example 1 swap).** Replaces `RequestHandler` with `BookManager` and `Handler` with `Mgr`. Grep clearance: `grep -niw 'BookManager\|Mgr' BENCHMARK_TABOO.md` → 0 hits; substring `book` → 0 hits (clean per substring audit); substring `manager` → only Safe SE Textbook references (lines 63, 66 — explicitly safe per BENCHMARK_TABOO.md §"confirmed not in benchmark", but planner prefers fresh names so the `Mgr` abbreviation is chosen instead of `BookManager`/`Manager` in the candidate alias slot). Proposed Example 1 rewrite:
> ```
> Example 1: Candidate = "Mgr", Component = "BookManager"
> Evidence: "The BookManager (hereafter Mgr) coordinates lookups against the catalog."
> Judgment: VALID — The document explicitly establishes "Mgr" as an alternate name for BookManager via parenthetical definition. The alias is distinctive and scoped to one component.
> ```
> Rationale-shape parity with the original: parenthetical-definition pattern → VALID; distinctive scoped alias preserved; no benchmark token introduced.

> **CUT-DKJ-03 detail (Family A, Example 2 swap — primary leak removal).** Replaces `CacheLayer` with `MailSender`. Grep clearance: `grep -niw 'MailSender' BENCHMARK_TABOO.md` → 0 hits; substring `mail` → 0 hits; substring `sender` → 0 hits (clean per substring audit). `MailSender` does NOT appear in the Safe SE Textbook Examples list (line 60–68) — fresh fully-novel synthetic name. Proposed Example 2 rewrite:
> ```
> Example 2: Candidate = "the system", Component = "MailSender"
> Evidence: "The system queues outgoing messages in the MailSender."
> Judgment: INVALID — "the system" refers to the overall application, not to MailSender specifically. It names a different entity (the whole system) rather than establishing MailSender as an alias.
> ```
> Rationale-shape parity with the original: INVALID because the candidate phrase overshoots the referent. The leak (`cache` substring via `CacheLayer`) is removed; the calibration shape is preserved.

> **CUT-DKJ-04 detail (Family A, combined both-example rewrite).** Single coherent rewrite using one component domain (a synthetic mail/catalog system). Grep-clearance per CUT-DKJ-02 and CUT-DKJ-03 already established for `BookManager`/`Mgr`/`MailSender`. Proposed combined rewrite:
> ```
> Example 1: Candidate = "Mgr", Component = "BookManager"
> Evidence: "The BookManager (hereafter Mgr) coordinates lookups against the catalog."
> Judgment: VALID — The document explicitly establishes "Mgr" as an alternate name for BookManager via parenthetical definition. The alias is distinctive and scoped to one component.
>
> Example 2: Candidate = "the system", Component = "MailSender"
> Evidence: "The system queues outgoing messages in the MailSender."
> Judgment: INVALID — "the system" refers to the overall application, not to MailSender specifically. It names a different entity (the whole system) rather than establishing MailSender as an alias.
> ```
> Variant rationale: tests whether the judge generalizes alias-validity from the (parenthetical-definition → VALID, whole-system overshoot → INVALID) rule-shape without rotation across two component domains. If Phase 46 byte-equality on this variant succeeds, downstream cuts can prefer it over CUT-DKJ-02 + CUT-DKJ-03 (one cut, two pattern coverages).

> **CUT-DKJ-05 detail (Family B, concept-only Example 1).** Name-stripped abstract restatement of the parenthetical-definition VALID pattern. Proposed:
> ```
> Pattern (VALID): When a short term in the document is bound to a longer component name via an explicit parenthetical introduction such as "(hereafter X)" — and X uniquely refers to one named component — treat the short term as a valid alias for that component. The equivalence must be anchored in the document; do not infer it from prose context alone.
> ```
> Variant rationale: removes ALL component names from the VALID anchor; tests whether the judge can apply the parenthetical-definition rule from rule prose alone, without a worked few-shot. If Phase 46 confirms byte-equality survives, the prompt loses zero benchmark surface area but also loses the worked example — risk tier reflects this.

> **CUT-DKJ-06 detail (Family B, concept-only Example 2).** Name-stripped abstract restatement of the whole-system / different-referent INVALID pattern. Proposed:
> ```
> Pattern (INVALID): When the candidate phrase refers to the overall application (e.g. "the system", "the application", "the platform" used as a wrap-all referent) rather than to one named component, reject the alias. The candidate is invalid because it names a different entity (the whole) rather than establishing the named component as having an alternate surface form.
> ```
> Variant rationale: removes `CacheLayer` and all benchmark-adjacent component names from the INVALID anchor; the failure mode is described abstractly. CUT-DKJ-05 and CUT-DKJ-06 may be applied together (full Family B rewrite of the constant) or separately, at Phase 46's discretion.

> **Reviewer judgment (DKJ).** CUT-DKJ-01 risk is `high` for the same reason CUT-AMB-01 was `high` plus an additional consideration: the judge stage has higher leverage than ambiguity classification on the end-to-end pipeline (it gates which proposed aliases reach extraction), so a calibration shift here propagates further. CUT-DKJ-02 / CUT-DKJ-03 / CUT-DKJ-04 are `med` because name-only swaps preserve the worked few-shot's calibration shape and the synthetic candidates have been grep-cleared against the full taboo list. CUT-DKJ-05 / CUT-DKJ-06 are `med-high` because removing the concrete anchor makes the judge fall back to rule prose alone — this is a larger semantic delta than Family A but smaller than CUT-DKJ-01 (the rule still references the pattern, just without an example). CUT-DKJ-07 is `med`: the "architectural tier or technology platform" span is the only specific failure mode the rule constant calls out beyond the generic equivalence/whole-system/different-entity triad, so a rewording must preserve "multi-element grouping" semantics — Phase 46 empirical loop handles this per REQ-V264-07.

> **DKJ Phase 46 implication.** DKJ is the highest-yield section in the audit per 45-RESEARCH.md §3.3 ("Of 29 Universal Taboo terms, only `cache` produces a confirmed body-text hit"). Phase 46 should prioritize the DKJ cut set ahead of other sections — the benchmark-leak verdict is empirically confirmed (not just inferred), and the candidate rewordings are pre-grep-cleared. Cheap-then-coverage ordering: CUT-DKJ-04 (combined Family A, one snapshot batch covers both examples) → CUT-DKJ-02 → CUT-DKJ-03 → CUT-DKJ-05/06 (Family B, higher semantic risk) → CUT-DKJ-01 (drop-block, last resort) → CUT-DKJ-07 (domain-loaded, separate empirical loop). Phase 46 may stop at the first family that survives byte-equality.

<!-- SECTION:DKJ:END -->

## Phase 2 — Extraction (EXT)

<!-- SECTION:EXT:START -->

### Items

| Item | Type | Verdict | LOC | Notes |
|---|---|---|---|---|
| `ENTITY_EXTRACTION_RULES` | constant | clean | 1 | Single sentence (`prompts_v5.py:67`). Mechanical Universal-Taboo grep on body returns one hit: `component` ("refers to the component by name"). The v2.1 GATE-06 cross-dataset isolation check passes — `component` is used as the generic SE noun, not identifying any specific dataset's component (same precedent as `AMBIGUITY_RULES` per 45-02, `DOC_KNOWLEDGE_EXTRACTION_RULES` per 45-03, `DOC_KNOWLEDGE_JUDGE_RULES` per 45-04). Load-bearing terms per D-01: `code-level path`, `compound identifier`, `architectural intent`, and `Favor inclusion` are the task semantics (the dotted-path FP-exclusion behavior empirically validated by `experiment_dotted_path_rename.py` and recorded in `prompts_v5.py` module docstring lines 9–18; the v1.0/v2.0 Spike-003 dotted-path finding logged in PROJECT.md key decisions). Per-dataset taboo hits: zero. |
| `_prompt_extraction` (prose) | builder | domain-loaded ("software architecture components") | 15 total (1 audit-relevant prose line: 323) | Opener at `s_linker19.py:323`: `Extract ALL references to software architecture components from this document.` Per D-01 pragmatic rubric: the qualifier `software architecture` is pleonastic — the `COMPONENTS:` slot at line 325 (interpolating `comp_names`) already constrains the universe to the component list passed in; `components` alone (or `entities`/`named elements`) would carry the same instruction. Same `software architecture` pleonasm pattern as `_prompt_ambiguity` opener at line 266 (per 45-02). Mechanical Universal-Taboo grep on this opener returns zero whole-word hits (`components` plural does not match `\bcomponent\b`; `architecture` is not in Universal Taboo). Line 325 (`COMPONENTS:` slot), line 326 (conditional `KNOWN ALIASES` interpolation — structural per D-03), line 328 (`ENTITY_EXTRACTION_RULES` interpolation audited above), lines 330–331 (`DOCUMENT:` slot), and lines 333–335 (JSON-schema `Return JSON: {…}` + `JSON only:` suffix) are excluded from rewrite per D-03. |

> **EXT inventory note:** LOC counts confirmed against frozen source — `ENTITY_EXTRACTION_RULES` is `prompts_v5.py:67` (1 line), `_prompt_extraction` spans `s_linker19.py:321–335` (15 lines, of which line 323 is the only audit-relevant prose under D-03). No discrepancies vs the inspection priors copied into the top-of-doc Verdict Summary by 45-01.

> **EXT benchmark-leak audit (mechanical grep results, verbatim):**
> - `sed -n '67p' prompts_v5.py | grep -iwoE 'logic|client|storage|server|cache|model|component|adapter|processor|event|socket|layer|database|registry|auth|persistence|facade|recording|cascade|conversion|validation|dedicated|preferences|config|internal|order|common|UI|DB'` → 1 hit: `component`. Passes v2.1 GATE-06 isolation as generic SE noun (same precedent as `AMBIGUITY_RULES` per 45-02).
> - `sed -n '323p' s_linker19.py | grep -iwoE …` (same pattern) → 0 hits. The plural `components` does not whole-word-match `\bcomponent\b`; `architecture` / `references` / `software` / `document` are not in Universal Taboo or any per-dataset section.
> - `grep -niwE 'code-level|compound|architectural|favor' BENCHMARK_TABOO.md` → 0 hits. The `ENTITY_EXTRACTION_RULES` exclusion-clause vocabulary is leak-free; the terms are load-bearing per the dotted-path FP behavior recorded in `prompts_v5.py` module docstring and PROJECT.md key decisions.

### Cut Candidates

| cut_id | file:lines | trigger | before | after | risk | gated_by |
|---|---|---|---|---|---|---|
| CUT-EXT-01 | src/llm_sad_sam/linkers/experimental/s_linker19.py:323 | domain-loaded ("software architecture components") | `Extract ALL references to software architecture components from this document.` | `[Phase 46 empirical loop]` | low-med — the `COMPONENTS:` slot at line 325 already scopes the task to the passed-in component list, so the `software architecture` qualifier is pleonastic per D-01; impact is bounded because the qualifier is opener-only (one prose line out of 15 total in the builder); however, `_prompt_extraction` is the MOST snapshot-diverse builder in this section (18 snapshots across two phase tags `phase_2_framing_c_pass1` and `phase_2_framing_c_pass2`), so byte-equality is harder to preserve than for lower-snapshot sections — Phase 46 may collapse to `components`, `named elements`, or `entities` | tests/test_s_linker20_prompt_extraction.py @ phase_2_framing_c_pass1, phase_2_framing_c_pass2 |

> ENTITY_EXTRACTION_RULES: no cut rows (verdict: clean — body text load-bearing per PROJECT.md key decisions; `code-level path`, `compound identifier`, `architectural intent` encode the v1.0/v2.0 Spike-003 dotted-path FP-exclusion behavior and the `Favor inclusion` tiebreaker is empirically tuned; Phase 46 should not propose cuts here).

> **Reviewer judgment (EXT).** CUT-EXT-01 risk is `low-med`. The `low` side: the qualifier is provably non-load-bearing once the COMPONENTS slot is read (the slot enumerates the exact component vocabulary the model should extract; the opener qualifier only restates that scope in domain words). The `med` side: `_prompt_extraction` carries 18 snapshots across two phase tags — more snapshot diversity than any other Phase-1/Phase-2 builder audited so far (AMB has 5, DKX has 5, DKJ has 5; only VAL with 24 and COR with 40 are higher), so a rewording that the model treats as semantically equivalent must still survive 18 byte-equal replays. No `benchmark-leak` verdict was assigned for either EXT item (mechanical grep cleared all body tokens against the 5 per-dataset sections; the standing `component` Universal-Taboo overlap passes v2.1 GATE-06 isolation as a generic SE noun), so the Phase-46 Family-A/B rewording slot stays empty for this section.

> **EXT cross-section observation.** The `software architecture components` opener pattern recurs in three builders so far — `_prompt_ambiguity` line 266 (CUT-AMB-02), `_prompt_extraction` line 323 (CUT-EXT-01), and is anticipated for `_prompt_validation` (45-06) — making it the strongest **Phase-46 batching opportunity** in the audit: a single approved replacement vocabulary (e.g. `components` alone, or `named elements`) could resolve all three `domain-loaded` flags with one harness run per affected gate. Phase 46 should evaluate the three opener cuts as a coordinated batch rather than independently.

<!-- SECTION:EXT:END -->

## Phase 4 — Validation (VAL)

<!-- SECTION:VAL:START -->
<!-- TBD: filled by .planning/phases/45-audit/45-06-PLAN.md (Wave 1) — header table for VALIDATION_RULES + _prompt_validation, then cut table CUT-VAL-NN. Bonus rows for P1_FOCUS, P2_FOCUS, COREF_VALIDATION_FOCUS folded in here per CD-6 (CONTEXT.md §<decisions> Claude's Discretion). -->
<!-- SECTION:VAL:END -->

## Phase 5 — Coref (COR)

<!-- SECTION:COR:START -->
<!-- TBD: filled by .planning/phases/45-audit/45-07-PLAN.md (Wave 1) — header table for COREF_RULES + ANTECEDENT_ALIAS_RULES + _prompt_coref, then cut table CUT-COR-NN. ALIAS_SCOPE_RULES is audited under DKX (it is imported by _prompt_doc_knowledge_extract, not _prompt_coref); see 45-RESEARCH.md §6.1. -->
<!-- SECTION:COR:END -->

## Phase Close Notes

<!-- FINAL:SUMMARY:START -->
<!-- TBD: filled by .planning/phases/45-audit/45-08-PLAN.md (Wave 2) -->
<!-- FINAL:SUMMARY:END -->

<!-- FINAL:GATE01:START -->
<!-- TBD: filled by .planning/phases/45-audit/45-08-PLAN.md (Wave 2) -->
<!-- FINAL:GATE01:END -->
