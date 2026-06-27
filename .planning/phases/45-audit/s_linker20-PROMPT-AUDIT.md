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
| `VALIDATION_RULES` | constant | 1 | domain-loaded | 1 | 45-06 (VAL) |
| `_prompt_validation` | builder | 14 | domain-loaded | 1 | 45-06 (VAL) |
| `P1_FOCUS` *(folded into VAL per CD-6)* | constant | 7 | behavioral-protected | 1 | 45-06 (VAL) |
| `P2_FOCUS` *(folded into VAL per CD-6)* | constant | 6 | clean | 0 | 45-06 (VAL) |
| `COREF_VALIDATION_FOCUS` *(folded into VAL per CD-6)* | constant | 7 | domain-loaded | 1 | 45-06 (VAL) |
| `COREF_RULES` | constant | 1 | domain-loaded | 2 | 45-07 (COR) |
| `_prompt_coref` | builder | 27 | domain-loaded | 3 | 45-07 (COR) |
| **Total (18 items)** | — | — | clean: 8 / domain-loaded: 8 / benchmark-leak: 1 / behavioral-protected: 1 | cut rows: 19 | — |

LOC values are inspection priors copied from 45-RESEARCH.md §1 and §2; Wave-1 plans confirm or correct each via their own read of the frozen source. The 3 CD-6 fold-in rows (P1_FOCUS, P2_FOCUS, COREF_VALIDATION_FOCUS) are imported by `_prompt_validation` and audited inside section VAL; they are not part of the REQ-V264-03 enumerated 9 PROMPT CONSTANTS but are recorded here for Phase 46 input fidelity per 45-CONTEXT.md decision CD-6. The `behavioral-protected` verdict on P1_FOCUS reflects the `prompts_v5.py` module-docstring-protected X.Y.Z clause (CUT-VAL-04 tombstone) — Phase 46 MUST NOT cut this clause; see §VAL.

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

### Items

| Item | Type | Verdict | LOC | Notes |
|---|---|---|---|---|
| `VALIDATION_RULES` | constant | domain-loaded ("counterparts") | 1 | Single sentence (`prompts_v5.py:94`). Mechanical Universal-Taboo grep on body returns one hit: `component` ("treats the component as an architectural participant") — passes v2.1 GATE-06 cross-dataset isolation as generic SE noun (same precedent as `AMBIGUITY_RULES` per 45-02, `DOC_KNOWLEDGE_EXTRACTION_RULES` per 45-03, `DOC_KNOWLEDGE_JUDGE_RULES` per 45-04, `ENTITY_EXTRACTION_RULES` per 45-05). `architectural participant` and `technique that merely shares the component's name` are load-bearing per D-01 (they encode the approve/reject semantics specific to SAD/SAM matching). The trailing `counterparts` is `domain-loaded` candidate per D-01 — universal noun "matching entities" would carry the same instruction. Per-dataset taboo hits: zero. |
| `_prompt_validation` (prose) | builder | domain-loaded ("software architecture document") | 14 total (1 audit-relevant prose line: 339) | Opener at `s_linker19.py:339`: `Validate component references in a software architecture document. {focus}` — same `software architecture` pleonasm as `_prompt_ambiguity` line 266 (CUT-AMB-02 per 45-02) and `_prompt_extraction` line 323 (CUT-EXT-01 per 45-05). The `COMPONENTS:` slot at line 341 (interpolating `comp_names`) already constrains universe; the `{focus}` interpolation carries P1/P2/COREF_VALIDATION_FOCUS instructions audited below. Mechanical Universal-Taboo grep on this opener returns one hit: `component` plural (`components`) — does not whole-word-match `\bcomponent\b`; passes second-pass as generic SE noun in any case. Line 341 (`VALIDATION_RULES` interpolation audited above), line 343 (`COMPONENTS:` slot), lines 345–346 (`CASES:` slot + case loop), and lines 348–350 (JSON-schema `Return JSON: {…}` + `JSON only:` suffix) are excluded from rewrite per D-03. |
| `P1_FOCUS` (folded per CD-6) | constant | clean (with docstring-protected clause inside) | 7 (multi-line tuple: `prompts_v5.py:80–86`) | Architectural-participation question + trailing `qualified-name identifier (e.g. a package- or member-access path X.Y.Z)` anchor. Mechanical Universal-Taboo grep returns one hit: `component` ("does the sentence name this component as an architectural participant") — passes v2.1 GATE-06 isolation as generic SE noun. The trailing X.Y.Z clause is **empirically validated load-bearing** per `prompts_v5.py` module docstring lines 5–22 (catches 2/3 code-path FPs on gpt-5.4, 1/3 on Claude Sonnet, 0 collateral damage on 4-TP control set via `experiment_dotted_path_rename.py`; strict joint improvement over the original "dotted-path identifier" wording per docstring lines 14–16). Per 45-RESEARCH.md §7.4 and Phase-45 threat T-45-VAL-02 (mitigate): **DO NOT propose cutting**. Single visibility-only row (CUT-VAL-04) emitted with `after = DO NOT CUT — empirically validated load-bearing` and `risk = high`. The opener phrase `architectural participation` and the action enumeration `performing operations, providing services, or taking part in the described system behavior` are load-bearing for SAD/SAM task framing per D-01 — clean. Per-dataset taboo hits: zero. |
| `P2_FOCUS` (folded per CD-6) | constant | clean | 6 (multi-line tuple: `prompts_v5.py:88–93`) | Referential-specificity question (`is the component name used to identify this specific architectural element, or does it serve as a generic technical term in this sentence?`). Mechanical Universal-Taboo grep returns one hit: `component` — passes v2.1 GATE-06 isolation as generic SE noun. Load-bearing per D-01: `referential specificity` is the P2 epistemic frame (not replaceable with a universal noun without losing the question's discriminator) and `generic technical term` is the load-bearing rejection criterion (its absence would erase the "name vs. concept" distinction the pass exists to make). No `domain-loaded` candidate available. Per-dataset taboo hits: zero. No cut rows emitted. |
| `COREF_VALIDATION_FOCUS` (folded per CD-6) | constant | domain-loaded ("role-referential phrase") | 7 (multi-line tuple: `prompts_v5.py:106–112`) | Single-pass coref-focused validator. Mechanical Universal-Taboo grep returns one hit: `service` (`'the service'`, used as an example role-referential phrase) AND `component` (`refer to the named component`) — both pass v2.1 GATE-06 isolation as generic SE nouns. `service` matches BBB §Components compound `Recording Service` case-insensitive whole-word, but the prompt uses bare `the service` exactly as a generic role-referential phrase in a quoted enumeration alongside `'it'`, `'they'`, `'the service'` — per 45-RESEARCH.md §1.6 explicit second-pass dismissal, this is the generic-SE-noun case, not a BBB leak. The phrase `role-referential phrase` is `domain-loaded` per D-01 — linguistics jargon; universal alternative `noun phrase that refers back` would carry the same instruction. The asymmetric single-pass design (vs `_prompt_validation`'s twopass under P1+P2) is **empirically load-bearing** per `prompts_v5.py` docstring lines 101–105: "entity twopass leaks ~4 FPs on bigbluebutton coref" → per 45-RESEARCH.md §7.4, **do NOT propose symmetrizing** (that is a code-architecture change, not a prompt-text cut). Per-dataset taboo hits: zero (after second-pass isolation). |

> **VAL inventory note:** LOC counts confirmed against frozen source — `VALIDATION_RULES` is `prompts_v5.py:94` (1 line); `_prompt_validation` spans `s_linker19.py:337–350` (14 lines, of which line 339 is the only audit-relevant prose under D-03; line 341 is the `VALIDATION_RULES` interpolation audited above; line 343 is the `COMPONENTS:` slot; lines 345–346 are the `CASES:` slot + case loop; lines 348–350 are JSON-schema). The three folded constants per CD-6: `P1_FOCUS` spans `prompts_v5.py:80–86` (7 lines, multi-line tuple form), `P2_FOCUS` spans `prompts_v5.py:88–93` (6 lines), `COREF_VALIDATION_FOCUS` spans `prompts_v5.py:106–112` (7 lines). No discrepancies vs the inspection priors in §1.4/§1.6 of 45-RESEARCH.md.

> **VAL benchmark-leak audit (mechanical grep results, verbatim):**
> - `sed -n '94p' prompts_v5.py | grep -iwoE 'logic|client|storage|server|cache|model|component|adapter|processor|event|socket|layer|database|registry|auth|persistence|facade|recording|cascade|conversion|validation|dedicated|preferences|config|internal|order|common|UI|DB|service'` → 1 hit: `component` (×2: "the component", "component's name"). Both pass v2.1 GATE-06 isolation as generic SE noun.
> - `sed -n '339p' s_linker19.py | grep -iwoE …` (same pattern) → 0 hits (the plural `components` does not whole-word-match `\bcomponent\b`; `architecture`/`software`/`document` not in Universal Taboo).
> - `sed -n '80,86p;88,93p;106,112p' prompts_v5.py | grep -iwoE …` (same pattern) → 4 hits: `component` (×3 across P1/P2/COREF_VALIDATION_FOCUS, all generic SE noun) and `service` (×1, in COREF_VALIDATION_FOCUS `'the service'` quoted role-referential exemplar).
> - `grep -niw 'service' BENCHMARK_TABOO.md` → 1 hit (line 22, BBB §Components compound `Recording Service`). The bare `the service` in the prompt is the generic-SE-noun second-pass case per 45-RESEARCH.md §1.6 explicit dismissal — not a benchmark leak.
> - `grep -niwE 'counterparts|participant|architectural|technique|referential|specificity|generic|technical|qualified|package|member|access|coref|resolution|pronoun|role-referential|grammatical|topic' BENCHMARK_TABOO.md` → 0 hits across all VAL-section vocabulary. Every load-bearing token clears the taboo list.

### Cut Candidates

| cut_id | file:lines | trigger | before | after | risk | gated_by |
|---|---|---|---|---|---|---|
| CUT-VAL-01 | src/llm_sad_sam/linkers/experimental/prompts_v5.py:94 | domain-loaded ("counterparts") | `Approve when the sentence treats the component as an architectural participant, including counterparts.` | `[Phase 46 empirical loop]` | med — 24 snapshots across 3 phase tags is medium-high diversity (most conservatively gated builder in the audit per 45-RESEARCH.md §4.2); `counterparts` is a single noun and universal alternative `matching entities` carries the same instruction per D-01, but the rewording must survive 24 byte-equal replays across 3 phase tags so any subtle semantic shift will be caught early in Phase 46 | tests/test_s_linker20_prompt_validation.py @ phase_4_twopass_p1, phase_4_twopass_p2, phase_5_coref_validation |
| CUT-VAL-02 | src/llm_sad_sam/linkers/experimental/s_linker19.py:339 | domain-loaded ("software architecture document") | `Validate component references in a software architecture document. {focus}` | `[Phase 46 empirical loop]` | low — same `software architecture` pleonasm as `_prompt_ambiguity` line 266 (CUT-AMB-02) and `_prompt_extraction` line 323 (CUT-EXT-01); the `COMPONENTS:` slot at line 341 already constrains scope; Phase 46 should batch these three openers together per the cross-section observation closing 45-05's EXT section — a single approved replacement (e.g. `components` alone, or `named elements`) resolves all three flags with one harness run per affected gate | tests/test_s_linker20_prompt_validation.py @ phase_4_twopass_p1, phase_4_twopass_p2, phase_5_coref_validation |
| CUT-VAL-03 | src/llm_sad_sam/linkers/experimental/prompts_v5.py:106-112 (COREF_VALIDATION_FOCUS) | domain-loaded ("role-referential phrase") | `…or similar role-referential phrase in this sentence actually refer to the named component…` | `[Phase 46 empirical loop]` | med-high — COREF_VALIDATION_FOCUS gates `phase_5_coref_validation` which reuses `_prompt_validation` (24 validation snapshots) AND drives 40 downstream `phase_5_coref` snapshots transitively through the coref→coref-validation pipeline; `role-referential` is linguistics jargon, but it may be load-bearing for distinguishing the asymmetric coref-only validator from the P1/P2 entity twopass (per docstring lines 101–105 the narrower focus prevents ~4 FPs on bigbluebutton coref); universal alternative `noun phrase that refers back` is plausibly equivalent per D-01 but the semantic risk is non-trivial | tests/test_s_linker20_prompt_validation.py @ phase_4_twopass_p1, phase_4_twopass_p2, phase_5_coref_validation |
| CUT-VAL-04 | src/llm_sad_sam/linkers/experimental/prompts_v5.py:84-85 (P1_FOCUS) | behavioral-protected (`prompts_v5.py` module docstring lines 5–22) | `…and not just as a qualified-name identifier (e.g. a package- or member-access path X.Y.Z)?` | `DO NOT CUT — empirically validated load-bearing` | high — `experiment_dotted_path_rename.py` empirical record (cited in docstring lines 9–18): catches 2/3 code-path FPs on gpt-5.4 AND 1/3 on Claude Sonnet with 0 collateral damage on the 4-TP control set; strict joint improvement over the prior `dotted-path identifier` wording (which catches 0/3 on Sonnet); cutting reintroduces the FPs that v2.6.3 documents as fixed and that motivated P1_FOCUS's extension over s17f; visibility-only row — Phase 46 MUST skip this clause per Phase-45 threat T-45-VAL-02 mitigation | tests/test_s_linker20_prompt_validation.py @ phase_4_twopass_p1, phase_4_twopass_p2 |

> **VAL Family A / Family B note:** D-06 mandates Family A (synthetic-neutral) + Family B (concept-only) rewordings only for `benchmark-leak` findings. No VAL-section item escalates to `benchmark-leak` after second-pass isolation — the only Universal-Taboo hit (`service` in COREF_VALIDATION_FOCUS's `'the service'` quoted role-referential exemplar) is dismissed per 45-RESEARCH.md §1.6 as generic-SE-noun isolation (`the service` is used as a generic anaphor exemplar alongside `'it'` and `'they'`, not to identify any specific BBB Recording Service instance). Therefore Family A/B rewording slots stay empty for this section; the 3 domain-loaded flags + 1 visibility-only protected-clause row are the complete cut output.

> **P1_FOCUS protected clause record (per CD-6 + 45-RESEARCH.md §7.4).** The clause `"and not just as a qualified-name identifier (e.g. a package- or member-access path X.Y.Z)"` is behaviorally protected by the `prompts_v5.py` module docstring (lines 5–22). The explicit `X.Y.Z` schema is the structural anchor that empirically carries the clause across LLM backends (docstring line 11–13: "the chosen wording … catches 2 of 3 code-path FPs on gpt-5.4 and 1 of 3 on Claude Sonnet with 0 collateral damage on the 4-TP control set"). The earlier `dotted-path identifier` wording catches 0/3 on Sonnet — the qualified-name framing with parenthetical X.Y.Z example is a strict joint improvement, not stylistic. CUT-VAL-04 records this for Phase 46 visibility with `after = DO NOT CUT`; the audit pipeline must surface the clause as `risk = high` and prevent any Phase-46 rewording proposal from touching it.

> **COREF_VALIDATION_FOCUS asymmetric-design record (per CD-6 + 45-RESEARCH.md §7.4).** The asymmetric single-pass design (vs `_prompt_validation`'s twopass under P1+P2) is **empirically load-bearing** per `prompts_v5.py` docstring lines 101–105: "anaphoric resolution asks a different epistemic question than name disambiguation. The narrower focus is empirically load-bearing (cleanup E experiment showed entity twopass leaks ~4 FPs on bigbluebutton coref)." This is a code-architecture decision (how `_validate_coref_links` calls `_run_validation_pass`), not a prompt-text cut. The audit does NOT propose symmetrizing the call site, splitting COREF_VALIDATION_FOCUS into a P1/P2 pair, or any other structural change. Only the `role-referential phrase` lexical span carries a `domain-loaded` flag (CUT-VAL-03); the asymmetric calling convention stays put.

> **Reviewer judgment (VAL).** CUT-VAL-01 risk is `med`: `counterparts` is a single replaceable noun and the universal alternative `matching entities` is straightforward, but the 3-phase-tag gating (24 snapshots) means any rewording must survive replays across `phase_4_twopass_p1`, `phase_4_twopass_p2`, AND `phase_5_coref_validation` — the most conservative gating in the audit. CUT-VAL-02 risk is `low`: the third instance of the `software architecture` opener pleonasm pattern (AMB → EXT → VAL); Phase 46 should approve one replacement vocabulary across all three sites. CUT-VAL-03 risk is `med-high`: `role-referential phrase` gates the single-pass coref-validation route that the empirical record (docstring lines 104–105, ~4 FP reduction on BBB) singles out as load-bearing — rewording must preserve "noun phrase that refers back to a prior antecedent" semantics, and the harness must catch any drift. CUT-VAL-04 risk is `high`: it is the only row in the entire VAL section that carries `behavioral-protected` trigger; Phase 46 MUST treat the row as a tombstone, not a candidate.

> **VAL Phase 46 implication.** VAL is gated on 3 phase tags (24 snapshots in `tests/test_s_linker20_prompt_validation.py`) — the most conservative gating of any builder. Phase 46 ordering for this section: CUT-VAL-02 (low risk, batches with AMB/EXT openers — single replacement vocabulary across three call sites) → CUT-VAL-01 (med risk, `counterparts` → `matching entities`) → CUT-VAL-03 (med-high risk, `role-referential phrase` → `noun phrase that refers back`) → CUT-VAL-04 (DO NOT EXECUTE, tombstone). Three of the four cuts emit `after = [Phase 46 empirical loop]` per D-05 (no in-Phase-45 rewordings for `domain-loaded`); the fourth is a tombstone. The `software architecture` opener pattern recurring three times (AMB CUT-AMB-02 + EXT CUT-EXT-01 + VAL CUT-VAL-02) remains the strongest Phase-46 batching opportunity in the audit.

> **VAL closing reviewer paragraph.**
> The three `software architecture …` opener spans (AMB line 266, EXT line 323, VAL line 339) are a single coordinated Phase-46 batching opportunity — one approved replacement vocabulary resolves all three `domain-loaded` flags in one harness run per affected gate. The P1_FOCUS `X.Y.Z` qualified-name clause is behaviorally protected by the module docstring (CUT-VAL-04 tombstone) — Phase 46 MUST skip it; no rewording proposal may touch it. The COREF_VALIDATION_FOCUS asymmetric single-pass design is a code-architecture decision (docstring lines 101–105 empirically load-bearing per cleanup-E ~4 FP reduction on BBB) — not a prompt-text cut; only the lexical `role-referential phrase` span carries a flag (CUT-VAL-03).

<!-- SECTION:VAL:END -->

## Phase 5 — Coref (COR)

<!-- SECTION:COR:START -->

> **Cross-section reference (ALIAS_SCOPE_RULES):** ALIAS_SCOPE_RULES is imported by `_prompt_doc_knowledge_extract` only (s_linker19.py:292), not by `_prompt_coref`. Its canonical audit row lives in section DKX above. This COR section does not duplicate the audit; Phase 46 references the DKX cut_ids when minimizing alias-scope text.

### Items

| Item | Type | Verdict | LOC | Notes |
|---|---|---|---|---|
| `COREF_RULES` | constant | domain-loaded (linguistics jargon spans) | 1 (long sentence) | Single dense sentence (`prompts_v5.py:114`). Mechanical Universal-Taboo grep on the body returns three overlap classes: `component` (×3 — "a component named or aliased earlier", "two or more equally plausible antecedents", "only one component has been introduced", "the component" in the quoted role-referential enumeration), `service` (×1 — `"the service"` in the same quoted enumeration), `module` (×1 — `"the module"` in the same enumeration), `system` (×1 — `"the system"` in the same enumeration). Per 45-RESEARCH.md §1.6 explicit second-pass dismissal and the precedent set by VAL CUT-VAL-03's COREF_VALIDATION_FOCUS reading (45-06): the enumeration `"it", "the module", "the service", "the component", "the system"` is a quoted role-referential exemplar list — every entry functions as a generic SE noun (bare anaphor with definite article), passing v2.1 GATE-06 isolation. None identifies any one project's component. The `domain-loaded` flag covers two distinct jargon spans per D-01: `"role-referential noun phrase"` (linguistics jargon — universal alternative `"noun phrase that refers back to a component"` carries the same instruction) and `"section-established topic"` (encodes the second resolution condition; universal alternative `"topic of the surrounding section"` plausibly equivalent). Per-dataset taboo hits: zero. |
| `ANTECEDENT_ALIAS_RULES` | constant | clean | 9 | Nine lines (`prompts_v5.py:116–124`). The two examples use `TaskScheduler` and `scheduler` — both names appear in BENCHMARK_TABOO.md line 63 Safe SE Textbook Examples list (`Operating systems: Scheduler, MemoryManager, FileSystem, ProcessTable, Dispatcher`), explicitly listed under §"confirmed not in benchmark". Mechanical grep clearance of every body token (`TaskScheduler`, `scheduler`, `queues`, `jobs`, `antecedent`, `alias`, `terminal`, `abbreviation`, `hyphenated`, `canonical`) against all 5 per-dataset sections + Universal Taboo: zero whole-word hits except `Scheduler` in the Safe-SE list (which is the affirmative-safe case). The body prose (`terminal word of a multi-word name`, `abbreviation`, `hyphenated form`, `documented alternate name`) restates ALIAS_SCOPE_RULES vocabulary (audited under DKX) — load-bearing per D-01, no `domain-loaded` candidate. The `Default to true` tiebreaker at line 124 is task semantics, not a domain-loaded over-specification. No cut row. Per-dataset taboo hits: zero. |
| `_prompt_coref` (prose) | builder | domain-loaded (multiple jargon spans) | 27 (~6 audit-relevant prose lines: 354, 358–364) | Opener at `s_linker19.py:354` (`Resolve anaphoric references (pronouns and role-referential noun phrases) to architecture components.`) and inline prose block at `s_linker19.py:358–364` (`For each TARGET sentence below, identify any pronoun or role-referential noun phrase that refers back to a component listed above. If a target sentence has no anaphoric reference to a listed component, return no resolution for it. Be conservative — only include resolutions you are CERTAIN about.`). Mechanical Universal-Taboo grep on the prose returns one overlap: `component` plural (`components` — does not whole-word-match `\bcomponent\b`) and singular (`component`, ×2 in inline prose) — both pass v2.1 GATE-06 isolation as generic SE noun (same precedent as `_prompt_doc_knowledge_extract` opener line 286 per 45-03 DKX, `_prompt_validation` opener line 339 per 45-06 VAL). Three jargon spans share a single sentence at line 354 per D-01: `"anaphoric references"`, `"role-referential noun phrases"`, `"architecture components"` — all over-specified universal-noun candidates. The inline prose at lines 358–360 repeats `"anaphoric reference"` jargon. Line 361 (`Be conservative — only include resolutions you are CERTAIN about`) is **behavioral** per 45-RESEARCH.md §7.4 — risk = high, visibility-only row, no rewording proposed (Phase 46 must skip). Lines 366–368 are structural case-loop labels (`--- Case {i+1}: S{...} ---`, `CONTEXT:`, `TARGET:`) excluded per D-03. Lines 374–377 are JSON-schema (`Return JSON: {…}` + `JSON only:` suffix) excluded per D-03. Per-dataset taboo hits: zero. |

> **COR inventory note:** LOC counts confirmed against frozen source — `COREF_RULES` is `prompts_v5.py:114` (1 line, triple-quoted long sentence), `ANTECEDENT_ALIAS_RULES` spans `prompts_v5.py:116–124` (9 lines, triple-quoted block with 2 examples + default-true tiebreaker), `_prompt_coref` spans `s_linker19.py:352–378` (27 lines, of which lines 354 + 358–364 are audit-relevant prose under D-03 — ~6 prose lines total; line 356 is the `COMPONENTS:` slot; lines 366–368 are structural case-loop labels; lines 370 and 372 are constant interpolations of `COREF_RULES` and `ANTECEDENT_ALIAS_RULES` audited above; lines 374–377 are JSON-schema). No discrepancies vs the inspection priors copied into the top-of-doc Verdict Summary by 45-01.

> **COR benchmark-leak audit (mechanical grep results, verbatim):**
> - `grep -niwE 'TaskScheduler|scheduler' BENCHMARK_TABOO.md` → 1 hit (line 63, §Safe SE Textbook Examples — `Operating systems: Scheduler, ...`). **Confirms ANTECEDENT_ALIAS_RULES clean.** The example names `TaskScheduler` and `scheduler` are in the affirmative-safe list; per 45-RESEARCH.md Open Question 3, this is the expected outcome.
> - `grep -niwE 'queues|jobs|module|service|system|pronoun|anaphoric|role-referential|antecedent' BENCHMARK_TABOO.md` → 2 hits: line 22 (BBB §Components compound `Recording Service`) and line 52 (Universal Taboo entry `internal (BBB/Teammates — "X.internal module")`). Both pass v2.1 GATE-06 isolation: COREF_RULES uses bare `the service`/`the module` in a quoted role-referential exemplar enumeration `"it", "the module", "the service", "the component", "the system"` — every entry functions as a generic SE noun (definite-article anaphor placeholder), not as identifier of any specific project's component. Same precedent as VAL CUT-VAL-03 reading of COREF_VALIDATION_FOCUS's `'the service'` per 45-RESEARCH.md §1.6 explicit dismissal.
> - `grep -niwE 'component|name|reference|architecture' BENCHMARK_TABOO.md` → only the standing `component` Universal-Taboo references (lines 32–58); `name`, `reference`, `architecture` zero whole-word hits. The `_prompt_coref` prose opener + inline block are clean against per-dataset hits (the `component` overlap passes second-pass isolation as generic SE noun, same precedent as DKX line 286 / VAL line 339).
> - `grep -niwE 'role-referential|anaphoric|antecedent|grammatical|topic|coref' BENCHMARK_TABOO.md` → 0 hits. The linguistics jargon spans carry NO benchmark surface; the `domain-loaded` verdict is pure over-specification (D-01), not leakage (D-02).

### Cut Candidates

| cut_id | file:lines | trigger | before | after | risk | gated_by |
|---|---|---|---|---|---|---|
| CUT-COR-01 | src/llm_sad_sam/linkers/experimental/prompts_v5.py:114 | domain-loaded ("role-referential noun phrase") | `…decide whether a pronoun or role-referential noun phrase in the target sentence refers back to a component…` | `[Phase 46 empirical loop]` | med-high — 40 snapshots is the highest-diversity gating in the audit (per 45-RESEARCH.md §5.3 + 44-CONTEXT §D-03); `role-referential` may be jargon-load-bearing for distinguishing pronouns from named role mentions (the alternative `noun phrase that refers back to a component` is plausibly equivalent per D-01 but must survive 40 byte-equal replays); same lexical span as VAL CUT-VAL-03's COREF_VALIDATION_FOCUS reading — Phase 46 should batch this with CUT-VAL-03 since the two prompts share the jargon | tests/test_s_linker20_prompt_coref.py @ phase_5_coref |
| CUT-COR-02 | src/llm_sad_sam/linkers/experimental/prompts_v5.py:114 | domain-loaded ("section-established topic") | `…treat it as the section-established topic and resolve role-referential phrases ("it", "the module", "the service", "the component", "the system") to that topic…` | `[Phase 46 empirical loop]` | med — the phrase encodes the second resolution condition (only one component introduced in immediately preceding sentences); rewording risk depends on whether the universal alternative `"topic of the surrounding section"` captures the section-scope constraint without losing the "no direct name repetition" exemption; impact bounded by the explicit context-sentence window already implied by the rule | tests/test_s_linker20_prompt_coref.py @ phase_5_coref |
| CUT-COR-03 | src/llm_sad_sam/linkers/experimental/s_linker19.py:354 | domain-loaded ("anaphoric references" + "role-referential noun phrases" + "architecture components") | `Resolve anaphoric references (pronouns and role-referential noun phrases) to architecture components.` | `[Phase 46 empirical loop]` | med-high — 40 snapshots (highest in audit); three jargon spans share one sentence; the candidate universal-noun rewrite (`pronouns and noun phrases that refer back to a component`) collapses all three spans into a single rewording — Phase 46 should test the line 354 + lines 358–364 prose as one coordinated rewrite, not piecewise, to keep semantic consistency across the opener and the inline restatement | tests/test_s_linker20_prompt_coref.py @ phase_5_coref |
| CUT-COR-04 | src/llm_sad_sam/linkers/experimental/s_linker19.py:358-360 | domain-loaded ("anaphoric reference" repeated; "role-referential noun phrase" repeated) | `For each TARGET sentence below, identify any pronoun or role-referential noun phrase that refers back to a component listed above. If a target sentence has no anaphoric reference to a listed component, return no resolution for it.` | `[Phase 46 empirical loop]` | med — repeat of the opener jargon at line 354 (CUT-COR-03); rewrite must stay lexically consistent with CUT-COR-03's chosen replacement vocabulary or the prompt becomes self-contradictory; Phase 46 must batch CUT-COR-03 + CUT-COR-04 (the two cuts are a single semantic unit split only by the CASES loop interpolation in between) | tests/test_s_linker20_prompt_coref.py @ phase_5_coref |
| CUT-COR-05 | src/llm_sad_sam/linkers/experimental/s_linker19.py:361 | behavioral-protected (45-RESEARCH.md §7.4) | `Be conservative — only include resolutions you are CERTAIN about.` | `DO NOT CUT — no evidence safe` | high — behavioral conservatism dial; coref Phase 5 has the highest FP sensitivity in the pipeline (v2.6.2 s17e showed FP 43→14 via validation gating per CLAUDE.md milestone notes); removing the conservatism instruction risks reintroducing the FP class that the validated-coref breakthrough closed; visibility-only row — Phase 46 MUST skip this line per Phase-45 threat T-45-COR-02 mitigation | tests/test_s_linker20_prompt_coref.py @ phase_5_coref |

> ANTECEDENT_ALIAS_RULES: no cut rows (TaskScheduler + scheduler verified clean across all 5 dataset sections + Universal Taboo; both names appear in §Safe SE Textbook Examples at BENCHMARK_TABOO.md line 63 — `Operating systems: Scheduler, MemoryManager, FileSystem, ProcessTable, Dispatcher`). The constant's body prose (`terminal word of a multi-word name`, `abbreviation`, `hyphenated form`, `documented alternate name`, `Default to true`) is task semantics — load-bearing per D-01 with no `domain-loaded` candidate.

> **COR Family A / Family B note:** D-06 mandates Family A (synthetic-neutral) + Family B (concept-only) rewordings only for `benchmark-leak` findings. No COR-section item escalates to `benchmark-leak` after second-pass isolation (the `the module` / `the service` / `the system` overlaps in COREF_RULES's quoted role-referential enumeration are dismissed as generic-SE-noun anaphor placeholders per 45-RESEARCH.md §1.6, same precedent as VAL CUT-VAL-03; `TaskScheduler`/`scheduler` in ANTECEDENT_ALIAS_RULES grep-clear via the Safe-SE-Textbook list). Therefore Family A/B rewording slots stay empty for this section; the 4 domain-loaded flags + 1 visibility-only behavioral-protected row are the complete cut output.

> **CUT-COR-05 behavioral-protected record (per 45-RESEARCH.md §7.4 + threat T-45-COR-02).** The clause `"Be conservative — only include resolutions you are CERTAIN about"` is behaviorally protected. Coref is the FP-sensitive stage of the pipeline (per CLAUDE.md v2.6.2 milestone notes, the validated-coref breakthrough drove FP 43→14 — a 67% FP reduction at the coref gate); the conservatism instruction at line 361 is the prompt-side counterpart of the validation-side gating. Removing it risks reintroducing the FP class that v2.6.2 documents as fixed. CUT-COR-05 records this for Phase 46 visibility with `after = DO NOT CUT — no evidence safe`; the audit pipeline must surface the line as `risk = high` and prevent any Phase-46 rewording proposal from touching it.

> **CUT-COR-03 + CUT-COR-04 batching note.** The opener (line 354) and inline prose (lines 358–360) share the `anaphoric reference` / `role-referential noun phrase` jargon. Phase 46 MUST batch the two cuts as a single semantic unit — the CASES-loop interpolation between them (lines 365–368) does not break the lexical link, and a Phase 46 rewrite that touches only one of the two sites will produce a self-contradictory prompt (opener says "anaphoric references"; inline says something different, or vice versa). The chosen universal-noun replacement (e.g. `"pronouns and noun phrases that refer back to a component"`) must apply to BOTH sites in lockstep.

> **Reviewer judgment (COR).** CUT-COR-01 risk is `med-high`: 40 snapshots is the highest-diversity gating in the audit (more than EXT's 18 and VAL's 24); `role-referential` is the linguistics-jargon span that VAL CUT-VAL-03 also flags in COREF_VALIDATION_FOCUS, so the two cuts share a Phase-46 batching opportunity — one approved replacement vocabulary (`noun phrase that refers back`) resolves both flags. CUT-COR-02 risk is `med`: `section-established topic` encodes the second resolution condition (the only-one-component-introduced clause); the universal alternative `topic of the surrounding section` is plausible but the empirical check must verify it preserves the "no direct name repetition" exemption that the rule explicitly grants. CUT-COR-03 risk is `med-high`: three jargon spans in one sentence at the highest-diversity gate; combined with CUT-COR-04 (mandatory batching). CUT-COR-04 risk is `med`: the repeat of CUT-COR-03's jargon — risk is the same shape but lower magnitude because the CASES loop between the two sites blunts any independent semantic drift. CUT-COR-05 risk is `high`: the only row in COR that carries `behavioral-protected` trigger; Phase 46 MUST treat the row as a tombstone, not a candidate.

> **COR Phase 46 implication.** COR is gated on 40 snapshots — the highest-diversity section in the audit (more than 1.6× VAL's 24, more than 2× any other section). Phase 46 ordering for this section: (CUT-COR-03 + CUT-COR-04) batched as one rewrite of the line 354 + 358–360 jargon, batched in turn with VAL CUT-VAL-03 (`role-referential phrase` shared lexicon) → CUT-COR-01 (med-high, `role-referential noun phrase` in COREF_RULES — same vocabulary, same batch) → CUT-COR-02 (med, `section-established topic` standalone) → CUT-COR-05 (DO NOT EXECUTE, tombstone). Four of the five cuts emit `after = [Phase 46 empirical loop]` per D-05 (no in-Phase-45 rewordings for `domain-loaded`); the fifth is a tombstone. The 4-cut COR jargon batch + 1-cut VAL CUT-VAL-03 is the second-largest coordinated Phase-46 batching opportunity in the audit after the 3-cut `software architecture` opener batch (AMB CUT-AMB-02 + EXT CUT-EXT-01 + VAL CUT-VAL-02).

> **COR closing reviewer paragraph.**
> COR is the most-diverse-snapshot section in the audit (40 snapshots) — wording changes here get the broadest empirical pressure, so any rewording that survives byte-equality in COR is strong evidence the change is semantically equivalent for the model. The multiple `anaphoric` / `role-referential` jargon spans (CUT-COR-01, CUT-COR-03, CUT-COR-04) appear together across COREF_RULES and `_prompt_coref` prose and must be tested as a coordinated rewrite, not piecewise — Phase 46 must batch them with VAL CUT-VAL-03 since the same `role-referential` lexicon is shared. The line 361 conservatism instruction (CUT-COR-05) is behavioral per §7.4 and tombstoned — Phase 46 MUST preserve it; removing it risks reintroducing the FP class that v2.6.2 s17e documents as fixed (FP 43→14 via validation gating). The ALIAS_SCOPE_RULES back-reference at the top of the section blocks Phase 46 from issuing duplicate cut_ids — its canonical audit row lives in DKX (45-03), and `_prompt_coref` does not import it.

<!-- SECTION:COR:END -->

## Phase Close Notes

<!-- FINAL:SUMMARY:START -->

### REQ-V264-03 Coverage Tick-Off (9 PROMPT CONSTANTS, in REQUIREMENTS.md order)

- [x] REQ-V264-03 / AMBIGUITY_FEW_SHOT — section AMB, verdict `clean`, 1 cut row (CUT-AMB-01 drop-block per REQ-V264-06)
- [x] REQ-V264-03 / AMBIGUITY_RULES — section AMB, verdict `clean`, 0 cut rows
- [x] REQ-V264-03 / DOC_KNOWLEDGE_EXTRACTION_RULES — section DKX, verdict `clean`, 0 cut rows
- [x] REQ-V264-03 / ALIAS_SCOPE_RULES — section DKX (canonical row; back-reference from COR per 45-RESEARCH.md §6.1), verdict `clean`, 0 cut rows
- [x] REQ-V264-03 / DOC_KNOWLEDGE_JUDGE_EXAMPLES — section DKJ, verdict `benchmark-leak`, 6 cut rows (CUT-DKJ-01 drop-block + CUT-DKJ-02/03/04 Family A + CUT-DKJ-05/06 Family B)
- [x] REQ-V264-03 / DOC_KNOWLEDGE_JUDGE_RULES — section DKJ, verdict `domain-loaded`, 1 cut row (CUT-DKJ-07)
- [x] REQ-V264-03 / ENTITY_EXTRACTION_RULES — section EXT, verdict `clean`, 0 cut rows
- [x] REQ-V264-03 / VALIDATION_RULES — section VAL, verdict `domain-loaded`, 1 cut row (CUT-VAL-01)
- [x] REQ-V264-03 / COREF_RULES — section COR, verdict `domain-loaded`, 2 cut rows (CUT-COR-01, CUT-COR-02)

### REQ-V264-04 Coverage Tick-Off (6 in-class f-string scaffold builders)

- [x] REQ-V264-04 / _prompt_ambiguity — section AMB, verdict `domain-loaded`, 1 cut row (CUT-AMB-02)
- [x] REQ-V264-04 / _prompt_doc_knowledge_extract — section DKX, verdict `clean`, 0 cut rows
- [x] REQ-V264-04 / _prompt_doc_knowledge_judge — section DKJ, verdict `clean`, 0 cut rows
- [x] REQ-V264-04 / _prompt_extraction — section EXT, verdict `domain-loaded`, 1 cut row (CUT-EXT-01)
- [x] REQ-V264-04 / _prompt_validation — section VAL, verdict `domain-loaded`, 1 cut row (CUT-VAL-02)
- [x] REQ-V264-04 / _prompt_coref — section COR, verdict `domain-loaded`, 3 cut rows (CUT-COR-03, CUT-COR-04, CUT-COR-05 behavioral-protected tombstone)

### CD-6 Fold-In Coverage Tick-Off (informational; not enumerated in REQ-V264-03/04)

- [x] CD-6 / P1_FOCUS — folded into section VAL, verdict `behavioral-protected` (qualified-name X.Y.Z clause), 1 cut row (CUT-VAL-04 — DO NOT CUT tombstone)
- [x] CD-6 / P2_FOCUS — folded into section VAL, verdict `clean`, 0 cut rows
- [x] CD-6 / COREF_VALIDATION_FOCUS — folded into section VAL, verdict `domain-loaded` ("role-referential phrase"), 1 cut row (CUT-VAL-03)

### Cross-Check Verifications

- [x] **D-06: every `benchmark-leak` verdict has at least one Family A AND one Family B rewording row.**
  Evidence: DKJ is the only section with a `benchmark-leak` verdict (`DOC_KNOWLEDGE_JUDGE_EXAMPLES`, line 152). Family A rows present: CUT-DKJ-02, CUT-DKJ-03, CUT-DKJ-04 (3 synthetic-neutral swap variants). Family B rows present: CUT-DKJ-05, CUT-DKJ-06 (2 concept-only variants). Both families ≥1 row. CUT-DKJ-01 is the drop-block row (REQ-V264-06) and CUT-DKJ-07 covers the sibling `DOC_KNOWLEDGE_JUDGE_RULES` domain-loaded clause. Confirmed.
- [x] **D-05: no `after` rewording text for any `domain-loaded` row (each domain-loaded `after` = `[Phase 46 empirical loop]`).**
  Evidence: grep over the audit doc for `\| domain-loaded ` cut rows → 9 hits: CUT-AMB-02, CUT-DKJ-07, CUT-EXT-01, CUT-VAL-01, CUT-VAL-02, CUT-VAL-03, CUT-COR-01, CUT-COR-02, CUT-COR-03, CUT-COR-04. Each row's `after` cell literally reads `[Phase 46 empirical loop]`. Confirmed.
- [x] **D-07: every cut row has non-empty `gated_by`, `risk`, and risk-justification cells.**
  Evidence: 19 cut rows total (AMB:2 + DKX:0 + DKJ:7 + EXT:1 + VAL:4 + COR:5). Per-row scan: every `risk` cell carries a tier (`low` / `low-med` / `med` / `med-high` / `high`) PLUS an inline justification after the em-dash (the schema collapses risk and justification into one cell per D-07/D-08); every `gated_by` cell lists the test module path + phase tag(s). Confirmed.
- [x] **REQ-V264-06 drop-block applied to `AMBIGUITY_FEW_SHOT` (CUT-AMB-01) and `DOC_KNOWLEDGE_JUDGE_EXAMPLES` (CUT-DKJ-01).**
  Evidence: CUT-AMB-01 trigger is `drop-block (REQ-V264-06, not benchmark-leak)`, `after = ""`. CUT-DKJ-01 trigger is `benchmark-leak (drop-block, REQ-V264-06)`, `after = ""`. Both rows are the FIRST cut row of their section's cut table per the Drop-block convention recorded in §"Cut ID Scheme". Confirmed.

### ROADMAP.md Phase 45 Success Criteria Tick-Off

- [x] **SC1:** `s_linker20-PROMPT-AUDIT.md` exists at `.planning/phases/45-audit/s_linker20-PROMPT-AUDIT.md` and covers all 9 imported PROMPT CONSTANTS with LOC + verdict + line-level cut candidates per §Verdict Summary above (rows 1–4 cover AMB+DKX constants; rows 7–8 cover DKJ constants; rows 10, 12, 16 cover EXT/VAL/COR constants).
- [x] **SC2:** All 6 in-class f-string scaffolds (`_prompt_ambiguity`, `_prompt_doc_knowledge_extract`, `_prompt_doc_knowledge_judge`, `_prompt_extraction`, `_prompt_validation`, `_prompt_coref`) covered with same columns (rows 3, 6, 9, 11, 13, 17 of §Verdict Summary).
- [x] **SC3:** Every `benchmark-leak` finding has a proposed neutral rewording included. The only `benchmark-leak` verdict in the doc is `DOC_KNOWLEDGE_JUDGE_EXAMPLES` (section DKJ); it carries Family A (CUT-DKJ-02/03/04) and Family B (CUT-DKJ-05/06) rewordings inline with per-cut detail blocks giving the full proposed rewrite text.
- [x] **SC4:** Zero code changes to `s_linker19`, `s_linker13_min`, or any imported prompt module — verified empirically in the FINAL:GATE01 anchor below (`git diff --stat` returns empty, exit code 0).

### Section Verdict Tally (post Wave-2)

| Section | Items | clean | domain-loaded | benchmark-leak | behavioral-protected | Cut rows |
|---|---|---|---|---|---|---|
| AMB | 3 | 2 | 1 | 0 | 0 | 2 |
| DKX | 3 | 3 | 0 | 0 | 0 | 0 |
| DKJ | 3 | 1 | 1 | 1 | 0 | 7 |
| EXT | 2 | 1 | 1 | 0 | 0 | 1 |
| VAL (incl. 3 CD-6 fold-ins) | 5 | 1 | 3 | 0 | 1 | 4 |
| COR | 2 | 0 | 2 | 0 | 0 | 5 |
| **Total** | **18** | **8** | **8** | **1** | **1** | **19** |

<!-- FINAL:SUMMARY:END -->

<!-- FINAL:GATE01:START -->

## GATE-01 Byte-Equal Verification

**Date:** 2026-06-08
**Command:**
```
git diff --stat src/llm_sad_sam/linkers/experimental/s_linker19.py src/llm_sad_sam/linkers/experimental/prompts_v5.py src/llm_sad_sam/linkers/experimental/s_linker13_min.py
```
**Exit code:** 0
**Output:** (empty)
**Verdict:** PASS
**Phase 45 close requires:** PASS (empty git-diff). Any non-empty diff is a phase failure.

GATE-01 source files (`s_linker19.py`, `prompts_v5.py`, `s_linker13_min.py`) are byte-equal vs HEAD at phase close. No code edits occurred during Phase 45 — the audit is a read-only planning artefact per the phase scope boundary in 45-CONTEXT.md §<domain>.

<!-- FINAL:GATE01:END -->
