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
| `DOC_KNOWLEDGE_EXTRACTION_RULES` | constant | 1 | TBD | TBD | 45-03 (DKX) |
| `ALIAS_SCOPE_RULES` | constant | 4 | TBD | TBD | 45-03 (DKX) |
| `_prompt_doc_knowledge_extract` | builder | 19 | TBD | TBD | 45-03 (DKX) |
| `DOC_KNOWLEDGE_JUDGE_EXAMPLES` | constant | 7 | TBD | TBD | 45-04 (DKJ) |
| `DOC_KNOWLEDGE_JUDGE_RULES` | constant | 1 | TBD | TBD | 45-04 (DKJ) |
| `_prompt_doc_knowledge_judge` | builder | 16 | TBD | TBD | 45-04 (DKJ) |
| `ENTITY_EXTRACTION_RULES` | constant | 1 | TBD | TBD | 45-05 (EXT) |
| `_prompt_extraction` | builder | 15 | TBD | TBD | 45-05 (EXT) |
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
<!-- TBD: filled by .planning/phases/45-audit/45-03-PLAN.md (Wave 1) — header table for DOC_KNOWLEDGE_EXTRACTION_RULES + ALIAS_SCOPE_RULES + _prompt_doc_knowledge_extract, then cut table CUT-DKX-NN. -->
<!-- SECTION:DKX:END -->

## Phase 1 — Doc-Knowledge Judge (DKJ)

<!-- SECTION:DKJ:START -->
<!-- TBD: filled by .planning/phases/45-audit/45-04-PLAN.md (Wave 1) — header table for DOC_KNOWLEDGE_JUDGE_EXAMPLES + DOC_KNOWLEDGE_JUDGE_RULES + _prompt_doc_knowledge_judge, then cut table CUT-DKJ-NN. -->
<!-- SECTION:DKJ:END -->

## Phase 2 — Extraction (EXT)

<!-- SECTION:EXT:START -->
<!-- TBD: filled by .planning/phases/45-audit/45-05-PLAN.md (Wave 1) — header table for ENTITY_EXTRACTION_RULES + _prompt_extraction, then cut table CUT-EXT-NN. -->
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
