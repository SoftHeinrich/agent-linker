---
phase: 46-minimize
artifact_type: minimize-log
milestone: v2.6.4
produced_by: Phase 46 (MINIMIZE)
consumed_by: Phase 47 (SHIP) — kept-cut after-text inlined from tests/scratch/
gate_status: GATE-01 byte-equal — re-verified per per-cut commit, finalised at phase close (46-08-PLAN)
---

# s_linker20 Minimize Log (v2.6.4 Phase 46)

## Schema header

Every row carries `| cut_id | verdict | snapshot_delta | gate06_isolation | loc_saved | commit_sha | reasoning |` per CONTEXT D-04 and 46-RESEARCH §7.5. Verdicts are one of `kept`, `reverted`, `unsafe`, `protected`, `superseded-by-drop`, `superseded-by-A`, `superseded-by-B`, `kept-original`. One atomic commit is produced per row per D-04 — superseded rows fold into their parent commit's SHA. The `cut_id` column is the foreign key into `.planning/phases/45-audit/s_linker20-PROMPT-AUDIT.md`; every MINIMIZE-LOG row uses an audit `CUT-{TAG}-NN` identifier.

Under `SAD_SAM_LINKER_SOURCE=scratch` the harness adapter swaps the `SLinker19` import to `tests.scratch.s_linker19` (which itself binds `tests.scratch.prompts_v5`), and the step-6 prompt-equality assertion is gated off in all six `tests/test_s_linker20_prompt_*.py` modules (per 46-RESEARCH §2.3). The meaningful signals in scratch-mode collapse to (i) the harness import succeeds (no `ImportError`, no `RuntimeError` from the toggle), (ii) all 97 parsed-output snapshots still pass, and (iii) the GATE-06 re-grep on the after-text is clean against `BENCHMARK_TABOO.md`. Standing caveat: parsed-output snapshots are invariant under prompt cuts because replay parsing depends only on cached `response_text` — Phase 46 verdicts measure harness compatibility + vocabulary cleanliness, NOT model behavior. Phase 48 sweep measures behavior.

## Cross-section pleonasm-batch decision

The recurring `software architecture …` opener pleonasm (audit rows CUT-AMB-02, CUT-EXT-01, CUT-VAL-02) is targeted by plans 46-02 / 46-05 / 46-06 against a SINGLE pre-decided replacement vocabulary: **`components` bare** (collapses to the noun the `COMPONENTS:` slot already names downstream). For CUT-VAL-02 specifically the full replacement opener is `Validate components in a document. {focus}` — wired through `reconstruct_validation_inputs` (`tests/harness/inputs.py`) by plan 46-01 via the `ACCEPTED_PREFIXES` tuple so cuts to scratch don't trivially break the harness. All three cuts get separate commits per D-04; the Pareto Summary (46-08) cross-references them as one conceptual batch.

## VAL-03 ↔ COR-01 shared-lexicon decision

Audit rows CUT-VAL-03 and CUT-COR-01 both target the `role-referential phrase` / `role-referential noun phrase` jargon. Plan 46-06 trials CUT-VAL-03 first, picks the replacement vocabulary, and writes it to its MINIMIZE-LOG row (`reasoning` cell). Plan 46-07 reads that row and uses the same vocabulary for CUT-COR-01 so the two cuts stay lexically aligned. Recommended target vocabulary (per 46-RESEARCH §9 Q4): **`noun phrase that refers back`** — but the empirical trial outcome in 46-06 is authoritative; 46-07 follows 46-06's choice, not the recommendation, if they diverge.

## CUT-DKJ-07 vocabulary

Audit row CUT-DKJ-07 targets the `compound concept` / `multi-element grouping` jargon used in the doc-knowledge-judge examples block. The pre-decided replacement vocabulary is **`grouping that encompasses multiple elements`** (lower-jargon paraphrase). Plan 46-04 records the empirical trial outcome and writes the chosen vocabulary into its MINIMIZE-LOG row; downstream Phase 47 inlining reads that row, not this header.

## Scratch-mode bootstrap note

The import-line rewrite in `tests/scratch/s_linker19.py` (`from llm_sad_sam.linkers.experimental.prompts_v5 import` → `from tests.scratch.prompts_v5 import`) applied by plan 46-01 is a one-time wiring change, NOT a cut row. The MINIMIZE-LOG has 19 rows total at phase close (17 trial + 2 protected) — none for the import rewrite. Phase 46 cuts mutate `tests/scratch/{s_linker19.py, prompts_v5.py}` on top of this wiring change; the wiring change itself is permanent for the duration of Phase 46.

## Verdict vocabulary

| Verdict | Meaning | Commit type |
|---|---|---|
| kept | Cut applied to scratch; harness passes; GATE-06 clean | feat(46-NN): keep CUT-... — ... |
| reverted | Cut applied; harness failed (≥1 snapshot crashed/diverged); scratch rolled back | chore(46-NN): revert CUT-... — ... |
| unsafe | Cut applied; harness passes; GATE-06 re-grep hit benchmark vocabulary; scratch rolled back | chore(46-NN): revert CUT-... — unsafe: taboo:{section}:{term} |
| protected | Tombstone — not trialled (CUT-VAL-04, CUT-COR-05) | docs(46-NN): protect CUT-... — ... |
| superseded-by-drop | Family A/B row moot because parent drop-block passed | folds into parent commit |
| superseded-by-A | Family B row (or Family A peer) moot because a Family A row passed | folds into parent commit |
| superseded-by-B | Family A/B peers moot because this Family B row passed | folds into parent commit |
| kept-original | Drop, all Family A, all Family B rows failed; block preserved verbatim | chore(46-NN): kept-original CUT-... — no replacement passed |

## Pareto Summary

<!-- FINAL:PARETO:START -->
<!-- TBD: filled by .planning/phases/46-minimize/46-08-PLAN.md (Wave 3) — totals (kept/reverted/unsafe/protected), per-section LOC totals, drop-block smallest-passing identifiers (CUT-AMB-01, CUT-DKJ-01), cross-section pleonasm batch cross-references (CUT-AMB-02 + CUT-EXT-01 + CUT-VAL-02), VAL-03 ↔ COR-01 lexical-share note. -->
<!-- FINAL:PARETO:END -->

## AMB — Phase 1 Ambiguity

<!-- SECTION:AMB:START -->

| cut_id | verdict | snapshot_delta | gate06_isolation | loc_saved | commit_sha | reasoning |
|---|---|---|---|---|---|---|
| CUT-AMB-02 | kept | 0/5 | clean | 0 | 0710510 | Pleonasm `Classify these software architecture component names.` → `Classify these component names.` at `tests/scratch/s_linker19.py:274` (`_prompt_ambiguity` opener). Pre-decided batch vocab (46-01) is `components` bare. `reconstruct_ambiguity_inputs` anchors on `^NAMES:` so opener change is harness-safe; 5/5 snapshots passed under `SAD_SAM_LINKER_SOURCE=scratch`. GATE-06 grep on `classify\|these\|component\|names` against `BENCHMARK_TABOO.md`: only hits are bare `component` as generic SE noun anaphor in per-dataset sections (e.g. "Teammates component"), cleared per Phase 45 v2.1 isolation precedent (CUT-VAL-03 / CUT-COR-01); `classify`, `these`, `names` have zero per-dataset hits. Cross-section batch member: 1 of 3 with CUT-EXT-01 (46-05) and CUT-VAL-02 (46-06), all targeting the recurring `software architecture …` opener pleonasm with the same shared replacement vocabulary. |
| CUT-AMB-01 | kept | 0/5 | clean (no after-text) | 7 | dfad56a | DROP-BLOCK passed: `AMBIGUITY_FEW_SHOT` body in `tests/scratch/prompts_v5.py:30-36` replaced with `""` (drop-by-empty per 46-RESEARCH §9 Q9 (a)). Constant binding preserved so the scratch `s_linker19.py` `from tests.scratch.prompts_v5 import (..., AMBIGUITY_FEW_SHOT, ...)` import still resolves. `reconstruct_ambiguity_inputs` anchors on `^NAMES:` and `NOW CLASSIFY THE NAMES ABOVE.` — both still present in the post-cut prompt — so 5/5 snapshots pass under `SAD_SAM_LINKER_SOURCE=scratch` (snapshot_delta = 0/5). After-text is the empty string ⇒ GATE-06 trivially clean (no tokens to grep). LOC saved = 7 (the multi-line triple-quoted body). Behavioral caveat per 46-RESEARCH §4.4: drop verdict reflects harness compatibility only; behavioral effect on judge calibration (zero-shot vs few-shot) is NOT observable in this phase. Phase 48 sweep validates behavior. |
<!-- SECTION:AMB:END -->

> **AMB section closing note (46-02).** Smallest-passing identifier for `AMBIGUITY_FEW_SHOT` is **`drop`** (drop-by-empty per 46-RESEARCH §9 Q9 (a)); no Family A / Family B rewordings were emitted at audit time per D-06 (AMB verdict is `clean`, not `benchmark-leak`), so no further trial branches exist for this block. Total LOC saved in AMB section: **7** (CUT-AMB-02 = 0 LOC + CUT-AMB-01 = 7 LOC). CUT-AMB-02 closes 1 of 3 sites in the cross-section pleonasm batch (`software architecture …` opener); CUT-EXT-01 (46-05) and CUT-VAL-02 (46-06) close the remaining 2 sites with the same `components` bare replacement vocabulary set by 46-01.

## DKX — Phase 1 Doc-Knowledge Extract

<!-- SECTION:DKX:START -->

| cut_id | verdict | snapshot_delta | gate06_isolation | loc_saved | commit_sha | reasoning |
|---|---|---|---|---|---|---|
| (none) | no-cuts-attempted | n/a | n/a | 0 | 27bc025 | Phase 45 audit assigned `clean` verdict to all 3 DKX items; no benchmark-leak or domain-loaded findings to trial. |

> **DKX section: 0 cut rows.** All 3 DKX items — `DOC_KNOWLEDGE_EXTRACTION_RULES` (prompts_v5.py:40),
> `ALIAS_SCOPE_RULES` (prompts_v5.py:42-45), and `_prompt_doc_knowledge_extract` (s_linker19.py:294-310 prose) —
> received verdict `clean` at audit time per `.planning/phases/45-audit/45-03-SUMMARY.md` (no benchmark-leak tokens
> across all 5 dataset sections + Universal Taboo + Safe SE Textbook; no domain-loaded spans flagged after D-01
> pragmatic review). The audit emitted zero cut rows for the section per D-05 (`domain-loaded` rewordings are
> deferred to Phase 46 only when an audit row exists; no row, no cut). Phase 46 therefore attempts zero trials
> on DKX and logs this single completeness row to preserve section symmetry AMB → DKX → DKJ → EXT → VAL → COR
> per 46-RESEARCH §7.1. Phase 47 (SHIP) needs no DKX inlining beyond what was byte-equal at phase open.

<!-- SECTION:DKX:END -->

## DKJ — Phase 1 Doc-Knowledge Judge

<!-- SECTION:DKJ:START -->

| cut_id | verdict | snapshot_delta | gate06_isolation | loc_saved | commit_sha | reasoning |
|---|---|---|---|---|---|---|
| CUT-DKJ-01 | kept | 0/5 | clean (no after-text) | 7 | 74ec3bd | DROP-BLOCK passed: full `DOC_KNOWLEDGE_JUDGE_EXAMPLES` body in `tests/scratch/prompts_v5.py:41-47` replaced with `""` (drop-by-empty per 46-RESEARCH §9 Q9 (a)). Constant binding preserved so the scratch `s_linker19.py:109` `from tests.scratch.prompts_v5 import (..., DOC_KNOWLEDGE_JUDGE_EXAMPLES, ...)` import still resolves. `reconstruct_doc_judge_inputs` terminates on the blank line after `PROPOSED MAPPINGS` — terminator preserved under empty examples body. 5/5 snapshots pass under `SAD_SAM_LINKER_SOURCE=scratch` (snapshot_delta = 0/5). After-text is the empty string ⇒ GATE-06 trivially clean (no tokens to grep). LOC saved = 7 (the 7-line triple-quoted body collapses to `""`). **BEHAVIORAL CAVEAT (46-RESEARCH §4.4 + §6.3):** harness verdict reflects harness compatibility only; the few-shot drives judge calibration (VALID/INVALID rationale shape) and removal may shift model behavior on real LLM calls. Phase 48 sweep validates behavioral safety. This was the only `benchmark-leak` section in the entire audit (45-04-SUMMARY: `CacheLayer` was the sole confirmed body-text Universal Taboo hit via the `cache` substring) — drop directly removes the leak. |
| CUT-DKJ-02 | superseded-by-drop | n/a | n/a | 0 | 74ec3bd | Family A Example-1 swap (`RequestHandler`→`BookManager`, `Handler`→`Mgr`) moot — drop-block CUT-DKJ-01 passed under D-03; the entire examples body is now empty, so Family A name swap has no surface to apply to. |
| CUT-DKJ-03 | superseded-by-drop | n/a | n/a | 0 | 74ec3bd | Family A Example-2 swap (`CacheLayer`→`MailSender`, primary leak removal) moot — drop-block CUT-DKJ-01 passed under D-03; the `CacheLayer` leak is removed by the drop itself rather than by the name swap. |
| CUT-DKJ-04 | superseded-by-drop | n/a | n/a | 0 | 74ec3bd | Family A combined-rewrite (both examples in a single coherent synthetic mail/catalog domain) moot — drop-block CUT-DKJ-01 passed under D-03; no examples body remains. |
| CUT-DKJ-05 | superseded-by-drop | n/a | n/a | 0 | 74ec3bd | Family B Example-1 concept-only (name-stripped abstract parenthetical-definition rule) moot — drop-block CUT-DKJ-01 passed under D-03; no examples body remains. |
| CUT-DKJ-06 | superseded-by-drop | n/a | n/a | 0 | 74ec3bd | Family B Example-2 concept-only (name-stripped abstract whole-system overshoot rule) moot — drop-block CUT-DKJ-01 passed under D-03; no examples body remains. |
| CUT-DKJ-07 | kept | 0/5 | clean | 0 | 8a83bda | `DOC_KNOWLEDGE_JUDGE_RULES` (`tests/scratch/prompts_v5.py:43`) clause `architectural tier or technology platform` -> `grouping` per 46-RESEARCH §9 Q5 audit-suggested vocabulary. After-text: `An alias is also invalid when it names a grouping that encompasses multiple elements, because it identifies a grouping rather than a single named unit.` Multi-element exclusion semantics preserved (`that encompasses multiple elements` stays verbatim; trailing `because it identifies a grouping rather than a single named unit` clause unchanged). `reconstruct_doc_judge_inputs` anchors on `PROPOSED MAPPINGS` terminator — unaffected by a substring rewording inside the rules string. 5/5 snapshots pass under `SAD_SAM_LINKER_SOURCE=scratch`. GATE-06 re-grep: `grep -niwE 'grouping\|encompasses\|elements\|invalid\|names' BENCHMARK_TABOO.md` -> 0 hits across all per-dataset KEYWORDS sections; `alias` shows the standing meta-references to the linker `alias` mechanism (BENCHMARK_TABOO lines 24, 55-58 — `alias` is the linker-architectural noun, cleared per v2.1 isolation precedent). LOC saved = 0 (substring rewording within a single dense line; no whole-line removal). Separate commit from the drop-block parent per D-04 (CUT-DKJ-07 is NOT part of the drop-block tree — sibling constant). |

<!-- SECTION:DKJ:END -->

> **DKJ section closing note (46-04).** Smallest-passing identifier for `DOC_KNOWLEDGE_JUDGE_EXAMPLES` is **`drop`** (drop-by-empty per 46-RESEARCH §9 Q9 (a)). The DROP trial passed harness compatibility on first attempt — Family A (CUT-DKJ-02/03/04 synthetic-neutral name swaps using the 45-04 pre-cleared `BookManager`/`Mgr`/`MailSender` evidence) and Family B (CUT-DKJ-05/06 concept-only rewrites) were never trialled; all five log as `superseded-by-drop` per D-03. CUT-DKJ-07 (a separate-from-drop-block §3-loop cut targeting `DOC_KNOWLEDGE_JUDGE_RULES` `architectural tier or technology platform` clause) trialled independently and `kept` per the pre-decided `grouping that encompasses multiple elements` vocabulary set in the MINIMIZE-LOG header. Total LOC saved in DKJ section: **7** (CUT-DKJ-01 = 7 LOC for the dropped examples body + CUT-DKJ-07 = 0 LOC for the substring rewording). **BEHAVIORAL CAVEAT (46-RESEARCH §4.4 + §6.3):** the DKJ section was the only `benchmark-leak` finding in the entire audit (45-04-SUMMARY: `CacheLayer` -> `cache` substring was the sole confirmed Universal Taboo hit); both kept cuts pass byte-equal under cached-replay scratch mode, which validates harness compatibility only. The few-shot examples drive judge calibration (VALID/INVALID rationale shape) on real LLM calls — Phase 48 sweep validates behavioral safety. CUT-DKJ-07 commits separately from the drop-block parent per D-04 (sibling constant — not part of the EXAMPLES drop-block tree).

## EXT — Phase 2 Extraction

<!-- SECTION:EXT:START -->

| cut_id | verdict | snapshot_delta | gate06_isolation | loc_saved | commit_sha | reasoning |
|---|---|---|---|---|---|---|
| CUT-EXT-01 | kept | 0/18 | clean | 0 | fbfbcb9 | Pleonasm `Extract ALL references to software architecture components from this document.` -> `Extract ALL references to components from this document.` at `tests/scratch/s_linker19.py:331` (`_prompt_extraction` opener; drifted from audit-time line 323 because upstream Wave-2 plan 46-02 mutated nearby AMB lines). Pre-decided batch vocab (46-01 MINIMIZE-LOG header) is `components` bare — collapses to the noun the `COMPONENTS:` slot already names downstream. `reconstruct_extraction_inputs` (`tests/harness/inputs.py`) anchors on `^COMPONENTS:` and `\nDOCUMENT:\n` — opener change is harness-safe per 46-RESEARCH §6.2 (kept HIGH confidence). 18/18 snapshots pass under `SAD_SAM_LINKER_SOURCE=scratch` (largest single-cut gating in the audit: `phase_2_framing_c_pass1` + `phase_2_framing_c_pass2` × 5 projects + extra parametrize axes = 18 collected). GATE-06 re-grep `grep -niwE 'extract\|references\|components\|document' BENCHMARK_TABOO.md`: `extract`/`references` 0 hits; `components` 5 hits all in per-dataset `Components:` schema-section column headers (lines 7/12/17/22/27 — generic SE noun anaphor cleared per Phase 45 v2.1 isolation precedent / CUT-AMB-02 reasoning); `document` 1 hit at line 100 in methodology prose (`document the inspection`, not dataset vocab). LOC saved = 0 (substring rewording within a single line). Cross-section batch member: 2 of 3 — paired with CUT-AMB-02 (kept, 46-02, sha `0710510`) and CUT-VAL-02 (pending in 46-06); all three share the `components` bare replacement vocabulary. Behavioral caveat per 46-RESEARCH §4.4: harness verdict reflects harness compatibility only (cached-replay); Phase 48 sweep validates behavioral safety on live LLM calls. |

> **EXT section closing note (46-05).** Total LOC saved in EXT section: **0** (CUT-EXT-01 is a substring rewording within a single line — no whole-line removal). EXT contained exactly 1 trial cut per the Phase 45 audit (CUT-EXT-01 is the sole EXT row), so ordering within the section was trivial under D-02. CUT-EXT-01 closes 2 of 3 sites in the cross-section pleonasm batch (`software architecture …` opener) — upstream sibling CUT-AMB-02 closed site 1 of 3 in plan 46-02 (sha `0710510`); the remaining VAL site is pending in plan 46-06 (CUT-VAL-02). All three batch members share the `components` bare replacement vocabulary pre-decided in the 46-01 MINIMIZE-LOG header. The Pareto Summary (46-08, Wave 3) will roll the three commits together as one conceptual batch with a combined LOC-saved and cross-reference footer.

<!-- SECTION:EXT:END -->

## VAL — Phase 4 Validation

<!-- SECTION:VAL:START -->

| cut_id | verdict | snapshot_delta | gate06_isolation | loc_saved | commit_sha | reasoning |
|---|---|---|---|---|---|---|
| CUT-VAL-02 | kept | 0/24 | clean | 0 | d82e5a9 | Pleonasm `Validate component references in a software architecture document. {focus}` -> `Validate components in a document. {focus}` at `tests/scratch/s_linker19.py:347` (`_prompt_validation` opener; drifted from audit-time line 339 because upstream Wave-2 plans 46-02/46-04/46-05 mutated nearby lines). Pre-decided batch vocab (46-01 MINIMIZE-LOG header) is `components` bare. `reconstruct_validation_inputs` (`tests/harness/inputs.py:279-291`) consumes the new opener via the `ACCEPTED_PREFIXES` tuple pre-wired by 46-01 (entry 2: `Validate components in a document.`) so the harness accepts both pre/post opener variants. 24/24 snapshots passed under `SAD_SAM_LINKER_SOURCE=scratch` (phase_4_twopass_p1 + phase_4_twopass_p2 + phase_5_coref_validation — the most conservative gating in the audit per Phase 45 §VAL). GATE-06 re-grep `grep -niwE 'validate\|components\|document' BENCHMARK_TABOO.md`: `validate` 0 hits; `components` 5 hits, all in per-dataset `Components:` schema-section column headers (lines 7/12/17/22/27 — generic SE noun anaphor cleared per Phase 45 v2.1 isolation precedent / CUT-AMB-02 + CUT-EXT-01 reasoning); `document` 1 hit at line 100 methodology prose (`document the inspection`, not dataset vocab). LOC saved = 0 (substring rewording within a single line). Cross-section batch member: **3 of 3** — paired with CUT-AMB-02 (kept, 46-02, sha `0710510`) and CUT-EXT-01 (kept, 46-05, sha `fbfbcb9`); all three share the `components` bare replacement vocabulary. Batch closes the recurring `software architecture …` opener pleonasm across AMB/EXT/VAL. Behavioral caveat per 46-RESEARCH §4.4: harness verdict reflects harness compatibility only (cached-replay); Phase 48 sweep validates behavioral safety on live LLM calls. |
| CUT-VAL-01 | kept | 0/24 | clean | 0 | 5118c32 | Domain-loaded noun swap in `VALIDATION_RULES` (`tests/scratch/prompts_v5.py:82`): `including counterparts.` -> `including matching entities.` per 46-RESEARCH §6.2 universal-noun mapping (`counterparts` is the single domain-loaded noun; remainder of `VALIDATION_RULES` preserved verbatim). After-text: `Approve when the sentence treats the component as an architectural participant, including matching entities. Reject when the matching word is generic, names a different entity, or describes a technique that merely shares the component's name.` `VALIDATION_RULES` is body content (not opener), so `reconstruct_validation_inputs` is unaffected by the rewording — confirmed by 24/24 snapshot pass under `SAD_SAM_LINKER_SOURCE=scratch` (phase_4_twopass_p1 + phase_4_twopass_p2 + phase_5_coref_validation; the most conservative gating in the audit). GATE-06 re-grep `grep -niwE 'approve\|sentence\|treats\|component\|architectural\|participant\|including\|matching\|entities' BENCHMARK_TABOO.md`: only hits are bare `component` as generic SE noun anaphor across per-dataset Aliases / Ambiguity sections + meta methodology prose (lines 18/32-37/47/50/53-54/77/83-90/95 — generic SE noun anaphor cleared per Phase 45 v2.1 isolation precedent / CUT-AMB-02 reasoning); `approve`/`sentence`/`treats`/`architectural`/`participant`/`including`/`matching`/`entities` all 0 hits. LOC saved = 0 (single-noun substring rewording within a single dense line). Behavioral caveat per 46-RESEARCH §4.4: harness verdict reflects harness compatibility only (cached-replay); Phase 48 sweep validates behavioral safety on live LLM calls. |
| CUT-VAL-03 | kept | 0/24 | clean | 0 | 8c195bc | Domain-loaded jargon swap in `COREF_VALIDATION_FOCUS` (`tests/scratch/prompts_v5.py:94-100`): `or similar role-referential phrase in this sentence actually refer to ` -> `or similar noun phrase that refers back in this sentence actually refer to ` per 46-RESEARCH §9 Q4 audit-suggested vocabulary. After-text (full COREF_VALIDATION_FOCUS): `Check coref resolution: does the pronoun, 'it', 'they', 'the service', or similar noun phrase that refers back in this sentence actually refer to the named component as an architectural participant — performing operations, providing services, or being the grammatical topic of the sentence?` Syntactic agreement preserved (plural `refer` agrees with the disjunction subject "the pronoun, 'it', 'they', 'the service', or similar noun phrase…"). The asymmetric single-pass design (per Phase 45 audit's `COREF_VALIDATION_FOCUS asymmetric-design record`) is untouched — only the lexical `role-referential phrase` span is reworded; the empirically load-bearing narrower focus (cleanup E experiment, ~4 FP reduction on bigbluebutton coref per prompts_v5.py:90-93 docstring) stays intact. `COREF_VALIDATION_FOCUS` is body content read from prompt body (not opener), so `reconstruct_validation_inputs` is unaffected — confirmed by 24/24 snapshot pass under `SAD_SAM_LINKER_SOURCE=scratch` (phase_4_twopass_p1 + phase_4_twopass_p2 + phase_5_coref_validation; the 5 of 24 snapshots driven by `phase_5_coref_validation` exercise this constant). GATE-06 re-grep `grep -niwE 'noun\|phrase\|refers\|back\|coref\|resolution\|pronoun\|sentence\|named\|component' BENCHMARK_TABOO.md`: `noun`/`phrase`/`refers`/`back`/`coref`/`resolution`/`pronoun`/`sentence`/`named` all 0 hits; `component` hits are bare generic SE noun anaphor (cleared per Phase 45 v2.1 isolation precedent / CUT-AMB-02 reasoning). LOC saved = 0 (substring rewording within a single dense multi-line literal). **VAL-03 → COR-01 lexicon handoff (integration contract for plan 46-07):** Replacement vocabulary `noun phrase that refers back` chosen and committed in this row; plan 46-07's CUT-COR-01 reads this row and applies the SAME string to the `role-referential noun phrase` span in `COREF_RULES` (`tests/scratch/prompts_v5.py:102`) so the two cuts stay lexically aligned across COREF_VALIDATION_FOCUS and COREF_RULES. Note that the unmutated `role-referential phrases` occurrence at line 102 inside `COREF_RULES` is the 46-07 CUT-COR-01 target — out of scope for CUT-VAL-03. Behavioral caveat per 46-RESEARCH §4.4: harness verdict reflects harness compatibility only (cached-replay); Phase 48 sweep validates behavioral safety on live LLM calls. |

<!-- SECTION:VAL:END -->

> **VAL section closing note (46-06).** Per-row verdicts: CUT-VAL-02 = **kept** (d82e5a9), CUT-VAL-01 = **kept** (5118c32), CUT-VAL-03 = **kept** (8c195bc); CUT-VAL-04 = **protected** (eec7fb8, tombstone — visibility-only, not trialled). Total LOC saved in VAL section: **0** (all three trialled cuts are substring rewordings within single dense lines / multi-line literals — no whole-line removal). **Cross-section pleonasm-batch state (CLOSED 3/3):** CUT-VAL-02 closes site 3 of 3 in the recurring `software architecture …` opener pleonasm batch — paired with CUT-AMB-02 (kept, 46-02, sha `0710510`, site 1/3) and CUT-EXT-01 (kept, 46-05, sha `fbfbcb9`, site 2/3); all three share the `components` bare replacement vocabulary pre-decided in the 46-01 MINIMIZE-LOG header. The Pareto Summary (46-08, Wave 3) rolls the three commits together as one conceptual batch. **VAL-03 → COR-01 lexicon handoff (integration contract for plan 46-07):** CUT-VAL-03 was kept; the chosen replacement vocabulary is **`noun phrase that refers back`** — plan 46-07's Task 2 (CUT-COR-01) reads this string from the CUT-VAL-03 reasoning cell and from this blockquote, then applies the SAME wording to the `role-referential noun phrase` span in `COREF_RULES` (`tests/scratch/prompts_v5.py:102`) so the two cuts stay lexically aligned across `COREF_VALIDATION_FOCUS` (this section, VAL-03) and `COREF_RULES` (COR section, COR-01). GATE-01 byte-equal preserved after every commit in this plan (5 commits total: 3 trial commits + 1 docs protect commit + 1 bookkeeping backfill commit; the closing-note commit immediately following this row makes 6 total).

## COR — Phase 5 Coref

<!-- SECTION:COR:START -->

| cut_id | verdict | snapshot_delta | gate06_isolation | loc_saved | commit_sha | reasoning |
|---|---|---|---|---|---|---|
| CUT-COR-02 | kept | 0/40 | clean | 0 | (sha) | Domain-loaded jargon swap in `COREF_RULES` (`tests/scratch/prompts_v5.py:102`): `treat it as the section-established topic` -> `treat it as the topic of the surrounding section` per 46-RESEARCH §6.2 audit-suggested vocabulary. Surrounding context: full clause now reads `…only one component has been introduced in the immediately preceding sentences — treat it as the topic of the surrounding section and resolve role-referential phrases ("it", "the module", "the service", "the component", "the system") to that topic even without a direct name repetition.` The "no direct name repetition" exemption and the quoted role-referential placeholders (`"it"`, `"the module"`, `"the service"`, `"the component"`, `"the system"`) stay intact per audit row 337 + Phase 45 v2.1 isolation precedent. `COREF_RULES` is body content (not opener), so `reconstruct_coref_inputs` is unaffected. 40/40 snapshots pass under `SAD_SAM_LINKER_SOURCE=scratch` (the highest-diversity gating in the audit). GATE-06 re-grep `grep -niwE 'topic\|surrounding\|section' BENCHMARK_TABOO.md` -> 0 hits across all per-dataset sections + Universal Taboo + Safe SE Textbook. LOC saved = 0 (substring rewording within a single dense line). Behavioral caveat per 46-RESEARCH §4.4: harness verdict reflects harness compatibility only (cached-replay); Phase 48 sweep validates behavioral safety on live LLM calls. |
<!-- SECTION:COR:END -->

## Protected Tombstones (visibility-only — not trialled per CONTEXT in-scope §)

| cut_id | verdict | snapshot_delta | gate06_isolation | loc_saved | commit_sha | reasoning |
|---|---|---|---|---|---|---|
| CUT-VAL-04 | protected | n/a | n/a | 0 | eec7fb8 | Behaviorally-protected per prompts_v5.py docstring lines 5–22 + experiment_dotted_path_rename.py record. The qualified-name X.Y.Z clause in P1_FOCUS catches 2/3 code-path FPs on gpt-5.4 + 1/3 on Claude Sonnet with 0 collateral damage on the 4-TP control set; cutting reintroduces the FPs that v2.6.3 documents as fixed and that motivated P1_FOCUS over s17f. Phase 46 MUST NOT cut per Phase 45 threat T-45-VAL-02. Commit assigned by 46-06 (docs(46-06): protect CUT-VAL-04 — qualified-name X.Y.Z clause behaviorally protected). |
| CUT-COR-05 | protected | n/a | n/a | 0 | (assigned by 46-07) | Behavioral conservatism dial at s_linker19.py:361 (`Be conservative — only include resolutions you are CERTAIN about.`). Coref Phase 5 is the FP-sensitive stage; v2.6.2 s17e drove FP 43→14 via validation gating (CLAUDE.md milestone notes). Removing the conservatism instruction risks reintroducing the FP class the validated-coref breakthrough closed. Phase 46 MUST NOT cut per Phase 45 threat T-45-COR-02. Commit assigned by 46-07 (docs(46-07): protect CUT-COR-05 — coref conservatism behaviorally protected). |

## Phase Close Notes

<!-- FINAL:GATE01:START -->
<!-- TBD: filled by .planning/phases/46-minimize/46-08-PLAN.md (Wave 3) — final GATE-01 byte-equal git-diff record on s_linker19.py + prompts_v5.py + s_linker13_min.py at phase close. -->
<!-- FINAL:GATE01:END -->

<!-- FINAL:REQ:START -->
<!-- TBD: filled by .planning/phases/46-minimize/46-08-PLAN.md — REQ-V264-05/06/07 tick-off bullets + ROADMAP Phase 46 SC1..SC5 tick-off. -->
<!-- FINAL:REQ:END -->
