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
| CUT-AMB-01 | kept | 0/5 | clean (no after-text) | 7 | 20cbde3 | DROP-BLOCK passed: `AMBIGUITY_FEW_SHOT` body in `tests/scratch/prompts_v5.py:30-36` replaced with `""` (drop-by-empty per 46-RESEARCH §9 Q9 (a)). Constant binding preserved so the scratch `s_linker19.py` `from tests.scratch.prompts_v5 import (..., AMBIGUITY_FEW_SHOT, ...)` import still resolves. `reconstruct_ambiguity_inputs` anchors on `^NAMES:` and `NOW CLASSIFY THE NAMES ABOVE.` — both still present in the post-cut prompt — so 5/5 snapshots pass under `SAD_SAM_LINKER_SOURCE=scratch` (snapshot_delta = 0/5). After-text is the empty string ⇒ GATE-06 trivially clean (no tokens to grep). LOC saved = 7 (the multi-line triple-quoted body). Behavioral caveat per 46-RESEARCH §4.4: drop verdict reflects harness compatibility only; behavioral effect on judge calibration (zero-shot vs few-shot) is NOT observable in this phase. Phase 48 sweep validates behavior. |
<!-- SECTION:AMB:END -->

> **AMB section closing note (46-02).** Smallest-passing identifier for `AMBIGUITY_FEW_SHOT` is **`drop`** (drop-by-empty per 46-RESEARCH §9 Q9 (a)); no Family A / Family B rewordings were emitted at audit time per D-06 (AMB verdict is `clean`, not `benchmark-leak`), so no further trial branches exist for this block. Total LOC saved in AMB section: **7** (CUT-AMB-02 = 0 LOC + CUT-AMB-01 = 7 LOC). CUT-AMB-02 closes 1 of 3 sites in the cross-section pleonasm batch (`software architecture …` opener); CUT-EXT-01 (46-05) and CUT-VAL-02 (46-06) close the remaining 2 sites with the same `components` bare replacement vocabulary set by 46-01.

## DKX — Phase 1 Doc-Knowledge Extract

<!-- SECTION:DKX:START -->
<!-- TBD: filled by .planning/phases/46-minimize/46-03-PLAN.md (Wave 2). Zero cuts in audit. Single `no cuts attempted` log line for section symmetry. -->
<!-- SECTION:DKX:END -->

## DKJ — Phase 1 Doc-Knowledge Judge

<!-- SECTION:DKJ:START -->
<!-- TBD: filled by .planning/phases/46-minimize/46-04-PLAN.md (Wave 2). 7 cuts. CUT-DKJ-01 drop-block runs first per D-03; Family A (CUT-DKJ-02/03/04) and Family B (CUT-DKJ-05/06) follow only if drop fails. CUT-DKJ-07 trialled separately at end. Highest-yield section in the audit. -->
<!-- SECTION:DKJ:END -->

## EXT — Phase 2 Extraction

<!-- SECTION:EXT:START -->
<!-- TBD: filled by .planning/phases/46-minimize/46-05-PLAN.md (Wave 2). 1 cut: CUT-EXT-01 pleonasm (part of cross-section pleonasm batch). -->
<!-- SECTION:EXT:END -->

## VAL — Phase 4 Validation

<!-- SECTION:VAL:START -->
<!-- TBD: filled by .planning/phases/46-minimize/46-06-PLAN.md (Wave 2). 4 cuts: CUT-VAL-02 pleonasm (part of batch), CUT-VAL-01 (counterparts), CUT-VAL-03 (role-referential — picks vocabulary shared with COR-01). CUT-VAL-04 is the P1_FOCUS X.Y.Z tombstone — protected row only, populated below. -->
<!-- SECTION:VAL:END -->

## COR — Phase 5 Coref

<!-- SECTION:COR:START -->
<!-- TBD: filled by .planning/phases/46-minimize/46-07-PLAN.md (Wave 2). 5 cuts: CUT-COR-02 (section-established topic), CUT-COR-01 (role-referential — reads VAL-03's vocabulary), (CUT-COR-03 + CUT-COR-04 batched per audit lockstep). CUT-COR-05 conservatism tombstone — protected row only, populated below. -->
<!-- SECTION:COR:END -->

## Protected Tombstones (visibility-only — not trialled per CONTEXT in-scope §)

| cut_id | verdict | snapshot_delta | gate06_isolation | loc_saved | commit_sha | reasoning |
|---|---|---|---|---|---|---|
| CUT-VAL-04 | protected | n/a | n/a | 0 | (assigned by 46-06) | Behaviorally-protected per prompts_v5.py docstring lines 5–22 + experiment_dotted_path_rename.py record. The qualified-name X.Y.Z clause in P1_FOCUS catches 2/3 code-path FPs on gpt-5.4 + 1/3 on Claude Sonnet with 0 collateral damage on the 4-TP control set; cutting reintroduces the FPs that v2.6.3 documents as fixed and that motivated P1_FOCUS over s17f. Phase 46 MUST NOT cut per Phase 45 threat T-45-VAL-02. Commit assigned by 46-06 (docs(46-06): protect CUT-VAL-04 — qualified-name X.Y.Z clause behaviorally protected). |
| CUT-COR-05 | protected | n/a | n/a | 0 | (assigned by 46-07) | Behavioral conservatism dial at s_linker19.py:361 (`Be conservative — only include resolutions you are CERTAIN about.`). Coref Phase 5 is the FP-sensitive stage; v2.6.2 s17e drove FP 43→14 via validation gating (CLAUDE.md milestone notes). Removing the conservatism instruction risks reintroducing the FP class the validated-coref breakthrough closed. Phase 46 MUST NOT cut per Phase 45 threat T-45-COR-02. Commit assigned by 46-07 (docs(46-07): protect CUT-COR-05 — coref conservatism behaviorally protected). |

## Phase Close Notes

<!-- FINAL:GATE01:START -->
<!-- TBD: filled by .planning/phases/46-minimize/46-08-PLAN.md (Wave 3) — final GATE-01 byte-equal git-diff record on s_linker19.py + prompts_v5.py + s_linker13_min.py at phase close. -->
<!-- FINAL:GATE01:END -->

<!-- FINAL:REQ:START -->
<!-- TBD: filled by .planning/phases/46-minimize/46-08-PLAN.md — REQ-V264-05/06/07 tick-off bullets + ROADMAP Phase 46 SC1..SC5 tick-off. -->
<!-- FINAL:REQ:END -->
