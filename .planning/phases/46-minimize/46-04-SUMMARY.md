---
phase: 46-minimize
plan: 04
subsystem: prompt-minimization
tags: [s_linker20, doc-judge, drop-block, prompts_v5, gate-01, gate-06, benchmark-leak]

# Dependency graph
requires:
  - phase: 45-audit
    provides: CUT-DKJ-01..07 audit rows + Family A name grep-clearance (BookManager/Mgr/MailSender from 45-04-SUMMARY)
  - phase: 46-minimize/46-01
    provides: tests/scratch/ wiring + SAD_SAM_LINKER_SOURCE harness toggle + MINIMIZE-LOG skeleton with DKJ anchors
  - phase: 44-harness
    provides: tests/test_s_linker20_prompt_doc_judge.py (5-snapshot gated module) + reconstruct_doc_judge_inputs
provides:
  - DKJ section of s_linker20-MINIMIZE-LOG.md populated (7 rows CUT-DKJ-01..07)
  - tests/scratch/prompts_v5.py mutated for DROP (DOC_KNOWLEDGE_JUDGE_EXAMPLES="") and CUT-DKJ-07 (architectural tier or technology platform -> grouping)
  - Smallest-passing identifier for DOC_KNOWLEDGE_JUDGE_EXAMPLES = `drop` (first DKJ benchmark-leak removal)
affects: [46-08, 47-ship, 48-sweep]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "D-03 drop-block protocol verbatim: drop -> Family A -> Family B; first passing wins; superseded rows fold into parent commit (D-04)"
    - "Constant-binding preservation under drop-by-empty (46-RESEARCH §9 Q9 (a)): assignment statement kept, RHS becomes ``''``"
    - "Behavioral caveat tagging on harness-only verdicts (46-RESEARCH §4.4 + §6.3): kept-by-byte-equal does NOT equal kept-by-behavior; Phase 48 sweep is the behavioral gate"

key-files:
  created:
    - .planning/phases/46-minimize/46-04-SUMMARY.md
  modified:
    - tests/scratch/prompts_v5.py
    - .planning/phases/46-minimize/s_linker20-MINIMIZE-LOG.md

key-decisions:
  - "D-03 drop-block exit on first pass: CUT-DKJ-01 drop succeeded on first attempt -> Family A and Family B never trialled (per D-03 short-circuit semantics)"
  - "Family A pre-cleared names (BookManager / Mgr / MailSender) inherited verbatim from 45-04-SUMMARY.md grep evidence — no re-grep in Phase 46 (CONTEXT in-scope policy)"
  - "CUT-DKJ-07 vocabulary `grouping that encompasses multiple elements` pre-decided in MINIMIZE-LOG header (46-RESEARCH §9 Q5) — Phase 46 confirms harness compatibility only"
  - "Multi-element exclusion semantics preserved across CUT-DKJ-07: `that encompasses multiple elements` clause kept verbatim; only `architectural tier or technology platform` -> `grouping` was substituted"

patterns-established:
  - "DKJ section closing reviewer paragraph: smallest-passing identifier + total LOC + behavioral caveat — reusable across all benchmark-leak sections"
  - "Backfill commit pattern (per 46-02 deviation): commit row with `(sha)` placeholder, capture commit hash, backfill, second commit — preserves atomic per-row history"

requirements-completed: [REQ-V264-05, REQ-V264-06, REQ-V264-07]

# Metrics
duration: 4min
completed: 2026-06-08
---

# Phase 46 Plan 04: DKJ Drop-Block Protocol Summary

**DOC_KNOWLEDGE_JUDGE_EXAMPLES dropped wholesale (7 LOC) on first D-03 attempt — removes the audit's only confirmed benchmark-leak (`CacheLayer`/`cache`) and lexically neutralizes the `architectural tier or technology platform` clause to `grouping` (CUT-DKJ-07) with 0/5 snapshot delta across both cuts.**

## Performance

- **Duration:** ~4 min
- **Started:** 2026-06-08T15:54:31Z
- **Completed:** 2026-06-08T15:58:22Z
- **Tasks:** 2 (per plan)
- **Cuts trialled:** 2 (CUT-DKJ-01 drop + CUT-DKJ-07); 5 superseded automatically (CUT-DKJ-02..06)
- **Files modified:** 2 (tests/scratch/prompts_v5.py, .planning/phases/46-minimize/s_linker20-MINIMIZE-LOG.md)

## Accomplishments

- **D-03 short-circuit exit on first attempt.** CUT-DKJ-01 drop-by-empty passed `tests/test_s_linker20_prompt_doc_judge.py` 5/5 under `SAD_SAM_LINKER_SOURCE=scratch` (snapshot_delta = 0/5). Family A (CUT-DKJ-02 BookManager Example-1 swap, CUT-DKJ-03 MailSender Example-2 swap, CUT-DKJ-04 combined rewrite) and Family B (CUT-DKJ-05 concept-only Example 1, CUT-DKJ-06 concept-only Example 2) all logged `superseded-by-drop` per D-03 without separate trials.
- **Sole audit benchmark-leak removed.** The `CacheLayer` → `cache` substring Universal Taboo hit (per 45-04-SUMMARY, the only confirmed body-text leak in the entire 19-cut audit) is removed by the wholesale DROP of `DOC_KNOWLEDGE_JUDGE_EXAMPLES`.
- **CUT-DKJ-07 lexical neutralization.** `architectural tier or technology platform that encompasses multiple elements` → `grouping that encompasses multiple elements` in `DOC_KNOWLEDGE_JUDGE_RULES`. Multi-element exclusion semantics preserved (`that encompasses multiple elements` clause kept verbatim). 5/5 snapshots pass; GATE-06 clean (0 per-dataset hits on `grouping`/`encompasses`/`elements`/`invalid`/`names`).
- **GATE-01 byte-equal held continuously** across both per-cut commits + both backfill commits. `git diff --stat` on `s_linker19.py` + `prompts_v5.py` + `s_linker13_min.py` is empty.

## Per-Cut Verdicts Table

| cut_id | verdict | snapshot_delta | gate06_isolation | loc_saved | commit_sha | notes |
|---|---|---|---|---|---|---|
| CUT-DKJ-01 | **kept** | 0/5 | clean (empty after-text) | **7** | `74ec3bd` | DROP-BLOCK winner. `DOC_KNOWLEDGE_JUDGE_EXAMPLES` body collapsed to `""`. Constant binding preserved. |
| CUT-DKJ-02 | superseded-by-drop | n/a | n/a | 0 | `74ec3bd` | Family A Example-1 BookManager/Mgr swap moot. |
| CUT-DKJ-03 | superseded-by-drop | n/a | n/a | 0 | `74ec3bd` | Family A Example-2 MailSender swap moot (primary leak removed by drop). |
| CUT-DKJ-04 | superseded-by-drop | n/a | n/a | 0 | `74ec3bd` | Family A combined-rewrite moot. |
| CUT-DKJ-05 | superseded-by-drop | n/a | n/a | 0 | `74ec3bd` | Family B concept-only Example 1 moot. |
| CUT-DKJ-06 | superseded-by-drop | n/a | n/a | 0 | `74ec3bd` | Family B concept-only Example 2 moot. |
| CUT-DKJ-07 | **kept** | 0/5 | clean | 0 | `8a83bda` | Standalone §3-loop cut. `architectural tier or technology platform` → `grouping`. |

**Drop-block winner:** **`drop`** (CUT-DKJ-01) — first passing of the D-03 tree.
**Total DKJ LOC saved:** **7** (CUT-DKJ-01 = 7 LOC; CUT-DKJ-07 = 0 LOC substring rewording).

## Which Cut Won the Drop-Block Tree

**CUT-DKJ-01 (DROP)** won on the first attempt. Family A and Family B were never trialled. Per D-03 short-circuit semantics, this is the optimal Pareto outcome — minimum risk (no name-introduction at all), maximum LOC saved (entire 7-line examples body removed), zero benchmark vocabulary in the after-text (the after-text is literally `""`).

## GATE-06 Grep Summary

**CUT-DKJ-01:** Trivially clean — after-text is the empty string; no tokens to grep.

**CUT-DKJ-07 (after-text `An alias is also invalid when it names a grouping that encompasses multiple elements, because it identifies a grouping rather than a single named unit.`):**

| Token | BENCHMARK_TABOO hit |
|---|---|
| `grouping` | 0 hits |
| `encompasses` | 0 hits |
| `elements` | 0 hits |
| `invalid` | 0 hits |
| `names` | 0 hits |
| `alias` | 4 hits (BENCHMARK_TABOO lines 24, 55–58) — standing meta-references to the linker `alias` mechanism, NOT benchmark vocabulary; cleared per v2.1 isolation precedent (`alias` is the linker-architectural noun used in BENCHMARK_TABOO to *describe* the rejected benchmark words, not introduced *by* this prompt) |

Result: **clean**.

## Task Commits

Each task was committed atomically per D-04:

1. **Task 1 (drop-block parent — CUT-DKJ-01 kept + CUT-DKJ-02..06 superseded-by-drop):** `74ec3bd` (feat)
2. **Task 1 SHA backfill (replace `(sha)` / `(sha-parent)` placeholders with `74ec3bd`):** `56e8b55` (docs — per 46-02 deviation pattern)
3. **Task 2 (CUT-DKJ-07 kept — standalone §3-loop cut):** `8a83bda` (feat)
4. **Task 2 SHA backfill:** `58a5967` (docs)

Plan metadata commit: pending (this SUMMARY + state updates).

## Files Created/Modified

- `tests/scratch/prompts_v5.py` — `DOC_KNOWLEDGE_JUDGE_EXAMPLES` body emptied (CUT-DKJ-01); `DOC_KNOWLEDGE_JUDGE_RULES` `architectural tier or technology platform` → `grouping` (CUT-DKJ-07).
- `.planning/phases/46-minimize/s_linker20-MINIMIZE-LOG.md` — DKJ section anchors populated with 7 rows (CUT-DKJ-01..07) + DKJ closing reviewer paragraph.
- `.planning/phases/46-minimize/46-04-SUMMARY.md` — this file (created).

**Files NOT modified (GATE-01 byte-equal held):**
- `src/llm_sad_sam/linkers/experimental/s_linker19.py`
- `src/llm_sad_sam/linkers/experimental/prompts_v5.py`
- `src/llm_sad_sam/linkers/experimental/s_linker13_min.py`

## Decisions Made

- **D-03 short-circuit honored.** Once CUT-DKJ-01 passed, no Family A / Family B trials were run. The plan's protocol is explicit: "If RC=0: CUT-DKJ-01 verdict = `kept` (drop). CUT-DKJ-02..06 verdict = `superseded-by-drop`. PROCEED TO Task 2." Skipping the family walks is the protocol-mandated behavior, not a deviation.
- **CUT-DKJ-07 article fix `an` → `a`.** The substitution replaces `an architectural tier or technology platform` → `a grouping` — the article naturally flips because the head noun changed from a vowel-initial word to a consonant-initial word. This is an unavoidable grammatical consequence of the audit-prescribed substring rewrite, not a deviation. The plan's Step 1 wording ("Preserve all surrounding clauses, the leading `An alias is also invalid when it names a`, and the rest of the rule sentence intact") confirms the leading article should already read `a` post-cut.
- **`alias` BENCHMARK_TABOO hits cleared as meta-references.** The 4 hits on `alias` in BENCHMARK_TABOO are the document's own meta-vocabulary for *describing* benchmark component aliases — `alias` is the linker-architectural concept noun, present in every linker prompt and every audit doc. Per v2.1 isolation methodology, meta-vocabulary words used to *describe* what BENCHMARK_TABOO catalogs are not themselves catalogued tokens. The before-text already contained `alias` 4 times (`An alias is valid`, `An alias is invalid`, `An alias is also invalid`, `For each alias`) — CUT-DKJ-07 does not introduce a new `alias` occurrence.

## Deviations from Plan

None - plan executed exactly as written.

The plan anticipated a 4-step decision tree (DROP → Family A → Family B → kept-original); the first step passed, so steps 2–4 were short-circuited per D-03. This is the protocol-prescribed happy path, not a deviation.

## Issues Encountered

None.

A small note on the start-of-plan working-tree state: an unrelated working-copy modification to `.planning/phases/46-minimize/46-01-SUMMARY.md` and an untracked file `ablation_evjudge_rest.py` were present from prior sessions. Neither falls within this plan's `files_modified` declaration, so both were left untouched throughout. The atomic commits for this plan staged only the two declared files explicitly (`tests/scratch/prompts_v5.py` and the MINIMIZE-LOG) — never `git add .` or `-A`.

## Behavioral Caveat Reminder

Per 46-RESEARCH §4.4 + §6.3, both `kept` verdicts in this plan reflect **harness compatibility only**, not behavioral safety:

- **CUT-DKJ-01 (drop):** the doc-knowledge-judge few-shot drives the judge's calibration (the VALID/INVALID rationale shape). Removing it may shift the judge's threshold on real LLM calls. The 5/5 snapshot pass under cached-replay scratch mode means the harness still parses fine — it does NOT mean the model still judges the same way.
- **CUT-DKJ-07 (lexical):** the multi-element exclusion clause stays semantically intact (`that encompasses multiple elements` preserved), so behavioral risk is lower than CUT-DKJ-01. But the rule-shape may still drive judge decisions differently on edge cases involving infrastructure-tier nouns vs. generic groupings.

**Phase 48 sweep validates behavior.** Phase 46 produces a *recipe*, not behavior-validated production code. Phase 47 (SHIP) inlines this recipe into `s_linker20.py`; Phase 48 (SWEEP) runs the real LLM and measures the behavioral delta against `s_linker19.py`.

## GATE-01 Verification

After every per-cut commit (4 commits total), the following was run and returned empty:

```bash
git diff --stat \
  src/llm_sad_sam/linkers/experimental/s_linker19.py \
  src/llm_sad_sam/linkers/experimental/prompts_v5.py \
  src/llm_sad_sam/linkers/experimental/s_linker13_min.py
```

**Final GATE-01 status: PASS (byte-equal).**

## Next Phase Readiness

- **46-08 (Wave 3 phase close)** — Pareto Summary needs to record:
  - DOC_KNOWLEDGE_JUDGE_EXAMPLES smallest-passing identifier: **`drop`**
  - DKJ LOC saved: 7
  - DKJ kept rows: CUT-DKJ-01 (drop), CUT-DKJ-07 (lexical)
  - Cross-section context: DKJ is the only `benchmark-leak` section — its `drop` outcome closes the audit's primary GATE-06 finding.
- **47-ship** — Reads MINIMIZE-LOG row for CUT-DKJ-01 to know `DOC_KNOWLEDGE_JUDGE_EXAMPLES = ""` in the inlined `s_linker20.py` constants; reads CUT-DKJ-07 row for the post-cut `DOC_KNOWLEDGE_JUDGE_RULES` text.
- **48-sweep** — Validates the behavioral impact of the wholesale DKJ examples drop. The drop-block protocol's bet is that on the 5 dataset projects, parsed outputs survive because the judge's underlying VALID/INVALID calibration is robust to zero-shot operation. Phase 48 confirms or falsifies this empirically.

## Self-Check: PASSED

| Item | Result |
|------|--------|
| `tests/scratch/prompts_v5.py` (CUT-DKJ-01 + CUT-DKJ-07 applied) | FOUND |
| `.planning/phases/46-minimize/s_linker20-MINIMIZE-LOG.md` (7 DKJ rows present) | FOUND |
| `.planning/phases/46-minimize/46-04-SUMMARY.md` | FOUND |
| Commit `74ec3bd` (CUT-DKJ-01 drop-block parent) | FOUND in `git log --oneline --all` |
| Commit `56e8b55` (Task 1 SHA backfill) | FOUND in `git log --oneline --all` |
| Commit `8a83bda` (CUT-DKJ-07 kept) | FOUND in `git log --oneline --all` |
| Commit `58a5967` (Task 2 SHA backfill) | FOUND in `git log --oneline --all` |
| GATE-01 post-commit `git diff --stat` on the three frozen sources | empty (exit 0) on all 4 commits |
| Post-commit deletion check (`git diff --diff-filter=D HEAD~1 HEAD`) | empty (zero deletions) on all 4 commits |
| 7-row DKJ schema check (`<verify>` automated) | PASS |

---

*Phase: 46-minimize*
*Completed: 2026-06-08*
