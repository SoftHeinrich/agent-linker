---
phase: 45-audit
plan: 01
subsystem: audit-doc-bootstrap
tags: [skeleton, anchors, wave-1-enablement]
requires: []
provides:
  - .planning/phases/45-audit/s_linker20-PROMPT-AUDIT.md (skeleton with 5 D-08 section anchors, 15-row verdict table, gating reference, cut_id legend, drop-block convention note, phase-close anchors)
affects:
  - .planning/phases/45-audit/45-02-PLAN.md (SECTION:AMB:* anchors targetable)
  - .planning/phases/45-audit/45-03-PLAN.md (SECTION:DKX:* anchors targetable)
  - .planning/phases/45-audit/45-04-PLAN.md (SECTION:DKJ:* anchors targetable)
  - .planning/phases/45-audit/45-05-PLAN.md (SECTION:EXT:* anchors targetable)
  - .planning/phases/45-audit/45-06-PLAN.md (SECTION:VAL:* anchors targetable)
  - .planning/phases/45-audit/45-07-PLAN.md (SECTION:COR:* anchors targetable)
  - .planning/phases/45-audit/45-08-PLAN.md (FINAL:SUMMARY:*, FINAL:GATE01:* anchors targetable)
tech_stack:
  added: []
  patterns:
    - HTML-comment anchor pairs frame each Wave-1 section so plans can Edit between fixed markers without conflict.
    - Verbatim copy of Phase 44 §D-03 builder→phase-tag table (no paraphrase) — `phase_5_coref_validation` reuse path preserved literally.
    - Cut_id namespace declared up-front (`CUT-AMB-NN`..`CUT-COR-NN`) so Phase 46's MINIMIZE-LOG can reference cuts before they exist by row.
key_files:
  created:
    - .planning/phases/45-audit/s_linker20-PROMPT-AUDIT.md
  modified: []
decisions:
  - Followed plan task #8 enumeration of exactly 15 summary rows (9 constants + 6 builders per REQ-V264-03/04). `ANTECEDENT_ALIAS_RULES` is not in REQ-V264-03's enumerated 9 constants; the COR section anchor's TBD comment flags it for Wave-1 (45-07) to add as a section-header row, but it does not appear in the top-of-doc verdict summary.
  - Reworded the JSON-schema literal mention in the Out of Scope section to avoid the `Return JSON:` substring per the plan's grep-hygiene constraint while preserving the D-03 exclusion meaning.
metrics:
  duration: ~6 minutes
  completed_date: 2026-06-08
---

# Phase 45 Plan 01: Audit Doc Skeleton Summary

Bootstrapped the v2.6.4 prompt audit document at `.planning/phases/45-audit/s_linker20-PROMPT-AUDIT.md` with the full structural skeleton (rubric recap, scope, gating reference, cut_id legend, drop-block note, 15-row placeholder Verdict Summary, six pipeline-ordered section anchors, and two phase-close anchors) so Wave-1 plans 45-02 through 45-07 can fan out across non-overlapping HTML-comment anchor pairs.

## Anchor Markers Added (verbatim)

Pipeline-ordered section anchors (12 total, six START/END pairs):

```
<!-- SECTION:AMB:START -->
<!-- SECTION:AMB:END -->
<!-- SECTION:DKX:START -->
<!-- SECTION:DKX:END -->
<!-- SECTION:DKJ:START -->
<!-- SECTION:DKJ:END -->
<!-- SECTION:EXT:START -->
<!-- SECTION:EXT:END -->
<!-- SECTION:VAL:START -->
<!-- SECTION:VAL:END -->
<!-- SECTION:COR:START -->
<!-- SECTION:COR:END -->
```

Phase-close anchors (4 total, two START/END pairs):

```
<!-- FINAL:SUMMARY:START -->
<!-- FINAL:SUMMARY:END -->
<!-- FINAL:GATE01:START -->
<!-- FINAL:GATE01:END -->
```

## Placeholder Verdict Summary Table

Row count: **15** (9 imported constants + 6 in-class builders). Every `Verdict` and `Cut Rows` cell is `TBD`; the `Audited By` column attributes each row to the Wave-1 plan that will populate it (45-02 through 45-07).

Order (AMB → DKX → DKJ → EXT → VAL → COR):

1. `AMBIGUITY_FEW_SHOT` (constant, 7 LOC) — 45-02 (AMB)
2. `AMBIGUITY_RULES` (constant, 1 LOC) — 45-02 (AMB)
3. `_prompt_ambiguity` (builder, 19 LOC) — 45-02 (AMB)
4. `DOC_KNOWLEDGE_EXTRACTION_RULES` (constant, 1 LOC) — 45-03 (DKX)
5. `ALIAS_SCOPE_RULES` (constant, 4 LOC) — 45-03 (DKX)
6. `_prompt_doc_knowledge_extract` (builder, 19 LOC) — 45-03 (DKX)
7. `DOC_KNOWLEDGE_JUDGE_EXAMPLES` (constant, 7 LOC) — 45-04 (DKJ)
8. `DOC_KNOWLEDGE_JUDGE_RULES` (constant, 1 LOC) — 45-04 (DKJ)
9. `_prompt_doc_knowledge_judge` (builder, 16 LOC) — 45-04 (DKJ)
10. `ENTITY_EXTRACTION_RULES` (constant, 1 LOC) — 45-05 (EXT)
11. `_prompt_extraction` (builder, 15 LOC) — 45-05 (EXT)
12. `VALIDATION_RULES` (constant, 1 LOC) — 45-06 (VAL)
13. `_prompt_validation` (builder, 14 LOC) — 45-06 (VAL)
14. `COREF_RULES` (constant, 1 LOC) — 45-07 (COR)
15. `_prompt_coref` (builder, 27 LOC) — 45-07 (COR)

## Verification

- `grep -cE '^<!-- SECTION:(AMB|DKX|DKJ|EXT|VAL|COR):START -->$'` → `6` (expected 6).
- `grep -cE '^<!-- SECTION:(AMB|DKX|DKJ|EXT|VAL|COR):END -->$'` → `6` (expected 6).
- `grep -c 'phase_5_coref_validation'` → `3` (literal-copied into D-03 gating table row, the table footnote, and the cut_id legend's CUT-VAL gloss).
- `grep -c 'CUT-AMB-NN'` → `2`, `grep -c 'CUT-COR-NN'` → `2` (legend + reference points).
- `grep -c '^## Verdict Summary$'` → `1`; awk-scoped row count between header and the AMB section H2 → `15`.
- Gating table builder-row count (awk-scoped) → `6`.
- Final anchors → `4` lines (two START/END pairs).
- Taboo hygiene: `grep -v '^<!--' | grep -cE 'CacheLayer|RequestHandler'` → `0` (taboo tokens reserved for Wave-1 cut rows only).
- `grep -c 'Return JSON'` → `0` (D-02 hygiene; original draft had one instance in the Out-of-Scope bullet — reworded to drop the literal substring while preserving the D-03 exclusion meaning).
- **GATE-01 byte-equal:** `git diff --stat src/llm_sad_sam/linkers/experimental/{s_linker19.py,prompts_v5.py,s_linker13_min.py}` → empty (no edits to frozen source).

## Deviations from Plan

None of substance. Two minor planner-intent-preserving decisions:

1. **JSON-literal mention reworded (D-02 hygiene):** The plan's task #4 listed the JSON-schema exclusion example as `Return JSON: {...}` and the plan's task #1 final note required `no 'Return JSON:' text appears in the skeleton`. To satisfy both, the Out-of-Scope bullet refers to "the response-shape declaration and the `JSON only:` suffix" instead of the literal `Return JSON:` token. Meaning preserved; grep hygiene maintained.
2. **`ANTECEDENT_ALIAS_RULES` placement:** The plan's COR section header notes list `COREF_RULES, ANTECEDENT_ALIAS_RULES, _prompt_coref` (3 items) but the plan's task #8 summary-table enumeration includes only `COREF_RULES` + `_prompt_coref` for COR (2 items), keeping the table at exactly 15 rows (= REQ-V264-03's 9 constants + REQ-V264-04's 6 builders). Followed the explicit 15-row enumeration; the SECTION:COR:START TBD note already flags the COR header table will include `ANTECEDENT_ALIAS_RULES`, leaving 45-07 (COR) to decide whether to add it as a bonus row in the top-of-doc summary at Wave-2 finalization.

## Self-Check: PASSED

- File `.planning/phases/45-audit/s_linker20-PROMPT-AUDIT.md` exists.
- All 6 SECTION START/END anchor pairs in pipeline order (AMB → DKX → DKJ → EXT → VAL → COR).
- Verbatim Phase 44 §D-03 gating table (6 builder rows, `phase_5_coref_validation` preserved literally).
- 15-row TBD Verdict Summary table.
- Cut_id legend names all six prefixes plus the per-cut detail-block convention.
- Drop-block convention paragraph cites REQ-V264-06.
- FINAL:SUMMARY and FINAL:GATE01 anchor pairs present for Wave 2 (45-08).
- GATE-01 byte-equal verified (no diffs in frozen source).
