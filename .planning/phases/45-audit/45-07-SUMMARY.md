---
phase: 45-audit
plan: 07
subsystem: prompt-audit
tags: [audit, COR, coref, domain-loaded, behavioral-protected, ALIAS_SCOPE_RULES-back-reference]
requires:
  - 45-01 (Wave-0 skeleton + anchor scheme + Verdict Summary stub for COREF_RULES/_prompt_coref)
  - 45-03 (DKX canonical audit row for ALIAS_SCOPE_RULES — back-referenced from COR per §6.1)
  - 45-06 (VAL CUT-VAL-03 — shared `role-referential` lexicon; Phase 46 batching opportunity)
provides:
  - .planning/phases/45-audit/s_linker20-PROMPT-AUDIT.md (COR section between SECTION:COR:START/END anchors)
affects:
  - Phase 46 (MINIMIZE) — 4 domain-loaded cut rows + 1 visibility-only behavioral-protected tombstone gated by tests/test_s_linker20_prompt_coref.py @ phase_5_coref
tech-stack:
  added: []
  patterns:
    - "Behavioral-protected tombstone (CUT-COR-05) for line 361 conservatism — same pattern as VAL CUT-VAL-04"
    - "Cross-section back-reference (ALIAS_SCOPE_RULES → DKX) — prevents duplicate cut_ids"
    - "Multi-span jargon batching (CUT-COR-03 + CUT-COR-04 in lockstep, batched with VAL CUT-VAL-03)"
key-files:
  created:
    - .planning/phases/45-audit/45-07-SUMMARY.md
  modified:
    - .planning/phases/45-audit/s_linker20-PROMPT-AUDIT.md (COR section + Verdict Summary 2 rows TBD→final)
decisions:
  - "COREF_RULES verdict: domain-loaded (2 jargon spans — `role-referential noun phrase` + `section-established topic`)"
  - "ANTECEDENT_ALIAS_RULES verdict: clean (TaskScheduler + scheduler verified clean across all 5 datasets + Universal Taboo; both in §Safe SE Textbook Examples line 63) — no cut rows"
  - "_prompt_coref verdict: domain-loaded (3 cut rows: opener line 354, inline prose 358–360, behavioral-protected line 361 tombstone)"
  - "Line 361 `Be conservative` is behavioral per §7.4 — risk=high visibility-only row, after=`DO NOT CUT — no evidence safe`"
  - "ALIAS_SCOPE_RULES back-reference note placed before header table — points at DKX canonical audit row (45-03)"
  - "Phase 46 batching: CUT-COR-03 + CUT-COR-04 lockstep; further batched with COR CUT-COR-01 + VAL CUT-VAL-03 on shared `role-referential` lexicon"
metrics:
  duration: ~12min
  completed: 2026-06-08
---

# Phase 45 Plan 07: COR Section Audit Summary

One-liner: Audited COREF_RULES + ANTECEDENT_ALIAS_RULES + `_prompt_coref` and populated the COR section of `s_linker20-PROMPT-AUDIT.md` with 1 header table (3 rows) + 1 ALIAS_SCOPE_RULES back-reference note + 1 cut table (5 CUT-COR-NN rows: 4 domain-loaded + 1 behavioral-protected tombstone) — all gated by `tests/test_s_linker20_prompt_coref.py @ phase_5_coref`.

## Final Verdicts (3 rows)

| Item | Type | Verdict | LOC | Cut Rows |
|---|---|---|---|---|
| `COREF_RULES` | constant | domain-loaded (linguistics jargon spans) | 1 (long sentence) | 2 (CUT-COR-01, CUT-COR-02) |
| `ANTECEDENT_ALIAS_RULES` | constant | clean | 9 | 0 (TaskScheduler + scheduler grep-cleared via Safe SE Textbook list) |
| `_prompt_coref` (prose) | builder | domain-loaded (multiple jargon spans) | 27 (~6 prose lines: 354, 358–364) | 3 (CUT-COR-03, CUT-COR-04, CUT-COR-05) |

## CUT-COR-NN Row Count

**5 CUT-COR rows total** (plan minimum: 3; plan ceiling: 8 per §5.3 estimate):
- CUT-COR-01: `domain-loaded ("role-referential noun phrase")` on COREF_RULES — risk `med-high`
- CUT-COR-02: `domain-loaded ("section-established topic")` on COREF_RULES — risk `med`
- CUT-COR-03: `domain-loaded ("anaphoric references" + "role-referential noun phrases" + "architecture components")` on `_prompt_coref:354` opener — risk `med-high`
- CUT-COR-04: `domain-loaded ("anaphoric reference" repeated)` on `_prompt_coref:358-360` inline prose — risk `med`
- CUT-COR-05: `behavioral-protected (§7.4)` on `_prompt_coref:361` (`Be conservative …`) — risk `high`, `after = DO NOT CUT — no evidence safe`

All 5 rows have `gated_by = tests/test_s_linker20_prompt_coref.py @ phase_5_coref`.

## ANTECEDENT_ALIAS_RULES Grep-Clearance Result

**Verdict: clean — no cut rows.** Mandatory greps per Step-1:

1. `grep -niwE 'TaskScheduler|scheduler' BENCHMARK_TABOO.md` → 1 hit at line 63: `Operating systems: Scheduler, MemoryManager, FileSystem, ProcessTable, Dispatcher` (§Safe SE Textbook Examples — confirmed not in benchmark). **Both example names are in the affirmative-safe list.** This matches the expected outcome per 45-RESEARCH.md Open Question 3.
2. `grep -niwE 'queues|jobs|module|service|system|pronoun|anaphoric|role-referential|antecedent' BENCHMARK_TABOO.md` → 2 hits: line 22 (BBB §Components compound `Recording Service` — bare `the service` in COREF_RULES enumeration passes v2.1 GATE-06 isolation as generic SE noun) and line 52 (Universal Taboo entry `internal (BBB/Teammates — "X.internal module")` — bare `the module` in COREF_RULES enumeration passes the same isolation check). Same precedent as VAL CUT-VAL-03's reading of COREF_VALIDATION_FOCUS per 45-RESEARCH.md §1.6 explicit dismissal.

Per-dataset taboo hits on ANTECEDENT_ALIAS_RULES body: zero. No `benchmark-leak` escalation. No Family A / Family B rewordings emitted.

## ALIAS_SCOPE_RULES Back-Reference Confirmation

Back-reference note placed as the **first sub-element of the COR section** (before the header table), per Step-4. Text:

> **Cross-section reference (ALIAS_SCOPE_RULES):** ALIAS_SCOPE_RULES is imported by `_prompt_doc_knowledge_extract` only (s_linker19.py:292), not by `_prompt_coref`. Its canonical audit row lives in section DKX above. This COR section does not duplicate the audit; Phase 46 references the DKX cut_ids when minimizing alias-scope text.

Automated verification (`assert 'ALIAS_SCOPE_RULES' in body and 'DKX' in body`) **PASSED**.

## GATE-01 Git-Diff Verification

```
git diff --quiet src/llm_sad_sam/linkers/experimental/s_linker19.py \
                 src/llm_sad_sam/linkers/experimental/prompts_v5.py \
                 src/llm_sad_sam/linkers/experimental/s_linker13_min.py
```

Exit code: 0 → **GATE-01 byte-equal CONFIRMED.** No edits to source files.

## Verdict Summary Updates

Updated 2 TBD rows in the top-of-doc Verdict Summary table:
- `COREF_RULES`: TBD → `domain-loaded` / 2 cut rows
- `_prompt_coref`: TBD → `domain-loaded` / 3 cut rows

Per the plan deviation note + CD-6 / 45-01: ANTECEDENT_ALIAS_RULES is **NOT** added to the 15-row Verdict Summary; it is flagged inline in the COR section only (verdict: clean, 0 cut rows).

## Deviations from Plan

**None — plan executed exactly as written.**

- Step 1 grep produced the expected outcome (ANTECEDENT_ALIAS_RULES clean via Safe SE Textbook list); no escalation to benchmark-leak; no Family A / Family B rows emitted.
- Step 3 emitted three separate domain-loaded rows for the `_prompt_coref:354` opener jargon (one per cuttable span would have been three rows; the plan allowed "one row OR three separate rows — auditor picks granularity, preferring one row per cuttable span for Phase 46 actionability"). The auditor chose **one combined row** (CUT-COR-03) covering all three spans in the single sentence, because the universal-noun rewrite collapses the three spans into a single replacement (e.g. `pronouns and noun phrases that refer back to a component`) — splitting into three rows would over-fragment the Phase 46 batching unit. This is within the plan's documented latitude.
- Step 5 emitted the OPTIONAL CUT-COR-05 visibility-only behavioral-protected row (the plan listed it as OPTIONAL: "emit a visibility-only row with risk = high if flagged, OR skip with a one-line reviewer note"). The auditor chose the explicit row for Phase 46 visibility, consistent with the precedent set by VAL CUT-VAL-04 (P1_FOCUS `X.Y.Z` tombstone, 45-06).
- Total CUT-COR row count: 5 (within plan's 3–8 range; plan minimum 3 met).

## Self-Check

- [x] `.planning/phases/45-audit/s_linker20-PROMPT-AUDIT.md` modified (`git status --short` confirms `M`)
- [x] `.planning/phases/45-audit/45-07-SUMMARY.md` created (this file)
- [x] COR section body contains `COREF_RULES`, `ANTECEDENT_ALIAS_RULES`, `_prompt_coref` (all 3 header items)
- [x] COR section body contains `ALIAS_SCOPE_RULES` + `DKX` (back-reference note)
- [x] COR section body contains `phase_5_coref` (gated_by for all 5 cut rows)
- [x] 5 CUT-COR-NN rows present (≥3 plan minimum, ≤8 plan ceiling)
- [x] GATE-01 byte-equal confirmed (`git diff --quiet` on s_linker19.py + prompts_v5.py + s_linker13_min.py → exit 0)
- [x] Verdict Summary rows updated (COREF_RULES + _prompt_coref TBD→final)
- [x] ANTECEDENT_ALIAS_RULES NOT added to Verdict Summary per CD-6/45-01 deviation note (flagged inline only)

## Self-Check: PASSED
