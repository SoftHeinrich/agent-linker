---
phase: 45-audit
plan: 05
subsystem: prompt-audit
tags: [audit, ext, gate-01]
requires:
  - 45-01 (skeleton)
provides:
  - "EXT section of s_linker20-PROMPT-AUDIT.md (between SECTION:EXT anchors)"
  - "Verdict Summary rows for ENTITY_EXTRACTION_RULES + _prompt_extraction"
affects:
  - .planning/phases/45-audit/s_linker20-PROMPT-AUDIT.md
key-files:
  modified:
    - .planning/phases/45-audit/s_linker20-PROMPT-AUDIT.md
  created:
    - .planning/phases/45-audit/45-05-SUMMARY.md
decisions:
  - "Applied D-01 pragmatic rubric: ENTITY_EXTRACTION_RULES body text is load-bearing (code-level path / compound identifier / architectural intent encode Spike-003 dotted-path FP-exclusion behavior) → clean"
  - "Applied D-01 pragmatic rubric: _prompt_extraction opener `software architecture components` is pleonastic given the COMPONENTS slot at line 325 → domain-loaded"
  - "Per D-05, no rewording proposed for domain-loaded flag; after = [Phase 46 empirical loop]"
  - "Both phase_2_framing_c_pass1 AND phase_2_framing_c_pass2 listed in gated_by per Phase 44 §D-03 (18 snapshots across 2 tags)"
  - "Flagged cross-section batching opportunity: `software architecture components` opener pattern recurs in AMB (CUT-AMB-02), EXT (CUT-EXT-01), and anticipated VAL — Phase 46 should batch"
metrics:
  duration: "~10 minutes"
  completed: "2026-06-08"
  cut_rows: 1
  ext_items_audited: 2
---

# Phase 45 Plan 05: EXT Section Audit Summary

EXT-section audit of `ENTITY_EXTRACTION_RULES` (prompts_v5.py:67) and `_prompt_extraction` prose (s_linker19.py:323) — one `clean` constant, one `domain-loaded` builder opener, one cut row.

## What Shipped

- 2-row Items header table filled between `<!-- SECTION:EXT:START -->` and `<!-- SECTION:EXT:END -->` (verdict + LOC + notes per D-08).
- One `CUT-EXT-01` row: `domain-loaded` flag on `s_linker19.py:323` opener (`"Extract ALL references to software architecture components from this document."`).
- Per-cut risk tier (`low-med`) with short justification per CD-5.
- `gated_by` cell lists BOTH `phase_2_framing_c_pass1` AND `phase_2_framing_c_pass2` per Phase 44 §D-03 mapping.
- Top-of-doc Verdict Summary rows updated: `ENTITY_EXTRACTION_RULES` → `clean` / 0 cut rows; `_prompt_extraction` → `domain-loaded` / 1 cut row.
- EXT inventory note, EXT benchmark-leak audit (verbatim grep results), Reviewer judgment paragraph, and cross-section batching observation included per CD-6.

## Final Verdicts

| Item | Verdict | Cut rows |
|---|---|---|
| `ENTITY_EXTRACTION_RULES` | clean | 0 |
| `_prompt_extraction` (prose) | domain-loaded ("software architecture components") | 1 (CUT-EXT-01) |

## Cut Row Count

- `CUT-EXT-NN` rows emitted: **1** (CUT-EXT-01).
- Matches 45-RESEARCH.md §5.3 prior (0–2 expected) and the plan's must_haves ("at least 1 domain-loaded flag row").

## GATE-01 Verification

`git diff --quiet src/llm_sad_sam/linkers/experimental/s_linker19.py src/llm_sad_sam/linkers/experimental/prompts_v5.py src/llm_sad_sam/linkers/experimental/s_linker13_min.py` → **PASSED** (exit 0). No edits to frozen source artefacts. Final GATE-01 byte-equal is re-verified at phase close by 45-08.

## Plan Verification Output

```
CUT-EXT count=1
OK
GATE-01 byte-equal: PASSED
```

## Deviations from Plan

None — plan executed exactly as written. The mechanical Universal-Taboo grep on both items confirmed 45-RESEARCH.md §5.3 priors:
- `ENTITY_EXTRACTION_RULES` body: one `component` Universal-Taboo overlap, passes v2.1 GATE-06 isolation as generic SE noun (same precedent as AMBIGUITY_RULES, DOC_KNOWLEDGE_EXTRACTION_RULES, DOC_KNOWLEDGE_JUDGE_RULES from prior waves).
- `_prompt_extraction` line 323: zero Universal-Taboo whole-word hits; the `domain-loaded` flag is driven by D-01 pragmatic rubric (pleonasm against the COMPONENTS slot), not by grep.

No override rows were needed (no unexpected benchmark-leak hit).

## Self-Check: PASSED

- File modified: `.planning/phases/45-audit/s_linker20-PROMPT-AUDIT.md` ✓ (anchored between SECTION:EXT markers only)
- Verification script: `CUT-EXT count=1`, header items present, both `phase_2_framing_c_pass1` and `phase_2_framing_c_pass2` present in body, `domain-loaded` present ✓
- GATE-01 byte-equal: git diff --quiet returned exit 0 against the 3 frozen source files ✓
- Scope: zero edits outside `<!-- SECTION:EXT:START -->` ... `<!-- SECTION:EXT:END -->` AND the two Verdict Summary table rows for the audited items (Verdict Summary update is the canonical responsibility of each per-section plan per CD-6) ✓
