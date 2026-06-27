---
phase: 45-audit
plan: 06
subsystem: prompt-audit
tags: [audit, VAL, validation, phase-4, twopass, coref-validation, prompts_v5]
dependency_graph:
  requires:
    - 45-01 (audit doc skeleton; VAL anchors present)
  provides:
    - "s_linker20-PROMPT-AUDIT.md VAL section populated"
    - "CUT-VAL-01..04 cut row family (3 domain-loaded + 1 behavioral-protected tombstone)"
    - "Verdict Summary updated for VALIDATION_RULES (domain-loaded, 1 cut) and _prompt_validation (domain-loaded, 1 cut)"
    - "P1_FOCUS protected-clause record (X.Y.Z qualified-name; DO NOT CUT)"
    - "COREF_VALIDATION_FOCUS asymmetric-design record (do NOT symmetrize)"
  affects:
    - 45-08 (final gate / GATE-01 verification + FINAL:SUMMARY consolidation)
    - 46 (MINIMIZE) — Phase 46 ordering for VAL section + protected-clause tombstone
tech_stack:
  added: []
  patterns:
    - "Domain-loaded flag pattern (`software architecture <noun>` opener, third occurrence after AMB/EXT — single Phase-46 batching opportunity across three call sites)"
    - "Behavioral-protected tombstone pattern (CUT-VAL-04; `after = DO NOT CUT — empirically validated load-bearing`)"
    - "Generic-SE-noun second-pass isolation (Universal-Taboo `service` hit dismissed for COREF_VALIDATION_FOCUS `'the service'` quoted role-referential exemplar per 45-RESEARCH.md §1.6)"
key_files:
  created:
    - .planning/phases/45-audit/45-06-SUMMARY.md
  modified:
    - .planning/phases/45-audit/s_linker20-PROMPT-AUDIT.md (VAL section between anchors + Verdict Summary rows for VALIDATION_RULES / _prompt_validation)
decisions:
  - "Apply CD-6 fold-in: P1_FOCUS / P2_FOCUS / COREF_VALIDATION_FOCUS audited as bonus rows inside the VAL section header table (5 rows total). The original 15-row Verdict Summary at top of doc is NOT extended for these three constants per CD-6 — flagged inline only."
  - "CUT-VAL-01 (VALIDATION_RULES counterparts) gated by ALL THREE phase tags per Phase 44 §D-03 — VAL is the most conservatively gated builder in the audit."
  - "CUT-VAL-02 (opener `software architecture document`) is the third instance of the same opener pleonasm (AMB CUT-AMB-02 + EXT CUT-EXT-01 + VAL CUT-VAL-02) — Phase 46 should batch all three under a single replacement vocabulary."
  - "CUT-VAL-03 (COREF_VALIDATION_FOCUS `role-referential phrase`) carries risk = med-high because the constant gates the empirically-load-bearing asymmetric coref-validation pass (~4 FP reduction on BBB per docstring lines 101–105)."
  - "CUT-VAL-04 (P1_FOCUS qualified-name X.Y.Z clause) is a visibility-only tombstone with `after = DO NOT CUT — empirically validated load-bearing`. Risk = high; Phase 46 MUST skip it per Phase-45 threat T-45-VAL-02 mitigation and 45-RESEARCH.md §7.4."
  - "COREF_VALIDATION_FOCUS asymmetric single-pass design is NOT proposed for symmetrization (docstring lines 101–105 mark it empirically load-bearing); only the lexical `role-referential phrase` span carries a flag."
  - "Family A / Family B rewordings NOT emitted (D-06): no item escalates to benchmark-leak after second-pass isolation; the `service` Universal-Taboo hit in COREF_VALIDATION_FOCUS dismisses per 45-RESEARCH.md §1.6 as generic-SE-noun isolation."
metrics:
  duration_minutes: 1
  completed: 2026-06-08
---

# Phase 45 Plan 06: VAL (Validation) Section Audit Summary

VAL section audited: VALIDATION_RULES + _prompt_validation + 3 bonus folded constants (P1_FOCUS, P2_FOCUS, COREF_VALIDATION_FOCUS) per CD-6; 4 cut rows emitted (3 domain-loaded flags + 1 behavioral-protected tombstone); GATE-01 byte-equal preserved.

## Final Verdict per Item (5 rows)

| Item | Type | Verdict | Cut Rows |
|---|---|---|---|
| `VALIDATION_RULES` | constant | domain-loaded ("counterparts") | 1 (CUT-VAL-01) |
| `_prompt_validation` (prose) | builder | domain-loaded ("software architecture document") | 1 (CUT-VAL-02) |
| `P1_FOCUS` (folded per CD-6) | constant | clean (with docstring-protected X.Y.Z clause) | 1 (CUT-VAL-04, tombstone) |
| `P2_FOCUS` (folded per CD-6) | constant | clean | 0 |
| `COREF_VALIDATION_FOCUS` (folded per CD-6) | constant | domain-loaded ("role-referential phrase") | 1 (CUT-VAL-03) |

## Cut Row Count

**Total: 4 CUT-VAL-NN rows** (3 domain-loaded + 1 behavioral-protected tombstone).

| cut_id | trigger | risk | gated_by phase-tag count |
|---|---|---|---|
| CUT-VAL-01 | domain-loaded ("counterparts") | med | 3 (phase_4_twopass_p1, phase_4_twopass_p2, phase_5_coref_validation) |
| CUT-VAL-02 | domain-loaded ("software architecture document") | low | 3 (all 3) |
| CUT-VAL-03 | domain-loaded ("role-referential phrase") | med-high | 3 (all 3) |
| CUT-VAL-04 | behavioral-protected (docstring lines 5–22) | high | 2 (phase_4_twopass_p1, phase_4_twopass_p2 — phase_5_coref_validation N/A because COREF_VALIDATION_FOCUS, not P1_FOCUS, is injected at that phase) |

## P1_FOCUS Protected-Clause Acknowledgement

The clause `"and not just as a qualified-name identifier (e.g. a package- or member-access path X.Y.Z)"` at `prompts_v5.py:84–85` is documented as **behaviorally protected** by the `prompts_v5.py` module docstring (lines 5–22). Empirical record: `experiment_dotted_path_rename.py` shows the chosen wording catches 2/3 code-path FPs on gpt-5.4 AND 1/3 on Claude Sonnet with 0 collateral damage on the 4-TP control set; strict joint improvement over the prior `dotted-path identifier` wording (which catches 0/3 on Sonnet). CUT-VAL-04 records this as a Phase-46 tombstone with `after = DO NOT CUT — empirically validated load-bearing` and `risk = high`. Phase-45 threat T-45-VAL-02 (mitigate) is honored: the audit emits the row for visibility and explicitly bans cutting.

## COREF_VALIDATION_FOCUS Asymmetric-Design Acknowledgement

The asymmetric single-pass design of `_prompt_validation` when called with `COREF_VALIDATION_FOCUS` (vs the P1+P2 twopass for entity validation) is documented as **empirically load-bearing** per `prompts_v5.py` docstring lines 101–105 ("entity twopass leaks ~4 FPs on bigbluebutton coref"). The audit does NOT propose symmetrizing — that is a code-architecture change, not a prompt-text cut. Only the lexical `role-referential phrase` span carries a `domain-loaded` flag (CUT-VAL-03); the calling convention stays put.

## GATE-01 git-diff Verification

```
git diff --quiet \
  src/llm_sad_sam/linkers/experimental/s_linker19.py \
  src/llm_sad_sam/linkers/experimental/prompts_v5.py \
  src/llm_sad_sam/linkers/experimental/s_linker13_min.py
```

**Result:** PASS (exit 0; zero edits to frozen source artefacts).

## Cross-Section Observations (for 45-08 consolidation)

- **`software architecture` opener pleonasm — third occurrence:** AMB CUT-AMB-02 (line 266) + EXT CUT-EXT-01 (line 323) + VAL CUT-VAL-02 (line 339) are the same Phase-46 batching opportunity. A single approved replacement vocabulary (e.g. `components` alone, or `named elements`) resolves all three `domain-loaded` flags with one harness run per affected gate. This is the strongest batching opportunity in the entire audit so far.
- **VAL is the most conservatively gated section:** 24 snapshots across 3 phase tags (`phase_4_twopass_p1`, `phase_4_twopass_p2`, `phase_5_coref_validation`); every CUT-VAL-01/02/03 row carries all three tags in `gated_by`. CUT-VAL-04 is the only exception (P1_FOCUS does not appear in the `phase_5_coref_validation` path; that phase uses COREF_VALIDATION_FOCUS instead per Phase 44 §D-03).
- **No benchmark-leak verdict assigned:** all body-text tokens cleared the mechanical Universal-Taboo grep after second-pass isolation. The only ambiguous case (`service` in COREF_VALIDATION_FOCUS) dismisses per 45-RESEARCH.md §1.6 as generic-SE-noun isolation. Phase-46 Family A / Family B rewording slot stays empty for VAL.

## Deviations from Plan

None — plan executed exactly as written. The action specified emitting 3 domain-loaded flag rows (CUT-VAL-01/02/03) + optionally CUT-VAL-04 (visibility-only protected-clause tombstone). The audit emitted all four because the protected-clause tombstone is the cleanest Phase-46 input signal (the alternative — leaving P1_FOCUS without any cut row — would lose the explicit "do not cut" record). All other action steps (header table 5 rows, Verdict Summary update for VALIDATION_RULES + _prompt_validation only per CD-6, no symmetrize proposal for COREF_VALIDATION_FOCUS, closing reviewer paragraph) executed verbatim per the plan's Step 1–8 specification.

## Self-Check: PASSED

- FOUND: `.planning/phases/45-audit/s_linker20-PROMPT-AUDIT.md` (modified)
- FOUND: `.planning/phases/45-audit/45-06-SUMMARY.md` (created)
- VERIFIED: VAL section between `<!-- SECTION:VAL:START -->` and `<!-- SECTION:VAL:END -->` contains all 5 header items (VALIDATION_RULES, _prompt_validation, P1_FOCUS, P2_FOCUS, COREF_VALIDATION_FOCUS)
- VERIFIED: All 3 phase tags (`phase_4_twopass_p1`, `phase_4_twopass_p2`, `phase_5_coref_validation`) present in body
- VERIFIED: 4 CUT-VAL-NN rows present (≥3 minimum satisfied)
- VERIFIED: P1_FOCUS qualified-name `X.Y.Z` protected clause documented (CUT-VAL-04 tombstone + Notes column + protected-clause record blockquote)
- VERIFIED: COREF_VALIDATION_FOCUS asymmetric-design record present (Notes column + asymmetric-design record blockquote + closing reviewer paragraph)
- VERIFIED: GATE-01 — `git diff --quiet` on the 3 frozen source files returned exit 0
- VERIFIED: CUT-VAL-01/02/03 each carry all 3 phase tags in `gated_by`; CUT-VAL-04 carries 2 (P1_FOCUS-applicable tags only — correct per Phase 44 §D-03)
- VERIFIED: Verdict Summary updated for VALIDATION_RULES (domain-loaded, 1 cut) and _prompt_validation (domain-loaded, 1 cut) only — 3 folded constants flagged inline per CD-6, NOT added to the 15-row summary
