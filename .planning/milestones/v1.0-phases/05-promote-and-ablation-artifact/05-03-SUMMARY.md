---
phase: 05-promote-and-ablation-artifact
plan: 03
subsystem: docs
tags: [methodology, writeup, milestone-deliverable]

requires:
  - phase: 05-promote-and-ablation-artifact
    provides: ABLATION-TABLE.md (Plan 05-02) for inline citation
provides:
  - METHODOLOGY.md (7-section ~1834-word milestone writeup)
affects: []

tech-stack:
  added: []
  patterns: [third-person research-report voice; inline citations to .planning/ artifacts]

key-files:
  created:
    - .planning/phases/05-promote-and-ablation-artifact/METHODOLOGY.md
  modified: []

key-decisions:
  - "Voice: third-person research report; zero first-person pronouns in own prose (verified via grep)"
  - "Section ordering: 7-section orchestrator mapping (D-50 8 sections collapsed)"
  - "BENCHMARK_TABOO footnote added per Step 2f recommendation"

patterns-established:
  - "All numbers quoted from ABLATION-TABLE.md (single source of truth); no new numbers introduced"

requirements-completed: [PROMO-04]

duration: 12min
completed: 2026-05-29
---

# Phase 05-03: Methodology Writeup (PROMO-04) — Summary

**PROMO-04 satisfied: METHODOLOGY.md (1834 words, 7 sections) documents the rule-removal chain, standing-policy history, 13d failure mode, dual-hard-tier protocol for 13e, and deferred items EXT-01/EXT-02/EXT-03.**

## Performance

- **Duration:** ~12 min
- **Completed:** 2026-05-29
- **Tasks:** 2 of 2 (writing + verification)
- **Files modified:** 1 created

## Accomplishments

1. Wrote `METHODOLOGY.md` with the 7 LOCKED sections per orchestrator/D-50 mapping:
   - §1 Project Thesis (`Traceability Linking Without Hand-Crafted Rules`)
   - §2 The 13-Series Chain: Six Removed Rules (longest section, ~550 words)
   - §3 Standing-Policy History (BBB tolerance 2 pp → 4 pp → 6 pp)
   - §4 Negative Result: 13d / VAR-04 Retirement
   - §5 Dual-Hard-Tier Protocol for 13e (BBB Run 1 = 0.826, Run 2 = 0.818, |Δ| = 0.008)
   - §6 Final Result: `s_linker13` macro F1 = 0.9509 (with the 13b-was-higher-macro D-43 footnote)
   - §7 Deferred Items (EXT-01, EXT-02, EXT-03 + forward-pointer to STATE.md)
2. Word count 1834 (within 1500-2500 target band per D-50).
3. Inline citation to `ABLATION-TABLE.md` (4 references) and to source SUMMARYs (`03-01-SUMMARY.md`, `04-01-SUMMARY.md`, etc.) at every quoted-number site.
4. Voice spot-check: zero first-person pronouns in the planner's own prose (verified via `grep -iE '\b(we|our|us|I)\b' = 0 hits`).
5. BENCHMARK_TABOO scope footnote added (Step 2f optional, included per recommendation).

## Verification (D-54 SC-4)

- `wc -w METHODOLOGY.md` → 1834 (in band 1500-2500)
- `grep -cE '^## ' METHODOLOGY.md` → 7
- All 19 LOCKED tokens present (verified via the name-coverage script)
- `ABLATION-TABLE` cited 4 times (inline reference present)
- EXT-01/EXT-02/EXT-03 each named with one-line descriptions in §7
- First-person hits: 0

## Deviations

- (a) §2 word count slightly under 600 (~550); ROADMAP section balance is fine.
- (b) Slightly compressed §5 vs the recommended 200-250 words (uneventful pass narrative is naturally short).
- (c) Used `2 pp` (with space) rather than `2pp` (concatenated) for readability; grep coverage script accepts both.
- (d) BENCHMARK_TABOO footnote (Step 2f optional) added as the closing horizontal-rule note.

## SC mapping

- D-54 SC-4 (methodology writeup exists, covers 8 sections via 7-section mapping, 1500-2500 words) → PASS
