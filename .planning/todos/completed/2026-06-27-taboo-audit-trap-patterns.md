---
created: 2026-06-27T15:45:00.000Z
title: Taboo-audit the Mode 2 trap-pattern checklist for the no-reasoning validator
area: prompts
priority: medium
files:
  - BENCHMARK_TABOO.md
  - src/llm_sad_sam/linkers/experimental/s_linker20_union.py
  - .planning/spikes/004-nogap-validator-ab/README.md
  - .planning/notes/2026-06-27-nogap-validator-modes.md
---

## Task

Draft the Mode 2 trap-pattern red-flag checklist for the layered no-reasoning
validator, and verify every entry stays **structural/linguistic** rather than
**benchmark-derived**.

## Why

The trap-pattern rejecter is the highest-precision-recovery mode (kills the
systematic teammates enumeration FPs and the negation case) but also the most
overfit-prone. The patterns observed are structural — overview/header sentences,
negations ("X is not a …"), unresolved pronouns, package/module enumerations,
test-scaffolding descriptions — so they *should* pass BENCHMARK_TABOO, but this
must be confirmed, not assumed, before any of it goes into a prompt.

## Done when

- [x] Red-flag checklist drafted (each item = a linguistic structure + a generic,
      benchmark-free illustrative example). → `spikes/004-nogap-validator-ab/harness/traps.py`
      (overview_header, negation, qualified_path, deictic_pronoun, test_scaffolding).
- [x] Each item cross-checked against `BENCHMARK_TABOO.md` — confirmed structural/generic
      English only; no benchmark component name, alias, or project keyword.
- [x] Borderline examples kept stopword/generic.
- [x] Handed to spike 004.

## RESOLVED (2026-06-27) — Mode 2 REJECTED as hard rules

Spike 004 measured the trap checklist on cached links: applied as a hard sentence-level
post-filter, EVERY trap nets negative (ALL_TRAPS −7.5 macro; removes ~3× more true links
than false). Root cause: nothink's surplus FPs share sentences with true links (e.g.
"Architecture contains UI, Logic, Storage, … Test Driver Component" holds 4 true links but
trips test_scaffolding/overview_header). The patterns ARE taboo-safe and survive only as
*hints inside the LLM rubric* (the v3/v4 code-path + negation clauses), never as a blanket
filter. No further action — see `spikes/004-nogap-validator-ab/RESULTS.md` Stage 0b.
