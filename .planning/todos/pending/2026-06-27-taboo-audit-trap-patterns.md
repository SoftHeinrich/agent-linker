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

- [ ] Red-flag checklist drafted (each item = a linguistic structure + a generic,
      benchmark-free illustrative example).
- [ ] Each item cross-checked against `BENCHMARK_TABOO.md` — no component name,
      alias, or project-specific keyword from any of the 5 benchmark projects.
- [ ] Any borderline example rewritten with stopwords / generic English only.
- [ ] Checklist handed to spike 004 for implementation.
