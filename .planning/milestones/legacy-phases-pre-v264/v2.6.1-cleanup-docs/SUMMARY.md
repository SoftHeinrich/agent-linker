---
milestone: v2.6.1
phase: v2.6.1-03
title: Dead-Code Cleanup + Docs
status: complete
---

# Phase v2.6.1-03 SUMMARY

**One-liner:** Removed dead bank/training code inside s_linker15 (standalone s_linker-family style)
and updated docs (CLAUDE.md, MEMORY, STATE, planning) for the no-training direction.

Scope correction applied: "remove dead infra" = dead code in s15, NOT repo-wide voyager script
deletion (s14 + voyager_train_tlr*.py retained). Unused imports pruned. README/AGENTS left as-is
(no variant enumeration to update). Commit: 30e4a02 + docs commits.
