---
milestone: v2.6.1
phase: v2.6.1-03
title: Dead-Code Cleanup (in s15) + Docs
status: in progress
budget: $0
---

# Phase v2.6.1-03 — Dead-Code Cleanup + Docs

## Scope correction (user, 2026-06-02)

"Remove dead infra" = **remove dead bank/training code INSIDE s_linker15** so it is
structurally a clean standalone s_linker-family file — NOT delete the repo-wide
voyager training scripts. s_linker14_voyager and the voyager_train_tlr*.py scripts
are left in place (s_linker14 retained alongside).

## Tasks

- [x] Dead code removed from s_linker15 (bank/training surface — see Phase 02).
- [x] Unused imports pruned from s_linker15 (`Path`, `Any`).
- [x] Repo `CLAUDE.md`: document s_linker15 as current no-training axiom linker +
  backend reality (gpt-5.4 active).
- [ ] `MEMORY.md` (auto-memory): add v2.6.1 / s_linker15 entry.
- [ ] `README.md` / `AGENTS.md`: add s_linker15 mention if they enumerate variants.
- [ ] Planning: STATE.md progress + ROADMAP phase ticks once validation lands.
- [ ] (Deferred, not this phase) Repo-wide voyager training-script removal — only if
  user later requests it. Out of scope per scope correction above.

## Verification

- `python -m py_compile s_linker15.py` ✓
- No `_load_bank/_wrap/reload_bank/_slot_text/SLOT_NAMES/DEFAULT_BANK_PATH` in s15 ✓
- s_linker14_voyager.py + prompts_v4_axiom.py untouched ✓
- Docs name s_linker15 as the v2.6.1 linker.
