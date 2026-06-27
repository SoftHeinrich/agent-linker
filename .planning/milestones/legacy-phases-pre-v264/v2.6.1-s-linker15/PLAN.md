---
milestone: v2.6.1
phase: v2.6.1-02
title: s_linker15 — No-Training Axiom Linker
status: implemented (validation in flight)
budget: $0 code; validation run = gpt-5.4 5-project (shared with Phase 01)
---

# Phase v2.6.1-02 — s_linker15 (No-Training Axiom Linker)

## Goal

Ship `s_linker15.py`: `s_linker14_voyager` with ALL Voyager trained-bank machinery
removed, axiom prompts inlined (with the three Phase-01 FP fixes baked in), seed
extractor on empty rules. Standalone file (s_linker family style). Register
alongside s_linker14_voyager. Canonical s_linker13_min untouched.

## Design decisions (confirmed with user 2026-06-02)

- Seed extractor: **keep ILinker4**, constructed with `seed_extraction_rules=""`,
  `seed_actor_rules=""` (pure axiom seed, ILinker3-equivalent, no bank).
- Prompts: **inlined** into s_linker15 (NOT importing prompts_v4_axiom) so that
  s_linker14_voyager + prompts_v4_axiom stay byte-for-byte untouched and the FP
  fixes live only in s_linker15. Matches standalone-file preference.
- Role: **alongside s_linker14_voyager** (both registered, experimental=True).

## Tasks

- [x] Copy s_linker14_voyager.py → s_linker15.py.
- [x] Remove bank surface: `_load_bank`, `_wrap`, `reload_bank`, `_slot_text`,
  `SLOT_NAMES`, `_LEARNED_HEADER`, `DEFAULT_BANK_PATH`, `bank_path` param,
  `self._slot_patterns`, `self._bank_path`, pattern-count logging.
- [x] Inline 11 axiom prompts (B-variant) as module constants; assign directly in
  `__init__`. Drop `prompts_v4_axiom` import.
- [x] Bake the three FP fixes into the inlined prompts (DOC_KNOWLEDGE_JUDGE_RULES,
  ENTITY_EXTRACTION_RULES, SEED_DISAMBIGUATION_RULES).
- [x] Two empty-slot f-string sites (`_slot_text(...)`) → blank.
- [x] ILinker4 with empty seed rules.
- [x] Rename class `SLinker14Voyager` → `SLinker15`; `_VARIANT_NAME = "s_linker15"`;
  retarget docstring. Drop now-unused `Path`/`Any` imports.
- [x] Register in run_ablation.py (CANONICAL_VARIANTS + VARIANT_SPECS) alongside s_linker14.
- [x] Smoke test: import, instantiate, FP fixes present, bank symbols absent, registration resolves. PASS.
- [~] Validation: `python run_ablation.py --variants s_linker15` over 5 projects (gpt-5.4). IN FLIGHT.

## Verification

- s_linker15 runs with no bank file present (axiom-only). ✓ (instantiates clean)
- s_linker14_voyager.py + prompts_v4_axiom.py unchanged (git diff = none). ✓
- Axiom-only macro F1 ≥ s_linker14_voyager axiom-only floor. PENDING validation run.
- GATE-06: inlined axiom text free of benchmark vocabulary. ✓ (textbook SE terms only)
