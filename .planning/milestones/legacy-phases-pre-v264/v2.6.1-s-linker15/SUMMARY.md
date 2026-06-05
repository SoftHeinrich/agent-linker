---
milestone: v2.6.1
phase: v2.6.1-02
title: s_linker15 — No-Training Axiom Linker
status: complete
---

# Phase v2.6.1-02 SUMMARY

**One-liner:** Shipped `s_linker15` — `s_linker14_voyager` with all Voyager bank/training machinery
removed, axiom prompts inlined (B-variant + 3 FP fixes), ILinker4 on empty seed rules; registered
alongside s14, canonical s13_min untouched.

Built by copying s14 and excising `_load_bank`/`_wrap`/`reload_bank`/`_slot_text`/`SLOT_NAMES`/
`DEFAULT_BANK_PATH`/`bank_path`; prompts inlined so s14 + prompts_v4_axiom stay byte-unchanged.
Smoke-tested (import, instantiate, FP fixes present, bank symbols absent, registration resolves).
Validated dual-backend: GPT-5.4 89.1% macro, Claude 92.7% macro. No-training ties trained s14 on GPT.
Commits: 30e4a02 (build), 2f2c5e4 (dual-backend results).
