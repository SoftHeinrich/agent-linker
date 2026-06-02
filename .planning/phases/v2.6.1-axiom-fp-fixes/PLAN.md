---
milestone: v2.6.1
phase: v2.6.1-01
title: Axiom FP Root-Cause Fixes
status: not_started
budget: $0 (ablation harness only; cached results)
source_todo: .planning/todos/pending/2026-06-02-improve-prompts-v4-axiom-three-root-cause-fp-fixes.md
---

# Phase v2.6.1-01 — Axiom FP Root-Cause Fixes

## Goal

Patch three distinct axiom slots in `prompts_v4_axiom.py`, each targeting one root cause of residual
TM/BBB false positives in the B-variant axiom (hash `61e038`). Test each as an isolated ablation;
combine only if all three are individually improving-or-neutral.

**Success:** TM macro improves from 82.26% with no regression on MS / TS / JAB.

## Harness

- `scripts/ablation_validation_rules.py` + `scripts/run_ablation_variant.py`
- 5-project run, empty bank, keyed by axiom hash. Results cached in `results/voyager_v5/cache/`.
- Current B-variant hash: `61e038`. Pre-axiom baseline: `bcae0e`.

## Tasks

### Task 1 — Cause A: Tier/platform alias (DOC_KNOWLEDGE_JUDGE_RULES)
6 FPs (UI×4 + GAE Datastore×2). Alias names a broader TIER/PLATFORM containing the component, not
the component itself (PCM is flat — no containment). Fix at source: prevent tier aliases entering
the alias set.
- Patch `DOC_KNOWLEDGE_JUDGE_RULES`: add rule that an alias naming an architectural tier or
  technology platform encompassing multiple elements is invalid (grouping, not a single named unit).
- Run ablation → record TM FP delta. Must not regress MS/TS/JAB.

### Task 2 — Cause B: Code-path prefix leakage (ENTITY_EXTRACTION_RULES)
4 FPs (Logic S85, Storage S125/S136, Common S127). Component name appears as prefix in Java
package/class path; existing "code-level path" exclusion leaks via "semantically related → include"
reasoning.
- Strengthen `ENTITY_EXTRACTION_RULES`: add "even if the compound identifier is semantically
  related to the component."
- Run ablation → record TM FP delta. Must not regress MS/TS/JAB.

### Task 3 — Cause C: Functional alias as workflow subject (SEED_DISAMBIGUATION_RULES)
BBB Presentation-Conversion cluster (5+ FPs). "conversion process" is a valid alias but sentences
describe the WORKFLOW/ACTIVITY, not the component as an architectural unit. The "or activity"
variant (A) did NOT fix these — must use the functional-alias removal heuristic.
- Patch `SEED_DISAMBIGUATION_RULES`: functional-alias check — if removing the alias from the
  sentence still leaves an accurate description of a process step/activity → classify OTHER;
  COMPONENT only when the alias is clearly treated as the name of a specific architectural unit.
- Run ablation → record BBB FP delta. Must not regress MS/TS/JAB.

### Task 4 — Conditional combined test
If Tasks 1–3 are each individually improving-or-neutral → run combined-axiom ablation.
- Watch for negative interaction (cf. already-documented "or activity" + "+counterparts" = −3.2pp on
  TM). Record combined macro + per-dataset.
- Decide final v2.6.1 axiom: combined if no interaction, else best-individual subset.

## Verification

- TM macro > 82.26% on the chosen final axiom.
- MS / TS / JAB no regression vs B-variant (`61e038`).
- GATE-06: every new rule text free of benchmark vocabulary (abstract phrasing only).
- New axiom hash recorded; cache entry written.

## Notes / Anti-patterns

- Do NOT combine "or activity" with "+counterparts" in VALIDATION_RULES (documented −3.2pp TM).
- Cause C must use the functional-alias removal heuristic, NOT the failed "or activity" variant.
- Zero hardcoded/tailored values — `feedback-no-hardcoding`. Rules phrased as universal principles.
