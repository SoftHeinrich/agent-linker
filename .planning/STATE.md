---
gsd_state_version: 1.0
milestone: v2.5
milestone_name: Oracle Cache Fix + 15-Slot Expansion + Re-run
status: planning
last_updated: "2026-06-01T17:39:20.519Z"
last_activity: 2026-06-01
progress:
  total_phases: 6
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-01 for v2.5 kickoff)

**Core value:** Every rule removed and every prompt-rule trimmed must hold macro F1 ≥ 0.93 on Claude Sonnet AND gpt-5.4 macro within tolerance of the v2.0 baseline (0.9077) — or be rejected. Generality first (GATE-06).
**Current focus:** v2.5 — Oracle Cache Fix + 15-Slot Expansion + Re-run. Roadmap defined; Phase 25 is next.

## Current Position

Phase: 25 — Infrastructure Fixes
Plan: —
Status: planning
Last activity: 2026-06-01 — Roadmap v2.5 created (Phases 25–30)
Next action: Phase 25 — Infrastructure Fixes

```
Progress: [                              ] 0% (0/6 phases)
```

## v2.5 Design Decisions (Locked)

All locked in v2.5 requirements and milestone audit (2026-06-01). Full details in `.planning/milestones/v2.5-REQUIREMENTS.md`.

### A-1 Fix: Oracle Cache Contamination

**Root cause of v2.4 split-3 TM reuse**: Oracle cache key in `voyager_train_tlr_v4_beta.py` line 455 does not include `bank_content_hash`. Split-3 TM reuses mainline oracle outputs when bank state differs, invalidating cross-split diversity.

**Fix**: Add `bch = _bank_content_hash(bank)` to oracle key — same pattern as L cache fix already in place.

### A-3 Fix: Probation Variance

**Root cause**: `delta > 0` threshold allows ±3–4pp BBB LLM run-to-run variance to trigger false commits.

**Fix (user-selected)**: Raise threshold to `delta >= 0.005`. No extra LLM calls. Over-rejection risk accepted.

### A-4 Fix: D Slot Steering

**Problem**: D over-proposes DOC_KNOWLEDGE_EXTRACTION_RULES while ENTITY_EXTRACTION_RULES, AMBIGUITY_FEW_SHOT, DOC_KNOWLEDGE_JUDGE_EXAMPLES remain empty.

**Fix**: After per-slot pattern count display in D prompt, list zero-pattern slots by name as high-priority proposal targets.

### B-1: 15-Slot Expansion

**6 new slots**: `SEED_EXTRACTION_RULES`, `SEED_ACTOR_RULES`, `GENERIC_WORD_USAGE_RULES`, `ALIAS_SCOPE_RULES`, `ANTECEDENT_ALIAS_RULES`, `COREF_TERMINAL_SPECIFICITY_RULES`.

**`ilinker3.py` frozen**: `ILinker3Injected` subclass in `s_linker14_voyager.py` overrides `_prompt_extract()` and `_prompt_actor()` — frozen-artifact policy preserved.

**4 inline prompts wired**: ALIAS_SCOPE_SCHEMA, ANTECEDENT_ALIAS_GUIDE, generic filter, `_classify_specific_terminals()` prompt.

## Phase Sequence

| Phase | Name | Budget | Condition |
|-------|------|--------|-----------|
| 25 | Infrastructure Fixes | $0 | Unconditional |
| 26 | 15-Slot Expansion | $0 | After Phase 25 |
| 27 | Probe Tier | ≤ $10 gpt-5.4 | After Phase 26 |
| 28 | Range Tier | ≤ $25 gpt-5.4 | Phase 27 CONTINUE |
| 29 | Confirmation Tier | ≤ $60 gpt-5.4 | Phase 28 ≥ 0.87 |
| 30 | Milestone Close | $0 | Unconditional |

## Standing Gates (carried forward)

- **GATE-01**: `s_linker13_min` macro F1 ≥ 0.93 Claude AND gpt-5.4 ≥ 0.8977. Applies to canonical only. Formal check at Phase 29 and Phase 30.
- **GATE-06**: BENCHMARK_TABOO grep + reviewer_critic. Applies at bank-entry AND at all new slot seed text (Phase 26) AND all new Oracle + Distillator prompt text (Phase 26).
- **GATE-07**: `s_linker14_voyager` registered experimental=True; DEFAULT_BANK_PATH updated to v2.5 `cross_split_final_bank.json` after Phase 29 Confirmation.
- **GATE-08**: Total training budget cap ≤ $80 gpt-5.4 (Phases 27–29). Infrastructure phases 25, 26, 30 have zero LLM training cost.

## Canonical Artifact (current)

- **`src/llm_sad_sam/linkers/experimental/s_linker13_min.py`** (v2.1 PROMOTED, `canonical=True`, unchanged)
- Claude Sonnet macro F1: 0.9506 | gpt-5.4 macro F1: 0.9069

## v2.4 Summary (for context)

**Verdict:** WEAK — cross-split macro F1 = 90.5% (gpt-5.4, 5-dataset). Gap to canonical: −0.19pp.
**Key findings:** Oracle cache contamination invalidated cross-split diversity (all splits got identical Distillator proposals). Split-2 0/5 commits due to oracle key bug (not probation threshold, as v2.3 assumed). Slot steering gap: ENTITY_EXTRACTION_RULES never proposed — D always over-proposed to already-filled slots. 6 new static prompts outside axiom scope identified → v2.5 bank 9→15 expansion.
**See:** `.planning/milestones/v2.4-MILESTONE-AUDIT.md`

## Accumulated Context

### Decisions (locked for v2.5)

- Probation fix = raise threshold to `delta >= 0.005` (not multi-run averaging — cheaper, user-selected)
- `ilinker3.py` stays frozen — `ILinker3Injected` subclass in `s_linker14_voyager.py`
- gpt-5.4 default backend; Claude only if explicitly required
- β architecture (L + O + D-with-CoT-A + P) — same roles, same topology
- Bank format: 15 slots uniform, per-project during training, aggregated `cross_split_final_bank.json`
- Flex tier deferred (cost optimization, out of scope v2.5)

### Pending Todos from v2.4

- `.planning/todos/pending/260601-flex-tier-integration.md` — deferred, out of scope v2.5
- `.planning/todos/pending/260601-ilinker-prompts-not-axiomed.md` — addressed by REQ-V25-04/05/06 (Phase 26)

### Blockers/Concerns

None. All v2.5 requirements have locked designs; Phase 25 is implementation-ready.

## Deferred Items (v2.6+ candidates)

- Claude cross-model verification of v4 (per backend policy)
- OpenAI Flex tier integration (cost optimization)
- Per-document or runtime-adaptive bank building
- Complete v4 re-architecture beyond infra fix + slot expansion

## Session Continuity

Last session: 2026-06-01T17:39:20.519Z
Stopped at: v2.5 roadmap created — Phase 25 ready to plan
Resume file: None
Next action: Phase 25 — Infrastructure Fixes (`/gsd-plan-phase 25`)
