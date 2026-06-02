# P36 Probe Forensics — v2.7 Prep Tier-A Rollback Verification

**Date:** 2026-06-02
**Context:** Post Phase 34 KILL + Phase 35 overshoot rollback. Verifies Tier-A axiom rollback shipped in session-end commit 6109b58.

## Verdicts

| Run | Split | TRAIN | TEST | Verdict | Passes |
|---|---|---|---|---|---|
| P34 baseline | mainline | 0.9058 | 0.8486 | KILL | 3 |
| P35 overshoot | mainline_v27 | 0.8732 | 0.8369 | KILL | 2 |
| **P36 rollback** | **mainline_v28** | **0.9122** | **0.8501** | **KILL** | **2** |
| P36 second-run | mainline_v28 | 0.8971 | 0.8501 | CONTINUE | 3 |
| Range P36 | mainline_v28 | 0.8782 | — | — | 1 |

## TM Recovery (Tier-A Rollback Goal)

Pass 1→2 trajectory: 0.8160 → 0.8571 (+0.0411). Errors 33→24 (−9).

Identity statements (S88 "Logic is a Facade class", S87 "Logic API is represented by...") now reach assessor and are correctly classified. The "concrete artifact / flow required" gate was the overshoot source; removing it restored TM without re-introducing P34's gerund FPs.

Specific axiom slots filled in pass 2:
- `SEED_ACTOR_RULES` (appositional role inheritance — S87-88)
- `GENERIC_WORD_USAGE_RULES` (gerund fragment "Hiding the complexities..." — S122)
- `COREF_TERMINAL_SPECIFICITY_RULES` (conceptual/connector anaphora — S24/S27/S92/S94)
- `AMBIGUITY_RULES` (conceptual-package alias-of guard — S23-27)

Pass 2 had 8 assessor accepts, 0 rejects. No `removal_targets` fired (no regression to revert).

## BBB Ceiling — Tier C Target Evidence

BBB F1: 0.7273 (pass 2). 11 FN, no FP confirmed. **Two distinct surface-form patterns**, not one:

### Pattern A — Terminal-Partial / Modifier-Stripped

| Sentence | Surface form | Missed target |
|---|---|---|
| S12 "subscribes to ... collections on the server side" | "server side" | HTML5 Server |
| S12 "the client side ..." | "client side" | HTML5 Client |

Maps to original Tier-C scope (handoff): partial = 4th surface variant after pronoun/role-NP/alias.

### Pattern B — Service-Of / Sub-Component (NEW — not in original Tier-C scope)

| Sentence | Surface form | Missed target |
|---|---|---|
| S10 "The MongoDB database contains information..." | MongoDB | HTML5 Server (MongoDB is its storage subsystem) |
| S13 "Updates to MongoDB ... pushed to MiniMongo" | MongoDB / MiniMongo | HTML5 Server / HTML5 Client |
| S26 "send events to akka-apps" | akka-apps | Apps (alias/abbreviation) |

Requires knowing MongoDB IS HTML5 Server's persistence layer — runtime LLM discovery from doc text OR PCM sub-component projection. Current COREF (pronoun/role-NP/alias) cannot reach.

### Tier C Scope Revision

Two axiom slots, not one:

1. `COREF_TERMINAL_PARTIAL_RULES` — modifier-stripped + section-context anchor (original scope)
2. `SUB_COMPONENT_OF_RULES` — internal-service → parent-component (new finding)

Estimated lift: terminal-partial saves ~2 BBB links (+0.015 F1), service-of saves ~3-4 (+0.03 F1). Combined BBB → ~0.77. Still short of 0.87 gate alone.

## GATE-06 False Positive — 'persistence' Taboo

Pass 1 mediastore OD proposed a clean `SEED_EXTRACTION_RULES` pattern (predicate stored-in/retrieved-from disambiguation). Rejected by `scripts/voyager_train_tlr_v5.py:104-113` which hardcodes `Persistence` alongside actual benchmark project names.

Verification: `Persistence` is NOT a mediastore component name (no "Persistence" or "PersistenceManager" in gold). Generic SE term used in pattern's instructive example block.

**Fix:** Remove `Persistence` from taboo regex OR restrict to exact-case word-boundary match so common-noun usage passes.

## Loop Machinery Health

- **Best-bank checkpoint/restore**: P36-rollback correctly restored pass 2 (0.9122); P36-second correctly restored pass 1 (0.8971) when pass 2 regressed to 0.8782. Logic verified.
- **MAX_OUTER_PASSES=6**: early-stopped at pass 2 (KILL) / pass 3 (CONTINUE decay). Did not exhaust limit.
- **FM-exclusion (covered_fm_titles)**: pass 2 OD proposed novel FMs (COREF_TERMINAL_SPECIFICITY) not in pass 1. Diversity constraint working.
- **Slot quota**: empty `COREF_TERMINAL_SPECIFICITY_RULES` got populated in pass 2. Pressure working.
- **MIN_COMMIT_DELTA removed**: no variance wedge. Loop fully assessor-driven.

## Bank Composition (P36-rollback Pass 2 Final)

11/15 slots populated, 14 patterns total (identical across train projects via merge):

Filled: AMBIGUITY_RULES(2), COREF_RULES(1), DOC_KNOWLEDGE_EXTRACTION_RULES(1), VALIDATION_RULES(1), SEED_DISAMBIGUATION_RULES(1), SEED_EXTRACTION_RULES(1), SEED_ACTOR_RULES(1), GENERIC_WORD_USAGE_RULES(2), ALIAS_SCOPE_RULES(1), ANTECEDENT_ALIAS_RULES(1), COREF_TERMINAL_SPECIFICITY_RULES(2).

Empty: AMBIGUITY_FEW_SHOT, DOC_KNOWLEDGE_JUDGE_EXAMPLES, DOC_KNOWLEDGE_JUDGE_RULES, ENTITY_EXTRACTION_RULES.

v2.5 debt cleared: SEED_EXTRACTION_RULES + SEED_ACTOR_RULES now populated via Task-1 ilinker3 prompt lift.

## Headline Insights

1. **Tier-A rollback succeeded narrowly** — TM recovered (+4.1pp on pass 2) without re-introducing P35 overshoot symptoms. Identity statements flow through correctly.
2. **BBB ceiling needs TWO Tier-C axiom slots, not one** — terminal-partial (S12 "server side") + service-of/sub-component (S10/S13 MongoDB → HTML5 Server). Combined lift ~0.77 BBB, still short of 0.87 gate.
3. **GATE-06 has false-positive on 'persistence'** — costs one valid pattern per run. Cheap housekeeping fix; symptomatic of over-broad taboo regex.
4. **Loop machinery healthy** — best-bank restore + FM-exclusion + slot quota all firing correctly across both P36 runs.

## Follow-Up Actions

1. **GATE-06 fix** (5 min) — Remove `Persistence` from taboo regex in `scripts/voyager_train_tlr_v5.py:104-113`. Re-run pass 1 mediastore OD to reclaim pattern.
2. **TM error recount** — Read latest `results/llm_logs/s_linker14_voyager_teammates_*.json` pass-2 output; pass-2 patterns target ~7 of 17 reported errors; true remaining likely ~10.
3. **Tier C scope expansion** — Update `.continue-here.md` task 9 to design TWO axiom slots (terminal-partial + sub-component-of). Sub-component-of requires either runtime LLM discovery from SAD text or PCM hierarchy projection.
4. **Re-run probe** post Tier C — measure actual BBB lift.
