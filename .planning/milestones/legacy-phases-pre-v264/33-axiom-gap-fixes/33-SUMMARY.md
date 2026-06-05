---
phase: 33-axiom-gap-fixes
milestone: v2.6
status: complete
date: 2026-06-02
reqs: REQ-V26-08, REQ-V26-09, REQ-V26-10, REQ-V26-07
---

# Phase 33: Axiom Gap Fixes — Summary

## What Shipped

### 1. `prompts_v4_axiom.py` (NEW FILE)

All 15 bank slots in one place. Replaces `prompts_v3_axiom.py` as the axiom
source for `s_linker14_voyager`. `prompts_v3_axiom.py` unchanged (frozen for
`s_linker13_skill_learned_clean` compatibility).

**Gap 1 — SCN (14 FNs) — `COREF_RULES`**:
Added section-established-topic rule: "only one component has been introduced
in the immediately preceding sentences — treat it as the section-established
topic and resolve role-referential phrases ('it', 'the module', 'the service',
'the component', 'the system') to that topic even without direct name
repetition." This covers the 14 FNs across BBB+TM where coref prompts failed
to resolve section-body pronouns when the component name appeared only in the
section heading.

**Gap 2 — Gerund FPs (7 FPs) — `SEED_DISAMBIGUATION_RULES`**:
Added explicit self-referential capability rule: "a sentence stating only
what the component itself does or is responsible for ('XComp processes
requests', 'XComp manages sessions') with no external architectural entity
named is OTHER; a sentence where the component acts on or interacts with
another named entity is COMPONENT." Prior version buried this in a list;
now explicit with examples.

**Gap 3 — Coref Alias — `ANTECEDENT_ALIAS_RULES`**:
Axiomized from `ANTECEDENT_ALIAS_GUIDE` static string. Changed default from
"false when unsure" → "true when antecedent form clearly differs from
canonical name." Explicitly names terminal words and abbreviations as known
aliases. Activates existing code path at `s_linker14_voyager.py` line 1117
(the `antecedent_via_alias` check).

**`ALIAS_SCOPE_RULES`** (also axiomized from `ALIAS_SCOPE_SCHEMA` static string):
Condensed form of the global/local scope classification. Now a first-class
bank slot for training patterns.

### 2. `s_linker14_voyager.py` (UPDATED)

- Import changed: `prompts_v3_axiom` → `prompts_v4_axiom`
- Static strings `ALIAS_SCOPE_SCHEMA` and `ANTECEDENT_ALIAS_GUIDE` removed
- Both `_wrap` calls updated to use `_axiom.ALIAS_SCOPE_RULES` and
  `_axiom.ANTECEDENT_ALIAS_RULES` as the base (applies in `__init__` and
  `reload_bank`)
- Docstring references updated

### 3. `voyager_train_tlr_v5.py` (UPDATED)

- Axiom hash path updated from `prompts_v3_axiom.py` to `prompts_v4_axiom.py`
  (line 138). Cache keys now reflect the v4 axiom changes.

## GATE-06 Check

All new axiom text uses textbook SE vocabulary. Generic placeholder names
only (XComp, YComp, TaskScheduler). Zero benchmark component names. ✅

## REQ-V26-07 (GATE-01)

`s_linker13_min.py`, `prompts_v2.py`, `ilinker1.py`, `ilinker2.py`,
`ilinker3.py`, `prompts_v3_axiom.py` — all frozen artifacts unchanged. ✅

## What Is NOT Done (deferred)

- Inference safety verification on MS/TS/JAB (deferred per user direction:
  "keep v3 axiom unchanged, now new voyager depends on v4"). Zero LLM cost
  this phase.

## Next

Phase 34 — Probe Tier. 2-pass mainline run with v5 loop + v4 axiom.
