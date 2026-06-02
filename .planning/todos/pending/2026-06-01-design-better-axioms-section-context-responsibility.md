---
created: 2026-06-01T12:58:44.606Z
title: Design better axioms for section-context and responsibility-list gaps
area: tooling
resolves_phase: 33
files:
  - src/llm_sad_sam/linkers/experimental/s_linker14_voyager.py:1004
  - src/llm_sad_sam/linkers/experimental/s_linker14_voyager.py:224
  - src/llm_sad_sam/linkers/experimental/helper_v3.py:164
  - src/llm_sad_sam/linkers/experimental/prompts_v3_axiom.py
  - scripts/voyager_train_tlr_v4_beta.py
  - results/voyager_v4_beta/mainline/final_bank.json
---

## Problem

Phase 16 Range Tier ended at WEAK verdict (89.8% macro F1, 87.6% axiom-only floor, +2.2pp from bank).
BBB is stuck at 77.6% (17 FN), TM at 83.9% (15 FP, 5 FN). Distillation training loop saturates quickly
because all 9 bank slots are sentence-local syntactic rules; the remaining failures require document-structure
awareness the axiom vocabulary cannot express.

Empirical analysis of 210 complete 5-dataset runs + full FN/FP sentence inspection identified 3 root-cause gaps:

**Gap 1 — Section-context naming (SCN): 14 FNs across BBB+TM**
BBB FNs S6, S12, S13, S76, S79 = sentences using "the server"/"the client"/"server side" where
HTML5 Server/Client was established 1-8 sentences earlier as section context.
TM FNs S122, S138, S141 = "the datastore"/"database operation" where GAE Datastore established earlier.
Pattern: component named once in paragraph opener → subsequent sentences use role-referential definite NP
(head-noun matches component name suffix). Neither pronoun-coref (wrong trigger) nor entity-extraction
(no alias match) captures these. LLM adds only +1.6pp recall above TransArc on BBB (vs +38.7pp for MS).

Designed axiom direction (DOC_KNOWLEDGE_EXTRACTION_RULES):
> "When exactly one component name has terminal word W (case-insensitive), treat sentence containing
> 'the W' / 'W-side' / 'W side' as candidate for that component, subject to seed disambiguation.
> Only activate when the same component appeared by full name in preceding ≤10 sentences."

**Gap 2 — Responsibility-list FPs: 7 FPs in TM**
TM S82, S83, S136, S159, S182 = bare gerund/nominal fragments ("Providing a mechanism...",
"Connecting to GAE-provided APIs...", "Represented by the Db classes.") that ILinker3 Pass B extracts
as actor seeds. Gold standard does NOT count responsibility bullets as trace links. Seed disambiguation
approves them because prior sentences establish component context.

Designed axiom direction (SEED_DISAMBIGUATION_RULES):
> "Reject when sentence is bare gerund phrase (verb-ing without explicit subject) or nominal fragment
> describing functional capability, AND contains no explicit mention of any OTHER named component.
> Such sentences describe internal responsibilities, not architectural references."

**Gap 3 — `has_standalone_mention` gate: empirically confirmed LOAD-BEARING, do not remove**
Empirical study: analyzed all 24 BBB pronoun sentences.
Result: 0/24 pronoun sentences are gold FNs. 9 `gated_alias` cases (alias in ±5 window but not full name)
are ALL non-gold → gate correctly blocks FPs. BBB's 15/16 name_absent FNs are non-pronoun sentences
(definite NPs, not pronouns) → coref never reaches them regardless of gate.

Gate at s_linker14_voyager.py:1004: `if not (has_standalone_mention(comp, ant_sent.text) or res.get("antecedent_via_alias", False))`
CORRECT behavior. Removing = net FP increase.

Real fix: extend COREF axiom prompt to instruct LLM to set `antecedent_via_alias=True` when antecedent
sentence contains a known alias (not just full canonical name). The path already exists in code, just
not exercised because axiom says "named verbatim."

Current COREF axiom: "Resolve only when one component is the unambiguous antecedent."
Better: add "When the antecedent sentence contains a known alias of the component (see alias list above),
set antecedent_via_alias=true."

**Distributional shift finding:**
Training data (MS, TS) have high gold/sent density (0.84, 0.63) = many explicit name occurrences.
BBB has 0.71 gold/sent but role-referential style; TM has 0.29 gold/sent = sparse, long doc.
All 9 bank slots are sentence-scoped. The remaining FNs require paragraph/section-scoped evidence.
BBB-TM correlation r=0.829 across 210 runs (strong). Both are hardest datasets (avg 0.794/0.804).
At failure-mode level they share SCN but manifest differently: BBB = FN under-inference, TM = FP over-inference.

## Solution

Three implementation tasks in priority order:

1. **Coref axiom fix** (Gap 3, lowest risk, no structural change):
   Edit `prompts_v3_axiom.py` COREF_RULES: add one sentence instructing LLM to set
   `antecedent_via_alias=True` when antecedent has known alias. Then test on BBB pronoun sents.
   Expected: recover some BBB coref TPs without new FPs (gate stays, alias-backed only).

2. **DOC_KNOWLEDGE_EXTRACTION_RULES axiom** (Gap 1, medium risk):
   Add role-suffix definite-NP candidate generation. Requires checking "exactly one component has
   suffix W" to avoid ambiguity. Route through existing seed-disambiguation for validation.
   Must be tested on datasets without role-suffix component names to verify non-firing.

3. **SEED_DISAMBIGUATION_RULES axiom** (Gap 2, medium risk):
   Add gerund/nominal-fragment rejection rule. Test: should reject S82/S83/S136 TM FPs,
   should NOT reject cross-reference gerunds like "Providing auth to the Storage component".
   Risky only if training data has legitimate gerund trace links (check MS/TS before deploying).

All three should be proposed as bank patterns in the distillation loop (not hardcoded axioms),
so they can be probation-tested and rolled back if they regress.
