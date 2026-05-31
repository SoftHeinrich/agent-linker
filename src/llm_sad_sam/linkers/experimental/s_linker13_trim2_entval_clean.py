"""S-Linker13 Trim2 — Phase 12 Step 2: ENTITY_EXTRACTION_RULES + VALIDATION_RULES
merged via Technique 3 (lossless rubric distillation).

REMOVED_FROM: s_linker13_clean (subclass override; zero edits to parent module).
RULES_REMOVED: 4 (14 → 10 rules; merged duplicated architectural-participant rubric)
KEEP: ["all 9 semantic categories", "Favor inclusion tie-breaker (extraction header)"]
CLEAN: ["shared core ENTVAL_MERGED_RUBRIC_V3 + role-specific extraction/validation headers"]

Background
----------
The Phase 11 prompt-harness survey (§5 row 2) identified ENTITY_EXTRACTION_RULES
and VALIDATION_RULES as structurally redundant: rule 1 of EXTRACTION ("name
appears directly") mirrors APPROVE-clause-1 of VALIDATION ("named as
architectural participant"); EXTRACTION exclude 2 ("ordinary English word")
mirrors REJECT clauses 1 + 2 of VALIDATION. Technique 3 (lossless rubric
distillation) collapses the overlapping boundary while preserving every
semantic category.

Original rule counts:
  ENTITY_EXTRACTION_RULES: 6 include + 2 exclude  = 8
  VALIDATION_RULES:        3 APPROVE + 3 REJECT   = 6
                                              total = 14

Merged rubric (ENTVAL_MERGED_RUBRIC_V3): 10 numbered rules covering:
  - 6 inclusion criteria (direct name / alias / compound / synonym /
    interaction participation / passive-or-prepositional phrase)
  - 4 exclusion criteria (dotted path / ordinary English word /
    name modifies noun phrase as type-label /
    algorithm-or-subprocess sharing the name but not the component)

Coverage preservation
---------------------
Every semantic category present in the originals appears in the merged rubric:
alias, synonym, compound, interaction, passive, prepositional, dotted, heading,
ordinary. The heading criterion is folded into the inclusion side (rule 3:
"sentence or its section heading describes what a specific component does
by name or role").

Design decision: rubric-shared / decision-divergent
---------------------------------------------------
ENTVAL_MERGED_RUBRIC_V3 is the SHARED CORE — it does not contain the
decision-format directive (include/approve/reject) for either consumer.
Two role-specific framings wrap the core:

  ENTITY_EXTRACTION_RULES_V3 = _EXTRACTION_HEADER + RUBRIC
    proposer framing; preserves "Favor inclusion over exclusion" tie-breaker
    (V31 phase-contribution analysis showed this is load-bearing).

  VALIDATION_RULES_V3 = _VALIDATION_HEADER + RUBRIC
    judge framing; "APPROVE if any inclusion criterion holds, REJECT if any
    exclusion criterion holds".

Both wrappers append the same rubric body, so a single source of truth
governs both prompts.

Override surface
----------------
Two methods are overridden via parent-module-scope monkey-patch with
try/finally (same pattern as 12-03):

  - _run_single_extraction_pass: rebinds ENTITY_EXTRACTION_RULES in
    s_linker13_clean module scope before super() call, restores after.
  - _validate_with_evidence: rebinds VALIDATION_RULES in s_linker13_clean
    module scope before super() call, restores after.

NON-THREAD-SAFE: the monkey-patch modifies the parent module's global
namespace. The variant is safe under the single-variant-single-dataset
sequential invocation pattern enforced by the 12-02 harness; the inner
_run_parallel DAG inside s_linker13_clean only parallelizes across distinct
phase functions (seed_val, coref, entity), not across multiple instances
of the same phase function. Tracked as T-12-04-03 in plan threat model.

Reviewer-defensibility
----------------------
All examples in the merged rubric come from safe SE-textbook domains
(observer pattern, pipeline stage, com.example.name — same surface terms
as prompts_v2 line 203). Zero benchmark-component leakage (GATE-06 probe
runs in the registration test).

Inheritance contract
--------------------
All other prompts and pipeline phases inherit from SLinker13Clean unchanged.
DOC_KNOWLEDGE_*, COREF_RULES, SEED_DISAMBIGUATION_RULES, ANTECEDENT_ALIAS_GUIDE,
ALIAS_SCOPE_SCHEMA, AMBIGUITY_* — all unchanged.
"""

from __future__ import annotations

from llm_sad_sam.linkers.experimental.s_linker13_clean import SLinker13Clean


# ─────────────────────────────────────────────────────────────────────────────
# Shared core — Technique 3 lossless rubric distillation
# ─────────────────────────────────────────────────────────────────────────────

#: Shared architectural-participant rubric. 10 numbered rules collapsing the
#: 14-rule duplication across ENTITY_EXTRACTION_RULES and VALIDATION_RULES.
#: Both consumer prompts wrap this body with a role-specific decision header.
ENTVAL_MERGED_RUBRIC_V3 = """A sentence references a software architecture component when ANY of the
following inclusion criteria hold:

1. The component name (or a known alias) appears directly in the sentence.
2. A space-separated form of the name appears as a compound (e.g.,
   "Memory Manager" → MemoryManager).
3. The sentence — or its section heading — describes what a specific component
   does by name or role, naming the component as its subject or as an
   architectural participant (performs an operation, provides or receives a
   service, is being configured, is explicitly introduced as part of the
   system).
4. A known synonym for the component is used.
5. The component participates in an interaction described in the sentence
   (as sender, receiver, or target) — e.g., "X sends data to Y" references
   BOTH X and Y.
6. The component is mentioned in a passive or prepositional phrase — e.g.,
   "data is stored in X", "handled by X", "via X", "through X".

A sentence does NOT reference the component when ANY of the following
exclusion criteria hold:

7. The name appears only inside a dotted path (e.g., com.example.name).
8. The name is used as an ordinary English (or ordinary technical) word
   rather than as a reference to this specific component — e.g., "proxy"
   in "proxy pattern" refers to a design pattern concept, not a Proxy
   component.
9. The name modifies a noun phrase without being a standalone architectural
   reference — e.g., "observer pattern", "pipeline stage" — the word
   describes a type, not the component.
10. The sentence describes an algorithm, subprocess, or implementation
    technique that shares the component's name but is not the component
    itself."""


# Role-specific framing wrappers. Each appends the shared core verbatim.

_EXTRACTION_HEADER = """RULES — include a reference when the criteria below indicate a component
reference; favor inclusion over exclusion — later verification will filter
borderline cases."""

_VALIDATION_HEADER = """DECISION RULES — for each candidate, APPROVE if any of the inclusion
criteria below indicates this sentence references the component as an
architectural participant; REJECT if any of the exclusion criteria applies
(the sentence uses the name in a non-component sense)."""


#: Extraction-side prompt fragment. Replaces prompts_v2.ENTITY_EXTRACTION_RULES
#: in the variant's override of _run_single_extraction_pass.
ENTITY_EXTRACTION_RULES_V3 = _EXTRACTION_HEADER + "\n\n" + ENTVAL_MERGED_RUBRIC_V3


#: Validation-side prompt fragment. Replaces prompts_v2.VALIDATION_RULES in
#: the variant's override of _validate_with_evidence.
VALIDATION_RULES_V3 = _VALIDATION_HEADER + "\n\n" + ENTVAL_MERGED_RUBRIC_V3


# ─────────────────────────────────────────────────────────────────────────────
# Variant class — subclass of SLinker13Clean
# ─────────────────────────────────────────────────────────────────────────────

class SLinker13Trim2EntvalClean(SLinker13Clean):
    """Step 2 trim variant: ENTITY_EXTRACTION_RULES + VALIDATION_RULES merged via Technique 3.

    Override surface:
      - _run_single_extraction_pass rebinds parent module's ENTITY_EXTRACTION_RULES
        to ENTITY_EXTRACTION_RULES_V3 for the duration of the super() call.
      - _validate_with_evidence rebinds parent module's VALIDATION_RULES to
        VALIDATION_RULES_V3 for the duration of the super() call.

    Rule count: 10 rules in shared core (down from 14 across the two originals).
    Coverage: every semantic category in the originals is represented.
    Decision design: rubric-shared / decision-divergent (each consumer prepends
    a role-specific header to the same shared core).

    All other prompts and pipeline phases inherit from SLinker13Clean unchanged.
    """

    _VARIANT_NAME = "s_linker13_trim2_entval_clean"

    def _run_single_extraction_pass(self, sentences, comp_names, mappings,
                                    name_to_id, sent_map, pass_label=""):
        import llm_sad_sam.linkers.experimental.s_linker13_clean as _parent_mod
        orig = _parent_mod.ENTITY_EXTRACTION_RULES
        try:
            _parent_mod.ENTITY_EXTRACTION_RULES = ENTITY_EXTRACTION_RULES_V3
            return super()._run_single_extraction_pass(
                sentences, comp_names, mappings, name_to_id, sent_map, pass_label)
        finally:
            _parent_mod.ENTITY_EXTRACTION_RULES = orig

    def _validate_with_evidence(self, candidates, bundles, components, sent_map):
        import llm_sad_sam.linkers.experimental.s_linker13_clean as _parent_mod
        orig = _parent_mod.VALIDATION_RULES
        try:
            _parent_mod.VALIDATION_RULES = VALIDATION_RULES_V3
            return super()._validate_with_evidence(
                candidates, bundles, components, sent_map)
        finally:
            _parent_mod.VALIDATION_RULES = orig


__all__ = [
    "SLinker13Trim2EntvalClean",
    "ENTVAL_MERGED_RUBRIC_V3",
    "ENTITY_EXTRACTION_RULES_V3",
    "VALIDATION_RULES_V3",
]
