"""Prompt constants — v5 (s_linker19, paper variant).

Single source of truth for every static prompt fragment used by s_linker19.
Byte-identical to s_linker17f's inlined prompts where the design is unchanged;
the one substantive edit is P1_FOCUS — extended by a trailing clause to
absorb Phase 4b's code-path-rejection role into twopass's
architectural-participation question.

The trailing clause was empirically validated by
experiment_4b_prompt_absorption.py + experiment_dotted_path_rename.py:
the chosen wording ("qualified-name identifier (e.g. a package- or
member-access path X.Y.Z)") catches 2 of 3 code-path FPs on gpt-5.4 and
1 of 3 on Claude Sonnet with 0 collateral damage on the 4-TP control set
(strict joint improvement over the original "dotted-path identifier"
wording, which catches 0 of 3 on Sonnet). The third FP ("logic" as the
head of a code-path list, with no surrounding dotted neighbours) escapes
on both backends — accepted as a small limitation; no prompt modification
can reasonably encode that nuance.

GATE-06 (BENCHMARK_TABOO): all rule text uses textbook SE domain terms.
Zero benchmark component names. Zero project-specific vocabulary.
"""
from __future__ import annotations


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Ambiguity classification (model_knowledge)
# ─────────────────────────────────────────────────────────────────────────────

AMBIGUITY_FEW_SHOT = ""

AMBIGUITY_RULES = """A name is ARCHITECTURAL when it identifies a specific role or mechanism. A name is AMBIGUOUS when ordinary technical writing about any system would use it generically without naming a specific component."""


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Document knowledge (alias discovery)
# ─────────────────────────────────────────────────────────────────────────────

DOC_KNOWLEDGE_EXTRACTION_RULES = """Find surface forms the document uses to refer to a single named component (introduced short forms, alternate names, or words of multi-word names when they alone clearly mean the full name). Reject terms whose ordinary English use dominates."""

DOC_KNOWLEDGE_JUDGE_EXAMPLES = ""

DOC_KNOWLEDGE_JUDGE_RULES = """An alias is valid when the document establishes an equivalence between a phrase and a single named component. An alias is invalid when the phrase is generic vocabulary, names the whole system, or names a different entity. An alias is also invalid when it names a grouping that encompasses multiple elements, because it identifies a grouping rather than a single named unit. When uncertain, prefer APPROVE."""

ALIAS_SCOPE_RULES = """For each alias, classify its SCOPE:
- "global": distinctive enough to unambiguously name the component anywhere in the document. Typical shapes: multi-word forms, hyphenated forms, CamelCase, all-caps abbreviations of length >= 2, or names beginning with an uppercase letter.
- "local": a single all-lowercase word overlapping with ordinary English vocabulary. Safe only where the surrounding context already establishes which component is being discussed.
Qualified-name fragments (package- or member-access paths of the form X.Y or X.Y.Z) are NOT aliases — do not include them."""


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Entity extraction (Framing C)
# ─────────────────────────────────────────────────────────────────────────────

ENTITY_EXTRACTION_RULES = """Include a reference when the sentence refers to the component by name, alias, or as a participant in a described interaction. Exclude when the name appears only inside a code-level path — even if the compound identifier is semantically related to the component — or as ordinary English with no architectural intent. Favor inclusion."""


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — Unified evidence-bundle twopass validation
# ─────────────────────────────────────────────────────────────────────────────

# P1_FOCUS — architectural participation, V3 + qualified-name rename.
# The trailing clause "and not just as a qualified-name identifier (e.g. a
# package- or member-access path X.Y.Z)" absorbs Phase 4b's code-path-
# rejection role into the participation question. The explicit X.Y.Z schema
# is the structural anchor that empirically carries the clause across LLM
# backends (see experiment_dotted_path_rename.py).
P1_FOCUS = (
    "Check architectural participation: does the sentence name this "
    "component as an architectural participant — performing operations, "
    "providing services, or taking part in the described system behavior, "
    "and not just as a qualified-name identifier (e.g. a package- or "
    "member-access path X.Y.Z)?"
)

P2_FOCUS = (
    "Check referential specificity: is the component name used to identify "
    "this specific architectural element, or does it serve as a generic "
    "technical term in this sentence?"
)

VALIDATION_RULES = """Approve when the sentence treats the component as an architectural participant, including matching entities. Reject when the matching word is generic, names a different entity, or describes a technique that merely shares the component's name."""


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 — Coreference resolution
# ─────────────────────────────────────────────────────────────────────────────

# COREF_VALIDATION_FOCUS — single-pass coref-focused validator.
# KEPT asymmetric (vs entity twopass) on principled grounds: anaphoric
# resolution asks a different epistemic question than name disambiguation.
# The narrower focus is empirically load-bearing (cleanup E experiment
# showed entity twopass leaks ~4 FPs on bigbluebutton coref).
COREF_VALIDATION_FOCUS = (
    "Check coref resolution: does the pronoun, 'it', 'they', 'the service', "
    "or similar noun phrase that refers back in this sentence actually refer to "
    "the named component as an architectural participant — performing "
    "operations, providing services, or being the grammatical topic of the "
    "sentence?"
)

COREF_RULES = """For each case, decide whether a pronoun or role-referential noun phrase in the target sentence refers back to a component named or aliased earlier in the context. Resolve when: (a) the component's name or a known alias appears in the surrounding context sentences, or (b) only one component has been introduced in the immediately preceding sentences — treat it as the section-established topic and resolve role-referential phrases ("it", "the module", "the service", "the component", "the system") to that topic even without a direct name repetition. Avoid resolving when two or more equally plausible antecedents exist. Known aliases include the terminal word(s) of a multi-word name, documented abbreviations, and alternate forms used in the document. When the antecedent sentence uses a known alias rather than the full canonical name, set antecedent_via_alias=true."""

ANTECEDENT_ALIAS_RULES = """For each resolution, set antecedent_via_alias:
- true:  the antecedent quote refers to the component by an ALIAS — a terminal word of a multi-word name, an abbreviation, a hyphenated form, or any documented alternate name rather than the canonical name listed in COMPONENTS.
- false: the antecedent quote uses the canonical name verbatim as listed in COMPONENTS.

Examples:
- COMPONENTS contains "TaskScheduler"; antecedent: "The scheduler queues jobs" -> true (uses terminal "scheduler", not canonical "TaskScheduler").
- COMPONENTS contains "TaskScheduler"; antecedent: "TaskScheduler queues jobs" -> false (canonical name verbatim).

Default to true when the antecedent form clearly differs from the canonical name but unambiguously identifies the component."""
