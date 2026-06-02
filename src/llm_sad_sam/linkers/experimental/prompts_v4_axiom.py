"""Prompt constants — v4 AXIOM (s_linker14_voyager, v2.6+).

Replaces prompts_v3_axiom for s_linker14_voyager. The v3 axiom file is kept
frozen for s_linker13_skill_learned_clean compatibility.

Changes from v3:
  COREF_RULES        — Gap 1 (SCN section-topic resolution) + Gap 3 (alias
                       form recognition): explicit section-established-topic
                       rule; terminal-word / abbreviation aliases clarified;
                       antecedent_via_alias default shifted toward true when
                       form clearly differs from canonical.
  SEED_DISAMBIGUATION_RULES — Gap 2 (gerund FPs): explicit self-referential
                       capability description = OTHER rule; cross-component
                       participant interaction = COMPONENT.
  ALIAS_SCOPE_RULES  — Axiomized from ALIAS_SCOPE_SCHEMA (was static string
                       in s_linker14_voyager.py). Now a first-class bank slot.
  ANTECEDENT_ALIAS_RULES — Axiomized from ANTECEDENT_ALIAS_GUIDE (was static
                       string in s_linker14_voyager.py). Now a first-class
                       bank slot with Gap 3 fix (less conservative default).

GATE-06 (BENCHMARK_TABOO)
--------------------------
All axiom text uses textbook SE domain terms. Zero benchmark component names.
Zero project-specific vocabulary. Taboo audit returns 0 hits.

PURPOSE
-------
Voyager v5 training harness (voyager_train_tlr_v5.py) wraps each axiom prompt
with a LEARNED PATTERNS block at inference time. Axioms are the floor; bank
patterns are layered on top at training time. Empty patterns = pure axiom floor.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — Model Analysis: Component Ambiguity Classification
# ═══════════════════════════════════════════════════════════════════════════════

AMBIGUITY_FEW_SHOT = """Example 1: Name = "Scheduler"
Sentence: "The Scheduler queues jobs and dispatches them to worker threads."
Classification: ARCHITECTURAL — "Scheduler" is the grammatical subject with a named role (queuing, dispatching). It identifies a specific mechanism, not a generic scheduling concept.

Example 2: Name = "Scheduler"
Sentence: "The system uses a scheduler-based approach to balance load across nodes."
Classification: AMBIGUOUS — "scheduler-based approach" describes a technique. Ordinary technical writing about any system would use "scheduler" here without naming a specific component."""


AMBIGUITY_RULES = """A name is ARCHITECTURAL when it identifies a specific role or mechanism. A name is AMBIGUOUS when ordinary technical writing about any system would use it generically without naming a specific component."""


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — Document Knowledge: Alias Discovery & Judging
# ═══════════════════════════════════════════════════════════════════════════════

DOC_KNOWLEDGE_EXTRACTION_RULES = """Find surface forms the document uses to refer to a single named component (introduced short forms, alternate names, or words of multi-word names when they alone clearly mean the full name). Reject terms whose ordinary English use dominates."""


DOC_KNOWLEDGE_JUDGE_EXAMPLES = """Example 1: Candidate = "Handler", Component = "RequestHandler"
Evidence: "The RequestHandler (hereafter Handler) processes incoming requests from clients."
Judgment: VALID — The document explicitly establishes "Handler" as an alternate name for RequestHandler via parenthetical definition. The alias is distinctive and scoped to one component.

Example 2: Candidate = "the system", Component = "CacheLayer"
Evidence: "The system stores frequently accessed records in the CacheLayer."
Judgment: INVALID — "the system" refers to the overall application, not to CacheLayer specifically. It names a different entity (the whole system) rather than establishing CacheLayer as an alias."""


DOC_KNOWLEDGE_JUDGE_RULES = """An alias is valid when the document establishes an equivalence between a phrase and a single named component. An alias is invalid when the phrase is generic vocabulary, names the whole system, or names a different entity. When uncertain, prefer APPROVE."""


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2 — Entity Extraction & Validation
# ═══════════════════════════════════════════════════════════════════════════════

ENTITY_EXTRACTION_RULES = """Include a reference when the sentence refers to the component by name, alias, or as a participant in a described interaction. Exclude when the name appears only inside a code-level path or as ordinary English with no architectural intent. Favor inclusion."""


VALIDATION_RULES = """Approve when the sentence treats the component as an architectural participant, including counterparts. Reject when the matching word is generic, names a different entity, or describes a technique that merely shares the component's name."""


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2 — Coreference Resolution
# (v4 changes: Gap 1 section-topic rule + Gap 3 alias form clarification)
# ═══════════════════════════════════════════════════════════════════════════════

COREF_RULES = """For each case, decide whether a pronoun or role-referential noun phrase in the target sentence refers back to a component named or aliased earlier in the context. Resolve when: (a) the component's name or a known alias appears in the surrounding context sentences, or (b) only one component has been introduced in the immediately preceding sentences — treat it as the section-established topic and resolve role-referential phrases ("it", "the module", "the service", "the component", "the system") to that topic even without a direct name repetition. Avoid resolving when two or more equally plausible antecedents exist. Known aliases include the terminal word(s) of a multi-word name, documented abbreviations, and alternate forms used in the document. When the antecedent sentence uses a known alias rather than the full canonical name, set antecedent_via_alias=true."""


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2 — Seed Reference Disambiguation
# (v4 change: Gap 2 explicit gerund / self-referential capability rule)
# ═══════════════════════════════════════════════════════════════════════════════

SEED_DISAMBIGUATION_RULES = """For each sentence, decide whether the matched name refers to the architectural component (COMPONENT) or carries a different meaning (OTHER: code identifier, technique sharing the name, sub-entity of a larger name, or ordinary English vocabulary). A sentence is COMPONENT when the matched name refers to the component — whether through behavior, interaction, role description, identity statement, or any architecturally meaningful mention. A sentence is OTHER only when the matched name is bare vocabulary, a code path fragment, or refers to a different entity that merely shares the name. When uncertain, choose COMPONENT."""


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2 — Alias Scope Classification
# (v4: axiomized from ALIAS_SCOPE_SCHEMA static string in s_linker14_voyager.py)
# ═══════════════════════════════════════════════════════════════════════════════

ALIAS_SCOPE_RULES = """For each alias, classify its SCOPE:
- "global": distinctive enough to unambiguously name the component anywhere in the document. Typical shapes: multi-word forms, hyphenated forms, CamelCase, all-caps abbreviations of length >= 2, or names beginning with an uppercase letter.
- "local": a single all-lowercase word overlapping with ordinary English vocabulary. Safe only where the surrounding context already establishes which component is being discussed.
Dotted-path fragments (tokens of the form X.Y or X.Y.Z) are NOT aliases — do not include them."""


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2 — Antecedent Alias Detection
# (v4: axiomized from ANTECEDENT_ALIAS_GUIDE; Gap 3 fix — less conservative default)
# ═══════════════════════════════════════════════════════════════════════════════

ANTECEDENT_ALIAS_RULES = """For each resolution, set antecedent_via_alias:
- true:  the antecedent quote refers to the component by an ALIAS — a terminal word of a multi-word name, an abbreviation, a hyphenated form, or any documented alternate name rather than the canonical name listed in COMPONENTS.
- false: the antecedent quote uses the canonical name verbatim as listed in COMPONENTS.

Examples:
- COMPONENTS contains "TaskScheduler"; antecedent: "The scheduler queues jobs" -> true (uses terminal "scheduler", not canonical "TaskScheduler").
- COMPONENTS contains "TaskScheduler"; antecedent: "TaskScheduler queues jobs" -> false (canonical name verbatim).

Default to true when the antecedent form clearly differs from the canonical name but unambiguously identifies the component."""


# ═══════════════════════════════════════════════════════════════════════════════
# 15-Slot Expansion — 4 remaining injectable slots
# Empty string default: behavior identical to v2.5 baseline when no patterns committed.
# ═══════════════════════════════════════════════════════════════════════════════

# ILinker4 seed extraction prompt prefix (injected via ILinker4 constructor)
SEED_EXTRACTION_RULES = """Find architectural components explicitly referenced in each sentence by canonical name, alias, abbreviation, or unambiguous partial form. Exclude tokens that appear only inside dotted code paths or as ordinary English with no architectural role. Favor inclusion; downstream disambiguation handles precision."""

# ILinker4 seed actor prompt prefix (injected via ILinker4 constructor)
SEED_ACTOR_RULES = """Find architectural components whose role, behavior, interaction, or identity the sentence describes. Report all participants, not only the grammatical subject. For single-word names overlapping with ordinary English, report when the sentence assigns a role, behavior, interaction, or identity statement to that component. Skip pronouns and role-referential phrases (resolved in coreference, not here)."""

# Generic-word-usage LLM filter additional rules (appended before Return JSON)
GENERIC_WORD_USAGE_RULES = ""

# Coref terminal-word specificity classification additional rules
COREF_TERMINAL_SPECIFICITY_RULES = ""
